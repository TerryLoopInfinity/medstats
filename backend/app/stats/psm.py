"""倾向性得分匹配（Propensity Score Matching）模块。

支持：
  - PS 估计：logistic 回归，分类协变量自动 dummy 编码
  - 匹配方法：最近邻（greedy）/ caliper 约束 / 最优（linear_sum_assignment）
  - 匹配比例：1:1 / 1:2 / 1:3，有/无放回
  - 匹配质量：SMD（前后对比）、方差比 VR
  - 处理效应：连续（Wilcoxon）/ 二分类（McNemar）/ 生存（分层 Cox）
  - 图表：PS 核密度图、Love plot、SMD 条形图、KM 曲线（生存结局）
"""

import logging
import warnings as _warnings_mod
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats

from app.models.analysis import AnalysisResult, ChartResult, TableResult

logger = logging.getLogger(__name__)

_PALETTE = ["#5470c6", "#ee6666", "#91cc75", "#fac858", "#73c0de", "#3ba272"]


def _safe_exp(x: float) -> float:
    """np.clip 防止 exp 溢出。"""
    return float(np.exp(np.clip(float(x), -500.0, 500.0)))


# ─────────────────────────────────────────────────────────────────────────────
# 主入口
# ─────────────────────────────────────────────────────────────────────────────

def run(df: pd.DataFrame, params: dict) -> AnalysisResult:
    """
    参数
    ----
    params["treatment_col"]    str           处理变量（二分类）
    params["covariates"]       list[str]     协变量（用于 PS 估计）
    params["outcome_col"]      str           结局变量（可选）
    params["outcome_type"]     str           "continuous" | "binary" | "survival"
    params["time_col"]         str           时间变量（survival 类型）
    params["event_col"]        str           事件变量（survival 类型）
    params["method"]           str           "nearest" | "caliper" | "optimal"
    params["caliper"]          float|None    caliper 值（None 时用 0.2*SD(PS)）
    params["ratio"]            int           匹配比例（1/2/3）
    params["with_replacement"] bool          是否放回匹配
    """
    treatment_col: str = str(params.get("treatment_col", ""))
    covariates: list[str] = list(params.get("covariates") or [])
    outcome_col: str = str(params.get("outcome_col", ""))
    outcome_type: str = str(params.get("outcome_type", "continuous")).lower()
    time_col: str = str(params.get("time_col", ""))
    event_col: str = str(params.get("event_col", ""))
    method: str = str(params.get("method", "nearest")).lower()
    caliper_param = params.get("caliper", None)
    ratio: int = max(1, min(3, int(params.get("ratio", 1))))
    with_replacement: bool = bool(params.get("with_replacement", False))
    warnings: list[str] = []

    # ── 参数校验 ──────────────────────────────────────────────────────────────
    if not treatment_col:
        raise ValueError("请指定处理变量 (treatment_col)")
    if treatment_col not in df.columns:
        raise ValueError(f"处理变量 '{treatment_col}' 不存在于数据集中")
    if not covariates:
        raise ValueError("请至少选择一个协变量 (covariates)")
    if method not in ("nearest", "caliper", "optimal"):
        raise ValueError("method 必须为 nearest / caliper / optimal")
    if outcome_type not in ("continuous", "binary", "survival"):
        raise ValueError("outcome_type 必须为 continuous / binary / survival")
    if outcome_type == "survival":
        if not time_col or not event_col:
            raise ValueError("生存结局需要指定 time_col 和 event_col")
        for c in [time_col, event_col]:
            if c not in df.columns:
                raise ValueError(f"列 '{c}' 不存在于数据集中")

    # ── 有效协变量 ───────────────────────────────────────────────────────────
    valid_covs = [c for c in covariates if c in df.columns]
    missing_covs = [c for c in covariates if c not in df.columns]
    if missing_covs:
        warnings.append(f"以下协变量不存在，已忽略：{', '.join(missing_covs)}")
    if not valid_covs:
        raise ValueError("没有有效的协变量")

    # ── 需要的列 ─────────────────────────────────────────────────────────────
    need_cols = [treatment_col] + valid_covs
    if outcome_col and outcome_col in df.columns:
        need_cols.append(outcome_col)
    if outcome_type == "survival":
        need_cols.extend([time_col, event_col])
    need_cols = list(dict.fromkeys(need_cols))  # 去重保序

    df_work = df[need_cols].dropna().copy()

    # ── 二值化处理变量 ────────────────────────────────────────────────────────
    treat_vals = df_work[treatment_col].unique()
    if len(treat_vals) != 2:
        raise ValueError(f"处理变量 '{treatment_col}' 必须为二分类（当前有 {len(treat_vals)} 个唯一值）")
    treat_sorted = sorted(treat_vals, key=str)
    treat_map = {treat_sorted[0]: 0, treat_sorted[1]: 1}
    df_work["_treat"] = df_work[treatment_col].map(treat_map)
    treat_label_0, treat_label_1 = str(treat_sorted[0]), str(treat_sorted[1])

    n_total = len(df_work)
    n_treat = int(df_work["_treat"].sum())
    n_ctrl = n_total - n_treat
    if n_treat < 10 or n_ctrl < 10:
        raise ValueError(f"处理组（{n_treat}）或对照组（{n_ctrl}）样本量不足 10，无法进行匹配")

    n_before = len(df[need_cols].dropna(how="any") if True else df)
    if len(df) - n_total > 0:
        warnings.append(f"删除 {len(df) - n_total} 行含缺失值的记录，剩余 {n_total} 行")

    # ── 1. 倾向性得分估计 ─────────────────────────────────────────────────────
    ps, feature_names, ps_warnings = _estimate_ps(df_work, "_treat", valid_covs)
    warnings.extend(ps_warnings)
    df_work["_ps"] = ps

    # ── caliper 计算 ─────────────────────────────────────────────────────────
    ps_sd = float(np.std(ps, ddof=1))
    if caliper_param is not None:
        caliper_val: float | None = float(caliper_param)
    elif method == "caliper":
        caliper_val = 0.2 * ps_sd
    else:
        caliper_val = None  # nearest/optimal 不强制 caliper（optimal 内部处理）

    # ── 2. 匹配 ──────────────────────────────────────────────────────────────
    if method == "optimal":
        df_matched, pairs = _match_optimal(df_work, "_treat", "_ps", caliper_val)
    else:
        df_matched, pairs = _match_greedy(df_work, "_treat", "_ps", ratio, with_replacement, caliper_val)

    n_matched_treat = int((df_matched["_treat"] == 1).sum())
    n_matched_ctrl = int((df_matched["_treat"] == 0).sum())
    match_rate = n_matched_treat / n_treat if n_treat > 0 else 0.0

    if n_matched_treat == 0:
        raise ValueError("匹配失败，未能匹配到任何处理组样本。请调宽 caliper 或检查数据重叠情况")
    if match_rate < 0.3:
        warnings.append(f"匹配成功率较低（{match_rate:.1%}），PS 分布重叠不足，结果解读需谨慎")

    # ── 3. 协变量平衡评估 ─────────────────────────────────────────────────────
    balance_rows, smd_before_list, smd_after_list, cov_labels = _balance_table(
        df_work, df_matched, "_treat", valid_covs, treat_label_0, treat_label_1
    )

    # ── 4. 处理效应估计 ───────────────────────────────────────────────────────
    effect_table: TableResult | None = None
    effect_summary = ""
    km_chart: ChartResult | None = None

    if outcome_col and outcome_col in df_matched.columns:
        if outcome_type == "continuous":
            effect_table, effect_summary = _effect_continuous(
                df_matched, "_treat", outcome_col, treat_label_0, treat_label_1
            )
        elif outcome_type == "binary":
            effect_table, effect_summary = _effect_binary(
                df_matched, "_treat", outcome_col, treat_label_0, treat_label_1
            )
        elif outcome_type == "survival":
            effect_table, effect_summary, km_chart = _effect_survival(
                df_matched, "_treat", time_col, event_col, treat_label_0, treat_label_1, warnings
            )

    # ── 5. 组装输出 ───────────────────────────────────────────────────────────
    tables: list[TableResult] = []
    charts: list[ChartResult] = []

    tables.append(_table_match_summary(
        n_total, n_treat, n_ctrl, n_matched_treat, n_matched_ctrl,
        len(pairs), match_rate, method, ratio, with_replacement,
        caliper_val, treat_label_0, treat_label_1
    ))
    tables.append(_table_ps_summary(df_work, df_matched, "_treat", "_ps", treat_label_0, treat_label_1))
    tables.append(TableResult(
        title="协变量平衡评估（匹配前 → 匹配后 SMD）",
        headers=["协变量", f"匹配前均值（{treat_label_1}）", f"匹配前均值（{treat_label_0}）",
                 "匹配前 SMD", f"匹配后均值（{treat_label_1}）", f"匹配后均值（{treat_label_0}）",
                 "匹配后 SMD", "VR（匹配后）", "平衡状态"],
        rows=balance_rows,
    ))
    if effect_table:
        tables.append(effect_table)

    charts.append(_chart_ps_kde(df_work, "_treat", "_ps", treat_label_0, treat_label_1, matched=False))
    charts.append(_chart_ps_kde(df_matched, "_treat", "_ps", treat_label_0, treat_label_1, matched=True))
    charts.append(_chart_love_plot(cov_labels, smd_before_list, smd_after_list))
    charts.append(_chart_smd_bar(cov_labels, smd_before_list, smd_after_list))
    if km_chart:
        charts.append(km_chart)

    # ── 汇总结论 ──────────────────────────────────────────────────────────────
    n_balanced = sum(1 for r in balance_rows if isinstance(r[-1], str) and r[-1].startswith("✓"))
    n_imbalance = sum(1 for r in balance_rows if isinstance(r[-1], str) and r[-1].startswith("✗"))
    summary = (
        f"共 {n_total} 例（{treat_label_1} {n_treat} 例，{treat_label_0} {n_ctrl} 例），"
        f"{'最优' if method == 'optimal' else '最近邻'} 1:{ratio} 匹配后获得 {len(pairs)} 对匹配对"
        f"（成功率 {match_rate:.1%}）。"
        f"{n_balanced}/{len(cov_labels)} 个协变量匹配后 SMD < 0.2。"
    )
    if effect_summary:
        summary += " " + effect_summary
    if n_imbalance > 0:
        warnings.append(f"{n_imbalance} 个协变量匹配后 SMD ≥ 0.2，建议检查 PS 模型或放宽匹配条件")

    return AnalysisResult(
        method="psm",
        tables=tables,
        charts=charts,
        summary=summary,
        warnings=warnings,
    )


# ─────────────────────────────────────────────────────────────────────────────
# 倾向性得分估计
# ─────────────────────────────────────────────────────────────────────────────

def _estimate_ps(
    df: pd.DataFrame,
    treat_col: str,
    covariates: list[str],
) -> tuple[np.ndarray, list[str], list[str]]:
    """用 logistic 回归估计 PS，返回 (ps_array, feature_names, warnings)。"""
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler

    warn_list: list[str] = []

    cov_df = df[covariates].copy()
    cat_cols = [c for c in covariates if cov_df[c].dtype == object or str(cov_df[c].dtype) == "category"]
    num_cols = [c for c in covariates if c not in cat_cols]

    if cat_cols:
        dummies = pd.get_dummies(cov_df[cat_cols], drop_first=True, dtype=float)
        X = pd.concat([cov_df[num_cols].astype(float), dummies], axis=1)
    else:
        X = cov_df[num_cols].astype(float)

    y = df[treat_col].values

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    with _warnings_mod.catch_warnings():
        _warnings_mod.simplefilter("ignore")
        clf = LogisticRegression(max_iter=1000, solver="lbfgs", C=1.0)
        clf.fit(X_scaled, y)

    ps = clf.predict_proba(X_scaled)[:, 1]

    ps_t = ps[y == 1]
    ps_c = ps[y == 0]
    if len(ps_t) and len(ps_c):
        ol_lo = max(ps_t.min(), ps_c.min())
        ol_hi = min(ps_t.max(), ps_c.max())
        if ol_lo >= ol_hi:
            warn_list.append("处理组与对照组 PS 分布无重叠区域，匹配结果不可靠")

    return ps, list(X.columns), warn_list


# ─────────────────────────────────────────────────────────────────────────────
# Greedy 最近邻匹配
# ─────────────────────────────────────────────────────────────────────────────

def _match_greedy(
    df: pd.DataFrame,
    treat_col: str,
    ps_col: str,
    ratio: int,
    with_replacement: bool,
    caliper: float | None,
) -> tuple[pd.DataFrame, list[tuple[int, list[int]]]]:
    """Greedy 最近邻匹配，返回 (匹配后 DataFrame, pairs)。"""
    treat_idx = df.index[df[treat_col] == 1].tolist()
    ctrl_idx = df.index[df[treat_col] == 0].tolist()
    ps_vals = df[ps_col].to_dict()

    rng = np.random.default_rng(42)
    order = rng.permutation(len(treat_idx))

    used_ctrl: set[int] = set()
    pairs: list[tuple[int, list[int]]] = []

    ctrl_ps = np.array([ps_vals[c] for c in ctrl_idx])

    for rank in order:
        ti = treat_idx[int(rank)]
        ps_t = ps_vals[ti]

        if with_replacement:
            avail_local = list(range(len(ctrl_idx)))
        else:
            avail_local = [i for i, c in enumerate(ctrl_idx) if c not in used_ctrl]

        if not avail_local:
            break

        avail_ps = ctrl_ps[avail_local]
        dists = np.abs(avail_ps - ps_t)
        sorted_ranks = np.argsort(dists)

        matched: list[int] = []
        for r in sorted_ranks:
            local_i = avail_local[r]
            if caliper is not None and dists[r] > caliper:
                break
            matched.append(ctrl_idx[local_i])
            if len(matched) >= ratio:
                break

        if matched:
            pairs.append((ti, matched))
            if not with_replacement:
                used_ctrl.update(matched)

    return _build_matched_df(df, pairs), pairs


# ─────────────────────────────────────────────────────────────────────────────
# 最优匹配（1:1，bipartite assignment）
# ─────────────────────────────────────────────────────────────────────────────

def _match_optimal(
    df: pd.DataFrame,
    treat_col: str,
    ps_col: str,
    caliper: float | None,
) -> tuple[pd.DataFrame, list[tuple[int, list[int]]]]:
    """最优 1:1 匹配（scipy linear_sum_assignment）。"""
    from scipy.optimize import linear_sum_assignment

    treat_idx = df.index[df[treat_col] == 1].tolist()
    ctrl_idx = df.index[df[treat_col] == 0].tolist()
    ps_vals = df[ps_col].to_dict()

    ps_t = np.array([ps_vals[i] for i in treat_idx])
    ps_c = np.array([ps_vals[i] for i in ctrl_idx])

    cost = np.abs(ps_t[:, None] - ps_c[None, :])
    _BIG = 1e9
    if caliper is not None:
        cost[cost > caliper] = _BIG

    row_ind, col_ind = linear_sum_assignment(cost)

    pairs: list[tuple[int, list[int]]] = []
    for ri, ci in zip(row_ind, col_ind):
        if cost[ri, ci] < _BIG:
            pairs.append((treat_idx[ri], [ctrl_idx[ci]]))

    return _build_matched_df(df, pairs), pairs


def _build_matched_df(
    df: pd.DataFrame,
    pairs: list[tuple[int, list[int]]],
) -> pd.DataFrame:
    if not pairs:
        result = df.iloc[0:0].copy()
        result["_match_id"] = pd.Series([], dtype=int)
        return result

    rows: list[dict] = []
    for match_id, (ti, controls) in enumerate(pairs):
        row_t = df.loc[ti].to_dict()
        row_t["_match_id"] = match_id
        rows.append(row_t)
        for ci in controls:
            row_c = df.loc[ci].to_dict()
            row_c["_match_id"] = match_id
            rows.append(row_c)

    return pd.DataFrame(rows).reset_index(drop=True)


# ─────────────────────────────────────────────────────────────────────────────
# 协变量平衡评估
# ─────────────────────────────────────────────────────────────────────────────

def _smd(x_t: np.ndarray, x_c: np.ndarray) -> float:
    if len(x_t) < 2 or len(x_c) < 2:
        return float("nan")
    m_t, m_c = np.nanmean(x_t), np.nanmean(x_c)
    v_t = float(np.nanvar(x_t, ddof=1))
    v_c = float(np.nanvar(x_c, ddof=1))
    pooled = np.sqrt((v_t + v_c) / 2)
    if pooled < 1e-12:
        return 0.0
    return float(abs(m_t - m_c) / pooled)


def _vr(x_t: np.ndarray, x_c: np.ndarray) -> float:
    if len(x_t) < 2 or len(x_c) < 2:
        return float("nan")
    v_t = float(np.nanvar(x_t, ddof=1))
    v_c = float(np.nanvar(x_c, ddof=1))
    if v_c < 1e-12:
        return float("nan")
    return float(v_t / v_c)


def _balance_table(
    df_before: pd.DataFrame,
    df_after: pd.DataFrame,
    treat_col: str,
    covariates: list[str],
    label0: str,
    label1: str,
) -> tuple[list[list], list[float], list[float], list[str]]:
    rows: list[list] = []
    smd_before_list: list[float] = []
    smd_after_list: list[float] = []
    cov_labels: list[str] = []

    treat_b = df_before[treat_col] == 1
    ctrl_b = ~treat_b
    treat_a = df_after[treat_col] == 1 if len(df_after) > 0 else pd.Series(dtype=bool)
    ctrl_a = ~treat_a if len(df_after) > 0 else pd.Series(dtype=bool)

    for col in covariates:
        if col not in df_before.columns:
            continue

        s_b = df_before[col]
        is_cat = s_b.dtype == object or str(s_b.dtype) == "category"

        if is_cat:
            levels = sorted(s_b.dropna().unique(), key=str)
            # 类别头行
            rows.append([col, "", "", "", "", "", "", "", ""])
            for lv in levels:
                lv_label = f"{col} = {lv}"
                ind_b = (s_b == lv).astype(float)

                x_t_b = ind_b[treat_b].values
                x_c_b = ind_b[ctrl_b].values
                m_t_b = float(np.mean(x_t_b))
                m_c_b = float(np.mean(x_c_b))
                s_b_val = _smd(x_t_b, x_c_b)

                if col in df_after.columns and len(df_after) > 0:
                    ind_a = (df_after[col] == lv).astype(float)
                    x_t_a = ind_a[treat_a].values
                    x_c_a = ind_a[ctrl_a].values
                    m_t_a = float(np.mean(x_t_a)) if len(x_t_a) else float("nan")
                    m_c_a = float(np.mean(x_c_a)) if len(x_c_a) else float("nan")
                    s_a_val = _smd(x_t_a, x_c_a)
                    vr_a_val = _vr(x_t_a, x_c_a)
                else:
                    m_t_a = m_c_a = s_a_val = vr_a_val = float("nan")

                balance = _balance_label(s_a_val)
                rows.append([
                    f"  {lv_label}",
                    f"{m_t_b:.3f}", f"{m_c_b:.3f}", f"{s_b_val:.3f}" if not np.isnan(s_b_val) else "—",
                    f"{m_t_a:.3f}" if not np.isnan(m_t_a) else "—",
                    f"{m_c_a:.3f}" if not np.isnan(m_c_a) else "—",
                    f"{s_a_val:.3f}" if not np.isnan(s_a_val) else "—",
                    f"{vr_a_val:.3f}" if not np.isnan(vr_a_val) else "—",
                    balance,
                ])
                smd_before_list.append(s_b_val if not np.isnan(s_b_val) else 0.0)
                smd_after_list.append(s_a_val if not np.isnan(s_a_val) else 0.0)
                cov_labels.append(lv_label)

        else:
            x_t_b = df_before.loc[treat_b, col].astype(float).values
            x_c_b = df_before.loc[ctrl_b, col].astype(float).values
            m_t_b = float(np.nanmean(x_t_b))
            m_c_b = float(np.nanmean(x_c_b))
            s_b_val = _smd(x_t_b, x_c_b)

            if col in df_after.columns and len(df_after) > 0:
                x_t_a = df_after.loc[treat_a, col].astype(float).values
                x_c_a = df_after.loc[ctrl_a, col].astype(float).values
                m_t_a = float(np.nanmean(x_t_a)) if len(x_t_a) else float("nan")
                m_c_a = float(np.nanmean(x_c_a)) if len(x_c_a) else float("nan")
                s_a_val = _smd(x_t_a, x_c_a)
                vr_a_val = _vr(x_t_a, x_c_a)
            else:
                m_t_a = m_c_a = s_a_val = vr_a_val = float("nan")

            balance = _balance_label(s_a_val)
            rows.append([
                col,
                f"{m_t_b:.3f}", f"{m_c_b:.3f}", f"{s_b_val:.3f}" if not np.isnan(s_b_val) else "—",
                f"{m_t_a:.3f}" if not np.isnan(m_t_a) else "—",
                f"{m_c_a:.3f}" if not np.isnan(m_c_a) else "—",
                f"{s_a_val:.3f}" if not np.isnan(s_a_val) else "—",
                f"{vr_a_val:.3f}" if not np.isnan(vr_a_val) else "—",
                balance,
            ])
            smd_before_list.append(s_b_val if not np.isnan(s_b_val) else 0.0)
            smd_after_list.append(s_a_val if not np.isnan(s_a_val) else 0.0)
            cov_labels.append(col)

    return rows, smd_before_list, smd_after_list, cov_labels


def _balance_label(smd: float) -> str:
    if np.isnan(smd):
        return "—"
    if smd < 0.1:
        return "✓ 良好"
    if smd < 0.2:
        return "✓ 可接受"
    return "✗ 失衡"


# ─────────────────────────────────────────────────────────────────────────────
# 处理效应估计
# ─────────────────────────────────────────────────────────────────────────────

def _effect_continuous(
    df_m: pd.DataFrame,
    treat_col: str,
    outcome_col: str,
    label0: str,
    label1: str,
) -> tuple[TableResult, str]:
    """配对样本 Wilcoxon 符号秩检验（连续结局）。"""
    treat_vals: dict[int, float] = {}
    ctrl_vals: dict[int, list[float]] = {}
    for _, row in df_m.iterrows():
        mid = int(row["_match_id"])
        v = float(row[outcome_col])
        if row[treat_col] == 1:
            treat_vals[mid] = v
        else:
            ctrl_vals.setdefault(mid, []).append(v)

    diffs = [
        treat_vals[mid] - float(np.mean(ctrl_vals[mid]))
        for mid in treat_vals
        if mid in ctrl_vals
    ]

    if len(diffs) < 5:
        return TableResult(
            title="处理效应（连续结局）",
            headers=["指标", "值"],
            rows=[["配对数量不足（< 5 对），无法检验", "—"]],
        ), ""

    diffs_arr = np.array(diffs)
    mean_diff = float(np.mean(diffs_arr))
    se_diff = float(np.std(diffs_arr, ddof=1) / np.sqrt(len(diffs_arr)))
    ci_lo = mean_diff - 1.96 * se_diff
    ci_hi = mean_diff + 1.96 * se_diff

    with _warnings_mod.catch_warnings():
        _warnings_mod.simplefilter("ignore")
        stat, p = stats.wilcoxon(diffs_arr, alternative="two-sided")

    p_str = "< 0.001" if p < 0.001 else f"{p:.3f}"
    table = TableResult(
        title="处理效应估计（连续结局 — 配对 Wilcoxon 检验）",
        headers=["指标", f"{label1} vs {label0}", "95% CI", "统计量", "p 值"],
        rows=[[
            "均值差（处理 − 对照）",
            f"{mean_diff:.3f}",
            f"[{ci_lo:.3f}, {ci_hi:.3f}]",
            f"W = {stat:.1f}",
            p_str,
        ]],
    )
    direction = "高于" if mean_diff > 0 else "低于"
    summary = (
        f"匹配后 {label1} 组结局均值{direction} {label0} 组 {abs(mean_diff):.3f}"
        f"（Wilcoxon p {p_str}）。"
    )
    return table, summary


def _effect_binary(
    df_m: pd.DataFrame,
    treat_col: str,
    outcome_col: str,
    label0: str,
    label1: str,
) -> tuple[TableResult, str]:
    """McNemar 检验（二分类结局）。"""
    ev = df_m[outcome_col].dropna().unique()
    df_m = df_m.copy()
    if len(ev) == 2:
        ev_sorted = sorted(ev, key=str)
        df_m["_outbin"] = df_m[outcome_col].map({ev_sorted[0]: 0, ev_sorted[1]: 1})
    elif pd.api.types.is_numeric_dtype(df_m[outcome_col]):
        df_m["_outbin"] = (df_m[outcome_col] != 0).astype(int)
    else:
        return TableResult(
            title="处理效应（二分类结局）",
            headers=["指标", "值"],
            rows=[["结局变量无法识别为二分类", "—"]],
        ), ""

    pairs_t: dict[int, int] = {}
    pairs_c: dict[int, int] = {}
    for _, row in df_m.iterrows():
        mid = int(row["_match_id"])
        v = int(row["_outbin"])
        if row[treat_col] == 1:
            pairs_t[mid] = v
        elif mid not in pairs_c:
            pairs_c[mid] = v

    a = b = c = d = 0
    for mid in pairs_t:
        if mid not in pairs_c:
            continue
        tv, cv = pairs_t[mid], pairs_c[mid]
        if tv == 1 and cv == 1:
            a += 1
        elif tv == 1 and cv == 0:
            b += 1
        elif tv == 0 and cv == 1:
            c += 1
        else:
            d += 1

    n_pairs = a + b + c + d
    p_treat = (a + b) / n_pairs if n_pairs else float("nan")
    p_ctrl = (a + c) / n_pairs if n_pairs else float("nan")
    cond_or = (b / c) if c > 0 else float("nan")

    if b + c == 0:
        chi2, mcnemar_p = 0.0, 1.0
    else:
        chi2 = float((abs(b - c) - 1) ** 2 / (b + c))
        mcnemar_p = float(stats.chi2.sf(chi2, df=1))

    p_str = "< 0.001" if mcnemar_p < 0.001 else f"{mcnemar_p:.3f}"
    table = TableResult(
        title="处理效应估计（二分类结局 — McNemar 检验）",
        headers=["指标", "值"],
        rows=[
            [f"事件率（{label1}）", f"{p_treat:.3f}" if not np.isnan(p_treat) else "—"],
            [f"事件率（{label0}）", f"{p_ctrl:.3f}" if not np.isnan(p_ctrl) else "—"],
            ["配对一致（两组均阳）", str(a)],
            ["配对一致（两组均阴）", str(d)],
            ["不一致（处理+ 对照−）", str(b)],
            ["不一致（处理− 对照+）", str(c)],
            ["条件 OR（b/c）", f"{cond_or:.3f}" if not np.isnan(cond_or) else "—"],
            ["McNemar χ²", f"{chi2:.3f}"],
            ["p 值", p_str],
        ],
    )
    summary = (
        f"匹配后 {label1} 事件率 {p_treat:.1%} vs {label0} {p_ctrl:.1%}"
        f"，McNemar p {p_str}。"
    )
    return table, summary


def _effect_survival(
    df_m: pd.DataFrame,
    treat_col: str,
    time_col: str,
    event_col: str,
    label0: str,
    label1: str,
    warnings: list[str],
) -> tuple[TableResult, str, "ChartResult | None"]:
    """分层 Cox 回归 + Log-rank（生存结局）。"""
    try:
        from lifelines import CoxPHFitter, KaplanMeierFitter
        from lifelines.statistics import logrank_test
    except ImportError as e:
        raise RuntimeError("lifelines 未安装") from e

    df_s = df_m[[treat_col, time_col, event_col, "_match_id"]].copy()
    df_s[time_col] = pd.to_numeric(df_s[time_col], errors="coerce")
    df_s[event_col] = pd.to_numeric(df_s[event_col], errors="coerce")
    df_s = df_s.dropna()
    df_s = df_s[df_s[time_col] > 0]
    df_s = df_s[df_s[event_col].isin([0, 1])]

    if len(df_s) < 10:
        return TableResult(
            title="处理效应（生存结局）",
            headers=["指标", "值"],
            rows=[["有效数据不足，无法进行生存分析", "—"]],
        ), "", None

    t1 = df_s.loc[df_s[treat_col] == 1, time_col].values
    e1 = df_s.loc[df_s[treat_col] == 1, event_col].values
    t0 = df_s.loc[df_s[treat_col] == 0, time_col].values
    e0 = df_s.loc[df_s[treat_col] == 0, event_col].values

    with _warnings_mod.catch_warnings():
        _warnings_mod.simplefilter("ignore")
        lr = logrank_test(t1, t0, event_observed_A=e1, event_observed_B=e0)
    lr_p = float(lr.p_value)
    lr_p_str = "< 0.001" if lr_p < 0.001 else f"{lr_p:.3f}"

    hr = hr_lo = hr_hi = float("nan")
    cox_p_str = "—"
    try:
        with _warnings_mod.catch_warnings():
            _warnings_mod.simplefilter("ignore")
            cph = CoxPHFitter()
            cph.fit(df_s, duration_col=time_col, event_col=event_col,
                    formula=treat_col, strata="_match_id")
        beta = float(cph.params_[treat_col])
        se = float(cph.standard_errors_[treat_col])
        hr = _safe_exp(beta)
        hr_lo = _safe_exp(beta - 1.96 * se)
        hr_hi = _safe_exp(beta + 1.96 * se)
        cox_p = float(cph.summary["p"][treat_col])
        cox_p_str = "< 0.001" if cox_p < 0.001 else f"{cox_p:.3f}"
    except Exception as exc:
        warnings.append(f"分层 Cox 回归失败（{exc}），仅报告 Log-rank 结果")

    table = TableResult(
        title="处理效应估计（生存结局 — 分层 Cox + Log-rank）",
        headers=["分析方法", "效应量", "95% CI", "p 值"],
        rows=[
            [
                f"分层 Cox 回归（HR，{label1} vs {label0}）",
                f"{hr:.3f}" if not np.isnan(hr) else "—",
                f"[{hr_lo:.3f}, {hr_hi:.3f}]" if not np.isnan(hr) else "—",
                cox_p_str,
            ],
            [
                "Log-rank 检验",
                f"χ² = {lr.test_statistic:.3f}",
                "—",
                lr_p_str,
            ],
        ],
    )
    hr_str = f"{hr:.3f}" if not np.isnan(hr) else "—"
    summary = f"匹配后分层 Cox 回归 HR = {hr_str}（p {cox_p_str}），Log-rank p {lr_p_str}。"

    km_chart = _chart_km_matched(df_s, treat_col, time_col, event_col, label0, label1)
    return table, summary, km_chart


# ─────────────────────────────────────────────────────────────────────────────
# 汇总表格
# ─────────────────────────────────────────────────────────────────────────────

def _table_match_summary(
    n_total: int, n_treat: int, n_ctrl: int,
    n_mt: int, n_mc: int, n_pairs: int, match_rate: float,
    method: str, ratio: int, with_replacement: bool,
    caliper: float | None, label0: str, label1: str,
) -> TableResult:
    method_label = {
        "nearest": "最近邻（Greedy）",
        "caliper": "Caliper 约束最近邻",
        "optimal": "最优匹配（bipartite）",
    }.get(method, method)
    caliper_str = f"{caliper:.4f}" if caliper is not None else "无"
    return TableResult(
        title="匹配摘要",
        headers=["指标", "值"],
        rows=[
            ["匹配方法", method_label],
            ["匹配比例", f"1:{ratio}"],
            ["放回匹配", "是" if with_replacement else "否"],
            ["Caliper 值", caliper_str],
            [f"匹配前 — {label1}（处理组）", str(n_treat)],
            [f"匹配前 — {label0}（对照组）", str(n_ctrl)],
            ["匹配后 — 配对数", str(n_pairs)],
            [f"匹配后 — {label1}（处理组）", str(n_mt)],
            [f"匹配后 — {label0}（对照组）", str(n_mc)],
            ["匹配成功率（处理组）", f"{match_rate:.1%}"],
        ],
    )


def _table_ps_summary(
    df_before: pd.DataFrame,
    df_after: pd.DataFrame,
    treat_col: str,
    ps_col: str,
    label0: str,
    label1: str,
) -> TableResult:
    def desc(arr: np.ndarray) -> str:
        if len(arr) == 0:
            return "—"
        return f"{np.mean(arr):.3f} ± {np.std(arr, ddof=1):.3f}"

    ps_b_t = df_before.loc[df_before[treat_col] == 1, ps_col].values
    ps_b_c = df_before.loc[df_before[treat_col] == 0, ps_col].values

    ol_lo = max(ps_b_t.min(), ps_b_c.min()) if len(ps_b_t) and len(ps_b_c) else float("nan")
    ol_hi = min(ps_b_t.max(), ps_b_c.max()) if len(ps_b_t) and len(ps_b_c) else float("nan")
    overlap = f"[{ol_lo:.3f}, {ol_hi:.3f}]" if not np.isnan(ol_lo) and ol_lo < ol_hi else "无重叠"

    rows: list[list] = [
        [f"匹配前 {label1}（均值 ± SD）", desc(ps_b_t)],
        [f"匹配前 {label0}（均值 ± SD）", desc(ps_b_c)],
        ["PS 重叠区间（匹配前）", overlap],
    ]
    if ps_col in df_after.columns and len(df_after) > 0:
        ps_a_t = df_after.loc[df_after[treat_col] == 1, ps_col].values
        ps_a_c = df_after.loc[df_after[treat_col] == 0, ps_col].values
        rows += [
            [f"匹配后 {label1}（均值 ± SD）", desc(ps_a_t)],
            [f"匹配后 {label0}（均值 ± SD）", desc(ps_a_c)],
        ]
    return TableResult(title="倾向性得分分布摘要", headers=["指标", "值"], rows=rows)


# ─────────────────────────────────────────────────────────────────────────────
# 图表
# ─────────────────────────────────────────────────────────────────────────────

def _kde_series(values: np.ndarray, x_min: float, x_max: float, n: int = 100) -> list[list[float]]:
    """返回 ECharts line 所需 [[x, y], ...] 格式。"""
    if len(values) < 3:
        return []
    try:
        kde = stats.gaussian_kde(values, bw_method="scott")
        xs = np.linspace(x_min, x_max, n)
        ys = kde(xs)
        return [[round(float(x), 4), round(float(y), 6)] for x, y in zip(xs, ys)]
    except Exception:
        return []


def _chart_ps_kde(
    df: pd.DataFrame,
    treat_col: str,
    ps_col: str,
    label0: str,
    label1: str,
    matched: bool,
) -> ChartResult:
    prefix = "匹配后" if matched else "匹配前"
    if ps_col not in df.columns or len(df) == 0:
        return ChartResult(title=f"PS 分布（{prefix}）", chart_type="line", option={})

    ps_t = df.loc[df[treat_col] == 1, ps_col].values
    ps_c = df.loc[df[treat_col] == 0, ps_col].values
    all_ps = np.concatenate([ps_t, ps_c])
    x_lo = max(0.0, float(all_ps.min()) - 0.02)
    x_hi = min(1.0, float(all_ps.max()) + 0.02)

    return ChartResult(
        title=f"PS 核密度分布（{prefix}）",
        chart_type="line",
        option={
            "title": {"text": f"倾向性得分核密度分布（{prefix}）", "left": "center", "textStyle": {"fontSize": 14}},
            "tooltip": {"trigger": "axis"},
            "legend": {"bottom": 0},
            "grid": {"left": "12%", "right": "5%", "bottom": "15%"},
            "xAxis": {"type": "value", "name": "倾向性得分（PS）", "min": round(x_lo, 3), "max": round(x_hi, 3)},
            "yAxis": {"type": "value", "name": "核密度"},
            "series": [
                {
                    "name": label1,
                    "type": "line",
                    "data": _kde_series(ps_t, x_lo, x_hi),
                    "smooth": True,
                    "lineStyle": {"color": _PALETTE[0], "width": 2},
                    "areaStyle": {"color": _PALETTE[0], "opacity": 0.18},
                    "symbol": "none",
                },
                {
                    "name": label0,
                    "type": "line",
                    "data": _kde_series(ps_c, x_lo, x_hi),
                    "smooth": True,
                    "lineStyle": {"color": _PALETTE[1], "width": 2},
                    "areaStyle": {"color": _PALETTE[1], "opacity": 0.18},
                    "symbol": "none",
                },
            ],
        },
    )


def _chart_love_plot(
    cov_labels: list[str],
    smd_before: list[float],
    smd_after: list[float],
) -> ChartResult:
    """Love plot：协变量绝对 SMD 水平点图（匹配前后对比）。"""
    order = sorted(range(len(cov_labels)), key=lambda i: smd_before[i], reverse=True)
    labels = [cov_labels[i] for i in order]
    before = [round(float(smd_before[i]), 4) for i in order]
    after = [round(float(smd_after[i]), 4) for i in order]
    x_max = max(max(before, default=0.3), 0.3) * 1.15

    return ChartResult(
        title="Love Plot — 协变量平衡",
        chart_type="scatter",
        option={
            "title": {"text": "Love Plot — 协变量 |SMD|（匹配前 vs 匹配后）", "left": "center", "textStyle": {"fontSize": 14}},
            "tooltip": {"trigger": "item", "formatter": "{a}<br/>|SMD| = {c0}"},
            "legend": {"bottom": 0},
            "grid": {"left": "30%", "right": "8%", "bottom": "12%", "top": "50px"},
            "xAxis": {
                "type": "value",
                "name": "|SMD|",
                "min": 0,
                "max": round(x_max, 2),
                "splitLine": {"lineStyle": {"type": "dashed"}},
            },
            "yAxis": {
                "type": "category",
                "data": labels,
                "axisLabel": {"fontSize": 11},
            },
            "series": [
                {
                    "name": "匹配前",
                    "type": "scatter",
                    "data": [[b, i] for i, b in enumerate(before)],
                    "symbolSize": 13,
                    "symbol": "emptyCircle",
                    "itemStyle": {"color": _PALETTE[1], "borderWidth": 2},
                    "markLine": {
                        "symbol": ["none", "none"],
                        "label": {"formatter": "SMD=0.1", "position": "end", "fontSize": 10},
                        "data": [
                            {"xAxis": 0.1, "lineStyle": {"type": "dashed", "color": "#f0ad4e", "width": 2}},
                        ],
                    },
                },
                {
                    "name": "匹配后",
                    "type": "scatter",
                    "data": [[a, i] for i, a in enumerate(after)],
                    "symbolSize": 13,
                    "symbol": "circle",
                    "itemStyle": {"color": _PALETTE[0]},
                },
            ],
        },
    )


def _chart_smd_bar(
    cov_labels: list[str],
    smd_before: list[float],
    smd_after: list[float],
) -> ChartResult:
    """协变量匹配前后 |SMD| 分组水平条形图。"""
    return ChartResult(
        title="协变量平衡 — SMD 对比",
        chart_type="bar",
        option={
            "title": {"text": "协变量匹配前后 |SMD| 对比", "left": "center", "textStyle": {"fontSize": 14}},
            "tooltip": {"trigger": "axis", "axisPointer": {"type": "shadow"}},
            "legend": {"bottom": 0},
            "grid": {"left": "30%", "right": "8%", "bottom": "15%", "top": "50px"},
            "xAxis": {"type": "value", "name": "|SMD|", "min": 0},
            "yAxis": {"type": "category", "data": cov_labels, "axisLabel": {"fontSize": 11}},
            "series": [
                {
                    "name": "匹配前",
                    "type": "bar",
                    "data": [round(float(v), 4) for v in smd_before],
                    "itemStyle": {"color": _PALETTE[1], "opacity": 0.7},
                    "markLine": {
                        "symbol": ["none", "none"],
                        "label": {"formatter": "0.1"},
                        "data": [{"xAxis": 0.1, "lineStyle": {"type": "dashed", "color": "#f0ad4e", "width": 2}}],
                    },
                },
                {
                    "name": "匹配后",
                    "type": "bar",
                    "data": [round(float(v), 4) for v in smd_after],
                    "itemStyle": {"color": _PALETTE[0], "opacity": 0.85},
                },
            ],
        },
    )


def _chart_km_matched(
    df_s: pd.DataFrame,
    treat_col: str,
    time_col: str,
    event_col: str,
    label0: str,
    label1: str,
) -> ChartResult:
    """匹配后 KM 生存曲线。"""
    try:
        from lifelines import KaplanMeierFitter
    except ImportError:
        return ChartResult(title="KM 曲线", chart_type="kaplan_meier", option={})

    series = []
    nar_data = []
    for grp_val, grp_label, color in [(1, label1, _PALETTE[0]), (0, label0, _PALETTE[1])]:
        mask = df_s[treat_col] == grp_val
        t = df_s.loc[mask, time_col].values
        e = df_s.loc[mask, event_col].values
        if len(t) == 0:
            continue

        with _warnings_mod.catch_warnings():
            _warnings_mod.simplefilter("ignore")
            kmf = KaplanMeierFitter()
            kmf.fit(t, e, label=grp_label)

        timeline = kmf.timeline
        surv = kmf.survival_function_[grp_label].values

        pts: list[list[float]] = []
        for i, (x, y) in enumerate(zip(timeline, surv)):
            if i > 0:
                pts.append([float(x), round(float(surv[i - 1]), 4)])
            pts.append([float(x), round(float(y), 4)])

        series.append({
            "name": grp_label,
            "type": "line",
            "step": "end",
            "data": pts,
            "lineStyle": {"color": color, "width": 2},
            "symbol": "none",
        })

        time_pts = np.linspace(0, float(timeline.max()), 6).tolist()
        nar_data.append({
            "group": grp_label,
            "times": [round(tp, 1) for tp in time_pts],
            "counts": [int(np.sum(t >= tp)) for tp in time_pts],
        })

    return ChartResult(
        title="匹配后 KM 生存曲线",
        chart_type="kaplan_meier",
        option={
            "title": {"text": "匹配后 Kaplan-Meier 生存曲线", "left": "center", "textStyle": {"fontSize": 14}},
            "tooltip": {"trigger": "axis"},
            "legend": {"bottom": 0},
            "grid": {"left": "12%", "right": "5%", "bottom": "15%"},
            "xAxis": {"type": "value", "name": "时间", "min": 0},
            "yAxis": {"type": "value", "name": "生存概率", "min": 0, "max": 1},
            "series": series,
            "numberAtRisk": nar_data,
        },
    )
