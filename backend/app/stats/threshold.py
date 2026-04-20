"""阈值效应分析模块（分段线性回归 + 最优拐点检测）。

支持：
  - 模型类型：logistic / linear / cox
  - 网格搜索最优拐点（最大化对数似然）
  - Bootstrap 95% CI（100 次重抽样）
  - 似然比检验（非线性检验）
  - 拐点两侧效应量（OR / HR / β）及 95% CI
  - 图表：阈值效应曲线 + 对数似然曲线
"""

import logging
import math
import warnings as _warnings_mod
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats

from app.models.analysis import AnalysisResult, ChartResult, TableResult

logger = logging.getLogger(__name__)

# 防止 exp 溢出
_CLIP = 500.0


def _safe_exp(x: float) -> float:
    return float(np.exp(np.clip(float(x), -_CLIP, _CLIP)))


def _fmt_p(p: float) -> str:
    if math.isnan(p) or math.isinf(p):
        return "—"
    if p < 0.001:
        return "< 0.001"
    return f"{p:.3f}"


# ─────────────────────────────────────────────────────────────────────────────
# 主入口
# ─────────────────────────────────────────────────────────────────────────────

def run(df: pd.DataFrame, params: dict) -> AnalysisResult:
    """
    参数
    ----
    params["model_type"]    str             "logistic" | "cox" | "linear"
    params["outcome"]       str             因变量（logistic/linear）
    params["time_col"]      str             时间变量（cox）
    params["event_col"]     str             事件变量（cox）
    params["exposure"]      str             连续型暴露变量
    params["covariates"]    list[str]       调整协变量
    params["search_range"]  list[float,2]   拐点搜索范围，默认 [P10, P90]
    params["n_steps"]       int             候选拐点数量，默认 100
    params["n_bootstrap"]   int             Bootstrap 次数，默认 100
    """
    model_type: str = str(params.get("model_type", "logistic")).lower()
    exposure: str = str(params.get("exposure", ""))
    covariates: list[str] = list(params.get("covariates") or [])
    search_range = params.get("search_range", None)
    n_steps: int = int(params.get("n_steps", 100))
    n_bootstrap: int = int(params.get("n_bootstrap", 100))

    # ── 模型特有参数 ──────────────────────────────────────────────────────────
    outcome: str = str(params.get("outcome", ""))
    time_col: str = str(params.get("time_col", ""))
    event_col: str = str(params.get("event_col", ""))

    warnings: list[str] = []

    # ── 参数校验 ──────────────────────────────────────────────────────────────
    if model_type not in ("logistic", "linear", "cox"):
        raise ValueError("model_type 必须为 logistic / linear / cox")
    if not exposure:
        raise ValueError("请指定暴露变量 (exposure)")
    if exposure not in df.columns:
        raise ValueError(f"暴露变量 '{exposure}' 不存在于数据集中")
    if not pd.api.types.is_numeric_dtype(df[exposure]):
        raise ValueError(f"暴露变量 '{exposure}' 必须为数值型连续变量")

    if model_type in ("logistic", "linear"):
        if not outcome:
            raise ValueError("请指定因变量 (outcome)")
        if outcome not in df.columns:
            raise ValueError(f"因变量 '{outcome}' 不存在于数据集中")
    else:  # cox
        if not time_col:
            raise ValueError("Cox 模型请指定时间变量 (time_col)")
        if not event_col:
            raise ValueError("Cox 模型请指定事件变量 (event_col)")
        for col in [time_col, event_col]:
            if col not in df.columns:
                raise ValueError(f"列 '{col}' 不存在于数据集中")

    # ── 协变量过滤 ────────────────────────────────────────────────────────────
    valid_covs: list[str] = []
    for cov in covariates:
        if cov not in df.columns:
            warnings.append(f"协变量 '{cov}' 不存在，已忽略")
        elif cov == exposure:
            warnings.append(f"协变量 '{cov}' 与暴露变量相同，已忽略")
        else:
            valid_covs.append(cov)
    covariates = valid_covs

    # ── 完整案例过滤 ──────────────────────────────────────────────────────────
    if model_type in ("logistic", "linear"):
        needed_cols = [exposure, outcome] + covariates
    else:
        needed_cols = [exposure, time_col, event_col] + covariates

    df_full = df[needed_cols].dropna()
    n_orig = len(df)
    n_used = len(df_full)

    if n_orig > 0 and (n_orig - n_used) / n_orig > 0.20:
        warnings.append(
            f"缺失值过多：原始数据 {n_orig} 行，删除缺失后剩余 {n_used} 行"
            f"（删除比例 {(n_orig - n_used) / n_orig:.1%}）"
        )
    if n_used < 20:
        raise ValueError(f"有效样本量过少（n = {n_used}），至少需要 20 条完整记录")

    # ── 因变量编码（logistic） ────────────────────────────────────────────────
    if model_type == "logistic":
        y_raw = df_full[outcome]
        df_full = df_full.copy()
        df_full[outcome], enc_warnings = _encode_binary_outcome(y_raw, outcome)
        warnings.extend(enc_warnings)

    # ── 搜索范围 ──────────────────────────────────────────────────────────────
    exp_vals = df_full[exposure].values.astype(float)
    p10 = float(np.percentile(exp_vals, 10))
    p90 = float(np.percentile(exp_vals, 90))

    if search_range is not None:
        try:
            lo = float(search_range[0])
            hi = float(search_range[1])
            if lo >= hi:
                warnings.append("search_range 无效（lo >= hi），已改用 [P10, P90]")
                lo, hi = p10, p90
        except Exception:
            warnings.append("search_range 格式错误，已改用 [P10, P90]")
            lo, hi = p10, p90
    else:
        lo, hi = p10, p90

    if lo == hi:
        # Extremely low variance in exposure
        lo = float(np.percentile(exp_vals, 5))
        hi = float(np.percentile(exp_vals, 95))
        warnings.append("搜索范围过窄，已扩展至 [P5, P95]")

    n_steps = max(10, int(n_steps))
    n_bootstrap = max(0, int(n_bootstrap))

    # ── 网格搜索 ──────────────────────────────────────────────────────────────
    candidates = np.linspace(lo, hi, n_steps)
    candidates_list, ll_values = _grid_search(
        df_full, model_type, exposure, outcome, time_col, event_col,
        covariates, candidates, warnings,
    )

    if len(ll_values) < 3:
        raise ValueError(
            f"网格搜索中只有 {len(ll_values)} 个候选拐点拟合成功，"
            "请检查数据量或搜索范围"
        )

    best_idx = int(np.argmax(ll_values))
    c_best = float(candidates_list[best_idx])
    ll_best = float(ll_values[best_idx])
    logger.info("最优拐点 c_best = %.4f，对数似然 = %.4f", c_best, ll_best)

    # ── Bootstrap CI ──────────────────────────────────────────────────────────
    ci_lo, ci_hi = _bootstrap_ci(
        df_full, model_type, exposure, outcome, time_col, event_col,
        covariates, candidates, n_bootstrap, warnings,
    )

    # ── 在最优拐点处拟合分段模型 ──────────────────────────────────────────────
    seg_result = _fit_piecewise(
        df_full, model_type, exposure, outcome, time_col, event_col,
        covariates, c_best,
    )
    if seg_result is None:
        raise ValueError("在最优拐点处拟合分段模型失败，请检查数据")

    beta_left = seg_result["beta_left"]
    beta_right_add = seg_result["beta_right_add"]
    se_left = seg_result["se_left"]
    se_right_add = seg_result["se_right_add"]
    cov_lr = seg_result["cov_lr"]
    p_left = seg_result["p_left"]
    ll_piecewise = seg_result["llf"]

    # ── 线性模型（无拐点）对数似然 → 非线性检验 ──────────────────────────────
    ll_linear = _fit_linear_model(
        df_full, model_type, exposure, outcome, time_col, event_col, covariates
    )
    lrt_stat = 2.0 * (ll_piecewise - ll_linear) if ll_linear is not None else 0.0
    lrt_stat = max(0.0, lrt_stat)  # numerical safety
    p_nonlinear = float(stats.chi2.sf(lrt_stat, 1))

    # ── 效应量计算 ────────────────────────────────────────────────────────────
    beta_right_total = beta_left + beta_right_add
    var_right_total = (
        se_left ** 2 + se_right_add ** 2 + 2.0 * cov_lr
    )
    se_right_total = math.sqrt(max(var_right_total, 0.0))

    # p 值右侧：delta method（β_left + β_right_add = 0 的 Wald 检验）
    z_right = beta_right_total / se_right_total if se_right_total > 0 else float("nan")
    p_right = float(2.0 * stats.norm.sf(abs(z_right))) if not math.isnan(z_right) else float("nan")

    if model_type in ("logistic", "cox"):
        effect_label = "OR" if model_type == "logistic" else "HR"
        effect_left = _safe_exp(beta_left)
        effect_right = _safe_exp(beta_right_total)
        ci_left_lo = _safe_exp(beta_left - 1.96 * se_left)
        ci_left_hi = _safe_exp(beta_left + 1.96 * se_left)
        ci_right_lo = _safe_exp(beta_right_total - 1.96 * se_right_total)
        ci_right_hi = _safe_exp(beta_right_total + 1.96 * se_right_total)
        fmt_effect = lambda v: f"{v:.4f}"
        fmt_ci = lambda lo, hi: f"[{lo:.4f}, {hi:.4f}]"
        y_ref = 1.0
        y_axis_name = f"{effect_label}（每单位增加）"
    else:  # linear
        effect_label = "β"
        effect_left = float(beta_left)
        effect_right = float(beta_right_total)
        ci_left_lo = float(beta_left - 1.96 * se_left)
        ci_left_hi = float(beta_left + 1.96 * se_left)
        ci_right_lo = float(beta_right_total - 1.96 * se_right_total)
        ci_right_hi = float(beta_right_total + 1.96 * se_right_total)
        fmt_effect = lambda v: f"{v:.4f}"
        fmt_ci = lambda lo, hi: f"[{lo:.4f}, {hi:.4f}]"
        y_ref = 0.0
        y_axis_name = "β（每单位增加）"

    # ── 构造输出表格 ──────────────────────────────────────────────────────────
    tables: list[TableResult] = []

    # 表1：拐点估计
    tables.append(TableResult(
        title="拐点估计",
        headers=["统计量", "值"],
        rows=[
            ["最佳拐点", f"{c_best:.3f}"],
            ["Bootstrap 95% CI", (
                f"({ci_lo:.3f}, {ci_hi:.3f})"
                if not (math.isnan(ci_lo) or math.isnan(ci_hi))
                else "（Bootstrap 失败，见警告）"
            )],
            ["P for non-linearity", _fmt_p(p_nonlinear)],
            ["LRT 统计量（df=1）", f"{lrt_stat:.4f}"],
            ["模型", model_type],
            ["暴露变量", exposure],
            ["协变量", ", ".join(covariates) if covariates else "（无）"],
            ["有效样本量", str(n_used)],
        ],
    ))

    # 表2：拐点两侧效应
    tables.append(TableResult(
        title="拐点两侧效应",
        headers=["分段", "效应值", "95% CI", "p 值", "描述"],
        rows=[
            [
                f"拐点左侧 (x < {c_best:.2f})",
                fmt_effect(effect_left),
                fmt_ci(ci_left_lo, ci_left_hi),
                _fmt_p(p_left),
                "每单位增加的效应",
            ],
            [
                f"拐点右侧 (x ≥ {c_best:.2f})",
                fmt_effect(effect_right),
                fmt_ci(ci_right_lo, ci_right_hi),
                _fmt_p(p_right),
                "每单位增加的效应",
            ],
        ],
    ))

    # 表3：对数似然曲线数据
    ll_table_rows = [
        [f"{float(candidates_list[i]):.4f}", f"{float(ll_values[i]):.4f}"]
        for i in range(len(candidates_list))
    ]
    tables.append(TableResult(
        title="对数似然曲线数据",
        headers=["候选拐点", "对数似然"],
        rows=ll_table_rows,
    ))

    # ── 构造图表 ──────────────────────────────────────────────────────────────
    charts: list[ChartResult] = []
    charts.append(_build_effect_chart(
        exp_vals, c_best,
        beta_left, beta_right_add,
        se_left, se_right_total, se_left,
        cov_lr,
        model_type, exposure, effect_label, y_ref, y_axis_name,
    ))
    charts.append(_build_ll_chart(
        candidates_list, ll_values, c_best, ll_best,
    ))

    # ── 摘要文字 ──────────────────────────────────────────────────────────────
    ci_str = (
        f"{ci_lo:.3f}–{ci_hi:.3f}"
        if not (math.isnan(ci_lo) or math.isnan(ci_hi))
        else "计算失败"
    )
    summary = (
        f"阈值效应分析：{model_type} 模型，暴露变量 {exposure}，"
        f"最佳拐点 = {c_best:.3f}（Bootstrap 95% CI：{ci_str}），"
        f"P for non-linearity = {_fmt_p(p_nonlinear)}。"
        f"左侧效应：{fmt_effect(effect_left)}（{fmt_ci(ci_left_lo, ci_left_hi)}），"
        f"右侧效应：{fmt_effect(effect_right)}（{fmt_ci(ci_right_lo, ci_right_hi)}）。"
    )

    return AnalysisResult(
        method="threshold",
        tables=tables,
        charts=charts,
        summary=summary,
        warnings=warnings,
    )


# ─────────────────────────────────────────────────────────────────────────────
# 因变量编码（logistic 专用）
# ─────────────────────────────────────────────────────────────────────────────

def _encode_binary_outcome(
    col: pd.Series,
    col_name: str,
) -> tuple[pd.Series, list[str]]:
    """将因变量编码为 0/1 浮点型（保留 NaN），返回 (encoded, warnings)。"""
    enc_warnings: list[str] = []
    non_null = col.dropna()
    if len(non_null) == 0:
        raise ValueError(f"因变量 '{col_name}' 全为缺失值")

    if pd.api.types.is_numeric_dtype(col):
        unique_vals = sorted(float(v) for v in non_null.unique())
        if len(unique_vals) > 2:
            raise ValueError(
                f"因变量 '{col_name}' 有 {len(unique_vals)} 个唯一值，"
                "logistic 模型要求二分类"
            )
        if len(unique_vals) == 1:
            raise ValueError(f"因变量 '{col_name}' 只有一个唯一值")
        lo_v, hi_v = unique_vals[0], unique_vals[1]
        if set(unique_vals) == {0.0, 1.0}:
            return col.astype(float), enc_warnings
        enc_warnings.append(
            f"因变量编码：{lo_v} → 0，{hi_v} → 1"
        )
        mapping = {lo_v: 0.0, hi_v: 1.0}
        return (
            col.map(lambda x: mapping.get(float(x), float("nan")) if pd.notna(x) else float("nan")),
            enc_warnings,
        )

    col_str = col.where(col.isna(), col.astype(str))
    unique_str = sorted(col_str.dropna().unique())
    if len(unique_str) > 2:
        raise ValueError(
            f"因变量 '{col_name}' 有 {len(unique_str)} 个字符串类别，"
            "logistic 模型要求二分类"
        )
    if len(unique_str) == 1:
        raise ValueError(f"因变量 '{col_name}' 只有一个唯一值")
    neg_label, pos_label = unique_str[0], unique_str[1]
    enc_warnings.append(
        f"因变量字符串编码：'{neg_label}' → 0，'{pos_label}' → 1"
    )
    return col_str.map({neg_label: 0.0, pos_label: 1.0}), enc_warnings


# ─────────────────────────────────────────────────────────────────────────────
# 分段模型设计矩阵构建
# ─────────────────────────────────────────────────────────────────────────────

def _make_piecewise_df(
    df: pd.DataFrame,
    exposure: str,
    c: float,
    covariates: list[str],
) -> pd.DataFrame:
    """构造含 x_lin（线性项）和 x_right（右段附加斜率）的工作数据框。"""
    out = df.copy()
    out["x_lin"] = out[exposure].astype(float)
    out["x_right"] = np.maximum(out[exposure].astype(float) - c, 0.0)
    return out


# ─────────────────────────────────────────────────────────────────────────────
# 单点拟合（返回对数似然）
# ─────────────────────────────────────────────────────────────────────────────

def _fit_candidate_ll(
    df: pd.DataFrame,
    model_type: str,
    exposure: str,
    outcome: str,
    time_col: str,
    event_col: str,
    covariates: list[str],
    c: float,
) -> float | None:
    """在候选拐点 c 处拟合分段模型，返回对数似然（失败返回 None）。"""
    try:
        df_pw = _make_piecewise_df(df, exposure, c, covariates)
        feature_cols = ["x_lin", "x_right"] + covariates

        if model_type == "logistic":
            import statsmodels.api as sm
            needed = [outcome] + feature_cols
            sub = df_pw[needed].dropna()
            if len(sub) < len(feature_cols) + 2:
                return None
            y = sub[outcome].values.astype(float)
            X = sm.add_constant(sub[feature_cols].values.astype(float), prepend=True)
            with _warnings_mod.catch_warnings():
                _warnings_mod.simplefilter("ignore")
                res = sm.Logit(y, X).fit(disp=0, maxiter=200)
            return float(res.llf)

        elif model_type == "linear":
            import statsmodels.api as sm
            needed = [outcome] + feature_cols
            sub = df_pw[needed].dropna()
            if len(sub) < len(feature_cols) + 2:
                return None
            y = sub[outcome].values.astype(float)
            X = sm.add_constant(sub[feature_cols].values.astype(float), prepend=True)
            with _warnings_mod.catch_warnings():
                _warnings_mod.simplefilter("ignore")
                res = sm.OLS(y, X).fit()
            return float(res.llf)

        elif model_type == "cox":
            from lifelines import CoxPHFitter
            needed = [time_col, event_col] + feature_cols
            sub = df_pw[needed].dropna()
            if len(sub) < len(feature_cols) + 2:
                return None
            cph = CoxPHFitter()
            with _warnings_mod.catch_warnings():
                _warnings_mod.simplefilter("ignore")
                cph.fit(
                    sub,
                    duration_col=time_col,
                    event_col=event_col,
                    formula=" + ".join(feature_cols),
                    show_progress=False,
                )
            return float(cph.log_likelihood_)
    except Exception as exc:
        logger.debug("候选拐点 %.4f 拟合失败: %s", c, exc)
        return None


# ─────────────────────────────────────────────────────────────────────────────
# 网格搜索
# ─────────────────────────────────────────────────────────────────────────────

def _grid_search(
    df: pd.DataFrame,
    model_type: str,
    exposure: str,
    outcome: str,
    time_col: str,
    event_col: str,
    covariates: list[str],
    candidates: np.ndarray,
    warnings: list[str],
) -> tuple[list[float], list[float]]:
    """返回 (成功候选点列表, 对应对数似然列表)。"""
    good_c: list[float] = []
    good_ll: list[float] = []
    n_fail = 0

    for c in candidates:
        ll = _fit_candidate_ll(
            df, model_type, exposure, outcome, time_col, event_col, covariates, float(c)
        )
        if ll is not None and not math.isnan(ll) and not math.isinf(ll):
            good_c.append(float(c))
            good_ll.append(ll)
        else:
            n_fail += 1

    if n_fail > 0:
        warnings.append(f"网格搜索中有 {n_fail} 个候选拐点拟合失败，已跳过")

    return good_c, good_ll


# ─────────────────────────────────────────────────────────────────────────────
# Bootstrap CI
# ─────────────────────────────────────────────────────────────────────────────

def _bootstrap_ci(
    df: pd.DataFrame,
    model_type: str,
    exposure: str,
    outcome: str,
    time_col: str,
    event_col: str,
    covariates: list[str],
    candidates: np.ndarray,
    n_bootstrap: int,
    warnings: list[str],
) -> tuple[float, float]:
    """返回最优拐点的 Bootstrap 95% CI (lo, hi)；失败返回 (nan, nan)。"""
    if n_bootstrap <= 0:
        return float("nan"), float("nan")

    boot_c_best: list[float] = []
    n_rng = len(df)

    try:
        rng = np.random.default_rng(42)
        for i in range(n_bootstrap):
            try:
                idx = rng.integers(0, n_rng, size=n_rng)
                df_boot = df.iloc[idx].reset_index(drop=True)
                good_c, good_ll = _grid_search(
                    df_boot, model_type, exposure, outcome, time_col, event_col,
                    covariates, candidates, [],  # suppress inner warnings
                )
                if len(good_ll) >= 1:
                    best_idx = int(np.argmax(good_ll))
                    boot_c_best.append(float(good_c[best_idx]))
            except Exception as exc:
                logger.debug("Bootstrap 第 %d 次失败: %s", i, exc)
                continue  # skip failed resample

        if len(boot_c_best) < 10:
            warnings.append(
                f"Bootstrap 成功次数过少（{len(boot_c_best)} 次），CI 不可靠"
            )
            return float("nan"), float("nan")

        ci_lo = float(np.percentile(boot_c_best, 2.5))
        ci_hi = float(np.percentile(boot_c_best, 97.5))
        return ci_lo, ci_hi

    except Exception as exc:
        warnings.append(f"Bootstrap CI 计算失败：{exc}")
        return float("nan"), float("nan")


# ─────────────────────────────────────────────────────────────────────────────
# 最优拐点处完整拟合（含协方差信息）
# ─────────────────────────────────────────────────────────────────────────────

def _fit_piecewise(
    df: pd.DataFrame,
    model_type: str,
    exposure: str,
    outcome: str,
    time_col: str,
    event_col: str,
    covariates: list[str],
    c: float,
) -> dict[str, Any] | None:
    """
    在最优拐点 c 处拟合分段模型，返回：
      beta_left, beta_right_add, se_left, se_right_add, cov_lr, p_left, llf
    失败返回 None。
    """
    try:
        df_pw = _make_piecewise_df(df, exposure, c, covariates)
        feature_cols = ["x_lin", "x_right"] + covariates
        # indices in param vector (0 = const or first feature depending on model)
        # x_lin is always first feature → index 1 in statsmodels (after const), index 0 in lifelines
        # x_right is second feature → index 2 in statsmodels, index 1 in lifelines

        if model_type in ("logistic", "linear"):
            import statsmodels.api as sm
            needed = [outcome] + feature_cols
            sub = df_pw[needed].dropna()
            y = sub[outcome].values.astype(float)
            X = sm.add_constant(sub[feature_cols].values.astype(float), prepend=True)
            with _warnings_mod.catch_warnings():
                _warnings_mod.simplefilter("ignore")
                if model_type == "logistic":
                    res = sm.Logit(y, X).fit(disp=0, maxiter=300)
                else:
                    res = sm.OLS(y, X).fit()

            cov_matrix = np.asarray(res.cov_params())
            # param order: [const, x_lin, x_right, ...covariates]
            idx_lin = 1
            idx_right = 2
            beta_left = float(res.params[idx_lin])
            beta_right_add = float(res.params[idx_right])
            se_left = float(res.bse[idx_lin])
            se_right_add = float(res.bse[idx_right])
            cov_lr = float(cov_matrix[idx_lin, idx_right])
            p_left = float(res.pvalues[idx_lin])
            llf = float(res.llf)

        else:  # cox
            from lifelines import CoxPHFitter
            needed = [time_col, event_col] + feature_cols
            sub = df_pw[needed].dropna()
            cph = CoxPHFitter()
            with _warnings_mod.catch_warnings():
                _warnings_mod.simplefilter("ignore")
                cph.fit(
                    sub,
                    duration_col=time_col,
                    event_col=event_col,
                    formula=" + ".join(feature_cols),
                    show_progress=False,
                )

            # lifelines variance_matrix_ has index by feature name
            params = cph.params_
            beta_left = float(params["x_lin"])
            beta_right_add = float(params["x_right"])

            vm = cph.variance_matrix_
            se_left = float(math.sqrt(max(float(vm.loc["x_lin", "x_lin"]), 0.0)))
            se_right_add = float(math.sqrt(max(float(vm.loc["x_right", "x_right"]), 0.0)))
            cov_lr = float(vm.loc["x_lin", "x_right"])
            # p value for x_lin from summary
            summary_df = cph.summary
            p_left = float(summary_df.loc["x_lin", "p"])
            llf = float(cph.log_likelihood_)

        return {
            "beta_left": beta_left,
            "beta_right_add": beta_right_add,
            "se_left": se_left,
            "se_right_add": se_right_add,
            "cov_lr": cov_lr,
            "p_left": p_left,
            "llf": llf,
        }

    except Exception as exc:
        logger.warning("最优拐点处分段模型拟合失败: %s", exc)
        return None


# ─────────────────────────────────────────────────────────────────────────────
# 线性参照模型（无拐点）对数似然
# ─────────────────────────────────────────────────────────────────────────────

def _fit_linear_model(
    df: pd.DataFrame,
    model_type: str,
    exposure: str,
    outcome: str,
    time_col: str,
    event_col: str,
    covariates: list[str],
) -> float | None:
    """
    拟合单线性模型（不含拐点），返回对数似然。用于似然比检验。
    """
    try:
        feature_cols = [exposure] + covariates

        if model_type in ("logistic", "linear"):
            import statsmodels.api as sm
            needed = [outcome] + feature_cols
            sub = df[needed].dropna()
            y = sub[outcome].values.astype(float)
            X = sm.add_constant(sub[feature_cols].values.astype(float), prepend=True)
            with _warnings_mod.catch_warnings():
                _warnings_mod.simplefilter("ignore")
                if model_type == "logistic":
                    res = sm.Logit(y, X).fit(disp=0, maxiter=300)
                else:
                    res = sm.OLS(y, X).fit()
            return float(res.llf)

        else:  # cox
            from lifelines import CoxPHFitter
            needed = [time_col, event_col] + feature_cols
            sub = df[needed].dropna()
            cph = CoxPHFitter()
            with _warnings_mod.catch_warnings():
                _warnings_mod.simplefilter("ignore")
                cph.fit(
                    sub,
                    duration_col=time_col,
                    event_col=event_col,
                    formula=" + ".join(feature_cols),
                    show_progress=False,
                )
            return float(cph.log_likelihood_)

    except Exception as exc:
        logger.warning("线性参照模型拟合失败（非线性检验将跳过）: %s", exc)
        return None


# ─────────────────────────────────────────────────────────────────────────────
# 阈值效应曲线图
# ─────────────────────────────────────────────────────────────────────────────

def _build_effect_chart(
    exp_vals: np.ndarray,
    c_best: float,
    beta_left: float,
    beta_right_add: float,
    se_left: float,
    se_right_total: float,
    se_left_only: float,
    cov_lr: float,
    model_type: str,
    exposure: str,
    effect_label: str,
    y_ref: float,
    y_axis_name: str,
) -> ChartResult:
    """
    绘制以中位数为参考点的相对效应曲线（OR/HR/β vs exposure）。
    左段：相对于参考点 ref = median(exposure)，在 x 处的效应 = exp(beta_left * (x - ref))
    右段：x > c_best 时，右段额外斜率为 beta_right_add
    CI 用 95% 置信带。
    """
    x_min = float(np.min(exp_vals))
    x_max = float(np.max(exp_vals))
    ref_val = float(np.median(exp_vals))
    beta_right_total = beta_left + beta_right_add

    # 生成左段点
    n_pts = 80
    x_left = np.linspace(x_min, c_best, n_pts // 2 + 1)
    x_right_seg = np.linspace(c_best, x_max, n_pts // 2 + 1)

    def effect_left(x: float) -> float:
        dx = x - ref_val
        val = beta_left * dx
        return _safe_exp(val) if model_type in ("logistic", "cox") else float(val)

    def ci_left(x: float) -> tuple[float, float]:
        dx = x - ref_val
        margin = 1.96 * se_left * abs(dx)
        center_log = beta_left * dx
        if model_type in ("logistic", "cox"):
            return _safe_exp(center_log - margin), _safe_exp(center_log + margin)
        else:
            return float(center_log - margin), float(center_log + margin)

    def effect_right_fn(x: float) -> float:
        # from ref to c_best using left slope, then c_best to x using right_total slope
        dx_left = c_best - ref_val
        dx_right = x - c_best
        val = beta_left * dx_left + beta_right_total * dx_right
        return _safe_exp(val) if model_type in ("logistic", "cox") else float(val)

    def ci_right_fn(x: float) -> tuple[float, float]:
        dx_left = c_best - ref_val
        dx_right = x - c_best
        # variance: var(beta_left * dx_left + beta_right_total * dx_right)
        #   = dx_left^2 * var(beta_left) + dx_right^2 * var(beta_right_total)
        #     + 2 * dx_left * dx_right * cov(beta_left, beta_right_total)
        # cov(beta_left, beta_right_total) = cov(beta_left, beta_left + beta_right_add)
        #   = var(beta_left) + cov(beta_left, beta_right_add)
        var_bl = se_left_only ** 2
        var_brt = se_right_total ** 2
        cov_bl_brt = var_bl + cov_lr  # cov(bl, bl+br) = var(bl) + cov(bl, br)
        var_total = (
            dx_left ** 2 * var_bl
            + dx_right ** 2 * var_brt
            + 2.0 * dx_left * dx_right * cov_bl_brt
        )
        se_total = math.sqrt(max(var_total, 0.0))
        center_val = beta_left * dx_left + beta_right_total * dx_right
        margin = 1.96 * se_total
        if model_type in ("logistic", "cox"):
            return _safe_exp(center_val - margin), _safe_exp(center_val + margin)
        else:
            return float(center_val - margin), float(center_val + margin)

    left_data = [[float(x), effect_left(float(x))] for x in x_left]
    right_data = [[float(x), effect_right_fn(float(x))] for x in x_right_seg]

    left_ci_lo = [[float(x), ci_left(float(x))[0]] for x in x_left]
    left_ci_hi = [[float(x), ci_left(float(x))[1]] for x in x_left]
    right_ci_lo = [[float(x), ci_right_fn(float(x))[0]] for x in x_right_seg]
    right_ci_hi = [[float(x), ci_right_fn(float(x))[1]] for x in x_right_seg]

    # Combine CI bands (upper + reversed lower for area fill via ECharts stacked)
    ci_upper_all = left_ci_hi + right_ci_hi
    ci_lower_all = left_ci_lo + right_ci_lo

    ref_line_data = [[x_min, y_ref], [x_max, y_ref]]

    option: dict[str, Any] = {
        "title": {
            "text": f"阈值效应分析（{exposure}）",
            "subtext": f"拐点 = {c_best:.3f}，P for non-linearity（见表格）",
            "left": "center",
        },
        "tooltip": {"trigger": "axis"},
        "legend": {
            "data": ["左侧效应", "右侧效应", "95% CI 上界", "95% CI 下界", "参考线"],
            "top": "16%",
        },
        "grid": {"left": "12%", "right": "5%", "top": "28%", "bottom": "12%"},
        "xAxis": {
            "type": "value",
            "name": exposure,
            "nameLocation": "center",
            "nameGap": 28,
        },
        "yAxis": {
            "type": "value",
            "name": y_axis_name,
            "nameLocation": "center",
            "nameGap": 45,
        },
        "series": [
            {
                "name": "左侧效应",
                "type": "line",
                "data": left_data,
                "showSymbol": False,
                "lineStyle": {"color": "#5470c6", "width": 2.5},
                "itemStyle": {"color": "#5470c6"},
            },
            {
                "name": "右侧效应",
                "type": "line",
                "data": right_data,
                "showSymbol": False,
                "lineStyle": {"color": "#ee6666", "width": 2.5},
                "itemStyle": {"color": "#ee6666"},
            },
            {
                "name": "95% CI 上界",
                "type": "line",
                "data": ci_upper_all,
                "showSymbol": False,
                "lineStyle": {"opacity": 0},
                "areaStyle": {"color": "rgba(84,112,198,0.15)", "origin": "start"},
                "stack": "ci_band",
            },
            {
                "name": "95% CI 下界",
                "type": "line",
                "data": ci_lower_all,
                "showSymbol": False,
                "lineStyle": {"opacity": 0},
                "areaStyle": {"color": "rgba(84,112,198,0.15)", "origin": "start"},
                "stack": "ci_band",
            },
            {
                "name": "参考线",
                "type": "line",
                "data": ref_line_data,
                "showSymbol": False,
                "lineStyle": {"type": "dashed", "color": "#999", "width": 1.5},
                "itemStyle": {"color": "#999"},
            },
        ],
        "markLine": {
            "silent": True,
            "symbol": "none",
            "data": [[
                {"xAxis": c_best, "yAxis": "min", "label": {"show": False}},
                {"xAxis": c_best, "yAxis": "max", "label": {
                    "show": True,
                    "formatter": f"拐点 = {c_best:.2f}",
                    "position": "end",
                }},
            ]],
            "lineStyle": {"type": "dashed", "color": "#fa8c16", "width": 2},
        },
    }

    return ChartResult(title="阈值效应曲线", chart_type="line", option=option)


# ─────────────────────────────────────────────────────────────────────────────
# 对数似然曲线图
# ─────────────────────────────────────────────────────────────────────────────

def _build_ll_chart(
    candidates: list[float],
    ll_values: list[float],
    c_best: float,
    ll_best: float,
) -> ChartResult:
    ll_data = [[float(candidates[i]), float(ll_values[i])] for i in range(len(candidates))]

    option: dict[str, Any] = {
        "title": {
            "text": "对数似然曲线（拐点搜索）",
            "subtext": f"最优拐点 = {c_best:.3f}，最大对数似然 = {ll_best:.4f}",
            "left": "center",
        },
        "tooltip": {
            "trigger": "axis",
            "formatter": "候选拐点: {b}<br/>对数似然: {c}",
        },
        "grid": {"left": "12%", "right": "5%", "top": "26%", "bottom": "12%"},
        "xAxis": {
            "type": "value",
            "name": "候选拐点",
            "nameLocation": "center",
            "nameGap": 28,
        },
        "yAxis": {
            "type": "value",
            "name": "对数似然",
            "nameLocation": "center",
            "nameGap": 50,
        },
        "series": [
            {
                "name": "对数似然曲线",
                "type": "line",
                "data": ll_data,
                "showSymbol": False,
                "lineStyle": {"color": "#5470c6", "width": 2},
                "areaStyle": {"color": "rgba(84,112,198,0.08)"},
            },
            {
                "name": "最优拐点",
                "type": "scatter",
                "data": [[c_best, ll_best]],
                "symbolSize": 14,
                "itemStyle": {"color": "#ee6666"},
                "label": {
                    "show": True,
                    "formatter": f"拐点={c_best:.3f}",
                    "position": "top",
                    "fontSize": 12,
                    "fontWeight": "bold",
                    "color": "#ee6666",
                },
            },
        ],
    }

    return ChartResult(title="对数似然曲线", chart_type="line", option=option)
