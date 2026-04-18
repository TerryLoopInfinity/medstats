"""Cox 比例风险回归模块。

支持：
  - 单变量 Cox 回归（逐个自变量）
  - 多变量 Cox 回归（全部自变量同时进入）
  - HR（hazard ratio）= exp(β) + 95% CI，np.clip 防溢出
  - 比例风险假设检验（Schoenfeld 残差，全局 + 逐变量）
  - 模型拟合评估：C-index、AIC、Log-likelihood ratio test
  - VIF 共线性诊断
  - 图表：HR 森林图、调整后生存曲线、Schoenfeld 残差图、log-log 图
"""

import logging
import warnings as _warnings_mod
from typing import Any

import numpy as np
import pandas as pd

from app.models.analysis import AnalysisResult, ChartResult, TableResult

logger = logging.getLogger(__name__)

_PALETTE = [
    "#5470c6", "#91cc75", "#ee6666", "#fac858",
    "#73c0de", "#3ba272", "#fc8452", "#9a60b4",
]

# HR 允许的最大/最小指数参数（防止 exp 溢出）
_HR_CLIP = 500.0


# ─────────────────────────────────────────────────────────────────────────────
# 主入口
# ─────────────────────────────────────────────────────────────────────────────

def run(df: pd.DataFrame, params: dict) -> AnalysisResult:
    """
    参数
    ----
    params["time_col"]          str           时间变量
    params["event_col"]         str           事件变量（0=删失, 1=事件）
    params["predictors"]        list[str]     自变量列表
    params["categorical_vars"]  list[str]     分类变量
    params["ref_categories"]    dict[str,str] 参考组 {变量: 参考类别}
    params["mode"]              str           "univariate" | "multivariate" | "both"
    """
    try:
        from lifelines import CoxPHFitter
    except ImportError as exc:
        raise RuntimeError("lifelines 未安装，无法进行 Cox 回归") from exc

    time_col: str = str(params.get("time_col", ""))
    event_col: str = str(params.get("event_col", ""))
    predictors: list[str] = list(params.get("predictors") or [])
    categorical_vars: list[str] = list(params.get("categorical_vars") or [])
    ref_categories: dict[str, str] = dict(params.get("ref_categories") or {})
    mode: str = str(params.get("mode", "both")).lower()
    warnings: list[str] = []

    # ── 参数校验 ──────────────────────────────────────────────────────────────
    if not time_col:
        raise ValueError("请指定时间变量 (time_col)")
    if not event_col:
        raise ValueError("请指定事件变量 (event_col)")
    for col in [time_col, event_col]:
        if col not in df.columns:
            raise ValueError(f"列 '{col}' 不存在于数据集中")
    if not predictors:
        raise ValueError("请至少选择一个自变量 (predictors)")
    if mode not in ("univariate", "multivariate", "both"):
        raise ValueError("mode 必须为 univariate / multivariate / both")

    # ── 数据清洗 ──────────────────────────────────────────────────────────────
    valid_preds = [p for p in predictors if p in df.columns]
    missing_preds = [p for p in predictors if p not in df.columns]
    if missing_preds:
        warnings.append(f"以下自变量不存在，已忽略：{', '.join(missing_preds)}")
    if not valid_preds:
        raise ValueError("没有有效的自变量")

    needed = [time_col, event_col] + valid_preds
    df_work = df[needed].dropna().copy()
    df_work[time_col] = pd.to_numeric(df_work[time_col], errors="coerce")
    df_work[event_col] = pd.to_numeric(df_work[event_col], errors="coerce")
    df_work = df_work.dropna(subset=[time_col, event_col])
    df_work = df_work[df_work[time_col] > 0]
    df_work = df_work[df_work[event_col].isin([0, 1])]

    n_orig, n_clean = len(df), len(df_work)
    if n_orig - n_clean > 0:
        warnings.append(f"删除 {n_orig - n_clean} 行无效数据，剩余 {n_clean} 行")
    if n_clean < 20:
        raise ValueError(f"有效数据不足 20 行（当前 {n_clean} 行），无法进行 Cox 回归")

    # ── Dummy 编码分类变量 ────────────────────────────────────────────────────
    df_work, dummy_map, dum_warnings = _make_dummies(
        df_work, valid_preds, categorical_vars, ref_categories
    )
    warnings.extend(dum_warnings)

    # ── 展开所有设计矩阵列 ────────────────────────────────────────────────────
    all_expanded: list[str] = []
    cont_vars: list[str] = []
    skipped: list[str] = []
    for v in valid_preds:
        if v in dummy_map:
            all_expanded.extend(dummy_map[v])
        else:
            if not pd.api.types.is_numeric_dtype(df_work[v]):
                warnings.append(f"自变量 '{v}' 非数值型且未标记为分类变量，已忽略")
                skipped.append(v)
            else:
                all_expanded.append(v)
                cont_vars.append(v)

    valid_preds = [v for v in valid_preds if v not in skipped]
    if not all_expanded:
        raise ValueError("没有有效的自变量（数值型或已 dummy 编码），请检查配置")

    tables: list[TableResult] = []
    charts: list[ChartResult] = []
    multi_result: dict[str, Any] | None = None

    # ── 单变量分析 ────────────────────────────────────────────────────────────
    uni_rows: list[list[Any]] = []
    if mode in ("both", "univariate"):
        uni_rows, uni_warnings = _univariate_cox(
            df_work, time_col, event_col, valid_preds, dummy_map
        )
        warnings.extend(uni_warnings)
        tables.append(TableResult(
            title="单变量 Cox 回归",
            headers=["变量", "β", "SE", "Wald z", "p 值", "HR", "95% CI"],
            rows=uni_rows,
        ))

    # ── 多变量分析 ────────────────────────────────────────────────────────────
    if mode in ("both", "multivariate"):
        multi_result, multi_warnings = _multivariate_cox(
            df_work, time_col, event_col, all_expanded, dummy_map, valid_preds, cont_vars
        )
        warnings.extend(multi_warnings)

        tables.append(TableResult(
            title="多变量 Cox 回归 — 回归系数",
            headers=["变量", "β", "SE", "Wald z", "p 值", "HR", "95% CI"],
            rows=multi_result["coef_rows"],
        ))
        tables.append(TableResult(
            title="多变量 Cox 回归 — 模型拟合指标",
            headers=["指标", "值"],
            rows=multi_result["fit_rows"],
        ))
        if multi_result["ph_rows"]:
            tables.append(TableResult(
                title="比例风险假设检验（Schoenfeld 残差）",
                headers=["变量", "检验统计量", "df", "p 值", "结论"],
                rows=multi_result["ph_rows"],
            ))
        if multi_result["vif_rows"]:
            tables.append(TableResult(
                title="共线性诊断（VIF）",
                headers=["自变量", "VIF", "判断"],
                rows=multi_result["vif_rows"],
            ))

        # 警告 PH 假设违反
        ph_violations = [r[0] for r in multi_result["ph_rows"] if r[4] == "⚠ 违反"]
        if ph_violations:
            warnings.append(
                f"以下变量可能违反比例风险假设（Schoenfeld p < 0.05）：{', '.join(ph_violations)}。"
                "建议检查 log-log 图或考虑时依 Cox 模型。"
            )

        # 图表
        charts.extend(_build_cox_charts(
            multi_result, df_work, time_col, event_col,
            all_expanded, dummy_map, valid_preds, mode
        ))

    elif mode == "univariate" and uni_rows:
        # 单变量模式也生成森林图
        forest_chart = _build_forest_from_uni(uni_rows)
        if forest_chart:
            charts.append(forest_chart)

    # ── 摘要 ──────────────────────────────────────────────────────────────────
    n_used = len(df_work)
    n_events = int(df_work[event_col].sum())
    if multi_result:
        fit = {r[0]: r[1] for r in multi_result["fit_rows"]}
        cindex = fit.get("C-index", "—")
        summary = (
            f"Cox 回归，时间变量：{time_col}，事件变量：{event_col}，"
            f"纳入 {len(valid_preds)} 个自变量，有效样本 n = {n_used}，"
            f"事件 {n_events} 例，C-index = {cindex}。"
        )
        sig_vars = [
            r[0] for r in multi_result["coef_rows"]
            if r[0] and not str(r[0]).startswith("  ") and "参考" not in str(r[0])
            and len(r) > 4 and _parse_p(str(r[4])) < 0.05
        ]
        if sig_vars:
            summary += f" 显著变量（p < 0.05）：{', '.join(sig_vars)}。"
    else:
        summary = (
            f"单变量 Cox 回归，时间变量：{time_col}，事件变量：{event_col}，"
            f"共 {len(valid_preds)} 个自变量，有效样本 n = {n_used}，事件 {n_events} 例。"
        )

    return AnalysisResult(
        method="cox_reg",
        tables=tables,
        charts=charts,
        summary=summary,
        warnings=warnings,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Dummy 编码
# ─────────────────────────────────────────────────────────────────────────────

def _make_dummies(
    df: pd.DataFrame,
    predictors: list[str],
    categorical_vars: list[str],
    ref_categories: dict[str, str],
) -> tuple[pd.DataFrame, dict[str, list[str]], list[str]]:
    df = df.copy()
    dummy_map: dict[str, list[str]] = {}
    warnings: list[str] = []

    for v in categorical_vars:
        if v not in df.columns or v not in predictors:
            continue
        df[v] = df[v].astype(str)
        cats = sorted(df[v].unique().tolist())
        ref = str(ref_categories.get(v, cats[0]))
        if ref not in cats:
            warnings.append(f"参考组 '{ref}' 不在变量 '{v}' 的类别中，使用 '{cats[0]}'")
            ref = cats[0]
        non_ref = [c for c in cats if c != ref]
        cols_added: list[str] = []
        for c in non_ref:
            col_name = f"{v}__{c}"
            df[col_name] = (df[v] == c).astype(float)
            cols_added.append(col_name)
        dummy_map[v] = cols_added
        if not cols_added:
            warnings.append(f"分类变量 '{v}' 只有一个类别，已忽略")
    return df, dummy_map, warnings


# ─────────────────────────────────────────────────────────────────────────────
# 单变量 Cox
# ─────────────────────────────────────────────────────────────────────────────

def _univariate_cox(
    df: pd.DataFrame,
    time_col: str,
    event_col: str,
    valid_preds: list[str],
    dummy_map: dict[str, list[str]],
) -> tuple[list[list[Any]], list[str]]:
    from lifelines import CoxPHFitter

    rows: list[list[Any]] = []
    warnings: list[str] = []

    for v in valid_preds:
        cols = dummy_map.get(v, [v])
        fit_cols = [time_col, event_col] + cols
        sub = df[fit_cols].dropna()
        if len(sub) < 10 or sub[event_col].sum() < 5:
            warnings.append(f"变量 '{v}' 有效数据不足，跳过单变量 Cox")
            continue

        try:
            cph = CoxPHFitter(penalizer=0.01)
            with _warnings_mod.catch_warnings():
                _warnings_mod.simplefilter("ignore")
                cph.fit(sub, duration_col=time_col, event_col=event_col)

            if v in dummy_map:
                # 分类变量标题行
                rows.append([v, "", "", "", "", "", ""])
                for col in cols:
                    cat_label = col.split("__", 1)[-1] if "__" in col else col
                    row = _extract_coef_row(cph, col, f"  {cat_label}", v)
                    rows.append(row)
            else:
                row = _extract_coef_row(cph, v, v)
                rows.append(row)
        except Exception as exc:
            logger.warning("单变量 Cox '%s' 失败: %s", v, exc)
            warnings.append(f"变量 '{v}' 单变量 Cox 回归失败：{exc}")

    return rows, warnings


# ─────────────────────────────────────────────────────────────────────────────
# 多变量 Cox
# ─────────────────────────────────────────────────────────────────────────────

def _multivariate_cox(
    df: pd.DataFrame,
    time_col: str,
    event_col: str,
    all_expanded: list[str],
    dummy_map: dict[str, list[str]],
    valid_preds: list[str],
    cont_vars: list[str],
) -> tuple[dict[str, Any], list[str]]:
    from lifelines import CoxPHFitter

    warnings: list[str] = []
    fit_cols = [time_col, event_col] + all_expanded
    df_fit = df[fit_cols].dropna()

    if len(df_fit) < 20:
        raise ValueError(f"有效样本 ({len(df_fit)}) 不足以拟合多变量 Cox 模型")

    cph = CoxPHFitter(penalizer=0.01)
    with _warnings_mod.catch_warnings():
        _warnings_mod.simplefilter("ignore")
        cph.fit(df_fit, duration_col=time_col, event_col=event_col)

    # ── 系数表 ────────────────────────────────────────────────────────────────
    coef_rows: list[list[Any]] = []
    # 反转 dummy_map：col_name -> (原变量, 类别)
    col_to_orig: dict[str, tuple[str, str]] = {}
    for v, cols in dummy_map.items():
        for c in cols:
            cat = c.split("__", 1)[-1] if "__" in c else c
            col_to_orig[c] = (v, cat)

    emitted_headers: set[str] = set()
    for col in all_expanded:
        if col in col_to_orig:
            orig_v, cat = col_to_orig[col]
            if orig_v not in emitted_headers:
                coef_rows.append([orig_v, "", "", "", "", "", ""])
                emitted_headers.add(orig_v)
            row = _extract_coef_row(cph, col, f"  {cat}", orig_v)
        else:
            row = _extract_coef_row(cph, col, col)
        coef_rows.append(row)

    # ── 拟合指标 ──────────────────────────────────────────────────────────────
    cindex = float(np.clip(cph.concordance_index_, 0.0, 1.0))
    try:
        aic = float(cph.AIC_partial_)
    except Exception:
        aic = float("nan")
    try:
        llr_p = float(cph.log_likelihood_ratio_test().p_value)
    except Exception:
        llr_p = float("nan")
    try:
        ll = float(cph.log_likelihood_)
    except Exception:
        ll = float("nan")

    fit_rows: list[list[Any]] = [
        ["C-index（一致性指数）", f"{cindex:.4f}"],
        ["AIC", f"{aic:.2f}" if not np.isnan(aic) else "—"],
        ["对数似然", f"{ll:.4f}" if not np.isnan(ll) else "—"],
        ["LRT p 值（vs 空模型）", _fmt_p(llr_p) if not np.isnan(llr_p) else "—"],
        ["样本量", str(len(df_fit))],
        ["事件数", str(int(df_fit[event_col].sum()))],
    ]

    # ── PH 假设检验 ───────────────────────────────────────────────────────────
    ph_rows: list[list[Any]] = []
    try:
        from lifelines.statistics import proportional_hazard_test
        ph_result = proportional_hazard_test(cph, df_fit, time_transform="rank")
        ph_summary = ph_result.summary
        # 全局检验
        try:
            global_row = ph_summary.loc["km"]
        except Exception:
            global_row = None

        for col in all_expanded:
            if col in ph_summary.index:
                stat_val = float(ph_summary.loc[col, "test_statistic"])
                p_val = float(ph_summary.loc[col, "p"])
                df_v = int(ph_summary.loc[col, "degrees_of_freedom"])
                if col in col_to_orig:
                    orig_v, cat = col_to_orig[col]
                    label = f"{orig_v} ({cat})"
                else:
                    label = col
                conclusion = "⚠ 违反" if p_val < 0.05 else "满足"
                ph_rows.append([label, f"{stat_val:.3f}", df_v, _fmt_p(p_val), conclusion])

        # 全局行
        if global_row is not None:
            g_stat = float(global_row["test_statistic"])
            g_p = float(global_row["p"])
            g_df = int(global_row["degrees_of_freedom"])
            ph_rows.insert(0, ["全局检验", f"{g_stat:.3f}", g_df, _fmt_p(g_p),
                                "⚠ 违反" if g_p < 0.05 else "满足"])
    except Exception as exc:
        logger.warning("PH 假设检验失败: %s", exc)
        warnings.append(f"PH 假设检验失败：{exc}")

    # ── VIF ───────────────────────────────────────────────────────────────────
    vif_rows = _compute_vif(df_fit[all_expanded])
    if vif_rows:
        # 将 dummy 列名转回原变量名
        vif_rows_disp: list[list[Any]] = []
        for row in vif_rows:
            col = str(row[0])
            if col in col_to_orig:
                orig_v, cat = col_to_orig[col]
                vif_rows_disp.append([f"{orig_v} ({cat})", row[1], row[2]])
            else:
                vif_rows_disp.append(row)
        vif_rows = vif_rows_disp

    return {
        "cph": cph,
        "df_fit": df_fit,
        "coef_rows": coef_rows,
        "fit_rows": fit_rows,
        "ph_rows": ph_rows,
        "vif_rows": vif_rows,
        "all_expanded": all_expanded,
        "col_to_orig": col_to_orig,
    }, warnings


# ─────────────────────────────────────────────────────────────────────────────
# 系数行提取
# ─────────────────────────────────────────────────────────────────────────────

def _extract_coef_row(cph, col: str, display_name: str, parent_var: str = "") -> list[Any]:
    """从已拟合的 CoxPHFitter 中提取一行系数数据。"""
    try:
        summ = cph.summary
        if col not in summ.index:
            return [display_name, "—", "—", "—", "—", "—", "—"]
        row = summ.loc[col]
        beta = float(row.get("coef", row.get("beta", 0.0)))
        se = float(row.get("se(coef)", row.get("se", 0.0)))
        z = float(row.get("z", beta / se if se > 0 else 0.0))
        p = float(row.get("p", row.get("p-value", 1.0)))

        beta_c = float(np.clip(beta, -_HR_CLIP, _HR_CLIP))
        hr = float(np.exp(beta_c))
        # CI
        ci_lo_key = [k for k in row.index if "lower" in k.lower()]
        ci_hi_key = [k for k in row.index if "upper" in k.lower()]
        if ci_lo_key and ci_hi_key:
            lo_b = float(np.clip(row[ci_lo_key[0]], -_HR_CLIP, _HR_CLIP))
            hi_b = float(np.clip(row[ci_hi_key[0]], -_HR_CLIP, _HR_CLIP))
            hr_lo = float(np.exp(lo_b))
            hr_hi = float(np.exp(hi_b))
        else:
            hr_lo = float(np.exp(float(np.clip(beta - 1.96 * se, -_HR_CLIP, _HR_CLIP))))
            hr_hi = float(np.exp(float(np.clip(beta + 1.96 * se, -_HR_CLIP, _HR_CLIP))))

        return [
            display_name,
            f"{beta:.4f}",
            f"{se:.4f}",
            f"{z:.3f}",
            _fmt_p(p),
            f"{hr:.3f}",
            f"({hr_lo:.3f}, {hr_hi:.3f})",
        ]
    except Exception as exc:
        logger.debug("_extract_coef_row failed for '%s': %s", col, exc)
        return [display_name, "—", "—", "—", "—", "—", "—"]


# ─────────────────────────────────────────────────────────────────────────────
# VIF
# ─────────────────────────────────────────────────────────────────────────────

def _compute_vif(X: pd.DataFrame) -> list[list[Any]]:
    try:
        from statsmodels.stats.outliers_influence import variance_inflation_factor

        X_clean = X.dropna()
        if len(X_clean) < X_clean.shape[1] + 2:
            return []
        X_arr = X_clean.values.astype(float)
        rows = []
        for i, col in enumerate(X_clean.columns):
            try:
                vif_val = float(variance_inflation_factor(X_arr, i))
                vif_val = float(np.clip(vif_val, 0.0, 1e6))
                judgment = "严重共线" if vif_val > 10 else ("中度共线" if vif_val > 5 else "正常")
                rows.append([col, f"{vif_val:.2f}", judgment])
            except Exception:
                rows.append([col, "—", "—"])
        return rows
    except ImportError:
        return []


# ─────────────────────────────────────────────────────────────────────────────
# 图表
# ─────────────────────────────────────────────────────────────────────────────

def _build_cox_charts(
    multi_result: dict[str, Any],
    df_fit: pd.DataFrame,
    time_col: str,
    event_col: str,
    all_expanded: list[str],
    dummy_map: dict[str, list[str]],
    valid_preds: list[str],
    mode: str,
) -> list[ChartResult]:
    charts: list[ChartResult] = []
    cph = multi_result["cph"]
    coef_rows = multi_result["coef_rows"]

    # ── HR 森林图 ──────────────────────────────────────────────────────────────
    forest_chart = _build_hr_forest(coef_rows)
    if forest_chart:
        charts.append(forest_chart)

    # ── 调整后生存曲线（按第一个分类变量或最重要的连续变量分组）──────────────────
    adj_chart = _build_adjusted_survival(
        cph, df_fit, time_col, event_col, all_expanded, dummy_map, valid_preds
    )
    if adj_chart:
        charts.append(adj_chart)

    # ── Schoenfeld 残差图 ─────────────────────────────────────────────────────
    schoen_chart = _build_schoenfeld_chart(cph, df_fit, time_col, event_col, all_expanded)
    if schoen_chart:
        charts.append(schoen_chart)

    # ── log-log 图（PH 假设图形诊断）────────────────────────────────────────
    loglog_chart = _build_loglog_chart(df_fit, time_col, event_col, dummy_map, valid_preds)
    if loglog_chart:
        charts.append(loglog_chart)

    return charts


# ── HR 森林图 ─────────────────────────────────────────────────────────────────

def _build_hr_forest(coef_rows: list[list[Any]]) -> ChartResult | None:
    """从系数表构建 HR 森林图（forestData 格式，复用 ForestPlot 组件）。"""
    forest_data: list[dict] = []
    for row in coef_rows:
        label = str(row[0])
        if label == "" or len(row) < 7:
            continue
        if row[5] == "" or row[5] == "—":
            # 分类变量标题行
            forest_data.append({
                "label": label,
                "isHeader": True,
                "hr": None, "lo": None, "hi": None, "p": None,
            })
            continue
        try:
            hr = float(str(row[5]).replace("—", "nan"))
            ci_str = str(row[6]).strip("()")
            lo_str, hi_str = ci_str.split(",")
            lo = float(lo_str.strip())
            hi = float(hi_str.strip())
            p_str = str(row[4])
            p_val = 0.0001 if "0.001" in p_str else float(p_str) if p_str != "—" else 1.0
            forest_data.append({
                "label": label,
                "isHeader": False,
                "hr": round(hr, 4),
                "lo": round(lo, 4),
                "hi": round(hi, 4),
                "p": round(p_val, 4),
                "hrLabel": f"{hr:.3f} ({lo:.3f}–{hi:.3f})",
            })
        except Exception:
            continue

    if not forest_data:
        return None

    option: dict[str, Any] = {
        "forestData": forest_data,
        "refLine": 1.0,
        "xAxisLabel": "Hazard Ratio (HR)",
        "title": "多变量 Cox 回归 — HR 森林图",
    }
    return ChartResult(title="HR 森林图", chart_type="forest_plot", option=option)


# ── 调整后生存曲线 ────────────────────────────────────────────────────────────

def _build_adjusted_survival(
    cph, df_fit: pd.DataFrame, time_col: str, event_col: str,
    all_expanded: list[str],
    dummy_map: dict[str, list[str]],
    valid_preds: list[str],
) -> ChartResult | None:
    """按第一个分类变量（或连续变量高/低组）绘制模型预测生存曲线。"""
    try:
        # 选取分组变量
        group_var_orig: str | None = None
        for v in valid_preds:
            if v in dummy_map and len(dummy_map[v]) >= 1:
                group_var_orig = v
                break

        if group_var_orig is None:
            # 用第一个连续变量的中位数分高低组
            for v in valid_preds:
                if v in all_expanded:
                    group_var_orig = v
                    break

        if group_var_orig is None:
            return None

        series: list[dict] = []
        legend_data: list[str] = []

        if group_var_orig in dummy_map:
            # 分类变量：每类别一条曲线
            orig_col = df_fit.copy()
            # 找原始类别列（从 dummy 名中反推）
            cats = [c.split("__", 1)[-1] for c in dummy_map[group_var_orig]]
            # 参考类别
            ref_cat_label = "ref"
            all_cats = [ref_cat_label] + cats

            for color_idx, cat_label in enumerate(all_cats):
                # 构建 covariate profile（均值，除本 dummy 列外）
                profile = {c: float(df_fit[c].mean()) for c in all_expanded}
                for dc in dummy_map[group_var_orig]:
                    profile[dc] = 0.0
                if cat_label != ref_cat_label:
                    dc_name = f"{group_var_orig}__{cat_label}"
                    if dc_name in profile:
                        profile[dc_name] = 1.0
                    else:
                        continue

                cov = pd.DataFrame([profile])
                with _warnings_mod.catch_warnings():
                    _warnings_mod.simplefilter("ignore")
                    pred = cph.predict_survival_function(cov)

                times = pred.index.tolist()
                surv = [float(np.clip(v, 0.0, 1.0)) for v in pred.iloc[:, 0].tolist()]
                color = _PALETTE[color_idx % len(_PALETTE)]
                series.append({
                    "name": f"{group_var_orig}={cat_label}",
                    "type": "line",
                    "step": "end",
                    "data": [[t, s] for t, s in zip(times, surv)],
                    "lineStyle": {"color": color, "width": 2, "type": "dashed"},
                    "itemStyle": {"color": color},
                    "symbol": "none",
                })
                legend_data.append(f"{group_var_orig}={cat_label}")
        else:
            # 连续变量：中位数±1SD 两组
            col_vals = df_fit[group_var_orig].dropna()
            med = float(col_vals.median())
            sd = float(col_vals.std())
            for color_idx, (label, val) in enumerate([("低 (P50-SD)", med - sd), ("高 (P50+SD)", med + sd)]):
                profile = {c: float(df_fit[c].mean()) for c in all_expanded}
                profile[group_var_orig] = val
                cov = pd.DataFrame([profile])
                with _warnings_mod.catch_warnings():
                    _warnings_mod.simplefilter("ignore")
                    pred = cph.predict_survival_function(cov)
                times = pred.index.tolist()
                surv = [float(np.clip(v, 0.0, 1.0)) for v in pred.iloc[:, 0].tolist()]
                color = _PALETTE[color_idx % len(_PALETTE)]
                series.append({
                    "name": f"{group_var_orig} {label}",
                    "type": "line",
                    "step": "end",
                    "data": [[t, s] for t, s in zip(times, surv)],
                    "lineStyle": {"color": color, "width": 2, "type": "dashed"},
                    "itemStyle": {"color": color},
                    "symbol": "none",
                })
                legend_data.append(f"{group_var_orig} {label}")

        if not series:
            return None

        option: dict[str, Any] = {
            "title": {
                "text": f"调整后生存曲线（{group_var_orig}）",
                "left": "center",
                "textStyle": {"fontSize": 14},
            },
            "tooltip": {"trigger": "axis"},
            "legend": {"data": legend_data, "top": "28px"},
            "grid": {"left": "10%", "right": "5%", "top": "14%", "bottom": "10%"},
            "xAxis": {"type": "value", "name": "时间", "nameLocation": "middle", "nameGap": 28, "min": 0},
            "yAxis": {"type": "value", "name": "生存概率", "min": 0, "max": 1},
            "series": series,
        }
        return ChartResult(title="调整后生存曲线", chart_type="line", option=option)
    except Exception as exc:
        logger.warning("调整后生存曲线生成失败: %s", exc)
        return None


# ── Schoenfeld 残差图 ─────────────────────────────────────────────────────────

def _build_schoenfeld_chart(
    cph, df_fit: pd.DataFrame, time_col: str, event_col: str, all_expanded: list[str]
) -> ChartResult | None:
    """绘制第一个变量的 Schoenfeld 残差 vs 时间（散点 + 平滑趋势线）。"""
    try:
        if not all_expanded:
            return None
        col = all_expanded[0]

        residuals = cph.compute_residuals(df_fit, kind="schoenfeld")
        if col not in residuals.columns:
            return None

        t_vals = residuals.index.tolist()
        r_vals = residuals[col].tolist()

        # LOWESS 趋势线
        try:
            from statsmodels.nonparametric.smoothers_lowess import lowess
            smoothed = lowess(r_vals, t_vals, frac=0.4)
            trend = [[float(x), float(y)] for x, y in smoothed]
        except Exception:
            trend = []

        scatter_data = [[float(t), float(r)] for t, r in zip(t_vals, r_vals)]

        series: list[dict] = [
            {
                "name": f"Schoenfeld 残差（{col}）",
                "type": "scatter",
                "data": scatter_data,
                "symbolSize": 4,
                "itemStyle": {"color": "#5470c6", "opacity": 0.6},
            }
        ]
        if trend:
            series.append({
                "name": "LOWESS 趋势",
                "type": "line",
                "data": trend,
                "smooth": True,
                "lineStyle": {"color": "#ee6666", "width": 2},
                "symbol": "none",
                "itemStyle": {"color": "#ee6666"},
            })

        option: dict[str, Any] = {
            "title": {
                "text": f"Schoenfeld 残差图（{col}）",
                "subtext": "趋势线应呈水平以满足比例风险假设",
                "left": "center",
                "textStyle": {"fontSize": 14},
            },
            "tooltip": {"trigger": "axis"},
            "legend": {"data": [f"Schoenfeld 残差（{col}）", "LOWESS 趋势"], "top": "40px"},
            "grid": {"left": "10%", "right": "5%", "top": "16%", "bottom": "10%"},
            "xAxis": {"type": "value", "name": "时间", "nameLocation": "middle", "nameGap": 28},
            "yAxis": {"type": "value", "name": "Schoenfeld 残差"},
            "series": series,
        }
        return ChartResult(title="Schoenfeld 残差图", chart_type="scatter", option=option)
    except Exception as exc:
        logger.warning("Schoenfeld 残差图失败: %s", exc)
        return None


# ── log-log 图 ────────────────────────────────────────────────────────────────

def _build_loglog_chart(
    df_fit: pd.DataFrame,
    time_col: str,
    event_col: str,
    dummy_map: dict[str, list[str]],
    valid_preds: list[str],
) -> ChartResult | None:
    """log(-log(S(t))) vs log(t) — 平行则满足 PH 假设。"""
    try:
        from lifelines import KaplanMeierFitter

        # 找第一个分类变量
        group_var: str | None = None
        for v in valid_preds:
            if v in dummy_map:
                group_var = v
                break
        if group_var is None:
            return None

        # 从 df_fit 中恢复原始类别（逆 dummy 编码）
        orig_col = pd.Series("ref", index=df_fit.index, name=group_var)
        for dc in dummy_map[group_var]:
            cat_label = dc.split("__", 1)[-1]
            orig_col = orig_col.where(df_fit[dc] != 1, other=cat_label)

        temp = df_fit[[time_col, event_col]].copy()
        temp[group_var] = orig_col
        groups = sorted(temp[group_var].unique())

        series: list[dict] = []
        legend_data: list[str] = []
        for color_idx, g in enumerate(groups):
            mask = temp[group_var] == g
            sub = temp[mask]
            kmf = KaplanMeierFitter(label=str(g))
            with _warnings_mod.catch_warnings():
                _warnings_mod.simplefilter("ignore")
                kmf.fit(sub[time_col], sub[event_col])

            sf = kmf.survival_function_
            times = sf.index.tolist()
            sf_vals = sf.iloc[:, 0].tolist()

            data_pts: list[list[float]] = []
            for t, sv in zip(times, sf_vals):
                if t <= 0:
                    continue
                sv_c = float(np.clip(sv, 1e-10, 1.0 - 1e-10))
                lls = np.log(-np.log(sv_c))
                lt = np.log(float(t))
                lls = float(np.clip(lls, -50.0, 50.0))
                data_pts.append([round(lt, 5), round(lls, 5)])

            color = _PALETTE[color_idx % len(_PALETTE)]
            series.append({
                "name": str(g),
                "type": "line",
                "data": data_pts,
                "lineStyle": {"color": color, "width": 2},
                "itemStyle": {"color": color},
                "symbol": "none",
            })
            legend_data.append(str(g))

        option: dict[str, Any] = {
            "title": {
                "text": f"log(-log S(t)) vs log(t)  [{group_var}]",
                "subtext": "各组曲线平行则满足比例风险假设",
                "left": "center",
                "textStyle": {"fontSize": 14},
            },
            "tooltip": {"trigger": "axis"},
            "legend": {"data": legend_data, "top": "40px"},
            "grid": {"left": "10%", "right": "5%", "top": "16%", "bottom": "10%"},
            "xAxis": {"type": "value", "name": "log(t)", "nameLocation": "middle", "nameGap": 28},
            "yAxis": {"type": "value", "name": "log(-log S(t))"},
            "series": series,
        }
        return ChartResult(title="log-log 图（PH 假设诊断）", chart_type="line", option=option)
    except Exception as exc:
        logger.warning("log-log 图生成失败: %s", exc)
        return None


# ── 单变量森林图 ──────────────────────────────────────────────────────────────

def _build_forest_from_uni(uni_rows: list[list[Any]]) -> ChartResult | None:
    forest_data: list[dict] = []
    for row in uni_rows:
        label = str(row[0])
        if len(row) < 7 or row[5] in ("", "—"):
            if row[5] in ("", "—"):
                forest_data.append({"label": label, "isHeader": True,
                                    "hr": None, "lo": None, "hi": None, "p": None})
            continue
        try:
            hr = float(row[5])
            ci_str = str(row[6]).strip("()")
            lo_s, hi_s = ci_str.split(",")
            lo, hi = float(lo_s.strip()), float(hi_s.strip())
            p_str = str(row[4])
            p_val = 0.0001 if "0.001" in p_str else (float(p_str) if p_str != "—" else 1.0)
            forest_data.append({
                "label": label, "isHeader": False,
                "hr": round(hr, 4), "lo": round(lo, 4), "hi": round(hi, 4), "p": round(p_val, 4),
                "hrLabel": f"{hr:.3f} ({lo:.3f}–{hi:.3f})",
            })
        except Exception:
            continue

    if not forest_data:
        return None
    option = {
        "forestData": forest_data,
        "refLine": 1.0,
        "xAxisLabel": "Hazard Ratio (HR)",
        "title": "单变量 Cox 回归 — HR 森林图",
    }
    return ChartResult(title="HR 森林图（单变量）", chart_type="forest_plot", option=option)


# ─────────────────────────────────────────────────────────────────────────────
# 格式化辅助
# ─────────────────────────────────────────────────────────────────────────────

def _fmt_p(p: float) -> str:
    if p < 0.001:
        return "< 0.001"
    return f"{p:.3f}"


def _parse_p(s: str) -> float:
    s = s.strip()
    if s.startswith("<"):
        return float(s.replace("<", "").strip()) * 0.5
    try:
        return float(s)
    except Exception:
        return 1.0
