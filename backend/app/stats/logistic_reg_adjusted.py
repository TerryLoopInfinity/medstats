"""Logistic 回归控制混杂偏倚模块。

功能：
  1. 逐步调整模型 (Model 1 粗模型 → Model 2 部分调整 → Model 3 全调整)
  2. 暴露变量 OR 变化追踪 (crude OR → adjusted OR，log(OR) 变化 >10% 提示混杂)
  3. 分层分析 (按分层变量各水平分别拟合，Breslow-Day 同质性检验)
  4. 交互项检验 (暴露 × 修饰因子，输出交互 OR 和 p 值)
  5. 模型性能对比 (各模型 AUC)

输入 params：
  outcome           str           因变量（二分类）
  exposure          str           暴露变量（数值型）
  covariates        list[str]     候选协变量
  categorical_vars  list[str]     需要 dummy 编码的分类协变量
  ref_categories    dict[str,str] 参考组 {变量名: 参考类别}
  mode              str           "both"（默认）| "crude" | "adjusted"
  model2_covariates list[str]     Model 2 协变量（省略则自动取前半）
  stratify_var      str | None    分层变量
  interaction_var   str | None    交互项变量（数值型）

输出 AnalysisResult：
  tables:  模型对比表、完整系数表、混杂评估表、模型性能表、分层表、交互表
  charts:  OR森林图、分层森林图、AUC条形图、协变量贡献条形图
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


# ─────────────────────────────────────────────────────────────────────────────
# 溢出安全的 exp / 完全分离检测
# ─────────────────────────────────────────────────────────────────────────────

def _safe_exp(x: float) -> float:
    """np.clip 防止 exp 溢出 (math range error)。"""
    return float(np.exp(np.clip(float(x), -500.0, 500.0)))


def _check_separation(model_result: dict, model_label: str = "") -> list[str]:
    """检测完全分离：|β| > 10 或 SE > 100 时发出警告。"""
    sep_warnings: list[str] = []
    model = model_result["model"]
    prefix = f"[{model_label}] " if model_label else ""
    for name in model.params.index:
        if name in ("const", "Intercept"):
            continue
        beta = float(model.params[name])
        se = float(model.bse[name])
        if abs(beta) > 10 or se > 100:
            sep_warnings.append(
                f"{prefix}变量 '{name}' 可能存在完全分离，OR 估计不可靠"
                f"（β = {beta:.2f}，SE = {se:.2f}）"
            )
    return sep_warnings


# ─────────────────────────────────────────────────────────────────────────────
# 主入口
# ─────────────────────────────────────────────────────────────────────────────

def run(df: pd.DataFrame, params: dict) -> AnalysisResult:
    outcome: str = str(params.get("outcome", ""))
    exposure: str = str(params.get("exposure", ""))
    covariates: list[str] = list(params.get("covariates") or [])
    categorical_vars: list[str] = list(params.get("categorical_vars") or [])
    ref_categories: dict[str, str] = dict(params.get("ref_categories") or {})
    mode: str = str(params.get("mode", "both")).lower()
    model2_covariates: list[str] | None = params.get("model2_covariates")
    stratify_var: str | None = params.get("stratify_var") or None
    interaction_var: str | None = params.get("interaction_var") or None
    warnings: list[str] = []

    # ── 参数校验 ──────────────────────────────────────────────────────────────
    if not outcome:
        raise ValueError("请指定因变量 (outcome)")
    if outcome not in df.columns:
        raise ValueError(f"因变量 '{outcome}' 不存在于数据集中")
    if not exposure:
        raise ValueError("请指定暴露变量 (exposure)")
    if exposure not in df.columns:
        raise ValueError(f"暴露变量 '{exposure}' 不存在于数据集中")
    if not pd.api.types.is_numeric_dtype(df[exposure]):
        raise ValueError(f"暴露变量 '{exposure}' 必须为数值型变量")
    if exposure == outcome:
        raise ValueError("暴露变量与因变量不能相同")
    if mode not in ("both", "crude", "adjusted"):
        raise ValueError("mode 必须为 both / crude / adjusted")

    # ── 编码因变量 ────────────────────────────────────────────────────────────
    y_series, positive_label, enc_warnings = _encode_outcome(df[outcome])
    warnings.extend(enc_warnings)

    # ── 过滤有效协变量 ────────────────────────────────────────────────────────
    valid_raw_covs: list[str] = []
    for v in covariates:
        if v in (outcome, exposure):
            warnings.append(f"协变量 '{v}' 与因/暴露变量相同，已忽略")
            continue
        if v not in df.columns:
            warnings.append(f"协变量 '{v}' 不存在，已忽略")
            continue
        valid_raw_covs.append(v)

    # ── dummy 编码分类协变量 ──────────────────────────────────────────────────
    cat_covs = [v for v in categorical_vars if v in valid_raw_covs]
    df_work, dummy_map, dummy_w = _make_dummies(df, valid_raw_covs, cat_covs, ref_categories)
    warnings.extend(dummy_w)

    # 展开协变量列名
    expanded_covs: list[str] = []
    skipped_covs: set[str] = set()
    for v in valid_raw_covs:
        if v in dummy_map:
            expanded_covs.extend(dummy_map[v])
        elif pd.api.types.is_numeric_dtype(df[v]):
            expanded_covs.append(v)
        else:
            warnings.append(f"协变量 '{v}' 非数值型且未标记为分类变量，已忽略")
            skipped_covs.add(v)

    valid_raw_covs = [v for v in valid_raw_covs if v not in skipped_covs]

    # ── 确定各模型协变量（展开形式） ─────────────────────────────────────────
    if model2_covariates is None:
        n_half = max(1, math.ceil(len(expanded_covs) / 2)) if expanded_covs else 0
        m2_expanded: list[str] = expanded_covs[:n_half]
    else:
        m2_expanded = []
        for v in model2_covariates:
            if v in dummy_map:
                m2_expanded.extend(dummy_map[v])
            elif v in expanded_covs:
                m2_expanded.append(v)

    m3_expanded = expanded_covs

    # ── 分层/交互变量校验 ─────────────────────────────────────────────────────
    if stratify_var:
        if stratify_var not in df.columns:
            warnings.append(f"分层变量 '{stratify_var}' 不存在，已忽略分层分析")
            stratify_var = None

    if interaction_var:
        if interaction_var not in df.columns:
            warnings.append(f"交互变量 '{interaction_var}' 不存在，已忽略交互检验")
            interaction_var = None
        elif not pd.api.types.is_numeric_dtype(df[interaction_var]):
            warnings.append(f"交互变量 '{interaction_var}' 非数值型，已忽略交互检验")
            interaction_var = None

    # ── 准备主分析数据 ────────────────────────────────────────────────────────
    df_work = df_work.copy()
    df_work["__y__"] = y_series

    main_cols = ["__y__", exposure] + m3_expanded
    if stratify_var and stratify_var not in main_cols:
        main_cols.append(stratify_var)
    if interaction_var and interaction_var not in main_cols:
        main_cols.append(interaction_var)

    df_main = df_work[list(dict.fromkeys(main_cols))].dropna()
    n_main = len(df_main)

    min_needed = 2 + len(m3_expanded)
    if n_main < min_needed:
        raise ValueError(
            f"有效样本量 ({n_main}) 不足以拟合模型（需要至少 {min_needed} 个观测值）"
        )

    tables: list[TableResult] = []
    charts: list[ChartResult] = []

    # ─── 1. 拟合各模型 ────────────────────────────────────────────────────────
    m1 = _fit_logit(df_main, [exposure])

    m2: dict[str, Any] | None = None
    if m2_expanded and m2_expanded != m3_expanded:
        m2 = _fit_logit(df_main, [exposure] + m2_expanded)

    m3 = _fit_logit(df_main, [exposure] + m3_expanded)

    # ─── 完全分离检测 ─────────────────────────────────────────────────────────
    warnings.extend(_check_separation(m1, "粗模型"))
    if m2:
        warnings.extend(_check_separation(m2, "部分调整模型"))
    warnings.extend(_check_separation(m3, "全调整模型"))

    # ─── 2. 提取暴露 OR 统计 ──────────────────────────────────────────────────
    crude_or_stats = _get_exposure_or_stats(m1, exposure)
    m2_or_stats = _get_exposure_or_stats(m2, exposure) if m2 else None
    m3_or_stats = _get_exposure_or_stats(m3, exposure)
    crude_log_or: float = crude_or_stats["log_or"]

    # ─── 3. 各模型 AUC ───────────────────────────────────────────────────────
    auc1 = _compute_model_auc(m1, df_main)
    auc2 = _compute_model_auc(m2, df_main) if m2 else None
    auc3 = _compute_model_auc(m3, df_main)

    # ─── 4. 模型对比表 ───────────────────────────────────────────────────────
    models_for_comparison: list[dict[str, Any]] = []

    if mode in ("both", "crude"):
        models_for_comparison.append({
            "label": "Model 1（粗模型）",
            "covs": "无协变量",
            "n": n_main,
            **crude_or_stats,
            "auc": auc1,
            "log_or_pct": None,
        })

    if m2 and m2_or_stats and mode == "both":
        pct2 = _pct_change_log(crude_log_or, m2_or_stats["log_or"])
        models_for_comparison.append({
            "label": "Model 2（部分调整）",
            "covs": _covs_label(m2_expanded, dummy_map, valid_raw_covs),
            "n": n_main,
            **m2_or_stats,
            "auc": auc2,
            "log_or_pct": pct2,
        })

    if mode in ("both", "adjusted"):
        pct3 = _pct_change_log(crude_log_or, m3_or_stats["log_or"]) if mode == "both" else None
        m3_label = "Model 3（全调整）" if mode == "both" else "调整模型"
        models_for_comparison.append({
            "label": m3_label,
            "covs": _covs_label(m3_expanded, dummy_map, valid_raw_covs) or "无协变量",
            "n": n_main,
            **m3_or_stats,
            "auc": auc3,
            "log_or_pct": pct3,
        })

    tables.append(_build_model_comparison_table(models_for_comparison, exposure, outcome))

    # ─── 5. 最终模型完整系数表 ───────────────────────────────────────────────
    tables.append(_build_full_coef_table(m3, outcome, exposure))

    # ─── 6. 混杂评估表 ──────────────────────────────────────────────────────
    confounding_rows: list[list[Any]] = []
    if valid_raw_covs:
        confounding_rows = _compute_confounding(
            df_main, exposure, valid_raw_covs, dummy_map, crude_log_or, warnings
        )
        tables.append(TableResult(
            title="混杂评估——各协变量对暴露 OR 的影响",
            headers=["协变量", "单独调整后暴露 OR", "95% CI", "p 值", "log(OR) 变化 (%)", "混杂判断"],
            rows=confounding_rows,
        ))

    # ─── 7. 模型性能对比表 ──────────────────────────────────────────────────
    auc_table_rows: list[list[Any]] = []
    for m in models_for_comparison:
        auc_val = m.get("auc")
        auc_str = f"{auc_val:.4f}" if auc_val is not None and not math.isnan(auc_val) else "—"
        auc_table_rows.append([m["label"], auc_str])
    tables.append(TableResult(
        title="模型性能对比（AUC）",
        headers=["模型", "AUC"],
        rows=auc_table_rows,
    ))

    # ─── 8. 分层分析 ─────────────────────────────────────────────────────────
    strata_forest_data: list[dict] = []
    if stratify_var:
        strata_table_rows, strata_forest_data, hetero_p = _stratified_analysis(
            df_main, exposure, m3_expanded, stratify_var, warnings
        )
        tables.append(TableResult(
            title=f"分层分析（按 {stratify_var} 分层）",
            headers=["分层", "n", "OR", "95% CI", "p 值"],
            rows=strata_table_rows,
        ))
        if hetero_p is not None:
            hetero_note = (
                f"Breslow-Day 同质性检验 p = {_fmt_p(hetero_p)}；"
                + ("各层 OR 存在统计显著差异（效应不齐性）" if hetero_p < 0.05 else "各层 OR 无统计显著差异（效应齐性）")
            )
            warnings.append(hetero_note)

    # ─── 9. 交互项检验 ────────────────────────────────────────────────────────
    if interaction_var:
        interaction_rows, interaction_note = _interaction_test(
            df_main, exposure, interaction_var, m3_expanded
        )
        tables.append(TableResult(
            title=f"交互项检验（{exposure} × {interaction_var}）",
            headers=["交互项", "OR", "95% CI", "p 值", "结论"],
            rows=interaction_rows,
        ))
        warnings.append(interaction_note)

    # ─── 10. 图表 ─────────────────────────────────────────────────────────────
    # 10a. 模型对比 OR 森林图
    if len(models_for_comparison) >= 1:
        forest_data = [
            {
                "label": m["label"],
                "beta": round(m["or_val"], 4),
                "ci_lo": round(m["or_ci_lo"], 4),
                "ci_hi": round(m["or_ci_hi"], 4),
                "p": m["p_str"],
                "n": m["n"],
            }
            for m in models_for_comparison
        ]
        charts.append(ChartResult(
            title="多模型暴露效应 OR 森林图",
            chart_type="forest_plot",
            option={
                "forestData": forest_data,
                "nullLine": 1.0,
                "xLabel": f"{exposure} 对 {outcome} 的效应（OR，95% CI）",
                "title": "多模型暴露效应对比（OR）",
            },
        ))

    # 10b. 分层分析森林图
    if strata_forest_data:
        charts.append(ChartResult(
            title=f"分层分析 OR 森林图（{stratify_var}）",
            chart_type="forest_plot",
            option={
                "forestData": strata_forest_data,
                "nullLine": 1.0,
                "xLabel": f"{exposure} 对 {outcome} 的效应（OR，95% CI）",
                "title": f"分层分析：按 {stratify_var} 分层",
            },
        ))

    # 10c. AUC 对比条形图
    charts.append(_build_auc_bar_chart(models_for_comparison))

    # 10d. 协变量混杂贡献图
    if confounding_rows:
        charts.append(_build_covariate_bar_chart(confounding_rows, exposure))

    # ─── 摘要 ──────────────────────────────────────────────────────────────────
    pct_final = _pct_change_log(crude_log_or, m3_or_stats["log_or"])
    confounding_label = (
        "提示存在混杂偏倚" if not math.isnan(pct_final) and abs(pct_final) > 10
        else "混杂影响较小"
    )
    summary = (
        f"Logistic 回归控制混杂分析，因变量：{outcome}（阳性：{positive_label}），"
        f"暴露变量：{exposure}，有效样本量 n = {n_main}。"
        f"粗 OR = {crude_or_stats['or_val']:.4f}，"
        f"全调整后 OR = {m3_or_stats['or_val']:.4f}"
        f"（log(OR) 变化 {pct_final:+.1f}%，{confounding_label}）。"
        f"全调整模型 AUC = {auc3:.4f}。"
    )
    if stratify_var:
        summary += f" 已按 {stratify_var} 进行分层分析。"
    if interaction_var:
        summary += f" 已检验 {exposure} × {interaction_var} 交互效应。"

    return AnalysisResult(
        method="logistic_reg_adjusted",
        tables=tables,
        charts=charts,
        summary=summary,
        warnings=warnings,
    )


# ─────────────────────────────────────────────────────────────────────────────
# 因变量编码（与 logistic_reg.py 相同逻辑）
# ─────────────────────────────────────────────────────────────────────────────

def _encode_outcome(col: pd.Series) -> tuple[pd.Series, str, list[str]]:
    enc_warnings: list[str] = []
    non_null = col.dropna()
    if len(non_null) == 0:
        raise ValueError("因变量全为缺失值")

    if pd.api.types.is_numeric_dtype(col):
        unique_vals = sorted(float(v) for v in non_null.unique())
        if len(unique_vals) > 2:
            raise ValueError(
                f"因变量为数值型但有 {len(unique_vals)} 个唯一值，logistic 回归要求二分类变量"
            )
        if len(unique_vals) == 1:
            raise ValueError("因变量只有一个唯一值，无法进行 logistic 回归")
        if set(unique_vals) == {0.0, 1.0}:
            return col.astype(float), "1", enc_warnings
        lo, hi = unique_vals[0], unique_vals[1]
        enc_warnings.append(f"因变量数值编码：{lo} → 0，{hi} → 1（阳性类别：{hi}）")
        return (
            col.map(lambda x: {lo: 0.0, hi: 1.0}.get(float(x)) if pd.notna(x) else float("nan")),
            str(hi),
            enc_warnings,
        )

    col_str = col.where(col.isna(), col.astype(str))
    unique_str = sorted(col_str.dropna().unique())
    if len(unique_str) > 2:
        raise ValueError(
            f"因变量有 {len(unique_str)} 个唯一类别，logistic 回归要求二分类变量"
        )
    if len(unique_str) == 1:
        raise ValueError("因变量只有一个唯一值，无法进行 logistic 回归")

    if set(unique_str) == {"False", "True"}:
        return col_str.map({"False": 0.0, "True": 1.0}), "True", enc_warnings

    neg_label, pos_label = unique_str[0], unique_str[1]
    enc_warnings.append(
        f"因变量字符串编码：'{neg_label}' → 0，'{pos_label}' → 1（阳性类别：'{pos_label}'）"
    )
    return col_str.map({neg_label: 0.0, pos_label: 1.0}), pos_label, enc_warnings


# ─────────────────────────────────────────────────────────────────────────────
# dummy 编码（与 logistic_reg.py 相同逻辑）
# ─────────────────────────────────────────────────────────────────────────────

def _make_dummies(
    df: pd.DataFrame,
    predictors: list[str],
    categorical_vars: list[str],
    ref_categories: dict[str, str],
) -> tuple[pd.DataFrame, dict[str, list[str]], list[str]]:
    dummy_warnings: list[str] = []
    dummy_map: dict[str, list[str]] = {}
    df_out = df.copy()

    for var in categorical_vars:
        if var not in df.columns:
            continue

        non_null_mask = df_out[var].notna()
        col_str_vals = df_out.loc[non_null_mask, var].astype(str)
        unique_str = sorted(col_str_vals.unique())

        if len(unique_str) < 2:
            dummy_warnings.append(f"分类变量 '{var}' 唯一值不足，已忽略 dummy 编码")
            continue

        ref = str(ref_categories.get(var, unique_str[0]))
        if ref not in unique_str:
            dummy_warnings.append(
                f"变量 '{var}' 参考组 '{ref}' 不存在，已改用：'{unique_str[0]}'"
            )
            ref = unique_str[0]

        ordered = [ref] + sorted(v for v in unique_str if v != ref)
        col_cat = pd.Categorical(col_str_vals, categories=ordered)
        dummies_nonnull = pd.get_dummies(
            pd.Series(col_cat, index=col_str_vals.index),
            prefix=var, drop_first=True, dtype=float,
        )

        dummy_full = pd.DataFrame(
            np.nan, index=df_out.index, columns=dummies_nonnull.columns
        )
        dummy_full.loc[non_null_mask] = dummies_nonnull.values

        for col_name in dummy_full.columns:
            df_out[col_name] = dummy_full[col_name]

        dummy_map[var] = list(dummies_nonnull.columns)
        dummy_warnings.append(
            f"分类变量 '{var}' dummy 编码完成，参考组：'{ref}'，"
            f"哑变量：{list(dummies_nonnull.columns)}"
        )

    return df_out, dummy_map, dummy_warnings


# ─────────────────────────────────────────────────────────────────────────────
# Logistic 拟合辅助
# ─────────────────────────────────────────────────────────────────────────────

def _fit_logit(df: pd.DataFrame, predictors: list[str]) -> dict[str, Any]:
    """用 statsmodels Logit 拟合，df 须含 __y__ 和所有 predictors 列。"""
    import statsmodels.api as sm

    sub = df[["__y__"] + predictors].dropna()
    y = sub["__y__"].values.astype(float)
    X = sm.add_constant(sub[predictors], prepend=True, has_constant="add")

    with _warnings_mod.catch_warnings():
        _warnings_mod.simplefilter("ignore")
        model = sm.Logit(y, X).fit(disp=0, maxiter=300)

    return {"model": model, "n": len(y), "predictors": predictors}


def _get_exposure_or_stats(model_result: dict | None, exposure: str) -> dict[str, Any]:
    """从模型结果提取暴露变量 OR 及统计量。"""
    if model_result is None:
        return {}
    model = model_result["model"]
    beta = float(model.params[exposure])
    se = float(model.bse[exposure])
    ci = model.conf_int()
    ci_lo_log = float(ci.loc[exposure, 0])
    ci_hi_log = float(ci.loc[exposure, 1])
    p = float(model.pvalues[exposure])
    or_val = _safe_exp(beta)
    or_ci_lo = _safe_exp(ci_lo_log)
    or_ci_hi = _safe_exp(ci_hi_log)
    return {
        "log_or": beta,
        "se": se,
        "or_val": or_val,
        "or_ci_lo": or_ci_lo,
        "or_ci_hi": or_ci_hi,
        "p": p,
        "p_str": _fmt_p(p),
    }


def _pct_change_log(crude_log_or: float, adj_log_or: float) -> float:
    """计算 log(OR) 变化百分比。"""
    if math.isnan(crude_log_or) or crude_log_or == 0:
        return float("nan")
    return (adj_log_or - crude_log_or) / abs(crude_log_or) * 100


def _compute_model_auc(model_result: dict | None, df_main: pd.DataFrame) -> float:
    """计算模型 AUC。"""
    if model_result is None:
        return float("nan")
    import statsmodels.api as sm
    from sklearn.metrics import roc_auc_score

    model = model_result["model"]
    predictors = model_result["predictors"]
    sub = df_main[["__y__"] + predictors].dropna()
    if len(sub) == 0:
        return float("nan")
    X = sm.add_constant(sub[predictors], prepend=True, has_constant="add")
    y_pred = np.asarray(model.predict(X), dtype=float)
    y_true = sub["__y__"].values.astype(float)
    try:
        return float(roc_auc_score(y_true, y_pred))
    except Exception:
        return float("nan")


def _covs_label(
    expanded_cols: list[str],
    dummy_map: dict[str, list[str]],
    orig_covs: list[str],
) -> str:
    """将展开哑变量列名映射回原始变量名，返回逗号分隔字符串。"""
    seen = []
    for v in orig_covs:
        if v in dummy_map:
            if any(dc in expanded_cols for dc in dummy_map[v]):
                seen.append(v)
        elif v in expanded_cols:
            seen.append(v)
    return ", ".join(seen)


# ─────────────────────────────────────────────────────────────────────────────
# 模型对比表
# ─────────────────────────────────────────────────────────────────────────────

def _build_model_comparison_table(
    models: list[dict[str, Any]],
    exposure: str,
    outcome: str,
) -> TableResult:
    headers = ["模型", "协变量", "n", "OR", "95% CI", "p 值", "AUC", "log(OR) 变化 (%)"]
    rows: list[list[Any]] = []
    for m in models:
        pct = m.get("log_or_pct")
        if pct is None:
            pct_str = "参照"
        elif math.isnan(pct):
            pct_str = "—"
        else:
            pct_str = f"{pct:+.1f}%"
            if abs(pct) > 10:
                pct_str += " ⚠"

        auc_val = m.get("auc")
        auc_str = f"{auc_val:.4f}" if auc_val is not None and not math.isnan(auc_val) else "—"

        rows.append([
            m["label"],
            m.get("covs", "—"),
            str(m.get("n", "—")),
            f"{m['or_val']:.4f}",
            f"[{m['or_ci_lo']:.4f}, {m['or_ci_hi']:.4f}]",
            m["p_str"],
            auc_str,
            pct_str,
        ])

    return TableResult(
        title=f"逐步调整模型对比（暴露变量：{exposure}，因变量：{outcome}）",
        headers=headers,
        rows=rows,
    )


# ─────────────────────────────────────────────────────────────────────────────
# 完整系数表（最终模型）
# ─────────────────────────────────────────────────────────────────────────────

def _build_full_coef_table(
    model_result: dict[str, Any],
    outcome: str,
    exposure: str,
) -> TableResult:
    model = model_result["model"]
    n = model_result["n"]
    ci_arr = np.asarray(model.conf_int())
    rows: list[list[Any]] = []

    param_names = list(model.params.index)
    for idx, name in enumerate(param_names):
        beta = float(model.params.iloc[idx])
        se = float(model.bse.iloc[idx])
        wald = beta / se if se > 0 else float("nan")
        p = float(model.pvalues.iloc[idx])
        ci_lo = float(ci_arr[idx, 0])
        ci_hi = float(ci_arr[idx, 1])
        or_val = _safe_exp(beta)
        or_lo = _safe_exp(ci_lo)
        or_hi = _safe_exp(ci_hi)

        if name in ("const", "Intercept"):
            display_name = "截距（Intercept）"
            highlight = ""
        else:
            display_name = name
            highlight = "★ " if name == exposure else ""

        rows.append([
            f"{highlight}{display_name}",
            f"{beta:.4f}",
            f"{se:.4f}",
            f"{wald:.3f}",
            _fmt_p(p),
            f"{or_val:.4f}",
            f"[{or_lo:.4f}, {or_hi:.4f}]",
        ])

    llf = float(model.llf)
    llnull = float(model.llnull)
    mcfadden = 1.0 - llf / llnull if llnull != 0 else float("nan")
    fit_note = (
        f"McFadden R² = {mcfadden:.4f}；"
        f"AIC = {float(model.aic):.2f}；"
        f"n = {n}"
    )
    return TableResult(
        title=f"全调整模型系数表（因变量：{outcome}）— {fit_note}",
        headers=["变量", "β", "SE", "Wald z", "p 值", "OR", "95% CI"],
        rows=rows,
    )


# ─────────────────────────────────────────────────────────────────────────────
# 混杂评估（各协变量逐一加入）
# ─────────────────────────────────────────────────────────────────────────────

def _compute_confounding(
    df_main: pd.DataFrame,
    exposure: str,
    valid_raw_covs: list[str],
    dummy_map: dict[str, list[str]],
    crude_log_or: float,
    warnings: list[str],
) -> list[list[Any]]:
    rows: list[list[Any]] = []
    for v in valid_raw_covs:
        exp_cols = dummy_map[v] if v in dummy_map else [v]
        needed = ["__y__", exposure] + exp_cols
        missing = [c for c in needed if c not in df_main.columns]
        if missing:
            rows.append([v, "—", "—", "—", "—", "列缺失"])
            continue
        try:
            m = _fit_logit(df_main[needed], [exposure] + exp_cols)
            s = _get_exposure_or_stats(m, exposure)
            pct = _pct_change_log(crude_log_or, s["log_or"])
            if math.isnan(pct):
                judge = "无法计算"
            elif abs(pct) > 10:
                judge = "存在混杂 (>10%) ⚠"
            else:
                judge = "无明显混杂 (≤10%)"
            rows.append([
                v,
                f"{s['or_val']:.4f}",
                f"[{s['or_ci_lo']:.4f}, {s['or_ci_hi']:.4f}]",
                s["p_str"],
                f"{pct:+.1f}%" if not math.isnan(pct) else "—",
                judge,
            ])
        except Exception as exc:
            logger.warning("协变量 %s 混杂评估失败: %s", v, exc)
            rows.append([v, "—", "—", "—", "—", "计算失败"])
    return rows


# ─────────────────────────────────────────────────────────────────────────────
# 分层分析
# ─────────────────────────────────────────────────────────────────────────────

def _stratified_analysis(
    df_main: pd.DataFrame,
    exposure: str,
    m3_expanded: list[str],
    stratify_var: str,
    warnings: list[str],
) -> tuple[list[list[Any]], list[dict], float | None]:
    strata = sorted(df_main[stratify_var].dropna().unique())
    table_rows: list[list[Any]] = []
    forest_data: list[dict] = []
    log_ors: list[float] = []
    ses: list[float] = []

    # Within each stratum the stratify_var is constant → remove it and its dummies
    strata_exclude = {stratify_var} | {
        col for col in m3_expanded if col.startswith(f"{stratify_var}_")
    }
    per_stratum_covs = [c for c in m3_expanded if c not in strata_exclude]

    for stratum in strata:
        df_s = df_main[df_main[stratify_var] == stratum].copy()
        n_s = len(df_s)
        label = f"{stratify_var} = {stratum}（n = {n_s}）"
        predictors = [exposure] + per_stratum_covs
        min_obs = len(predictors) + 2

        if n_s < min_obs:
            warnings.append(f"分层 {stratify_var}={stratum} 样本量不足（n={n_s}），已跳过")
            table_rows.append([label, str(n_s), "—", "—", "样本量不足"])
            continue

        try:
            m = _fit_logit(df_s, predictors)
            s = _get_exposure_or_stats(m, exposure)
            table_rows.append([
                label,
                str(n_s),
                f"{s['or_val']:.4f}",
                f"[{s['or_ci_lo']:.4f}, {s['or_ci_hi']:.4f}]",
                s["p_str"],
            ])
            forest_data.append({
                "label": label,
                "beta": round(s["or_val"], 4),
                "ci_lo": round(s["or_ci_lo"], 4),
                "ci_hi": round(s["or_ci_hi"], 4),
                "p": s["p_str"],
                "n": n_s,
            })
            log_ors.append(s["log_or"])
            ses.append(s["se"])
        except Exception as exc:
            logger.warning("分层 %s=%s 分析失败: %s", stratify_var, stratum, exc)
            table_rows.append([label, str(n_s), "—", "—", "计算失败"])

    hetero_p: float | None = None
    if len(log_ors) >= 2:
        hetero_p = _breslow_day_test(log_ors, ses)
        q_str = f"Breslow-Day 同质性检验 p = {_fmt_p(hetero_p)}"
        table_rows.append(["── 同质性检验 ──", "", q_str, "", ""])
        forest_data.append({
            "label": f"─── {q_str}",
            "beta": float("nan"),
            "ci_lo": float("nan"),
            "ci_hi": float("nan"),
            "p": "—",
            "n": 0,
            "is_note": True,
        })

    return table_rows, forest_data, hetero_p


def _breslow_day_test(log_ors: list[float], ses: list[float]) -> float:
    """Cochran Q on log(OR)，作为 Breslow-Day 的回归版近似，返回 p 值。"""
    k = len(log_ors)
    weights = [1.0 / (se ** 2) for se in ses]
    w_sum = sum(weights)
    log_or_pooled = sum(w * b for w, b in zip(weights, log_ors)) / w_sum
    Q = sum(w * (b - log_or_pooled) ** 2 for w, b in zip(weights, log_ors))
    return float(1 - stats.chi2.cdf(Q, df=k - 1))


# ─────────────────────────────────────────────────────────────────────────────
# 交互项检验
# ─────────────────────────────────────────────────────────────────────────────

def _interaction_test(
    df_main: pd.DataFrame,
    exposure: str,
    interaction_var: str,
    m3_expanded: list[str],
) -> tuple[list[list[Any]], str]:
    df_int = df_main.copy()
    int_col = f"{exposure}_x_{interaction_var}"
    df_int[int_col] = df_int[exposure] * df_int[interaction_var]

    base_predictors = [exposure]
    if interaction_var not in base_predictors and interaction_var not in m3_expanded:
        base_predictors.append(interaction_var)
    other_covs = [c for c in m3_expanded if c not in base_predictors]
    all_predictors = base_predictors + other_covs + [int_col]

    try:
        m = _fit_logit(df_int, all_predictors)
        model = m["model"]
        beta = float(model.params[int_col])
        se = float(model.bse[int_col])
        ci = model.conf_int()
        ci_lo = float(ci.loc[int_col, 0])
        ci_hi = float(ci.loc[int_col, 1])
        p = float(model.pvalues[int_col])
        or_val = _safe_exp(beta)
        or_lo = _safe_exp(ci_lo)
        or_hi = _safe_exp(ci_hi)

        conclusion_text = (
            f"存在统计显著交互效应（p = {_fmt_p(p)}）"
            if p < 0.05
            else f"无统计显著交互效应（p = {_fmt_p(p)}）"
        )
        rows = [[
            f"{exposure} × {interaction_var}",
            f"{or_val:.4f}",
            f"[{or_lo:.4f}, {or_hi:.4f}]",
            _fmt_p(p),
            "有交互效应 ⚠" if p < 0.05 else "无显著交互",
        ]]
        note = f"交互检验（{exposure} × {interaction_var}）：{conclusion_text}"
        return rows, note
    except Exception as exc:
        logger.warning("交互项检验失败: %s", exc)
        rows = [[f"{exposure} × {interaction_var}", "—", "—", "—", "计算失败"]]
        note = f"交互检验失败：{exc}"
        return rows, note


# ─────────────────────────────────────────────────────────────────────────────
# 图表：AUC 对比条形图
# ─────────────────────────────────────────────────────────────────────────────

def _build_auc_bar_chart(models: list[dict[str, Any]]) -> ChartResult:
    labels: list[str] = []
    values: list[float] = []
    for m in models:
        auc_val = m.get("auc")
        if auc_val is None or math.isnan(auc_val):
            continue
        labels.append(m["label"])
        values.append(round(auc_val, 4))

    y_min = max(0.4, min(values) - 0.05) if values else 0.5

    option: dict[str, Any] = {
        "title": {
            "text": "各模型 AUC 对比",
            "left": "center",
            "textStyle": {"fontSize": 13},
        },
        "tooltip": {
            "trigger": "axis",
            "formatter": "{b}：AUC = {c}",
            "axisPointer": {"type": "shadow"},
        },
        "grid": {"left": "12%", "right": "8%", "top": "18%", "bottom": "18%"},
        "xAxis": {
            "type": "category",
            "data": labels,
            "axisLabel": {"rotate": 15, "fontSize": 11},
        },
        "yAxis": {
            "type": "value",
            "name": "AUC",
            "min": round(y_min, 2),
            "max": 1.0,
            "splitLine": {"lineStyle": {"type": "dashed"}},
        },
        "series": [{
            "type": "bar",
            "data": [
                {"value": v, "itemStyle": {"color": "#5470c6" if i == 0 else "#91cc75" if i < len(values) - 1 else "#ee6666"}}
                for i, v in enumerate(values)
            ],
            "barMaxWidth": 60,
            "label": {
                "show": True,
                "position": "top",
                "formatter": "{c}",
                "fontSize": 11,
            },
        }],
    }
    return ChartResult(title="各模型 AUC 对比", chart_type="bar", option=option)


# ─────────────────────────────────────────────────────────────────────────────
# 图表：协变量混杂贡献条形图
# ─────────────────────────────────────────────────────────────────────────────

def _build_covariate_bar_chart(
    confounding_rows: list[list[Any]],
    exposure: str,
) -> ChartResult:
    labels: list[str] = []
    values: list[float] = []
    colors: list[str] = []

    for row in confounding_rows:
        cov_name = str(row[0])
        pct_str = str(row[4])
        if pct_str in ("—", "计算失败"):
            continue
        try:
            pct_val = float(pct_str.replace("%", "").replace("⚠", "").strip())
        except ValueError:
            continue
        labels.append(cov_name)
        values.append(round(pct_val, 2))
        colors.append("#ee6666" if abs(pct_val) > 10 else "#5470c6")

    option: dict[str, Any] = {
        "title": {
            "text": f"协变量对 {exposure} OR 的影响（log(OR) 变化 %）",
            "left": "center",
            "textStyle": {"fontSize": 13},
        },
        "tooltip": {
            "trigger": "axis",
            "formatter": "{b}：{c}%",
            "axisPointer": {"type": "shadow"},
        },
        "grid": {"left": "18%", "right": "10%", "top": "15%", "bottom": "10%"},
        "xAxis": {
            "type": "value",
            "name": "log(OR) 变化 (%)",
            "axisLine": {"show": True},
            "splitLine": {"lineStyle": {"type": "dashed"}},
        },
        "yAxis": {
            "type": "category",
            "data": labels,
            "inverse": True,
        },
        "series": [{
            "type": "bar",
            "data": [
                {"value": v, "itemStyle": {"color": c}}
                for v, c in zip(values, colors)
            ],
            "barMaxWidth": 40,
            "label": {
                "show": True,
                "position": "right",
                "formatter": "{c}%",
                "fontSize": 11,
            },
        }],
    }
    return ChartResult(
        title="协变量混杂贡献图",
        chart_type="bar",
        option=option,
    )


# ─────────────────────────────────────────────────────────────────────────────
# 辅助格式化
# ─────────────────────────────────────────────────────────────────────────────

def _fmt_p(p: float) -> str:
    if math.isnan(p):
        return "—"
    if p < 0.001:
        return "< 0.001"
    return f"{p:.3f}"
