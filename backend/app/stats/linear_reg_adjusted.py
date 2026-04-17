"""线性回归控制混杂偏倚模块。

功能：
  1. 逐步调整模型 (Model 1 粗模型 → Model 2 部分调整 → Model 3 全调整)
  2. 暴露变量系数变化追踪 (crude β → adjusted β 变化百分比，>10% 提示混杂)
  3. 分层分析 (按分层变量各水平单独拟合，Cochran Q 异质性检验)
  4. 交互项检验 (暴露 × 修饰因子，输出交互 p 值)
  5. 协变量筛选辅助 (每个候选协变量单独加入时对暴露系数的影响)

输入 params：
  outcome          str           因变量（连续型）
  exposure         str           暴露变量（研究主自变量）
  covariates       list[str]     所有候选协变量
  mode             str           "both"（默认）| "crude" | "adjusted"
  model2_covariates list[str]   Model 2 协变量（省略则自动取前半）
  model3_covariates list[str]   Model 3 协变量（省略则用全部 covariates）
  stratify_var     str | None    分层变量（可选）
  interaction_var  str | None    交互项变量（可选）

输出 AnalysisResult：
  tables:  模型对比表、完整系数表、混杂评估表、分层分析表、交互检验表
  charts:  森林图 (forest_plot)、分层森林图 (forest_plot)、协变量贡献条形图
"""

import logging
import math
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats

from app.models.analysis import AnalysisResult, ChartResult, TableResult

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# 主入口
# ─────────────────────────────────────────────────────────────────────────────

def run(df: pd.DataFrame, params: dict) -> AnalysisResult:
    outcome: str = str(params.get("outcome", ""))
    exposure: str = str(params.get("exposure", ""))
    covariates: list[str] = list(params.get("covariates") or [])
    mode: str = str(params.get("mode", "both")).lower()
    model2_covariates: list[str] | None = params.get("model2_covariates")
    model3_covariates: list[str] | None = params.get("model3_covariates")
    stratify_var: str | None = params.get("stratify_var") or None
    interaction_var: str | None = params.get("interaction_var") or None
    warnings: list[str] = []

    # ── 参数校验 ──────────────────────────────────────────────────────────────
    if not outcome:
        raise ValueError("请指定因变量 (outcome)")
    if outcome not in df.columns:
        raise ValueError(f"因变量 '{outcome}' 不存在于数据集中")
    if not pd.api.types.is_numeric_dtype(df[outcome]):
        raise ValueError(f"因变量 '{outcome}' 必须为数值型连续变量")
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

    # ── 协变量过滤 ────────────────────────────────────────────────────────────
    valid_covs: list[str] = []
    for v in covariates:
        if v in (outcome, exposure):
            warnings.append(f"协变量 '{v}' 与因/暴露变量相同，已忽略")
            continue
        if v not in df.columns:
            warnings.append(f"协变量 '{v}' 不存在，已忽略")
            continue
        if not pd.api.types.is_numeric_dtype(df[v]):
            warnings.append(f"协变量 '{v}' 非数值型，已忽略（分类变量请先哑变量编码）")
            continue
        valid_covs.append(v)

    # ── 确定各模型协变量 ──────────────────────────────────────────────────────
    # Model 2: 部分调整（默认前半部分协变量）
    if model2_covariates is None:
        n_half = max(1, math.ceil(len(valid_covs) / 2)) if valid_covs else 0
        m2_covs = valid_covs[:n_half]
    else:
        m2_covs = [c for c in model2_covariates if c in valid_covs]

    # Model 3: 全调整（默认所有协变量）
    if model3_covariates is None:
        m3_covs = valid_covs
    else:
        m3_covs = [c for c in model3_covariates if c in valid_covs]

    # ── 分层变量校验 ──────────────────────────────────────────────────────────
    if stratify_var:
        if stratify_var not in df.columns:
            warnings.append(f"分层变量 '{stratify_var}' 不存在，已忽略分层分析")
            stratify_var = None

    # ── 交互变量校验 ──────────────────────────────────────────────────────────
    if interaction_var:
        if interaction_var not in df.columns:
            warnings.append(f"交互变量 '{interaction_var}' 不存在，已忽略交互检验")
            interaction_var = None
        elif not pd.api.types.is_numeric_dtype(df[interaction_var]):
            warnings.append(f"交互变量 '{interaction_var}' 非数值型，已忽略交互检验")
            interaction_var = None

    # ── 准备主分析数据（去 NA） ────────────────────────────────────────────────
    main_cols = [outcome, exposure] + m3_covs
    if stratify_var:
        main_cols.append(stratify_var)
    if interaction_var and interaction_var not in main_cols:
        main_cols.append(interaction_var)

    df_main = df[list(dict.fromkeys(main_cols))].dropna()
    n_main = len(df_main)

    min_needed = 2 + len(m3_covs)
    if n_main < min_needed:
        raise ValueError(
            f"有效样本量 ({n_main}) 不足以拟合模型（需要至少 {min_needed} 个观测值）"
        )

    tables: list[TableResult] = []
    charts: list[ChartResult] = []

    # ─── 1. 拟合各模型 ────────────────────────────────────────────────────────
    # Model 1: 粗模型
    m1_result = _fit_ols(df_main, outcome, [exposure])

    # Model 2: 部分调整
    m2_result = None
    if m2_covs and m2_covs != m3_covs:
        m2_predictors = [exposure] + m2_covs
        m2_result = _fit_ols(df_main, outcome, m2_predictors)

    # Model 3: 全调整
    m3_predictors = [exposure] + m3_covs
    m3_result = _fit_ols(df_main, outcome, m3_predictors)

    # ─── 2. 暴露系数提取 ─────────────────────────────────────────────────────
    crude_stats = _get_exposure_stats(m1_result, exposure)
    m2_stats = _get_exposure_stats(m2_result, exposure) if m2_result else None
    m3_stats = _get_exposure_stats(m3_result, exposure)
    crude_beta = crude_stats["beta"]

    # ─── 3. 模型对比表 ───────────────────────────────────────────────────────
    models_for_comparison: list[dict[str, Any]] = []

    if mode in ("both", "crude"):
        m1_r2 = float(m1_result["model"].rsquared)
        m1_r2_adj = float(m1_result["model"].rsquared_adj)
        models_for_comparison.append({
            "label": "Model 1（粗模型）",
            "covs": "无协变量",
            "n": n_main,
            **crude_stats,
            "r2": m1_r2,
            "r2_adj": m1_r2_adj,
            "beta_pct": None,
        })

    if m2_result and mode in ("both",):
        m2_r2 = float(m2_result["model"].rsquared)
        m2_r2_adj = float(m2_result["model"].rsquared_adj)
        pct2 = _pct_change(crude_beta, m2_stats["beta"])
        models_for_comparison.append({
            "label": "Model 2（部分调整）",
            "covs": ", ".join(m2_covs) if m2_covs else "无",
            "n": n_main,
            **m2_stats,
            "r2": m2_r2,
            "r2_adj": m2_r2_adj,
            "beta_pct": pct2,
        })

    if mode in ("both", "adjusted"):
        m3_r2 = float(m3_result["model"].rsquared)
        m3_r2_adj = float(m3_result["model"].rsquared_adj)
        pct3 = _pct_change(crude_beta, m3_stats["beta"]) if mode == "both" else None
        m3_label = "Model 3（全调整）" if mode == "both" else "调整模型"
        models_for_comparison.append({
            "label": m3_label,
            "covs": ", ".join(m3_covs) if m3_covs else "无",
            "n": n_main,
            **m3_stats,
            "r2": m3_r2,
            "r2_adj": m3_r2_adj,
            "beta_pct": pct3,
        })

    tables.append(_build_model_comparison_table(models_for_comparison, exposure, outcome))

    # ─── 4. 最终模型完整系数表 ───────────────────────────────────────────────
    tables.append(_build_full_coef_table(m3_result, outcome, exposure, m3_covs))

    # ─── 5. 混杂评估表 ──────────────────────────────────────────────────────
    if valid_covs:
        confounding_rows = _compute_confounding(df_main, outcome, exposure, valid_covs, crude_beta, warnings)
        tables.append(
            TableResult(
                title="混杂评估——各协变量对暴露效应的影响",
                headers=["协变量", "单独调整后暴露 β", "95% CI", "p 值", "β 变化 (%)", "混杂判断"],
                rows=confounding_rows,
            )
        )

    # ─── 6. 分层分析 ─────────────────────────────────────────────────────────
    strata_forest_data: list[dict] = []
    if stratify_var:
        strata_table_rows, strata_forest_data, hetero_p = _stratified_analysis(
            df_main, outcome, exposure, m3_covs, stratify_var, warnings
        )
        tables.append(
            TableResult(
                title=f"分层分析（按 {stratify_var} 分层）",
                headers=["分层", "n", "β", "SE", "95% CI", "p 值"],
                rows=strata_table_rows,
            )
        )
        # 异质性检验结论
        if hetero_p is not None:
            hetero_note = (
                f"Cochran Q 检验异质性 p = {_fmt_p(hetero_p)}；"
                + ("各层效应存在统计显著差异" if hetero_p < 0.05 else "各层效应无统计显著差异")
            )
            warnings.append(hetero_note)

    # ─── 7. 交互项检验 ────────────────────────────────────────────────────────
    if interaction_var:
        interaction_rows, interaction_note = _interaction_test(
            df_main, outcome, exposure, interaction_var, m3_covs
        )
        tables.append(
            TableResult(
                title=f"交互项检验（{exposure} × {interaction_var}）",
                headers=["交互项", "β", "SE", "95% CI", "p 值", "结论"],
                rows=interaction_rows,
            )
        )
        warnings.append(interaction_note)

    # ─── 8. 图表 ─────────────────────────────────────────────────────────────
    # 8a. 模型对比森林图
    if len(models_for_comparison) >= 1:
        forest_data = [
            {
                "label": m["label"],
                "beta": round(m["beta"], 4),
                "ci_lo": round(m["ci_lo"], 4),
                "ci_hi": round(m["ci_hi"], 4),
                "p": m["p_str"],
                "n": m["n"],
            }
            for m in models_for_comparison
        ]
        charts.append(
            ChartResult(
                title="多模型暴露效应森林图",
                chart_type="forest_plot",
                option={
                    "forestData": forest_data,
                    "nullLine": 0,
                    "xLabel": f"{exposure} 对 {outcome} 的效应（β，95% CI）",
                    "title": "多模型暴露效应对比",
                },
            )
        )

    # 8b. 分层分析森林图
    if strata_forest_data:
        charts.append(
            ChartResult(
                title=f"分层分析森林图（{stratify_var}）",
                chart_type="forest_plot",
                option={
                    "forestData": strata_forest_data,
                    "nullLine": 0,
                    "xLabel": f"{exposure} 对 {outcome} 的效应（β，95% CI）",
                    "title": f"分层分析：按 {stratify_var} 分层",
                },
            )
        )

    # 8c. 协变量贡献条形图
    if valid_covs:
        cov_bar_chart = _build_covariate_bar_chart(confounding_rows, exposure)
        charts.append(cov_bar_chart)

    # ─── 摘要 ──────────────────────────────────────────────────────────────────
    pct_final = _pct_change(crude_beta, m3_stats["beta"])
    confounding_label = "提示存在混杂偏倚" if abs(pct_final) > 10 else "混杂影响较小"
    summary = (
        f"线性回归控制混杂分析，因变量：{outcome}，暴露变量：{exposure}，"
        f"有效样本量 n = {n_main}。"
        f"粗效应 β = {crude_beta:.4f}，"
        f"全调整后 β = {m3_stats['beta']:.4f}（变化 {pct_final:+.1f}%，{confounding_label}）。"
        f"调整 R² = {float(m3_result['model'].rsquared_adj):.4f}。"
    )
    if stratify_var:
        summary += f" 已按 {stratify_var} 进行分层分析。"
    if interaction_var:
        summary += f" 已检验 {exposure} × {interaction_var} 交互效应。"

    return AnalysisResult(
        method="linear_reg_adjusted",
        tables=tables,
        charts=charts,
        summary=summary,
        warnings=warnings,
    )


# ─────────────────────────────────────────────────────────────────────────────
# OLS 拟合辅助
# ─────────────────────────────────────────────────────────────────────────────

def _fit_ols(df_clean: pd.DataFrame, outcome: str, predictors: list[str]) -> dict[str, Any]:
    """用 statsmodels OLS 拟合，返回含命名 params 的 model dict。"""
    import statsmodels.api as sm

    y = df_clean[outcome].values.astype(float)
    X = sm.add_constant(df_clean[predictors], prepend=True, has_constant="add")
    model = sm.OLS(y, X).fit()
    return {"model": model, "n": len(y), "predictors": predictors}


def _get_exposure_stats(model_result: dict | None, exposure: str) -> dict[str, Any]:
    """从模型结果中提取暴露变量统计量。"""
    if model_result is None:
        return {}
    model = model_result["model"]
    beta = float(model.params[exposure])
    se = float(model.bse[exposure])
    ci = model.conf_int()
    ci_lo = float(ci.loc[exposure, 0])
    ci_hi = float(ci.loc[exposure, 1])
    p = float(model.pvalues[exposure])
    return {
        "beta": beta,
        "se": se,
        "ci_lo": ci_lo,
        "ci_hi": ci_hi,
        "p": p,
        "p_str": _fmt_p(p),
    }


def _pct_change(crude_beta: float, adj_beta: float) -> float:
    """计算 β 变化百分比。"""
    if crude_beta == 0 or math.isnan(crude_beta):
        return float("nan")
    return (adj_beta - crude_beta) / abs(crude_beta) * 100


# ─────────────────────────────────────────────────────────────────────────────
# 模型对比表
# ─────────────────────────────────────────────────────────────────────────────

def _build_model_comparison_table(
    models: list[dict[str, Any]],
    exposure: str,
    outcome: str,
) -> TableResult:
    headers = ["模型", "协变量", "n", "β", "SE", "95% CI", "p 值", "R²", "调整 R²", "β 变化 (%)"]
    rows: list[list[Any]] = []
    for m in models:
        pct = m.get("beta_pct")
        if pct is None:
            pct_str = "参照"
        elif math.isnan(pct):
            pct_str = "—"
        else:
            pct_str = f"{pct:+.1f}%"
            if abs(pct) > 10:
                pct_str += " ⚠"

        rows.append([
            m["label"],
            m.get("covs", "—"),
            str(m.get("n", "—")),
            f"{m['beta']:.4f}",
            f"{m['se']:.4f}",
            f"[{m['ci_lo']:.4f}, {m['ci_hi']:.4f}]",
            m["p_str"],
            f"{m['r2']:.4f}",
            f"{m['r2_adj']:.4f}",
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
    covariates: list[str],
) -> TableResult:
    model = model_result["model"]
    predictors = model_result["predictors"]
    n = model_result["n"]

    y = model.model.endog
    y_std = float(np.std(y, ddof=1)) or 1.0

    # 对 X（不含截距）的标准差，用于标准化 β
    X_raw = model.model.exog[:, 1:]  # 去掉截距列
    X_stds = [float(np.std(X_raw[:, k], ddof=1)) or 1.0 for k in range(X_raw.shape[1])]

    ci_arr = np.asarray(model.conf_int())
    rows: list[list[Any]] = []

    param_names = list(model.params.index)
    for idx, name in enumerate(param_names):
        beta = float(model.params.iloc[idx])
        se = float(model.bse.iloc[idx])
        t_val = float(model.tvalues.iloc[idx])
        p_val = float(model.pvalues.iloc[idx])
        ci_lo = float(ci_arr[idx, 0])
        ci_hi = float(ci_arr[idx, 1])

        if name in ("const", "Intercept"):
            display_name = "截距（Intercept）"
            std_beta = "—"
            highlight = ""
        else:
            display_name = name
            pred_idx = predictors.index(name) if name in predictors else -1
            std_beta = (
                f"{beta * X_stds[pred_idx] / y_std:.4f}"
                if pred_idx >= 0
                else "—"
            )
            highlight = "★ " if name == exposure else ""

        rows.append([
            f"{highlight}{display_name}",
            f"{beta:.4f}",
            f"{se:.4f}",
            f"{t_val:.3f}",
            _fmt_p(p_val),
            f"[{ci_lo:.4f}, {ci_hi:.4f}]",
            std_beta,
        ])

    # 追加拟合指标行（分隔线）
    model_obj = model
    fit_note = (
        f"R² = {model_obj.rsquared:.4f}；"
        f"调整 R² = {model_obj.rsquared_adj:.4f}；"
        f"AIC = {model_obj.aic:.2f}；"
        f"n = {n}"
    )

    return TableResult(
        title=f"全调整模型系数表（因变量：{outcome}）— {fit_note}",
        headers=["变量", "β", "SE", "t 值", "p 值", "95% CI", "标准化 β"],
        rows=rows,
    )


# ─────────────────────────────────────────────────────────────────────────────
# 混杂评估（各协变量逐一加入）
# ─────────────────────────────────────────────────────────────────────────────

def _compute_confounding(
    df_main: pd.DataFrame,
    outcome: str,
    exposure: str,
    covariates: list[str],
    crude_beta: float,
    warnings: list[str],
) -> list[list[Any]]:
    rows: list[list[Any]] = []
    for cov in covariates:
        try:
            m = _fit_ols(df_main, outcome, [exposure, cov])
            s = _get_exposure_stats(m, exposure)
            pct = _pct_change(crude_beta, s["beta"])
            if math.isnan(pct):
                judge = "无法计算"
            elif abs(pct) > 10:
                judge = "存在混杂 (>10%) ⚠"
            else:
                judge = "无明显混杂 (≤10%)"

            rows.append([
                cov,
                f"{s['beta']:.4f}",
                f"[{s['ci_lo']:.4f}, {s['ci_hi']:.4f}]",
                s["p_str"],
                f"{pct:+.1f}%" if not math.isnan(pct) else "—",
                judge,
            ])
        except Exception as exc:
            logger.warning("协变量 %s 混杂评估失败: %s", cov, exc)
            rows.append([cov, "—", "—", "—", "—", "计算失败"])

    return rows


# ─────────────────────────────────────────────────────────────────────────────
# 分层分析
# ─────────────────────────────────────────────────────────────────────────────

def _stratified_analysis(
    df_main: pd.DataFrame,
    outcome: str,
    exposure: str,
    covariates: list[str],
    stratify_var: str,
    warnings: list[str],
) -> tuple[list[list[Any]], list[dict], float | None]:
    """按分层变量各水平分别拟合，返回 (table_rows, forest_data, hetero_p)。"""
    strata = sorted(df_main[stratify_var].dropna().unique())
    table_rows: list[list[Any]] = []
    forest_data: list[dict] = []
    betas: list[float] = []
    ses: list[float] = []

    for stratum in strata:
        df_s = df_main[df_main[stratify_var] == stratum].copy()
        n_s = len(df_s)
        label = f"{stratify_var} = {stratum}（n = {n_s}）"

        predictors = [exposure] + covariates
        min_obs = len(predictors) + 2
        if n_s < min_obs:
            warnings.append(f"分层 {stratify_var}={stratum} 样本量不足（n={n_s}），已跳过")
            table_rows.append([label, str(n_s), "—", "—", "—", "样本量不足"])
            continue

        try:
            m = _fit_ols(df_s, outcome, predictors)
            s = _get_exposure_stats(m, exposure)
            table_rows.append([
                label,
                str(n_s),
                f"{s['beta']:.4f}",
                f"{s['se']:.4f}",
                f"[{s['ci_lo']:.4f}, {s['ci_hi']:.4f}]",
                s["p_str"],
            ])
            forest_data.append({
                "label": label,
                "beta": round(s["beta"], 4),
                "ci_lo": round(s["ci_lo"], 4),
                "ci_hi": round(s["ci_hi"], 4),
                "p": s["p_str"],
                "n": n_s,
            })
            betas.append(s["beta"])
            ses.append(s["se"])
        except Exception as exc:
            logger.warning("分层 %s=%s 分析失败: %s", stratify_var, stratum, exc)
            table_rows.append([label, str(n_s), "—", "—", "—", "计算失败"])

    # ── Cochran Q 异质性检验 ──────────────────────────────────────────────────
    hetero_p: float | None = None
    if len(betas) >= 2:
        hetero_p = _cochran_q_test(betas, ses)
        q_str = f"Cochran Q 检验 p = {_fmt_p(hetero_p)}"
        table_rows.append(["── 异质性检验 ──", "", q_str, "", "", ""])
        forest_data.append({
            "label": f"─── 异质性 {q_str}",
            "beta": float("nan"),
            "ci_lo": float("nan"),
            "ci_hi": float("nan"),
            "p": "—",
            "n": 0,
            "is_note": True,
        })

    return table_rows, forest_data, hetero_p


def _cochran_q_test(betas: list[float], ses: list[float]) -> float:
    """Cochran Q 检验，返回 p 值。"""
    k = len(betas)
    weights = [1.0 / (se ** 2) for se in ses]
    w_sum = sum(weights)
    beta_pooled = sum(w * b for w, b in zip(weights, betas)) / w_sum
    Q = sum(w * (b - beta_pooled) ** 2 for w, b in zip(weights, betas))
    p = float(1 - stats.chi2.cdf(Q, df=k - 1))
    return p


# ─────────────────────────────────────────────────────────────────────────────
# 交互项检验
# ─────────────────────────────────────────────────────────────────────────────

def _interaction_test(
    df_main: pd.DataFrame,
    outcome: str,
    exposure: str,
    interaction_var: str,
    covariates: list[str],
) -> tuple[list[list[Any]], str]:
    """在模型中加入交互项，检验效应修饰。"""
    df_int = df_main.copy()
    int_col = f"{exposure}_x_{interaction_var}"
    df_int[int_col] = df_int[exposure] * df_int[interaction_var]

    # 确保交互变量本身在模型中
    base_predictors = [exposure]
    if interaction_var not in base_predictors:
        base_predictors.append(interaction_var)
    other_covs = [c for c in covariates if c not in base_predictors]
    all_predictors = base_predictors + other_covs + [int_col]

    try:
        m = _fit_ols(df_int, outcome, all_predictors)
        model = m["model"]
        beta = float(model.params[int_col])
        se = float(model.bse[int_col])
        ci = model.conf_int()
        ci_lo = float(ci.loc[int_col, 0])
        ci_hi = float(ci.loc[int_col, 1])
        p = float(model.pvalues[int_col])
        p_str = _fmt_p(p)

        if p < 0.05:
            conclusion = f"存在统计显著交互效应（p = {p_str}），{exposure} 对 {outcome} 的效应受 {interaction_var} 修饰"
        else:
            conclusion = f"无统计显著交互效应（p = {p_str}）"

        rows = [[
            f"{exposure} × {interaction_var}",
            f"{beta:.4f}",
            f"{se:.4f}",
            f"[{ci_lo:.4f}, {ci_hi:.4f}]",
            p_str,
            "有交互效应 ⚠" if p < 0.05 else "无显著交互",
        ]]
        note = f"交互检验（{exposure} × {interaction_var}）：{conclusion}"
        return rows, note

    except Exception as exc:
        logger.warning("交互项检验失败: %s", exc)
        rows = [[f"{exposure} × {interaction_var}", "—", "—", "—", "—", "计算失败"]]
        note = f"交互检验失败：{exc}"
        return rows, note


# ─────────────────────────────────────────────────────────────────────────────
# 图表：协变量贡献条形图
# ─────────────────────────────────────────────────────────────────────────────

def _build_covariate_bar_chart(
    confounding_rows: list[list[Any]],
    exposure: str,
) -> ChartResult:
    """条形图展示各协变量对暴露系数变化的贡献（%）。"""
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
            "text": f"协变量对 {exposure} 效应的影响（β 变化 %）",
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
            "name": "β 变化 (%)",
            "axisLine": {"show": True},
            "splitLine": {"lineStyle": {"type": "dashed"}},
        },
        "yAxis": {
            "type": "category",
            "data": labels,
            "inverse": True,
        },
        "markLine": {
            "data": [{"xAxis": 0, "lineStyle": {"type": "solid", "color": "#333"}}]
        },
        "series": [
            {
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
            }
        ],
    }

    # 添加参考线（10%）
    option["series"].append(  # type: ignore[attr-defined]
        {
            "type": "line",
            "data": [[10, labels[0]] if labels else [], [10, labels[-1]] if labels else []],
            "showSymbol": False,
            "lineStyle": {"type": "dashed", "color": "#ee6666", "width": 1},
            "silent": True,
        }
    )

    return ChartResult(
        title=f"协变量对暴露效应的影响",
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
