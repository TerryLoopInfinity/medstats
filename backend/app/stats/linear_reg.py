"""线性回归分析模块。

支持：
  - 单变量分析：每个自变量单独对因变量做简单线性回归
  - 多变量分析：所有自变量同时进入模型（强制进入法）
  - 输出：回归系数表（β、SE、t、p、95% CI、标准化 β）
          模型拟合指标（R²、调整 R²、F 统计量、AIC、BIC）
          残差诊断（正态性检验、Breusch-Pagan、Durbin-Watson）
          共线性诊断（VIF）
  - 图表：实际值 vs 预测值、残差 vs 拟合值、残差 QQ 图
"""

import logging
import math
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats

from app.models.analysis import AnalysisResult, ChartResult, TableResult

logger = logging.getLogger(__name__)

_SW_MAX_N = 5000


# ─────────────────────────────────────────────────────────────────────────────
# 主入口
# ─────────────────────────────────────────────────────────────────────────────

def run(df: pd.DataFrame, params: dict) -> AnalysisResult:
    """
    参数
    ----
    params["outcome"]      str          因变量（连续型）
    params["predictors"]   list[str]    自变量列表
    params["mode"]         str          "both"（默认）| "univariate" | "multivariate"
    """
    outcome: str = str(params.get("outcome", ""))
    predictors: list[str] = list(params.get("predictors") or [])
    mode: str = str(params.get("mode", "both")).lower()
    warnings: list[str] = []

    # ── 参数校验 ──────────────────────────────────────────────────────────────
    if not outcome:
        raise ValueError("请指定因变量 (outcome)")
    if outcome not in df.columns:
        raise ValueError(f"因变量 '{outcome}' 不存在于数据集中")
    if not pd.api.types.is_numeric_dtype(df[outcome]):
        raise ValueError(f"因变量 '{outcome}' 必须为数值型连续变量")
    if not predictors:
        raise ValueError("请至少选择一个自变量 (predictors)")
    if mode not in ("both", "univariate", "multivariate"):
        raise ValueError("mode 必须为 both / univariate / multivariate")

    # ── 过滤无效自变量 ────────────────────────────────────────────────────────
    valid_preds: list[str] = []
    for v in predictors:
        if v == outcome:
            warnings.append(f"自变量 '{v}' 与因变量相同，已忽略")
            continue
        if v not in df.columns:
            warnings.append(f"自变量 '{v}' 不存在，已忽略")
            continue
        if not pd.api.types.is_numeric_dtype(df[v]):
            warnings.append(f"自变量 '{v}' 非数值型，已忽略")
            continue
        valid_preds.append(v)

    if not valid_preds:
        raise ValueError("没有有效的自变量，请检查变量选择")

    # ── 准备完整数据（多变量分析用） ──────────────────────────────────────────
    all_cols = [outcome] + valid_preds
    df_full = df[all_cols].dropna()
    n_full = len(df_full)
    if n_full < len(valid_preds) + 2:
        raise ValueError(
            f"有效样本量 ({n_full}) 不足，无法进行回归分析（自变量数：{len(valid_preds)}）"
        )

    tables: list[TableResult] = []
    charts: list[ChartResult] = []

    # ── 单变量分析 ────────────────────────────────────────────────────────────
    if mode in ("both", "univariate"):
        uni_rows = _univariate_analysis(df, outcome, valid_preds, warnings)
        tables.append(
            TableResult(
                title="单变量线性回归",
                headers=["自变量", "β", "SE", "t 值", "p 值", "95% CI", "R²"],
                rows=uni_rows,
            )
        )

    # ── 多变量分析 ────────────────────────────────────────────────────────────
    if mode in ("both", "multivariate") and len(valid_preds) >= 1:
        multi_result = _multivariate_analysis(df_full, outcome, valid_preds)

        tables.append(
            TableResult(
                title="多变量线性回归 — 回归系数",
                headers=["变量", "β", "SE", "t 值", "p 值", "95% CI", "标准化 β"],
                rows=multi_result["coef_rows"],
            )
        )
        tables.append(
            TableResult(
                title="多变量线性回归 — 模型拟合指标",
                headers=["指标", "值"],
                rows=multi_result["fit_rows"],
            )
        )
        tables.append(
            TableResult(
                title="多变量线性回归 — 残差诊断",
                headers=["检验", "统计量", "p 值 / 判断"],
                rows=multi_result["diag_rows"],
            )
        )
        if len(valid_preds) > 1:
            tables.append(
                TableResult(
                    title="共线性诊断（VIF）",
                    headers=["自变量", "VIF", "判断"],
                    rows=multi_result["vif_rows"],
                )
            )

        # 图表
        y_arr = df_full[outcome].values.astype(float)
        charts.extend(
            _build_regression_charts(
                y_arr,
                multi_result["y_pred"],
                multi_result["residuals"],
                outcome,
            )
        )

    # ── 摘要 ──────────────────────────────────────────────────────────────────
    if mode in ("both", "multivariate"):
        sig_preds = [
            r[0] for r in multi_result["coef_rows"]
            if r[0] != "截距（Intercept）" and _parse_p(str(r[4])) < 0.05
        ]
        r2_val = next(
            (r[1] for r in multi_result["fit_rows"] if r[0] == "R²"), "—"
        )
        summary = (
            f"多变量线性回归，因变量：{outcome}，共纳入 {len(valid_preds)} 个自变量，"
            f"有效样本量 n = {n_full}，模型 R² = {r2_val}。"
        )
        if sig_preds:
            summary += f"统计显著的自变量（p < 0.05）：{', '.join(sig_preds)}。"
    else:
        summary = (
            f"单变量线性回归，因变量：{outcome}，"
            f"共分析 {len(valid_preds)} 个自变量，有效样本量 n = {n_full}。"
        )

    return AnalysisResult(
        method="linear_reg",
        tables=tables,
        charts=charts,
        summary=summary,
        warnings=warnings,
    )


# ─────────────────────────────────────────────────────────────────────────────
# 单变量分析
# ─────────────────────────────────────────────────────────────────────────────

def _univariate_analysis(
    df: pd.DataFrame,
    outcome: str,
    predictors: list[str],
    warnings: list[str],
) -> list[list[Any]]:
    rows: list[list[Any]] = []
    for pred in predictors:
        sub = df[[outcome, pred]].dropna()
        y = sub[outcome].values.astype(float)
        x = sub[pred].values.astype(float)
        n = len(y)
        if n < 4:
            warnings.append(f"'{pred}' 有效样本量不足（n={n}），已跳过单变量分析")
            continue

        # OLS via scipy
        slope, intercept, r_val, p_val, se_slope = stats.linregress(x, y)
        r2 = r_val ** 2
        df_resid = n - 2
        t_crit = float(stats.t.ppf(0.975, df_resid))
        ci_lo = slope - t_crit * se_slope
        ci_hi = slope + t_crit * se_slope

        rows.append([
            pred,
            f"{slope:.4f}",
            f"{se_slope:.4f}",
            f"{slope / se_slope:.3f}",
            _fmt_p(float(p_val)),
            f"[{ci_lo:.4f}, {ci_hi:.4f}]",
            f"{r2:.4f}",
        ])
    return rows


# ─────────────────────────────────────────────────────────────────────────────
# 多变量分析（使用 statsmodels OLS）
# ─────────────────────────────────────────────────────────────────────────────

def _multivariate_analysis(
    df_clean: pd.DataFrame,
    outcome: str,
    predictors: list[str],
) -> dict[str, Any]:
    import statsmodels.api as sm
    from statsmodels.stats.diagnostic import het_breuschpagan
    from statsmodels.stats.stattools import durbin_watson

    y = df_clean[outcome].values.astype(float)
    X_raw = df_clean[predictors].values.astype(float)
    X = sm.add_constant(X_raw, prepend=True)
    n = len(y)

    model = sm.OLS(y, X).fit()
    y_pred = np.asarray(model.fittedvalues, dtype=float)
    residuals = np.asarray(model.resid, dtype=float)

    # ── 回归系数表 ────────────────────────────────────────────────────────────
    coef_rows: list[list[Any]] = []

    # 标准化 β（对 X 和 y 均标准化后的系数，不含截距）
    y_std = float(np.std(y, ddof=1)) or 1.0
    X_stds = [float(np.std(X_raw[:, k], ddof=1)) or 1.0 for k in range(X_raw.shape[1])]

    param_names = ["截距（Intercept）"] + predictors
    for idx, name in enumerate(param_names):
        beta = float(model.params[idx])
        se = float(model.bse[idx])
        t_val = float(model.tvalues[idx])
        p_val = float(model.pvalues[idx])
        ci_arr = np.asarray(model.conf_int())
        ci_lo = float(ci_arr[idx, 0])
        ci_hi = float(ci_arr[idx, 1])

        if idx == 0:
            std_beta = "—"
        else:
            std_beta = f"{beta * X_stds[idx - 1] / y_std:.4f}"

        coef_rows.append([
            name,
            f"{beta:.4f}",
            f"{se:.4f}",
            f"{t_val:.3f}",
            _fmt_p(p_val),
            f"[{ci_lo:.4f}, {ci_hi:.4f}]",
            std_beta,
        ])

    # ── 模型拟合指标 ──────────────────────────────────────────────────────────
    fit_rows: list[list[Any]] = [
        ["R²",          f"{model.rsquared:.4f}"],
        ["调整 R²",     f"{model.rsquared_adj:.4f}"],
        ["F 统计量",    f"{model.fvalue:.3f}"],
        ["F p 值",      _fmt_p(float(model.f_pvalue))],
        ["AIC",         f"{model.aic:.2f}"],
        ["BIC",         f"{model.bic:.2f}"],
        ["样本量 n",    str(n)],
    ]

    # ── 残差诊断 ──────────────────────────────────────────────────────────────
    diag_rows: list[list[Any]] = []

    # 正态性（Shapiro-Wilk）
    if n <= _SW_MAX_N:
        sw_stat, sw_p = stats.shapiro(residuals)
        norm_str = "正态" if sw_p > 0.05 else "非正态"
        diag_rows.append(["残差正态性（Shapiro-Wilk）", f"{sw_stat:.4f}", f"{_fmt_p(sw_p)} ({norm_str})"])
    else:
        mu_r, sig_r = float(np.mean(residuals)), float(np.std(residuals))
        ks_stat, ks_p = stats.kstest((residuals - mu_r) / (sig_r or 1.0), "norm")
        norm_str = "正态" if ks_p > 0.05 else "非正态"
        diag_rows.append(["残差正态性（K-S）", f"{ks_stat:.4f}", f"{_fmt_p(ks_p)} ({norm_str})"])

    # Breusch-Pagan 方差齐性
    try:
        bp_lm, bp_p, bp_f, bp_fp = het_breuschpagan(residuals, X)
        homo_str = "方差齐性" if bp_p > 0.05 else "存在异方差"
        diag_rows.append(["方差齐性（Breusch-Pagan）", f"{bp_lm:.4f}", f"{_fmt_p(bp_p)} ({homo_str})"])
    except Exception:
        diag_rows.append(["方差齐性（Breusch-Pagan）", "—", "计算失败"])

    # Durbin-Watson
    dw = float(durbin_watson(residuals))
    dw_judge = "无明显自相关" if 1.5 < dw < 2.5 else "可能存在自相关"
    diag_rows.append(["独立性（Durbin-Watson）", f"{dw:.4f}", dw_judge])

    # ── VIF 共线性诊断 ────────────────────────────────────────────────────────
    vif_rows: list[list[Any]] = []
    if len(predictors) > 1:
        try:
            from statsmodels.stats.outliers_influence import variance_inflation_factor
            for k, pred in enumerate(predictors):
                vif_val = float(variance_inflation_factor(X, k + 1))  # +1 跳过截距
                if math.isnan(vif_val) or math.isinf(vif_val):
                    vif_judge = "计算异常"
                elif vif_val < 5:
                    vif_judge = "无共线性"
                elif vif_val < 10:
                    vif_judge = "轻度共线性"
                else:
                    vif_judge = "严重共线性"
                vif_rows.append([pred, f"{vif_val:.3f}", vif_judge])
        except Exception as exc:
            logger.warning("VIF 计算失败: %s", exc)
            vif_rows = [[p, "—", "计算失败"] for p in predictors]

    return {
        "coef_rows": coef_rows,
        "fit_rows": fit_rows,
        "diag_rows": diag_rows,
        "vif_rows": vif_rows,
        "y_pred": y_pred,
        "residuals": residuals,
    }


# ─────────────────────────────────────────────────────────────────────────────
# 图表：实际值 vs 预测值 / 残差图 / QQ 图
# ─────────────────────────────────────────────────────────────────────────────

def _build_regression_charts(
    y_actual: np.ndarray,
    y_pred: np.ndarray,
    residuals: np.ndarray,
    outcome: str,
) -> list[ChartResult]:
    charts: list[ChartResult] = []
    n = len(y_actual)

    # ── 1. 实际值 vs 预测值 ────────────────────────────────────────────────────
    pts = [[float(y_pred[i]), float(y_actual[i])] for i in range(n)]
    y_min = float(min(min(y_actual), min(y_pred)))
    y_max = float(max(max(y_actual), max(y_pred)))
    ref_line = [[y_min, y_min], [y_max, y_max]]

    option_actual_pred: dict[str, Any] = {
        "title": {
            "text": "实际值 vs 预测值",
            "left": "center",
        },
        "tooltip": {"trigger": "item", "formatter": "{c}"},
        "legend": {"data": ["散点", "理想线"], "top": "8%"},
        "grid": {"left": "12%", "right": "5%", "top": "18%", "bottom": "12%"},
        "xAxis": {"type": "value", "name": f"预测值 ({outcome})", "scale": True},
        "yAxis": {"type": "value", "name": f"实际值 ({outcome})", "scale": True},
        "series": [
            {
                "name": "散点",
                "type": "scatter",
                "data": pts,
                "symbolSize": 5,
                "opacity": 0.6,
                "itemStyle": {"color": "#5470c6"},
            },
            {
                "name": "理想线",
                "type": "line",
                "data": ref_line,
                "showSymbol": False,
                "lineStyle": {"type": "dashed", "color": "#ee6666", "width": 2},
            },
        ],
    }
    charts.append(
        ChartResult(title="实际值 vs 预测值", chart_type="scatter", option=option_actual_pred)
    )

    # ── 2. 残差 vs 拟合值 ──────────────────────────────────────────────────────
    resid_pts = [[float(y_pred[i]), float(residuals[i])] for i in range(n)]
    zero_line = [[float(min(y_pred)), 0.0], [float(max(y_pred)), 0.0]]

    option_resid: dict[str, Any] = {
        "title": {
            "text": "残差 vs 拟合值",
            "left": "center",
        },
        "tooltip": {"trigger": "item", "formatter": "{c}"},
        "legend": {"data": ["残差", "零线"], "top": "8%"},
        "grid": {"left": "12%", "right": "5%", "top": "18%", "bottom": "12%"},
        "xAxis": {"type": "value", "name": "拟合值", "scale": True},
        "yAxis": {"type": "value", "name": "残差", "scale": True},
        "series": [
            {
                "name": "残差",
                "type": "scatter",
                "data": resid_pts,
                "symbolSize": 5,
                "opacity": 0.6,
                "itemStyle": {"color": "#91cc75"},
            },
            {
                "name": "零线",
                "type": "line",
                "data": zero_line,
                "showSymbol": False,
                "lineStyle": {"type": "dashed", "color": "#ee6666", "width": 2},
            },
        ],
    }
    charts.append(
        ChartResult(title="残差 vs 拟合值", chart_type="scatter", option=option_resid)
    )

    # ── 3. 残差 QQ 图 ──────────────────────────────────────────────────────────
    sorted_resid = np.sort(residuals)
    std_resid = (sorted_resid - float(np.mean(sorted_resid))) / (
        float(np.std(sorted_resid, ddof=1)) or 1.0
    )
    # 理论分位数
    probs = (np.arange(1, n + 1) - 0.375) / (n + 0.25)
    theoretical_q = np.array([float(stats.norm.ppf(p)) for p in probs])

    qq_pts = [[float(theoretical_q[i]), float(std_resid[i])] for i in range(n)]
    # 参考线：理论 vs 理论（45°）
    q_min, q_max = float(theoretical_q[0]), float(theoretical_q[-1])
    qq_ref = [[q_min, q_min], [q_max, q_max]]

    option_qq: dict[str, Any] = {
        "title": {
            "text": "残差正态 QQ 图",
            "left": "center",
        },
        "tooltip": {"trigger": "item", "formatter": "{c}"},
        "legend": {"data": ["样本分位数", "正态参考线"], "top": "8%"},
        "grid": {"left": "12%", "right": "5%", "top": "18%", "bottom": "12%"},
        "xAxis": {"type": "value", "name": "理论分位数", "scale": True},
        "yAxis": {"type": "value", "name": "标准化残差", "scale": True},
        "series": [
            {
                "name": "样本分位数",
                "type": "scatter",
                "data": qq_pts,
                "symbolSize": 5,
                "opacity": 0.65,
                "itemStyle": {"color": "#fac858"},
            },
            {
                "name": "正态参考线",
                "type": "line",
                "data": qq_ref,
                "showSymbol": False,
                "lineStyle": {"type": "dashed", "color": "#ee6666", "width": 2},
            },
        ],
    }
    charts.append(
        ChartResult(title="残差 QQ 图", chart_type="scatter", option=option_qq)
    )

    return charts


# ─────────────────────────────────────────────────────────────────────────────
# 辅助函数
# ─────────────────────────────────────────────────────────────────────────────

def _fmt_p(p: float) -> str:
    if math.isnan(p):
        return "—"
    if p < 0.001:
        return "< 0.001"
    return f"{p:.3f}"


def _parse_p(p_str: str) -> float:
    if p_str == "< 0.001":
        return 0.0005
    try:
        return float(p_str)
    except (ValueError, TypeError):
        return float("nan")
