"""中介分析 — Baron & Kenny 逐步法 + Sobel 检验 + Bootstrap 置信区间"""
import logging
import warnings
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats
import statsmodels.api as sm

from app.models.analysis import AnalysisResult, ChartResult, TableResult

logger = logging.getLogger(__name__)

_EXP_CLIP = 500  # np.exp(np.clip(x, -_EXP_CLIP, _EXP_CLIP)) 防溢出


# ─────────────────────────────────────────────────────────────────────────────
# 辅助函数
# ─────────────────────────────────────────────────────────────────────────────

def _fmt_p(p: float) -> str:
    if p < 0.001:
        return "< 0.001"
    return f"{p:.3f}"


def _fmt_stars(p: float) -> str:
    if p < 0.001:
        return "***"
    if p < 0.01:
        return "**"
    if p < 0.05:
        return "*"
    return "ns"


def _fit_regression(
    y_arr: np.ndarray,
    predictor_arrays: list[np.ndarray],
    is_binary: bool = False,
):
    """Fit OLS or logistic regression; return fitted statsmodels model."""
    X_data = sm.add_constant(
        np.column_stack(predictor_arrays), has_constant="add"
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        if is_binary:
            model = sm.Logit(y_arr, X_data).fit(disp=0, maxiter=200)
        else:
            model = sm.OLS(y_arr, X_data).fit()
    return model


# ─────────────────────────────────────────────────────────────────────────────
# 主分析函数
# ─────────────────────────────────────────────────────────────────────────────

def run(df: pd.DataFrame, params: dict[str, Any]) -> AnalysisResult:  # noqa: C901
    # ── 1. 解析参数 ──────────────────────────────────────────────────
    exposure: str = str(params.get("exposure", ""))
    outcome: str = str(params.get("outcome", ""))
    mediator: str = str(params.get("mediator", ""))
    covariates: list[str] = list(params.get("covariates", []))
    outcome_type: str = params.get("outcome_type", "continuous")
    mediator_type: str = params.get("mediator_type", "continuous")
    n_bootstrap: int = int(params.get("n_bootstrap", 5000))
    ci_level: float = float(params.get("ci_level", 0.95))

    if not exposure:
        raise ValueError("请指定暴露变量（exposure）")
    if not outcome:
        raise ValueError("请指定结局变量（outcome）")
    if not mediator:
        raise ValueError("请指定中介变量（mediator）")
    if exposure == mediator:
        raise ValueError("暴露变量和中介变量不能相同")
    if exposure == outcome or mediator == outcome:
        raise ValueError("结局变量不能与暴露或中介变量相同")

    all_cols = [exposure, outcome, mediator] + covariates
    for col in all_cols:
        if col not in df.columns:
            raise ValueError(f"变量 '{col}' 不在数据集中")

    # ── 2. 数据清洗 ──────────────────────────────────────────────────
    df_clean = df[all_cols].dropna().reset_index(drop=True)
    n = len(df_clean)
    warn_list: list[str] = []

    dropped = len(df) - n
    if dropped > 0:
        warn_list.append(f"已移除 {dropped} 行含缺失值记录，分析使用 {n} 条数据")
    if n < 50:
        warn_list.append("样本量较小（< 50），Bootstrap 置信区间不稳定")
    if n_bootstrap > 5000:
        warn_list.append(f"Bootstrap 次数 {n_bootstrap}，计算可能需要较长时间")
    if outcome_type == "binary":
        warn_list.append(
            "结局为二元变量，效应量为对数-OR（log-OR），"
            "中介比例仅供参考，推荐参考 VanderWeele 自然直接/间接效应方法"
        )

    X_arr = df_clean[exposure].astype(float).values
    Y_arr = df_clean[outcome].astype(float).values
    M_arr = df_clean[mediator].astype(float).values
    C_arr = df_clean[covariates].astype(float).values if covariates else None

    y_binary = outcome_type == "binary"
    m_binary = mediator_type == "binary"

    def _preds(*arrays: np.ndarray, C: np.ndarray | None = C_arr) -> list[np.ndarray]:
        result = list(arrays)
        if C is not None:
            for i in range(C.shape[1]):
                result.append(C[:, i])
        return result

    # ── 3. Baron & Kenny 路径 ────────────────────────────────────────
    # 路径 c: Y ~ X (+ 协变量) — 总效应
    m_c = _fit_regression(Y_arr, _preds(X_arr), is_binary=y_binary)
    c_coef = float(m_c.params[1])
    c_se = float(m_c.bse[1])
    c_p = float(m_c.pvalues[1])

    # 路径 a: M ~ X (+ 协变量)
    m_a = _fit_regression(M_arr, _preds(X_arr), is_binary=m_binary)
    a_coef = float(m_a.params[1])
    a_se = float(m_a.bse[1])
    a_p = float(m_a.pvalues[1])

    # 路径 b + c': Y ~ X + M (+ 协变量); X → c', M → b
    m_bc = _fit_regression(Y_arr, _preds(X_arr, M_arr), is_binary=y_binary)
    c_prime_coef = float(m_bc.params[1])
    c_prime_se = float(m_bc.bse[1])
    c_prime_p = float(m_bc.pvalues[1])
    b_coef = float(m_bc.params[2])
    b_se = float(m_bc.bse[2])
    b_p = float(m_bc.pvalues[2])

    # ── 4. 效应分解 ──────────────────────────────────────────────────
    indirect = a_coef * b_coef
    direct = c_prime_coef
    total = c_coef
    med_pct = (indirect / total * 100) if abs(total) > 1e-10 else float("nan")

    # ── 5. Sobel 检验 ────────────────────────────────────────────────
    sobel_var = b_coef**2 * a_se**2 + a_coef**2 * b_se**2
    sobel_se = float(np.sqrt(max(sobel_var, 1e-18)))
    sobel_z = indirect / sobel_se if sobel_se > 1e-12 else 0.0
    sobel_p = float(2 * (1 - stats.norm.cdf(abs(sobel_z))))

    # ── 6. Bootstrap ─────────────────────────────────────────────────
    rng = np.random.default_rng(42)
    boot_indirect = np.empty(n_bootstrap)

    for i in range(n_bootstrap):
        idx = rng.integers(0, n, size=n)
        Xi, Yi, Mi = X_arr[idx], Y_arr[idx], M_arr[idx]
        Ci = C_arr[idx] if C_arr is not None else None
        try:
            ma_b = _fit_regression(Mi, _preds(Xi, C=Ci), is_binary=m_binary)
            mbc_b = _fit_regression(Yi, _preds(Xi, Mi, C=Ci), is_binary=y_binary)
            boot_indirect[i] = float(ma_b.params[1]) * float(mbc_b.params[2])
        except Exception:
            boot_indirect[i] = indirect

    alpha_ci = 1 - ci_level

    # 百分位 CI
    lo_pct = float(np.quantile(boot_indirect, alpha_ci / 2))
    hi_pct = float(np.quantile(boot_indirect, 1 - alpha_ci / 2))

    # 偏差校正 CI (BC)
    prop_below = float(np.mean(boot_indirect < indirect))
    prop_below = float(np.clip(prop_below, 1e-6, 1 - 1e-6))
    z0 = float(stats.norm.ppf(prop_below))
    z_lo_q = float(stats.norm.ppf(alpha_ci / 2))
    z_hi_q = float(stats.norm.ppf(1 - alpha_ci / 2))
    lo_bc = float(np.quantile(boot_indirect, stats.norm.cdf(2 * z0 + z_lo_q)))
    hi_bc = float(np.quantile(boot_indirect, stats.norm.cdf(2 * z0 + z_hi_q)))

    p_below = float(np.mean(boot_indirect <= 0))
    boot_p = float(min(2 * min(p_below, 1 - p_below) + 1e-9, 1.0))

    indirect_sig = lo_bc > 0 or hi_bc < 0
    full_mediation = indirect_sig and c_prime_p >= 0.05
    partial_mediation = indirect_sig and c_prime_p < 0.05

    # ── 7. 表格 ─────────────────────────────────────────────────────
    ci_pct_str = f"{int(ci_level * 100)}%"
    effect_label = "β（log-OR）" if y_binary else "β（OLS）"

    def _or_str(coef: float) -> str:
        if not y_binary:
            return ""
        or_val = float(np.exp(np.clip(coef, -_EXP_CLIP, _EXP_CLIP)))
        return f" [OR={or_val:.3f}]"

    path_table = TableResult(
        title="Baron & Kenny 路径系数",
        headers=["路径", "描述", effect_label, "SE", "p 值", "显著性"],
        rows=[
            [
                "c",
                f"{exposure} → {outcome}（总效应）",
                f"{c_coef:.4f}{_or_str(c_coef)}",
                f"{c_se:.4f}",
                _fmt_p(c_p),
                _fmt_stars(c_p),
            ],
            [
                "a",
                f"{exposure} → {mediator}",
                f"{a_coef:.4f}",
                f"{a_se:.4f}",
                _fmt_p(a_p),
                _fmt_stars(a_p),
            ],
            [
                "b",
                f"{mediator} → {outcome}（控制 {exposure}）",
                f"{b_coef:.4f}{_or_str(b_coef)}",
                f"{b_se:.4f}",
                _fmt_p(b_p),
                _fmt_stars(b_p),
            ],
            [
                "c'",
                f"{exposure} → {outcome}（直接效应，控制 {mediator}）",
                f"{c_prime_coef:.4f}{_or_str(c_prime_coef)}",
                f"{c_prime_se:.4f}",
                _fmt_p(c_prime_p),
                _fmt_stars(c_prime_p),
            ],
        ],
    )

    effect_table = TableResult(
        title="效应分解",
        headers=[
            "效应类型",
            "估计值",
            f"Bootstrap {ci_pct_str} CI（百分位）",
            f"Bootstrap {ci_pct_str} CI（偏差校正）",
            "Bootstrap p 值",
        ],
        rows=[
            ["总效应 (c)", f"{total:.4f}", "—", "—", _fmt_p(c_p)],
            ["直接效应 (c')", f"{direct:.4f}", "—", "—", _fmt_p(c_prime_p)],
            [
                "间接效应 (a×b)",
                f"{indirect:.4f}",
                f"[{lo_pct:.4f}, {hi_pct:.4f}]",
                f"[{lo_bc:.4f}, {hi_bc:.4f}]",
                _fmt_p(boot_p),
            ],
            [
                "中介比例 (a×b / c)",
                f"{med_pct:.1f}%" if not np.isnan(med_pct) else "N/A",
                "—",
                "—",
                "—",
            ],
        ],
    )

    test_table = TableResult(
        title="间接效应显著性检验",
        headers=["检验方法", "统计量", "p 值", f"Bootstrap {ci_pct_str} CI", "结论"],
        rows=[
            [
                "Sobel 检验",
                f"z = {sobel_z:.4f}",
                _fmt_p(sobel_p),
                "—",
                "间接效应显著" if sobel_p < 0.05 else "间接效应不显著",
            ],
            [
                "Bootstrap（百分位）",
                f"{indirect:.4f}",
                _fmt_p(boot_p),
                f"[{lo_pct:.4f}, {hi_pct:.4f}]",
                "间接效应显著" if (lo_pct > 0 or hi_pct < 0) else "间接效应不显著",
            ],
            [
                "Bootstrap（偏差校正，推荐）",
                f"{indirect:.4f}",
                _fmt_p(boot_p),
                f"[{lo_bc:.4f}, {hi_bc:.4f}]",
                "间接效应显著" if indirect_sig else "间接效应不显著",
            ],
        ],
    )

    # ── 8. 图表 ─────────────────────────────────────────────────────
    def _coef_label(coef: float, p: float) -> str:
        return f"{coef:.3f} ({_fmt_stars(p)})"

    # 图表 1: 路径图（自定义类型，前端渲染 SVG）
    path_chart = ChartResult(
        title="中介效应路径图",
        chart_type="mediation_path",
        option={
            "nodes": {"X": exposure, "M": mediator, "Y": outcome},
            "paths": {
                "a": {
                    "coef": a_coef,
                    "se": a_se,
                    "p": a_p,
                    "label": _coef_label(a_coef, a_p),
                },
                "b": {
                    "coef": b_coef,
                    "se": b_se,
                    "p": b_p,
                    "label": _coef_label(b_coef, b_p),
                },
                "c": {
                    "coef": c_coef,
                    "se": c_se,
                    "p": c_p,
                    "label": _coef_label(c_coef, c_p),
                },
                "c_prime": {
                    "coef": c_prime_coef,
                    "se": c_prime_se,
                    "p": c_prime_p,
                    "label": _coef_label(c_prime_coef, c_prime_p),
                },
            },
            "indirect": indirect,
            "mediation_pct": float(med_pct) if not np.isnan(med_pct) else None,
            "ci": [lo_bc, hi_bc],
            "ci_level": ci_level,
        },
    )

    # 图表 2: Bootstrap 间接效应分布直方图
    hist_counts, hist_edges = np.histogram(boot_indirect, bins=50)
    bin_centers = [
        float((hist_edges[i] + hist_edges[i + 1]) / 2)
        for i in range(len(hist_counts))
    ]

    hist_chart = ChartResult(
        title="Bootstrap 间接效应分布",
        chart_type="bar",
        option={
            "title": {
                "text": "Bootstrap 间接效应分布",
                "subtext": (
                    f"观测值={indirect:.3f}，"
                    f"偏差校正 {ci_pct_str} CI=[{lo_bc:.3f}, {hi_bc:.3f}]"
                ),
                "left": "center",
                "subtextStyle": {"color": "#6b7280"},
            },
            "tooltip": {"trigger": "axis"},
            "xAxis": {
                "type": "category",
                "data": [f"{x:.3f}" for x in bin_centers],
                "axisLabel": {
                    "interval": max(1, len(bin_centers) // 8),
                    "rotate": 30,
                    "fontSize": 10,
                },
                "name": "间接效应 (a×b)",
                "nameLocation": "center",
                "nameGap": 45,
            },
            "yAxis": {"type": "value", "name": "频次"},
            "series": [
                {
                    "type": "bar",
                    "data": [
                        {
                            "value": int(c),
                            "itemStyle": {
                                "color": (
                                    "#91cc75"
                                    if lo_bc <= bin_centers[i] <= hi_bc
                                    else "#5470c6"
                                ),
                                "opacity": 0.85,
                            },
                        }
                        for i, c in enumerate(hist_counts)
                    ],
                    "barCategoryGap": "0%",
                    "name": "频次",
                }
            ],
            "grid": {"bottom": 70, "left": 60, "top": 80, "right": 20},
        },
    )

    # 图表 3: 效应分解饼图
    if not np.isnan(med_pct) and abs(direct) + abs(indirect) > 1e-10:
        tot_abs = abs(direct) + abs(indirect)
        ind_pct_plot = round(abs(indirect) / tot_abs * 100, 1)
        dir_pct_plot = round(100 - ind_pct_plot, 1)
        pie_data: list[dict] = [
            {"value": ind_pct_plot, "name": f"间接效应 {ind_pct_plot}%"},
            {"value": dir_pct_plot, "name": f"直接效应 {dir_pct_plot}%"},
        ]
    else:
        pie_data = [
            {"value": 50, "name": "间接效应"},
            {"value": 50, "name": "直接效应"},
        ]

    pie_chart = ChartResult(
        title="效应分解饼图",
        chart_type="pie",
        option={
            "title": {"text": "效应分解（间接 vs 直接）", "left": "center"},
            "tooltip": {"trigger": "item"},
            "legend": {"orient": "vertical", "left": "left"},
            "series": [
                {
                    "type": "pie",
                    "radius": ["40%", "70%"],
                    "data": pie_data,
                    "emphasis": {
                        "itemStyle": {
                            "shadowBlur": 10,
                            "shadowColor": "rgba(0,0,0,0.5)",
                        }
                    },
                    "label": {"formatter": "{b}\n{d}%"},
                }
            ],
            "color": ["#5470c6", "#91cc75"],
        },
    )

    # ── 9. 摘要 ─────────────────────────────────────────────────────
    if partial_mediation:
        med_type = "部分中介（间接效应显著，直接效应亦显著）"
    elif full_mediation:
        med_type = "完全中介（间接效应显著，直接效应不显著）"
    else:
        med_type = "中介效应不显著"

    summary = (
        f"中介分析（{exposure}→{mediator}→{outcome}）："
        f"路径 a β={a_coef:.3f}（{_fmt_stars(a_p)}），"
        f"路径 b β={b_coef:.3f}（{_fmt_stars(b_p)}），"
        f"间接效应={indirect:.3f}，"
        f"偏差校正 {ci_pct_str}CI=[{lo_bc:.3f},{hi_bc:.3f}]，"
        f"p={_fmt_p(boot_p)}。"
        + (f"中介比例≈{med_pct:.1f}%。" if not np.isnan(med_pct) else "")
        + f"结论：{med_type}。"
    )

    return AnalysisResult(
        method="mediation",
        tables=[path_table, effect_table, test_table],
        charts=[path_chart, hist_chart, pie_chart],
        summary=summary,
        warnings=warn_list,
    )
