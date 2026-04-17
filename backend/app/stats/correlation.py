"""相关分析模块。

支持：
  - Pearson / Spearman / Kendall's tau 相关
  - 自动根据正态性选择 Pearson 或 Spearman（method="auto"）
  - 输出：相关系数矩阵（r、p、95% CI、n）+ 显著性标记
  - 图表：热力图；变量 ≤ 5 时额外输出各对变量的散点图
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
_PALETTE = [
    "#5470c6", "#91cc75", "#fac858", "#ee6666",
    "#73c0de", "#3ba272", "#fc8452", "#9a60b4",
]


# ─────────────────────────────────────────────────────────────────────────────
# 主入口
# ─────────────────────────────────────────────────────────────────────────────

def run(df: pd.DataFrame, params: dict) -> AnalysisResult:
    """
    参数
    ----
    params["variables"]   list[str]   分析变量列表（至少 2 个）
    params["method"]      str         "auto"（默认）| "pearson" | "spearman" | "kendall"
    """
    variables: list[str] = list(params.get("variables") or [])
    method: str = str(params.get("method", "auto")).lower()
    warnings: list[str] = []

    # ── 参数校验 ──────────────────────────────────────────────────────────────
    if len(variables) < 2:
        raise ValueError("至少需要选择 2 个变量进行相关分析")
    if method not in ("auto", "pearson", "spearman", "kendall"):
        raise ValueError("method 必须为 auto / pearson / spearman / kendall")

    # ── 过滤无效变量 ──────────────────────────────────────────────────────────
    valid_vars: list[str] = []
    for v in variables:
        if v not in df.columns:
            warnings.append(f"变量 '{v}' 不存在，已忽略")
            continue
        if not pd.api.types.is_numeric_dtype(df[v]):
            warnings.append(f"变量 '{v}' 非数值型，已忽略")
            continue
        valid_vars.append(v)

    if len(valid_vars) < 2:
        raise ValueError("有效数值变量不足 2 个，无法进行相关分析")

    # ── 确定实际使用的方法 ────────────────────────────────────────────────────
    if method == "auto":
        all_normal = all(_is_normal(df[v].dropna()) for v in valid_vars)
        actual_method = "pearson" if all_normal else "spearman"
        method_label = (
            "Pearson r（自动选择，数据满足正态性）"
            if all_normal
            else "Spearman ρ（自动选择，数据不满足正态性）"
        )
    else:
        actual_method = method
        method_label = {
            "pearson": "Pearson r",
            "spearman": "Spearman ρ",
            "kendall": "Kendall τ",
        }[method]

    # ── 计算相关矩阵 ──────────────────────────────────────────────────────────
    n_vars = len(valid_vars)
    r_matrix = np.eye(n_vars)
    p_matrix = np.zeros((n_vars, n_vars))
    sig_matrix: list[list[str]] = [[""] * n_vars for _ in range(n_vars)]

    pair_rows: list[list[Any]] = []

    for i in range(n_vars):
        for j in range(i + 1, n_vars):
            s1 = df[valid_vars[i]].dropna()
            s2 = df[valid_vars[j]].dropna()
            common_idx = s1.index.intersection(s2.index)
            x = s1.loc[common_idx].values.astype(float)
            y = s2.loc[common_idx].values.astype(float)
            n = len(x)

            r, p, ci_lo, ci_hi = _compute_corr(x, y, actual_method, n)
            sig = _sig_mark(p)

            r_matrix[i, j] = r_matrix[j, i] = r
            p_matrix[i, j] = p_matrix[j, i] = p
            sig_matrix[i][j] = sig_matrix[j][i] = sig

            ci_str = (
                f"[{ci_lo:.3f}, {ci_hi:.3f}]"
                if not (math.isnan(ci_lo) or math.isnan(ci_hi))
                else "—"
            )
            pair_rows.append([
                valid_vars[i],
                valid_vars[j],
                f"{r:.3f}",
                _fmt_p(p),
                ci_str,
                str(n),
                sig if sig else "ns",
            ])

    # ── 相关矩阵展示表（方形，含显著性标记）──────────────────────────────────
    matrix_headers = ["变量"] + valid_vars
    matrix_rows: list[list[Any]] = []
    for i, vi in enumerate(valid_vars):
        row: list[Any] = [vi]
        for j in range(n_vars):
            if i == j:
                row.append("1.000")
            else:
                r_val = r_matrix[i, j]
                row.append(f"{r_val:.3f}{sig_matrix[i][j]}")
        matrix_rows.append(row)

    pair_headers = ["变量 1", "变量 2", "相关系数", "p 值", "95% CI", "样本量 n", "显著性"]
    tables = [
        TableResult(
            title=f"相关系数矩阵（{method_label}）",
            headers=matrix_headers,
            rows=matrix_rows,
        ),
        TableResult(
            title="两两相关详细结果",
            headers=pair_headers,
            rows=pair_rows,
        ),
    ]

    # ── 图表 ──────────────────────────────────────────────────────────────────
    charts: list[ChartResult] = [
        _build_heatmap(valid_vars, r_matrix, p_matrix, sig_matrix, method_label)
    ]
    if n_vars <= 5:
        charts.extend(_build_scatter_pairs(df, valid_vars, r_matrix, p_matrix))

    # ── 摘要 ──────────────────────────────────────────────────────────────────
    sig_pairs = [(r[0], r[1]) for r in pair_rows if r[6] not in ("", "ns")]
    summary = (
        f"共分析 {n_vars} 个变量的两两相关（{method_label}），"
        f"共 {len(pair_rows)} 对，"
        f"其中 {len(sig_pairs)} 对具有统计学意义（p < 0.05）。"
    )
    if sig_pairs:
        top = sig_pairs[:3]
        summary += "显著相关变量对：" + "、".join(f"{a}–{b}" for a, b in top)
        if len(sig_pairs) > 3:
            summary += f" 等 {len(sig_pairs)} 对。"
        else:
            summary += "。"

    return AnalysisResult(
        method="correlation",
        tables=tables,
        charts=charts,
        summary=summary,
        warnings=warnings,
    )


# ─────────────────────────────────────────────────────────────────────────────
# 相关系数计算（含 95% CI）
# ─────────────────────────────────────────────────────────────────────────────

def _compute_corr(
    x: np.ndarray, y: np.ndarray, method: str, n: int
) -> tuple[float, float, float, float]:
    """返回 (r, p, ci_lo, ci_hi)。"""
    if n < 3:
        return float("nan"), float("nan"), float("nan"), float("nan")

    if method == "pearson":
        r_val, p_val = stats.pearsonr(x, y)
        ci_lo, ci_hi = _fisher_z_ci(float(r_val), n)
    elif method == "spearman":
        r_obj = stats.spearmanr(x, y)
        r_val = float(r_obj.statistic)
        p_val = float(r_obj.pvalue)
        ci_lo, ci_hi = _fisher_z_ci(r_val, n)
    else:  # kendall
        kt = stats.kendalltau(x, y)
        r_val = float(kt.statistic)
        p_val = float(kt.pvalue)
        ci_lo, ci_hi = float("nan"), float("nan")

    return float(r_val), float(p_val), ci_lo, ci_hi


def _fisher_z_ci(r: float, n: int, alpha: float = 0.05) -> tuple[float, float]:
    """Fisher's z transform 95% CI（适用于 Pearson r 及 Spearman ρ 近似）。"""
    if n < 4 or abs(r) >= 1.0:
        return float("nan"), float("nan")
    z = math.atanh(r)
    se = 1.0 / math.sqrt(n - 3)
    z_crit = float(stats.norm.ppf(1 - alpha / 2))
    return math.tanh(z - z_crit * se), math.tanh(z + z_crit * se)


# ─────────────────────────────────────────────────────────────────────────────
# 热力图（ECharts heatmap）
# ─────────────────────────────────────────────────────────────────────────────

def _build_heatmap(
    var_names: list[str],
    r_matrix: np.ndarray,
    p_matrix: np.ndarray,
    sig_matrix: list[list[str]],
    method_label: str,
) -> ChartResult:
    n = len(var_names)
    data: list[dict[str, Any]] = []

    for i in range(n):
        for j in range(n):
            r_val = float(r_matrix[i, j])
            p_val = float(p_matrix[i, j])
            sig = sig_matrix[i][j]
            label_text = "1.000" if i == j else f"{r_val:.3f}{sig}"
            data.append({
                "value": [j, i, round(r_val, 4)],
                "label": {"formatter": label_text},
            })

    option: dict[str, Any] = {
        "title": {
            "text": "相关系数热力图",
            "subtext": method_label,
            "left": "center",
            "subtextStyle": {"color": "#888", "fontSize": 12},
        },
        "tooltip": {"trigger": "item"},
        "grid": {
            "left": "15%",
            "right": "12%",
            "top": "15%",
            "bottom": "18%",
        },
        "xAxis": {
            "type": "category",
            "data": var_names,
            "splitArea": {"show": True},
            "axisLabel": {"rotate": 30, "fontSize": 11},
        },
        "yAxis": {
            "type": "category",
            "data": var_names,
            "splitArea": {"show": True},
            "axisLabel": {"fontSize": 11},
        },
        "visualMap": {
            "min": -1,
            "max": 1,
            "calculable": True,
            "orient": "horizontal",
            "left": "center",
            "bottom": "2%",
            "inRange": {
                "color": ["#4575b4", "#91bfdb", "#ffffbf", "#fc8d59", "#d73027"]
            },
        },
        "series": [
            {
                "type": "heatmap",
                "data": data,
                "label": {"show": True, "fontSize": 11},
                "emphasis": {
                    "itemStyle": {
                        "shadowBlur": 10,
                        "shadowColor": "rgba(0,0,0,0.4)",
                    }
                },
            }
        ],
    }

    return ChartResult(title="相关系数热力图", chart_type="heatmap", option=option)


# ─────────────────────────────────────────────────────────────────────────────
# 散点图（每对变量一个图，变量 ≤ 5 时生成）
# ─────────────────────────────────────────────────────────────────────────────

def _build_scatter_pairs(
    df: pd.DataFrame,
    var_names: list[str],
    r_matrix: np.ndarray,
    p_matrix: np.ndarray,
) -> list[ChartResult]:
    charts: list[ChartResult] = []
    n_vars = len(var_names)
    color_idx = 0

    for i in range(n_vars):
        for j in range(i + 1, n_vars):
            vx, vy = var_names[j], var_names[i]
            common = df[[vx, vy]].dropna()
            pts = list(zip(common[vx].tolist(), common[vy].tolist()))

            if len(pts) > 80:
                rng = np.random.default_rng(42)
                idx_s = rng.choice(len(pts), 80, replace=False)
                pts = [pts[k] for k in idx_s]

            r_val = float(r_matrix[i, j])
            p_val = float(p_matrix[i, j])
            sig = _sig_mark(p_val)
            color = _PALETTE[color_idx % len(_PALETTE)]
            color_idx += 1

            option: dict[str, Any] = {
                "title": {
                    "text": f"{vx} vs {vy}",
                    "subtext": f"r = {r_val:.3f}{sig}  p {_fmt_p(p_val)}",
                    "left": "center",
                    "subtextStyle": {"color": "#888", "fontSize": 12},
                },
                "tooltip": {"trigger": "item", "formatter": "{c}"},
                "grid": {
                    "left": "12%",
                    "right": "5%",
                    "top": "20%",
                    "bottom": "12%",
                },
                "xAxis": {
                    "type": "value",
                    "name": vx,
                    "nameLocation": "end",
                    "scale": True,
                },
                "yAxis": {
                    "type": "value",
                    "name": vy,
                    "nameLocation": "end",
                    "scale": True,
                },
                "series": [
                    {
                        "type": "scatter",
                        "data": pts,
                        "symbolSize": 6,
                        "opacity": 0.65,
                        "itemStyle": {"color": color},
                    }
                ],
            }

            charts.append(
                ChartResult(title=f"{vx} vs {vy}", chart_type="scatter", option=option)
            )

    return charts


# ─────────────────────────────────────────────────────────────────────────────
# 辅助函数
# ─────────────────────────────────────────────────────────────────────────────

def _is_normal(series: pd.Series) -> bool:
    s = series.dropna()
    n = len(s)
    if n < 3:
        return False
    if n <= _SW_MAX_N:
        _, p = stats.shapiro(s)
    else:
        mu, sigma = float(s.mean()), float(s.std())
        _, p = stats.kstest((s - mu) / (sigma or 1.0), "norm")
    return bool(p > 0.05)


def _sig_mark(p: float) -> str:
    if math.isnan(p):
        return ""
    if p < 0.001:
        return "***"
    if p < 0.01:
        return "**"
    if p < 0.05:
        return "*"
    return ""


def _fmt_p(p: float) -> str:
    if math.isnan(p):
        return "—"
    if p < 0.001:
        return "< 0.001"
    return f"{p:.3f}"
