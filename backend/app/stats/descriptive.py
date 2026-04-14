"""统计描述与正态性检验模块。

接收 pd.DataFrame 和参数字典，返回标准化 AnalysisResult。
包含：
  - 描述统计表（n、缺失、均值、标准差、中位数、P25、P75、最小值、最大值）
  - 正态性检验表（Shapiro-Wilk / K-S，统计量、p 值、结论）
  - 每个变量的直方图 + QQ 图 ECharts option
"""

import logging
import math

import numpy as np
import pandas as pd
from scipy import stats

from app.models.analysis import AnalysisResult, ChartResult, TableResult

logger = logging.getLogger(__name__)

# Shapiro-Wilk 适用范围上限
_SW_MAX_N = 5000
# QQ 图最多绘制点数（降采样阈值）
_QQ_MAX_POINTS = 500


def run(df: pd.DataFrame, params: dict) -> AnalysisResult:
    """
    参数：
        params["variables"]  list[str]  要分析的列名；缺省时自动选全部数值列
    """
    variables: list[str] = params.get("variables") or []
    warnings: list[str] = []

    # ── 变量解析 ────────────────────────────────────────────────
    if not variables:
        variables = df.select_dtypes(include="number").columns.tolist()
        if not variables:
            raise ValueError("数据集中未找到任何数值型变量")

    missing_cols = [v for v in variables if v not in df.columns]
    if missing_cols:
        warnings.append(f"以下变量不存在，已忽略：{', '.join(missing_cols)}")
    variables = [v for v in variables if v in df.columns]

    non_numeric = [v for v in variables if not pd.api.types.is_numeric_dtype(df[v])]
    if non_numeric:
        warnings.append(f"以下变量为非数值型，已跳过：{', '.join(non_numeric)}")
    numeric_vars = [v for v in variables if v not in non_numeric]

    if not numeric_vars:
        raise ValueError("所选变量中无数值型列，请重新选择")

    # ── 逐变量计算 ────────────────────────────────────────────────
    desc_rows: list[list] = []
    norm_rows: list[list] = []
    charts: list[ChartResult] = []

    for col in numeric_vars:
        series = df[col].dropna()
        n = int(len(series))
        n_missing = int(df[col].isna().sum())

        if n == 0:
            warnings.append(f"变量 {col} 全部缺失，已跳过")
            continue

        mean   = float(series.mean())
        std    = float(series.std(ddof=1)) if n > 1 else float("nan")
        median = float(series.median())
        q1     = float(series.quantile(0.25))
        q3     = float(series.quantile(0.75))
        min_v  = float(series.min())
        max_v  = float(series.max())

        desc_rows.append([
            col, n, n_missing,
            _fmt(mean), _fmt(std), _fmt(median),
            _fmt(q1), _fmt(q3),
            _fmt(min_v), _fmt(max_v),
        ])

        # ── 正态性检验 ────────────────────────────────────────────
        if n < 3:
            norm_rows.append([col, n, "—", "—", "样本量不足 (n < 3)"])
        elif n <= _SW_MAX_N:
            sw_stat, sw_p = stats.shapiro(series)
            conclusion = "正态" if sw_p > 0.05 else "非正态"
            norm_rows.append([col, n, f"{sw_stat:.4f}", _fmt_p(sw_p),
                               f"Shapiro-Wilk，{conclusion}"])
        else:
            z = (series - mean) / (std or 1)
            ks_stat, ks_p = stats.kstest(z, "norm")
            conclusion = "正态" if ks_p > 0.05 else "非正态"
            norm_rows.append([col, n, f"{ks_stat:.4f}", _fmt_p(ks_p),
                               f"Kolmogorov-Smirnov，{conclusion}"])

        # ── 图表 ─────────────────────────────────────────────────
        charts.append(ChartResult(
            title=f"{col} — 直方图",
            chart_type="bar",
            option=_histogram_option(series, col),
        ))
        charts.append(ChartResult(
            title=f"{col} — QQ 图",
            chart_type="scatter",
            option=_qq_option(series, col),
        ))

    if not desc_rows:
        raise ValueError("所有变量均无有效数据，无法完成分析")

    # ── 文字结论 ──────────────────────────────────────────────────
    normal_vars: list[str] = []
    non_normal_vars: list[str] = []
    for row in norm_rows:
        conclusion_str = str(row[-1])
        if "非正态" in conclusion_str:
            non_normal_vars.append(str(row[0]))
        elif "正态" in conclusion_str:
            normal_vars.append(str(row[0]))

    summary_parts = [f"共分析 {len(desc_rows)} 个数值变量（数据集共 {len(df)} 行）。"]
    if normal_vars:
        summary_parts.append(
            f"符合正态分布（p > 0.05）：{', '.join(normal_vars)}，建议以均值 ± 标准差描述。"
        )
    if non_normal_vars:
        summary_parts.append(
            f"不符合正态分布（p ≤ 0.05）：{', '.join(non_normal_vars)}，"
            "建议以中位数（四分位数间距）描述，组间比较优先考虑非参数检验。"
        )

    return AnalysisResult(
        method="descriptive",
        tables=[
            TableResult(
                title="描述统计",
                headers=["变量", "有效 n", "缺失", "均值", "标准差",
                         "中位数", "P25", "P75", "最小值", "最大值"],
                rows=desc_rows,
            ),
            TableResult(
                title="正态性检验",
                headers=["变量", "n", "统计量", "p 值", "结论"],
                rows=norm_rows,
            ),
        ],
        charts=charts,
        summary=" ".join(summary_parts),
        warnings=warnings,
    )


# ── ECharts option 构建 ──────────────────────────────────────────


def _histogram_option(series: pd.Series, col: str) -> dict:
    values = series.values.astype(float)
    n_bins = min(int(math.ceil(math.log2(len(values)) + 1)), 30)
    counts, edges = np.histogram(values, bins=n_bins)

    labels = [f"{edges[i]:.3g}~{edges[i + 1]:.3g}" for i in range(len(counts))]

    return {
        "title": {"text": f"{col} 分布直方图", "left": "center",
                  "textStyle": {"fontSize": 13}},
        "tooltip": {"trigger": "axis", "axisPointer": {"type": "shadow"}},
        "grid": {"left": "12%", "right": "5%", "bottom": "20%", "top": "15%"},
        "xAxis": {
            "type": "category",
            "data": labels,
            "name": col,
            "nameLocation": "middle",
            "nameGap": 32,
            "axisLabel": {"rotate": 30, "fontSize": 10},
        },
        "yAxis": {"type": "value", "name": "频数"},
        "series": [{
            "type": "bar",
            "data": [int(c) for c in counts],
            "barWidth": "99%",
            "itemStyle": {"color": "#3b82f6", "borderRadius": [2, 2, 0, 0]},
        }],
    }


def _qq_option(series: pd.Series, col: str) -> dict:
    values = np.sort(series.values.astype(float))
    n = len(values)

    # 大样本降采样
    if n > _QQ_MAX_POINTS:
        idx = np.linspace(0, n - 1, _QQ_MAX_POINTS, dtype=int)
        values = values[idx]
        n = len(values)

    mu, sigma = float(values.mean()), float(values.std())
    std_vals = (values - mu) / (sigma if sigma > 0 else 1.0)

    # Filliben 公式计算理论分位数
    probs = (np.arange(1, n + 1) - 0.3175) / (n + 0.365)
    probs = np.clip(probs, 1e-9, 1 - 1e-9)
    theoretical = stats.norm.ppf(probs)

    scatter_data = [
        [round(float(theoretical[i]), 4), round(float(std_vals[i]), 4)]
        for i in range(n)
    ]

    lo, hi = float(theoretical[0]), float(theoretical[-1])
    ref_data = [[round(lo, 4), round(lo, 4)], [round(hi, 4), round(hi, 4)]]

    return {
        "title": {"text": f"{col} QQ 图", "left": "center",
                  "textStyle": {"fontSize": 13}},
        "legend": {"data": ["样本点", "参考线"], "bottom": 0},
        "grid": {"left": "13%", "right": "5%", "bottom": "15%", "top": "15%"},
        "xAxis": {"type": "value", "name": "理论分位数",
                  "nameLocation": "middle", "nameGap": 28},
        "yAxis": {"type": "value", "name": "样本分位数（标准化）",
                  "nameLocation": "middle", "nameGap": 48},
        "tooltip": {"trigger": "item"},
        "series": [
            {
                "name": "样本点",
                "type": "scatter",
                "data": scatter_data,
                "symbolSize": 4,
                "itemStyle": {"color": "#3b82f6", "opacity": 0.7},
            },
            {
                "name": "参考线",
                "type": "line",
                "data": ref_data,
                "symbol": "none",
                "lineStyle": {"color": "#ef4444", "type": "dashed", "width": 2},
            },
        ],
    }


# ── 格式化工具 ────────────────────────────────────────────────────


def _fmt(v: float) -> str:
    if math.isnan(v):
        return "—"
    if abs(v) >= 1000:
        return f"{v:.1f}"
    if abs(v) >= 10:
        return f"{v:.2f}"
    return f"{v:.3f}"


def _fmt_p(p: float) -> str:
    if p < 0.001:
        return "< 0.001"
    return f"{p:.3f}"
