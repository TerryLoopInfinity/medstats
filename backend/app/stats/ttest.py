"""定量资料批量差异性分析模块。

支持：
  - 两组独立比较：自动判断正态性 + 方差齐性 → 独立样本 t / Welch t / Mann-Whitney U
  - 多组独立比较：单因素 ANOVA / Kruskal-Wallis + 事后两两比较（Bonferroni 校正）
  - 配对比较：配对 t 检验 / Wilcoxon 符号秩检验
  - 效应量：Cohen's d（参数）/ rank-biserial r（非参数）/ η²（ANOVA）
  - 图表：每个变量生成箱线图（含原始散点）的 ECharts option
"""

import logging
import math
from itertools import combinations
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats

from app.models.analysis import AnalysisResult, ChartResult, TableResult

logger = logging.getLogger(__name__)

_SW_MAX_N = 5000  # Shapiro-Wilk 上限，超出改用 KS 检验


# ─────────────────────────────────────────────────────────────────────────────
# 主入口
# ─────────────────────────────────────────────────────────────────────────────

def run(df: pd.DataFrame, params: dict) -> AnalysisResult:
    """
    参数
    ----
    params["group_var"]      str          分组变量（必填）
    params["compare_vars"]   list[str]    待比较的连续变量列表（必填）
    params["compare_type"]   str          "independent"（默认）或 "paired"
    """
    group_var: str = params.get("group_var", "")
    compare_vars: list[str] = list(params.get("compare_vars") or [])
    compare_type: str = params.get("compare_type", "independent")
    warnings: list[str] = []

    # ── 参数校验 ──────────────────────────────────────────────────────────────
    if not group_var:
        raise ValueError("请指定分组变量 (group_var)")
    if group_var not in df.columns:
        raise ValueError(f"分组变量 '{group_var}' 不存在于数据集中")
    if not compare_vars:
        raise ValueError("请至少选择一个待比较变量 (compare_vars)")
    if compare_type not in ("independent", "paired"):
        raise ValueError("compare_type 必须为 'independent' 或 'paired'")

    # ── 分组信息 ──────────────────────────────────────────────────────────────
    group_series = df[group_var].dropna()
    group_labels = sorted(group_series.unique().tolist(), key=str)
    n_groups = len(group_labels)

    if n_groups < 2:
        raise ValueError(
            f"分组变量 '{group_var}' 只含 {n_groups} 个组别，至少需要 2 个"
        )
    if n_groups > 10:
        raise ValueError(
            f"分组变量组别过多（{n_groups}），请检查是否选错变量"
        )
    if compare_type == "paired" and n_groups != 2:
        raise ValueError("配对比较仅支持两组，请确认分组变量只含两个水平")

    group_dfs = {label: df[df[group_var] == label] for label in group_labels}

    # ── 表头（按比较类型动态构建）────────────────────────────────────────────
    if compare_type == "paired":
        headers = ["变量", "差值（均值 ± SD）", "统计量", "p 值", "效应量", "统计方法"]
    else:
        group_cols = [f"{group_var}={g}" for g in group_labels]
        headers = ["变量"] + group_cols + ["统计量", "p 值", "效应量", "统计方法"]

    # ── 逐变量分析 ────────────────────────────────────────────────────────────
    rows: list[list[Any]] = []
    charts: list[ChartResult] = []
    posthoc_rows: list[list[Any]] = []
    sig_vars: list[str] = []

    for col in compare_vars:
        if col not in df.columns:
            warnings.append(f"变量 '{col}' 不存在，已忽略")
            continue
        if not pd.api.types.is_numeric_dtype(df[col]):
            warnings.append(f"变量 '{col}' 非数值型，已忽略")
            continue

        if compare_type == "paired":
            row, var_warnings, chart = _analyze_paired(
                df, group_var, group_labels, col
            )
        elif n_groups == 2:
            row, var_warnings, chart = _analyze_two_groups(
                group_dfs, group_labels, group_var, col
            )
        else:
            row, var_warnings, chart, ph = _analyze_multi_groups(
                group_dfs, group_labels, group_var, col
            )
            posthoc_rows.extend(ph)

        warnings.extend(var_warnings)
        rows.append(row)
        charts.append(chart)

        p_str = str(row[-3])  # p 值列（倒数第三列）
        if _parse_p(p_str) <= 0.05:
            sig_vars.append(col)

    if not rows:
        raise ValueError("未生成任何有效分析行，请检查变量选择")

    # ── 结果表格 ──────────────────────────────────────────────────────────────
    tables: list[TableResult] = [
        TableResult(title="差异性分析结果", headers=headers, rows=rows)
    ]
    if posthoc_rows:
        ph_headers = ["变量", "组别 A", "组别 B", "统计量", "校正 p 值", "检验方法"]
        tables.append(
            TableResult(
                title="事后两两比较（Bonferroni 校正）",
                headers=ph_headers,
                rows=posthoc_rows,
            )
        )

    # ── 摘要 ──────────────────────────────────────────────────────────────────
    n_analyzed = len(rows)
    type_label = "配对" if compare_type == "paired" else f"{n_groups} 组独立"
    if sig_vars:
        summary = (
            f"{type_label}比较，共分析 {n_analyzed} 个变量，"
            f"{len(sig_vars)} 个变量组间差异有统计学意义（p < 0.05）："
            f"{', '.join(sig_vars)}。"
        )
    else:
        summary = (
            f"{type_label}比较，共分析 {n_analyzed} 个变量，"
            "各变量组间差异均无统计学意义（p ≥ 0.05）。"
        )

    return AnalysisResult(
        method="ttest",
        tables=tables,
        charts=charts,
        summary=summary,
        warnings=warnings,
    )


# ─────────────────────────────────────────────────────────────────────────────
# 两组独立比较
# ─────────────────────────────────────────────────────────────────────────────

def _analyze_two_groups(
    group_dfs: dict,
    group_labels: list,
    group_var: str,
    col: str,
) -> tuple[list[Any], list[str], ChartResult]:
    warnings: list[str] = []
    g0, g1 = group_labels[0], group_labels[1]
    s0 = group_dfs[g0][col].dropna()
    s1 = group_dfs[g1][col].dropna()

    if len(s0) < 2 or len(s1) < 2:
        row: list[Any] = [col, "—", "—", "—", "—", "—", "样本量不足"]
        return row, warnings, _build_boxplot(group_dfs, group_labels, col, "—", "—")

    normal0 = _is_normal(s0)
    normal1 = _is_normal(s1)

    if normal0 and normal1:
        _, p_levene = stats.levene(s0, s1)
        equal_var = bool(p_levene > 0.05)
        stat, p = stats.ttest_ind(s0, s1, equal_var=equal_var)
        method_str = "独立样本 t 检验" if equal_var else "Welch t 检验"
        effect = _cohens_d(s0, s1)
        effect_str = f"d = {effect:.3f}" if not math.isnan(effect) else "—"
        desc0 = _mean_sd_str(s0)
        desc1 = _mean_sd_str(s1)
    else:
        result = stats.mannwhitneyu(s0, s1, alternative="two-sided")
        stat, p = result.statistic, result.pvalue
        method_str = "Mann-Whitney U 检验"
        effect = _rank_biserial(s0, s1)
        effect_str = f"r = {effect:.3f}" if not math.isnan(effect) else "—"
        desc0 = _median_iqr_str(s0)
        desc1 = _median_iqr_str(s1)

    p_str = _fmt_p(float(p))
    row = [col, desc0, desc1, _fmt_stat(float(stat)), p_str, effect_str, method_str]
    chart = _build_boxplot(group_dfs, group_labels, col, p_str, method_str)
    return row, warnings, chart


# ─────────────────────────────────────────────────────────────────────────────
# 多组独立比较（≥3 组）
# ─────────────────────────────────────────────────────────────────────────────

def _analyze_multi_groups(
    group_dfs: dict,
    group_labels: list,
    group_var: str,
    col: str,
) -> tuple[list[Any], list[str], ChartResult, list[list[Any]]]:
    warnings: list[str] = []
    groups_data = [group_dfs[g][col].dropna() for g in group_labels]

    if any(len(g) < 2 for g in groups_data):
        desc_strs = ["—"] * len(group_labels)
        row: list[Any] = [col] + desc_strs + ["—", "—", "—", "样本量不足"]
        return row, warnings, _build_boxplot(group_dfs, group_labels, col, "—", "—"), []

    all_normal = all(_is_normal(g) for g in groups_data)
    pairs = list(combinations(range(len(group_labels)), 2))
    n_pairs = len(pairs)

    if all_normal:
        stat, p = stats.f_oneway(*groups_data)
        method_str = "单因素 ANOVA"
        desc_strs = [_mean_sd_str(g) for g in groups_data]
        grand_mean = float(pd.concat(groups_data).mean())
        ss_between = sum(
            len(g) * (float(g.mean()) - grand_mean) ** 2 for g in groups_data
        )
        ss_total = sum(float(((g - grand_mean) ** 2).sum()) for g in groups_data)
        eta2 = ss_between / ss_total if ss_total > 0 else float("nan")
        effect_str = f"η² = {eta2:.3f}" if not math.isnan(eta2) else "—"
        posthoc_rows: list[list[Any]] = []
        for i, j in pairs:
            gi, gj = groups_data[i], groups_data[j]
            s_val, pv = stats.ttest_ind(gi, gj, equal_var=False)
            pv_corrected = min(float(pv) * n_pairs, 1.0)
            posthoc_rows.append([
                col,
                str(group_labels[i]),
                str(group_labels[j]),
                _fmt_stat(float(s_val)),
                _fmt_p(pv_corrected),
                "Welch t（Bonferroni 校正）",
            ])
    else:
        stat, p = stats.kruskal(*groups_data)
        method_str = "Kruskal-Wallis 检验"
        desc_strs = [_median_iqr_str(g) for g in groups_data]
        effect_str = "—"
        posthoc_rows = []
        for i, j in pairs:
            gi, gj = groups_data[i], groups_data[j]
            mw = stats.mannwhitneyu(gi, gj, alternative="two-sided")
            pv_corrected = min(float(mw.pvalue) * n_pairs, 1.0)
            posthoc_rows.append([
                col,
                str(group_labels[i]),
                str(group_labels[j]),
                _fmt_stat(float(mw.statistic)),
                _fmt_p(pv_corrected),
                "Mann-Whitney U（Bonferroni 校正）",
            ])

    p_str = _fmt_p(float(p))
    row = [col] + desc_strs + [_fmt_stat(float(stat)), p_str, effect_str, method_str]
    chart = _build_boxplot(group_dfs, group_labels, col, p_str, method_str)
    return row, warnings, chart, posthoc_rows


# ─────────────────────────────────────────────────────────────────────────────
# 配对比较
# ─────────────────────────────────────────────────────────────────────────────

def _analyze_paired(
    df: pd.DataFrame,
    group_var: str,
    group_labels: list,
    col: str,
) -> tuple[list[Any], list[str], ChartResult]:
    warnings: list[str] = []
    g0, g1 = group_labels[0], group_labels[1]
    s0 = df[df[group_var] == g0][col].dropna().reset_index(drop=True)
    s1 = df[df[group_var] == g1][col].dropna().reset_index(drop=True)

    n = min(len(s0), len(s1))
    group_dfs_local = {g0: df[df[group_var] == g0], g1: df[df[group_var] == g1]}

    if n < 3:
        row: list[Any] = [col, "—", "—", "—", "—", "样本量不足（需 ≥ 3 对）"]
        return row, warnings, _build_boxplot(group_dfs_local, group_labels, col, "—", "—")

    if len(s0) != len(s1):
        warnings.append(
            f"'{col}' 两组样本量不等（{g0}组 n={len(s0)}, {g1}组 n={len(s1)}），"
            f"已取前 {n} 对进行配对分析"
        )
    s0, s1 = s0[:n], s1[:n]
    diff = s0 - s1

    if _is_normal(diff):
        stat, p = stats.ttest_rel(s0, s1)
        method_str = "配对 t 检验"
        std_diff = float(diff.std(ddof=1))
        d = float(diff.mean()) / std_diff if std_diff != 0 else float("nan")
        effect_str = f"d = {d:.3f}" if not math.isnan(d) else "—"
    else:
        result = stats.wilcoxon(s0, s1, alternative="two-sided")
        stat, p = result.statistic, result.pvalue
        method_str = "Wilcoxon 符号秩检验"
        effect_str = "—"

    diff_str = f"{float(diff.mean()):.2f} ± {float(diff.std(ddof=1)):.2f}"
    p_str = _fmt_p(float(p))
    row = [col, diff_str, _fmt_stat(float(stat)), p_str, effect_str, method_str]
    chart = _build_boxplot(group_dfs_local, group_labels, col, p_str, method_str)
    return row, warnings, chart


# ─────────────────────────────────────────────────────────────────────────────
# 图表：箱线图 + 散点叠加
# ─────────────────────────────────────────────────────────────────────────────

def _build_boxplot(
    group_dfs: dict,
    group_labels: list,
    col: str,
    p_str: str,
    method_str: str,
) -> ChartResult:
    palette = [
        "#5470c6", "#91cc75", "#fac858", "#ee6666",
        "#73c0de", "#3ba272", "#fc8452", "#9a60b4",
    ]
    box_data: list[list[float]] = []
    scatter_series: list[dict[str, Any]] = []

    for i, g in enumerate(group_labels):
        s = group_dfs[g][col].dropna()
        if len(s) == 0:
            box_data.append([0.0, 0.0, 0.0, 0.0, 0.0])
        else:
            box_data.append([
                float(s.min()),
                float(s.quantile(0.25)),
                float(s.median()),
                float(s.quantile(0.75)),
                float(s.max()),
            ])

        # 最多展示 60 个散点（随机抽样，seed 固定保证可复现）
        sampled = s.sample(min(len(s), 60), random_state=42) if len(s) > 60 else s
        scatter_series.append({
            "name": f"{g}",
            "type": "scatter",
            "data": [[i, float(v)] for v in sampled],
            "symbolSize": 5,
            "opacity": 0.55,
            "itemStyle": {"color": palette[i % len(palette)]},
            "z": 10,
        })

    p_label = f"p {p_str}" if p_str not in ("—", "") else ""
    subtext = f"{method_str}  {p_label}".strip()

    option: dict[str, Any] = {
        "title": {
            "text": col,
            "subtext": subtext,
            "left": "center",
            "subtextStyle": {"color": "#888", "fontSize": 12},
        },
        "tooltip": {"trigger": "item"},
        "grid": {"left": "12%", "right": "5%", "top": "20%", "bottom": "12%"},
        "xAxis": {
            "type": "category",
            "data": [str(g) for g in group_labels],
            "boundaryGap": True,
        },
        "yAxis": {
            "type": "value",
            "name": col,
            "nameLocation": "end",
        },
        "series": [
            {
                "name": "箱线图",
                "type": "boxplot",
                "data": box_data,
                "itemStyle": {"borderWidth": 2},
                "tooltip": {
                    "formatter": (
                        "最大值: {c[4]}<br/>Q3: {c[3]}<br/>"
                        "中位数: {c[2]}<br/>Q1: {c[1]}<br/>最小值: {c[0]}"
                    )
                },
            },
            *scatter_series,
        ],
    }

    return ChartResult(title=col, chart_type="boxplot", option=option)


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


def _cohens_d(s1: pd.Series, s2: pd.Series) -> float:
    n1, n2 = len(s1), len(s2)
    if n1 < 2 or n2 < 2:
        return float("nan")
    pooled_var = (
        (n1 - 1) * float(s1.var(ddof=1)) + (n2 - 1) * float(s2.var(ddof=1))
    ) / (n1 + n2 - 2)
    pooled_std = math.sqrt(pooled_var) if pooled_var > 0 else 0.0
    if pooled_std == 0:
        return float("nan")
    return abs(float(s1.mean()) - float(s2.mean())) / pooled_std


def _rank_biserial(s1: pd.Series, s2: pd.Series) -> float:
    n1, n2 = len(s1), len(s2)
    if n1 == 0 or n2 == 0:
        return float("nan")
    result = stats.mannwhitneyu(s1, s2, alternative="two-sided")
    return abs(1.0 - (2.0 * float(result.statistic)) / (n1 * n2))


def _mean_sd_str(s: pd.Series) -> str:
    s = s.dropna()
    if len(s) == 0:
        return "—"
    if len(s) == 1:
        return f"{float(s.iloc[0]):.2f}"
    return f"{float(s.mean()):.2f} ± {float(s.std(ddof=1)):.2f}"


def _median_iqr_str(s: pd.Series) -> str:
    s = s.dropna()
    if len(s) == 0:
        return "—"
    return (
        f"{float(s.median()):.2f} "
        f"[{float(s.quantile(0.25)):.2f}, {float(s.quantile(0.75)):.2f}]"
    )


def _fmt_stat(v: float, digits: int = 3) -> str:
    return "—" if math.isnan(v) else f"{v:.{digits}f}"


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
