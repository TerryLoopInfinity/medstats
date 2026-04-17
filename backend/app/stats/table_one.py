"""三线表（Table 1）生成模块。

接收 pd.DataFrame + 参数字典，返回标准化 AnalysisResult。
输出：
  - 三线表（连续变量：均值±SD 或 中位数[IQR]，分类变量：n(%)）
  - 组间比较 p 值（独立 t / Mann-Whitney U / ANOVA / Kruskal-Wallis / χ² / Fisher精确检验）
  - 无图表，三线表本身即核心结果
"""

import logging
import math
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats

from app.models.analysis import AnalysisResult, ChartResult, TableResult

logger = logging.getLogger(__name__)

_SW_MAX_N = 5000  # Shapiro-Wilk 上限


# ─────────────────────────────────────────────────────────────────────────────
# 主入口
# ─────────────────────────────────────────────────────────────────────────────

def run(df: pd.DataFrame, params: dict) -> AnalysisResult:
    """
    参数
    ----
    params["group_var"]         str         分组变量（必填）
    params["continuous_vars"]   list[str]   连续变量列表
    params["categorical_vars"]  list[str]   分类变量列表
    """
    group_var: str = params.get("group_var", "")
    continuous_vars: list[str] = list(params.get("continuous_vars") or [])
    categorical_vars: list[str] = list(params.get("categorical_vars") or [])
    warnings: list[str] = []

    # ── 校验参数 ──────────────────────────────────────────────────────────────
    if not group_var:
        raise ValueError("请指定分组变量 (group_var)")
    if group_var not in df.columns:
        raise ValueError(f"分组变量 '{group_var}' 不存在于数据集中")
    if not continuous_vars and not categorical_vars:
        raise ValueError("请至少选择一个连续变量或分类变量")

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

    group_dfs = {label: df[df[group_var] == label] for label in group_labels}
    n_total = int(group_series.notna().sum())
    n_per_group = {label: int((df[group_var] == label).sum()) for label in group_labels}

    # ── 表头 ──────────────────────────────────────────────────────────────────
    headers: list[str] = ["变量", f"整体 (n={n_total})"]
    for label in group_labels:
        headers.append(f"{group_var}={label} (n={n_per_group[label]})")
    headers += ["p 值", "统计方法"]

    # ── 构建行 ────────────────────────────────────────────────────────────────
    rows: list[list[Any]] = []
    sig_vars: list[str] = []

    # ── 连续变量 ──────────────────────────────────────────────────────────────
    for col in continuous_vars:
        if col not in df.columns:
            warnings.append(f"连续变量 '{col}' 不存在，已忽略")
            continue
        if not pd.api.types.is_numeric_dtype(df[col]):
            warnings.append(f"'{col}' 非数值型，已移至分类变量")
            categorical_vars.append(col)
            continue

        series_all = df[col].dropna()
        if len(series_all) == 0:
            warnings.append(f"变量 '{col}' 全部缺失，已跳过")
            continue

        # 正态性：各组均通过才用参数检验
        all_normal = _all_groups_normal(group_labels, group_dfs, col)

        if all_normal:
            overall_str = _mean_sd_str(series_all)
            group_strs = [_mean_sd_str(group_dfs[g][col].dropna()) for g in group_labels]
            p_val, method_str = _parametric_test(group_labels, group_dfs, col, n_groups)
        else:
            overall_str = _median_iqr_str(series_all)
            group_strs = [_median_iqr_str(group_dfs[g][col].dropna()) for g in group_labels]
            p_val, method_str = _nonparametric_test(group_labels, group_dfs, col, n_groups)

        p_str = _fmt_p(p_val)
        row: list[Any] = [col, overall_str] + group_strs + [p_str, method_str]
        rows.append(row)
        if not math.isnan(p_val) and p_val <= 0.05:
            sig_vars.append(col)

    # ── 分类变量 ──────────────────────────────────────────────────────────────
    for col in categorical_vars:
        if col not in df.columns:
            warnings.append(f"分类变量 '{col}' 不存在，已忽略")
            continue

        col_series = df[col].dropna()
        if len(col_series) == 0:
            warnings.append(f"变量 '{col}' 全部缺失，已跳过")
            continue

        categories = sorted(col_series.unique().tolist(), key=str)
        n_col_total = len(col_series)

        # 列联表
        ct: dict[Any, dict[Any, int]] = {
            cat: {g: int((group_dfs[g][col] == cat).sum()) for g in group_labels}
            for cat in categories
        }
        ct_matrix = np.array([[ct[cat][g] for g in group_labels] for cat in categories])

        p_val, method_str = _categorical_test(ct_matrix)
        p_str = _fmt_p(p_val)

        # 变量名行（空统计列，仅保留 p 值）
        header_row: list[Any] = [col, ""] + ["" for _ in group_labels] + [p_str, method_str]
        rows.append(header_row)

        if not math.isnan(p_val) and p_val <= 0.05:
            sig_vars.append(col)

        # 各分类子行
        cat_counts_total: dict[Any, int] = {
            k: int(v) for k, v in col_series.value_counts().items()
        }
        for cat in categories:
            n_cat_total = cat_counts_total.get(cat, 0)
            overall_pct = f"{n_cat_total} ({100 * n_cat_total / n_col_total:.1f}%)"
            group_strs = []
            for g in group_labels:
                n_cat_g = ct[cat][g]
                pct = 100 * n_cat_g / n_per_group[g] if n_per_group[g] > 0 else 0
                group_strs.append(f"{n_cat_g} ({pct:.1f}%)")
            sub_row: list[Any] = [f"  {cat}", overall_pct] + group_strs + ["", ""]
            rows.append(sub_row)

    if not rows:
        raise ValueError("未生成任何有效行，请检查变量选择")

    # ── 摘要 ──────────────────────────────────────────────────────────────────
    group_desc = " / ".join(
        f"{g}组 (n={n_per_group[g]})" for g in group_labels
    )
    summary_parts = [
        f"三线表：{len(continuous_vars)} 个连续变量、{len(categorical_vars)} 个分类变量，"
        f"分组变量 {group_var}（{group_desc}）。"
    ]
    if sig_vars:
        summary_parts.append(
            f"组间差异有统计学意义（p < 0.05）的变量：{', '.join(sig_vars)}。"
        )
    else:
        summary_parts.append("各变量组间差异均无统计学意义（p ≥ 0.05）。")

    return AnalysisResult(
        method="table_one",
        tables=[
            TableResult(
                title="Table 1 — 基线特征表",
                headers=headers,
                rows=rows,
            )
        ],
        charts=[],
        summary=" ".join(summary_parts),
        warnings=warnings,
    )


# ─────────────────────────────────────────────────────────────────────────────
# 辅助：正态性
# ─────────────────────────────────────────────────────────────────────────────

def _is_normal(series: pd.Series) -> bool:
    """单组正态性检验，p > 0.05 视为正态。"""
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


def _all_groups_normal(
    group_labels: list,
    group_dfs: dict,
    col: str,
) -> bool:
    return all(_is_normal(group_dfs[g][col]) for g in group_labels)


# ─────────────────────────────────────────────────────────────────────────────
# 辅助：统计检验
# ─────────────────────────────────────────────────────────────────────────────

def _parametric_test(
    group_labels: list, group_dfs: dict, col: str, n_groups: int
) -> tuple[float, str]:
    groups_data = [group_dfs[g][col].dropna() for g in group_labels]
    if any(len(g) < 2 for g in groups_data):
        return float("nan"), "样本量不足"
    if n_groups == 2:
        _, p = stats.ttest_ind(groups_data[0], groups_data[1], equal_var=False)
        return float(p), "独立样本 t 检验（Welch）"
    _, p = stats.f_oneway(*groups_data)
    return float(p), "单因素 ANOVA"


def _nonparametric_test(
    group_labels: list, group_dfs: dict, col: str, n_groups: int
) -> tuple[float, str]:
    groups_data = [group_dfs[g][col].dropna() for g in group_labels]
    if any(len(g) < 2 for g in groups_data):
        return float("nan"), "样本量不足"
    if n_groups == 2:
        _, p = stats.mannwhitneyu(groups_data[0], groups_data[1], alternative="two-sided")
        return float(p), "Mann-Whitney U 检验"
    _, p = stats.kruskal(*groups_data)
    return float(p), "Kruskal-Wallis 检验"


def _categorical_test(ct_matrix: np.ndarray) -> tuple[float, str]:
    """列联表检验：2×2 优先 Fisher 精确检验（期望值 < 5 时），其余 χ²。"""
    if ct_matrix.shape[0] < 2 or ct_matrix.shape[1] < 2:
        return float("nan"), "—"
    total = ct_matrix.sum()
    if total == 0:
        return float("nan"), "—"
    if ct_matrix.shape == (2, 2):
        row_sums = ct_matrix.sum(axis=1, keepdims=True)
        col_sums = ct_matrix.sum(axis=0, keepdims=True)
        expected = row_sums * col_sums / total
        if (expected < 5).any():
            _, p = stats.fisher_exact(ct_matrix)
            return float(p), "Fisher 精确检验"
    chi2, p, _, _ = stats.chi2_contingency(ct_matrix)
    return float(p), "χ² 检验"


# ─────────────────────────────────────────────────────────────────────────────
# 辅助：格式化
# ─────────────────────────────────────────────────────────────────────────────

def _mean_sd_str(s: pd.Series) -> str:
    s = s.dropna()
    if len(s) == 0:
        return "—"
    if len(s) == 1:
        return f"{float(s.iloc[0]):.2f}"
    return f"{s.mean():.2f} ± {s.std(ddof=1):.2f}"


def _median_iqr_str(s: pd.Series) -> str:
    s = s.dropna()
    if len(s) == 0:
        return "—"
    return f"{s.median():.2f} [{s.quantile(0.25):.2f}, {s.quantile(0.75):.2f}]"


def _fmt_p(p: float) -> str:
    if math.isnan(p):
        return "—"
    if p < 0.001:
        return "< 0.001"
    return f"{p:.3f}"
