"""基本假设检验模块。

支持：
  - normality   正态性检验：Shapiro-Wilk、Kolmogorov-Smirnov（D 检验）
  - variance    方差齐性检验：Levene 检验、Bartlett 检验
  - chi2        卡方检验（含 Yates 校正）& Fisher 精确检验
  - onesample   单样本检验：单样本 t 检验 / Wilcoxon 符号秩检验

输出：标准化 AnalysisResult（与 ttest.py 一致）
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
# 主入口（按 test_type 路由）
# ─────────────────────────────────────────────────────────────────────────────

def run(df: pd.DataFrame, params: dict) -> AnalysisResult:
    """
    公共参数
    --------
    params["test_type"]    str   "normality" | "variance" | "chi2" | "onesample"

    normality 额外参数
    ------------------
    params["variables"]    list[str]   待检验的连续变量列表

    variance 额外参数
    -----------------
    params["variables"]    list[str]   待检验的连续变量列表
    params["group_var"]    str         分组变量

    chi2 额外参数
    -------------
    params["row_var"]      str         行变量（第一个分类变量）
    params["col_var"]      str         列变量（第二个分类变量）
    params["yates"]        bool        是否强制使用 Yates 校正（默认 False，2×2 表自动判断）

    onesample 额外参数
    ------------------
    params["variables"]    list[str]   待检验的连续变量列表
    params["mu"]           float       假设均值（默认 0）
    params["alternative"]  str         "two-sided"（默认）| "less" | "greater"
    """
    test_type: str = params.get("test_type", "normality")
    warnings: list[str] = []

    dispatch = {
        "normality": _run_normality,
        "variance":  _run_variance,
        "chi2":      _run_chi2,
        "onesample": _run_onesample,
    }
    if test_type not in dispatch:
        raise ValueError(
            f"未知检验类型 '{test_type}'，支持：normality, variance, chi2, onesample"
        )
    return dispatch[test_type](df, params, warnings)


# ─────────────────────────────────────────────────────────────────────────────
# 正态性检验
# ─────────────────────────────────────────────────────────────────────────────

def _run_normality(
    df: pd.DataFrame, params: dict, warnings: list[str]
) -> AnalysisResult:
    variables: list[str] = list(params.get("variables") or [])
    if not variables:
        raise ValueError("请至少选择一个变量 (variables)")

    headers = ["变量", "n", "均值", "SD", "S-W 统计量", "S-W p 值", "K-S 统计量", "K-S p 值", "结论"]
    rows: list[list[Any]] = []
    non_normal: list[str] = []

    for col in variables:
        if col not in df.columns:
            warnings.append(f"变量 '{col}' 不存在，已忽略")
            continue
        if not pd.api.types.is_numeric_dtype(df[col]):
            warnings.append(f"变量 '{col}' 非数值型，已忽略")
            continue

        s = df[col].dropna()
        n = len(s)
        if n < 3:
            rows.append([col, n, "—", "—", "—", "—", "—", "—", "样本量不足"])
            continue

        mean_str = f"{float(s.mean()):.3f}"
        sd_str = f"{float(s.std(ddof=1)):.3f}"

        # Shapiro-Wilk（n ≤ 5000）
        if n <= _SW_MAX_N:
            sw_stat, sw_p = stats.shapiro(s)
            sw_stat_str = f"{float(sw_stat):.4f}"
            sw_p_str = _fmt_p(float(sw_p))
        else:
            sw_stat_str = "—（n 过大）"
            sw_p_str = "—"
            sw_p = float("nan")

        # Kolmogorov-Smirnov（标准化后与理论正态比较）
        mu, sigma = float(s.mean()), float(s.std())
        ks_stat, ks_p = stats.kstest((s - mu) / (sigma or 1.0), "norm")
        ks_stat_str = f"{float(ks_stat):.4f}"
        ks_p_str = _fmt_p(float(ks_p))

        # 综合结论：以 S-W 为主（n ≤ 5000 时），否则以 K-S 为准
        primary_p = float(sw_p) if n <= _SW_MAX_N else float(ks_p)
        is_normal = primary_p > 0.05
        conclusion = "正态分布" if is_normal else "非正态分布"
        if not is_normal:
            non_normal.append(col)

        rows.append([col, n, mean_str, sd_str, sw_stat_str, sw_p_str, ks_stat_str, ks_p_str, conclusion])

    if not rows:
        raise ValueError("未生成任何有效行，请检查变量选择")

    n_non = len(non_normal)
    n_all = len(rows)
    summary = (
        f"正态性检验：共检验 {n_all} 个变量。"
        + (
            f"其中 {n_non} 个变量不符合正态分布：{', '.join(non_normal)}，"
            "建议使用非参数检验。"
            if non_normal
            else "所有变量均符合正态分布（p > 0.05）。"
        )
    )

    return AnalysisResult(
        method="hypothesis",
        tables=[TableResult(title="正态性检验结果", headers=headers, rows=rows)],
        charts=[],
        summary=summary,
        warnings=warnings,
    )


# ─────────────────────────────────────────────────────────────────────────────
# 方差齐性检验
# ─────────────────────────────────────────────────────────────────────────────

def _run_variance(
    df: pd.DataFrame, params: dict, warnings: list[str]
) -> AnalysisResult:
    variables: list[str] = list(params.get("variables") or [])
    group_var: str = params.get("group_var", "")

    if not variables:
        raise ValueError("请至少选择一个变量 (variables)")
    if not group_var:
        raise ValueError("方差齐性检验需要指定分组变量 (group_var)")
    if group_var not in df.columns:
        raise ValueError(f"分组变量 '{group_var}' 不存在")

    group_series = df[group_var].dropna()
    group_labels = sorted(group_series.unique().tolist(), key=str)
    n_groups = len(group_labels)
    if n_groups < 2:
        raise ValueError(f"分组变量 '{group_var}' 需要至少 2 个组别")

    group_dfs = {g: df[df[group_var] == g] for g in group_labels}

    headers = ["变量", "Levene 统计量", "Levene p 值", "Bartlett 统计量", "Bartlett p 值", "结论"]
    rows: list[list[Any]] = []
    non_homogeneous: list[str] = []

    for col in variables:
        if col not in df.columns:
            warnings.append(f"变量 '{col}' 不存在，已忽略")
            continue
        if not pd.api.types.is_numeric_dtype(df[col]):
            warnings.append(f"变量 '{col}' 非数值型，已忽略")
            continue

        groups_data = [group_dfs[g][col].dropna() for g in group_labels]
        if any(len(g) < 2 for g in groups_data):
            rows.append([col, "—", "—", "—", "—", "样本量不足"])
            continue

        lev_stat, lev_p = stats.levene(*groups_data)
        try:
            bar_stat, bar_p = stats.bartlett(*groups_data)
            bar_stat_str = f"{float(bar_stat):.4f}"
            bar_p_str = _fmt_p(float(bar_p))
        except Exception:
            bar_stat_str = "—"
            bar_p_str = "—"
            bar_p = float("nan")

        lev_stat_str = f"{float(lev_stat):.4f}"
        lev_p_str = _fmt_p(float(lev_p))

        # Levene 为主（对非正态更稳健）
        homogeneous = float(lev_p) > 0.05
        conclusion = "方差齐性" if homogeneous else "方差不齐"
        if not homogeneous:
            non_homogeneous.append(col)

        rows.append([col, lev_stat_str, lev_p_str, bar_stat_str, bar_p_str, conclusion])

    if not rows:
        raise ValueError("未生成任何有效行，请检查变量选择")

    summary = (
        f"方差齐性检验（分组变量：{group_var}，{n_groups} 组），共检验 {len(rows)} 个变量。"
        + (
            f"以下变量方差不齐（Levene p ≤ 0.05）：{', '.join(non_homogeneous)}，"
            "建议使用 Welch t 检验或非参数方法。"
            if non_homogeneous
            else "所有变量方差齐性良好（Levene p > 0.05）。"
        )
    )

    return AnalysisResult(
        method="hypothesis",
        tables=[TableResult(title="方差齐性检验结果", headers=headers, rows=rows)],
        charts=[],
        summary=summary,
        warnings=warnings,
    )


# ─────────────────────────────────────────────────────────────────────────────
# 卡方检验 / Fisher 精确检验
# ─────────────────────────────────────────────────────────────────────────────

def _run_chi2(
    df: pd.DataFrame, params: dict, warnings: list[str]
) -> AnalysisResult:
    row_var: str = params.get("row_var", "")
    col_var: str = params.get("col_var", "")
    force_yates: bool = bool(params.get("yates", False))

    if not row_var or not col_var:
        raise ValueError("卡方检验需要指定 row_var 和 col_var")
    for v in (row_var, col_var):
        if v not in df.columns:
            raise ValueError(f"变量 '{v}' 不存在于数据集中")

    sub = df[[row_var, col_var]].dropna()
    if len(sub) == 0:
        raise ValueError("去除缺失值后无有效数据")

    ct = pd.crosstab(sub[row_var], sub[col_var])
    ct_matrix = ct.values
    n_total = int(ct_matrix.sum())
    n_rows, n_cols = ct_matrix.shape

    if n_rows < 2 or n_cols < 2:
        raise ValueError("列联表维度不足（行或列少于 2 个类别）")

    # 期望频数矩阵
    row_sums = ct_matrix.sum(axis=1, keepdims=True)
    col_sums = ct_matrix.sum(axis=0, keepdims=True)
    expected = row_sums * col_sums / n_total
    min_expected = float(expected.min())
    low_expected_pct = float((expected < 5).mean()) * 100

    if low_expected_pct > 0:
        warnings.append(
            f"有 {low_expected_pct:.0f}% 的格子期望频数 < 5（最小期望值 {min_expected:.2f}），"
            "已自动选用 Fisher 精确检验（2×2）或报告警告（>2×2）"
        )

    method_str: str
    stat_str: str
    p_str: str
    p_val: float

    if ct_matrix.shape == (2, 2):
        if min_expected < 5 or force_yates:
            _, p_val = stats.fisher_exact(ct_matrix)
            method_str = "Fisher 精确检验"
            stat_str = "—"
        elif force_yates:
            chi2, p_val, _, _ = stats.chi2_contingency(ct_matrix, correction=True)
            method_str = "χ² 检验（Yates 校正）"
            stat_str = f"{float(chi2):.4f}"
        else:
            chi2, p_val, _, _ = stats.chi2_contingency(ct_matrix, correction=False)
            method_str = "χ² 检验"
            stat_str = f"{float(chi2):.4f}"
    else:
        chi2, p_val, dof, _ = stats.chi2_contingency(ct_matrix, correction=False)
        method_str = f"χ² 检验（{n_rows-1}×{n_cols-1} df={dof}）"
        stat_str = f"{float(chi2):.4f}"

    p_str = _fmt_p(float(p_val))

    # 构建列联表展示
    row_labels = [str(r) for r in ct.index.tolist()]
    col_labels = [str(c) for c in ct.columns.tolist()]

    ct_headers = [f"{row_var} \\ {col_var}"] + col_labels + ["合计"]
    ct_rows: list[list[Any]] = []
    for i, rl in enumerate(row_labels):
        row_data: list[Any] = [rl]
        for j in range(n_cols):
            obs = int(ct_matrix[i, j])
            exp = float(expected[i, j])
            row_data.append(f"{obs} ({exp:.1f})")
        row_data.append(int(row_sums[i, 0]))
        ct_rows.append(row_data)
    # 合计行
    totals: list[Any] = ["合计"] + [int(col_sums[0, j]) for j in range(n_cols)] + [n_total]
    ct_rows.append(totals)

    result_headers = ["检验方法", "统计量", "p 值", "结论"]
    conclusion = "两变量存在统计关联（p < 0.05）" if float(p_val) <= 0.05 else "尚无统计关联（p ≥ 0.05）"
    result_rows: list[list[Any]] = [[method_str, stat_str, p_str, conclusion]]

    summary = (
        f"卡方检验：{row_var} × {col_var}，"
        f"总样本量 n={n_total}，{method_str}，"
        f"p = {p_str}，{conclusion}。"
    )

    return AnalysisResult(
        method="hypothesis",
        tables=[
            TableResult(
                title=f"列联表（观测值，括号内为期望值）",
                headers=ct_headers,
                rows=ct_rows,
            ),
            TableResult(title="检验结果", headers=result_headers, rows=result_rows),
        ],
        charts=[],
        summary=summary,
        warnings=warnings,
    )


# ─────────────────────────────────────────────────────────────────────────────
# 单样本检验
# ─────────────────────────────────────────────────────────────────────────────

def _run_onesample(
    df: pd.DataFrame, params: dict, warnings: list[str]
) -> AnalysisResult:
    variables: list[str] = list(params.get("variables") or [])
    mu: float = float(params.get("mu", 0.0))
    alternative: str = params.get("alternative", "two-sided")

    if not variables:
        raise ValueError("请至少选择一个变量 (variables)")
    if alternative not in ("two-sided", "less", "greater"):
        raise ValueError("alternative 必须为 'two-sided'、'less' 或 'greater'")

    alt_label = {"two-sided": "双侧", "less": "左侧（< μ）", "greater": "右侧（> μ）"}[alternative]
    headers = ["变量", "n", "均值", "SD", "假设均值 μ", "统计量", "p 值", "效应量 d", "统计方法"]
    rows: list[list[Any]] = []
    sig_vars: list[str] = []

    for col in variables:
        if col not in df.columns:
            warnings.append(f"变量 '{col}' 不存在，已忽略")
            continue
        if not pd.api.types.is_numeric_dtype(df[col]):
            warnings.append(f"变量 '{col}' 非数值型，已忽略")
            continue

        s = df[col].dropna()
        n = len(s)
        if n < 3:
            rows.append([col, n, "—", "—", mu, "—", "—", "—", "样本量不足"])
            continue

        mean_str = f"{float(s.mean()):.3f}"
        sd_str = f"{float(s.std(ddof=1)):.3f}"

        if _is_normal(s):
            stat, p = stats.ttest_1samp(s, popmean=mu, alternative=alternative)
            method_str = f"单样本 t 检验（{alt_label}）"
            sd_val = float(s.std(ddof=1))
            d = (float(s.mean()) - mu) / sd_val if sd_val != 0 else float("nan")
            effect_str = f"d = {d:.3f}" if not math.isnan(d) else "—"
        else:
            # Wilcoxon 符号秩检验（与 mu 的偏差）
            diff = s - mu
            # 去掉差值为 0 的观测（Wilcoxon 要求）
            diff_nonzero = diff[diff != 0]
            if len(diff_nonzero) < 3:
                rows.append([col, n, mean_str, sd_str, mu, "—", "—", "—", "有效样本量不足"])
                continue
            result = stats.wilcoxon(diff_nonzero, alternative=alternative)
            stat, p = result.statistic, result.pvalue
            method_str = f"单样本 Wilcoxon 符号秩检验（{alt_label}）"
            effect_str = "—"

        p_str = _fmt_p(float(p))
        stat_str = f"{float(stat):.4f}"
        row: list[Any] = [col, n, mean_str, sd_str, mu, stat_str, p_str, effect_str, method_str]
        rows.append(row)
        if _parse_p(p_str) <= 0.05:
            sig_vars.append(col)

    if not rows:
        raise ValueError("未生成任何有效行，请检查变量选择")

    summary = (
        f"单样本检验（μ₀ = {mu}，{alt_label}），共检验 {len(rows)} 个变量。"
        + (
            f"以下变量与假设均值差异有统计学意义（p < 0.05）：{', '.join(sig_vars)}。"
            if sig_vars
            else "各变量与假设均值无统计学差异（p ≥ 0.05）。"
        )
    )

    return AnalysisResult(
        method="hypothesis",
        tables=[TableResult(title="单样本检验结果", headers=headers, rows=rows)],
        charts=[],
        summary=summary,
        warnings=warnings,
    )


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
