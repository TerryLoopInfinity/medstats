"""PSM 模块测试 — 使用 demo_survival.csv。"""

import pytest
import pandas as pd
import numpy as np
from pathlib import Path

DATA_PATH = Path(__file__).parent.parent.parent / "data" / "examples" / "demo_survival.csv"

BASE_PARAMS = {
    "treatment_col": "treatment",
    "covariates": ["age", "gender", "stage", "biomarker"],
    "outcome_col": "time",
    "outcome_type": "survival",
    "time_col": "time",
    "event_col": "event",
    "method": "nearest",
    "ratio": 1,
    "with_replacement": False,
}


@pytest.fixture(scope="module")
def df():
    return pd.read_csv(DATA_PATH)


def _run(df, **overrides):
    from app.stats.psm import run
    params = {**BASE_PARAMS, **overrides}
    return run(df, params)


# ── 基础运行 ──────────────────────────────────────────────────────────────────

def test_run_returns_result(df):
    result = _run(df)
    assert result.method == "psm"
    assert len(result.tables) >= 3
    assert len(result.charts) >= 3


# ── PS 分布有重叠 ─────────────────────────────────────────────────────────────

def test_ps_overlap(df):
    from app.stats.psm import _estimate_ps
    df_work = df[["treatment", "age", "gender", "stage", "biomarker"]].dropna().copy()
    treat_vals = df_work["treatment"].unique()
    treat_sorted = sorted(treat_vals, key=str)
    df_work["_treat"] = df_work["treatment"].map({treat_sorted[0]: 0, treat_sorted[1]: 1})

    ps, _, warns = _estimate_ps(df_work, "_treat", ["age", "gender", "stage", "biomarker"])
    ps_t = ps[df_work["_treat"].values == 1]
    ps_c = ps[df_work["_treat"].values == 0]
    overlap_lo = max(ps_t.min(), ps_c.min())
    overlap_hi = min(ps_t.max(), ps_c.max())
    assert overlap_lo < overlap_hi, "处理组与对照组 PS 应有重叠"


# ── 匹配后 SMD 整体下降，caliper 方法全部 < 0.2 ──────────────────────────────

def test_smd_after_matching(df):
    """验证 caliper 匹配后所有连续协变量 SMD < 0.2，分类变量 SMD 整体下降。"""
    result = _run(df, method="caliper")
    balance_table = next(t for t in result.tables if "平衡" in t.title)
    before_idx = balance_table.headers.index("匹配前 SMD")
    after_idx  = balance_table.headers.index("匹配后 SMD")

    smds_before, smds_after = [], []
    for row in balance_table.rows:
        # 跳过类别头行（无数值）
        if row[0] == "" or (not str(row[0]).startswith("  ") and row[after_idx] in ("", "—")):
            continue
        try:
            smds_before.append(float(row[before_idx]))
            smds_after.append(float(row[after_idx]))
        except (ValueError, TypeError):
            pass

    assert len(smds_after) > 0, "应有 SMD 数据行"
    # 匹配后 SMD 均值应显著低于匹配前
    assert sum(smds_after) < sum(smds_before), "匹配后总体 SMD 应低于匹配前"
    # 连续变量（age, biomarker）匹配后 SMD < 0.2
    age_row = next((r for r in balance_table.rows if r[0] == "age"), None)
    bio_row = next((r for r in balance_table.rows if r[0] == "biomarker"), None)
    for row in [age_row, bio_row]:
        if row is None:
            continue
        smd_val = float(row[after_idx])
        assert smd_val < 0.2, f"连续协变量 '{row[0]}' 匹配后 SMD = {smd_val:.3f} ≥ 0.2"


# ── 匹配成功率 > 80% ──────────────────────────────────────────────────────────

def test_match_rate(df):
    result = _run(df)
    summary_table = next(t for t in result.tables if t.title == "匹配摘要")
    rate_row = next(r for r in summary_table.rows if "匹配成功率" in str(r[0]))
    rate_str = rate_row[1].replace("%", "")
    rate = float(rate_str) / 100
    assert rate > 0.8, f"匹配成功率 {rate:.1%} 应 > 80%"


# ── 分层 Cox 输出 HR ──────────────────────────────────────────────────────────

def test_stratified_cox_hr(df):
    result = _run(df)
    effect_table = next((t for t in result.tables if "Cox" in t.title), None)
    assert effect_table is not None, "应存在处理效应表"
    cox_row = next((r for r in effect_table.rows if "Cox" in str(r[0])), None)
    assert cox_row is not None
    hr_str = cox_row[1]
    assert hr_str != "—", "HR 不应为缺失"
    hr_val = float(hr_str)
    assert hr_val > 0, "HR 应为正值"


# ── 1:1 与 1:2 结果格式一致 ───────────────────────────────────────────────────

def test_ratio_1_vs_2_consistent_format(df):
    res1 = _run(df, ratio=1)
    res2 = _run(df, ratio=2)

    for res in [res1, res2]:
        assert len(res.tables) >= 3
        assert len(res.charts) >= 3
        headers_balance = next(t for t in res.tables if "平衡" in t.title).headers
        assert "匹配前 SMD" in headers_balance
        assert "匹配后 SMD" in headers_balance

    # 1:2 匹配对照组样本量应约为 1:1 的两倍
    def n_ctrl(res):
        sum_t = next(t for t in res.tables if t.title == "匹配摘要")
        row = next(r for r in sum_t.rows if "对照组" in str(r[0]) and "匹配后" in str(r[0]))
        return int(row[1])

    assert n_ctrl(res2) >= n_ctrl(res1), "1:2 匹配对照组样本量应 ≥ 1:1"


# ── caliper 匹配 ──────────────────────────────────────────────────────────────

def test_caliper_method(df):
    result = _run(df, method="caliper")
    assert result.method == "psm"
    assert len(result.tables) >= 3


# ── optimal 匹配 ─────────────────────────────────────────────────────────────

def test_optimal_method(df):
    result = _run(df, method="optimal")
    assert result.method == "psm"
    match_table = next(t for t in result.tables if t.title == "匹配摘要")
    method_row = next(r for r in match_table.rows if r[0] == "匹配方法")
    assert "最优" in method_row[1]


# ── 图表数量与类型 ────────────────────────────────────────────────────────────

def test_charts(df):
    result = _run(df)
    chart_types = [c.chart_type for c in result.charts]
    assert "line" in chart_types, "应有 PS 核密度图（line）"
    assert "scatter" in chart_types, "应有 Love plot（scatter）"
    assert "bar" in chart_types, "应有 SMD 条形图（bar）"
    assert "kaplan_meier" in chart_types, "应有 KM 曲线（kaplan_meier）"


# ── 无结局变量时也能运行 ──────────────────────────────────────────────────────

def test_no_outcome(df):
    result = _run(df, outcome_col="", outcome_type="continuous")
    assert result.method == "psm"
    assert len(result.tables) >= 3
    assert len(result.charts) >= 3
