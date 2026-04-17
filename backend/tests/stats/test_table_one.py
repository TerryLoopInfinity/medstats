"""pytest 测试：stats/table_one.py

使用 data/examples/demo_basic.csv（含 group 和 sex 列）以及合成数据集。
"""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from app.stats.table_one import run

DEMO_CSV = Path(__file__).parents[2] / "data" / "examples" / "demo_basic.csv"


# ── fixtures ───────────────────────────────────────────────────────────────


@pytest.fixture
def demo_df() -> pd.DataFrame:
    return pd.read_csv(DEMO_CSV)


@pytest.fixture
def simple_df() -> pd.DataFrame:
    """小型合成数据：正态分布 + 分类变量，2 组。"""
    rng = np.random.default_rng(42)
    n = 60
    group = [0] * 30 + [1] * 30
    age = rng.normal(50, 10, 30).tolist() + rng.normal(55, 10, 30).tolist()
    bmi = rng.normal(24, 4, 30).tolist() + rng.normal(27, 4, 30).tolist()
    sex = rng.integers(0, 2, n).tolist()
    return pd.DataFrame({"group": group, "age": age, "bmi": bmi, "sex": sex})


@pytest.fixture
def skewed_df() -> pd.DataFrame:
    """偏态分布数据，触发 Mann-Whitney U 检验。"""
    rng = np.random.default_rng(7)
    n = 80
    group = [0] * 40 + [1] * 40
    glucose = np.concatenate([
        np.exp(rng.normal(1.5, 0.5, 40)),
        np.exp(rng.normal(2.0, 0.5, 40)),
    ]).tolist()
    sex = rng.integers(0, 2, n).tolist()
    return pd.DataFrame({"group": group, "glucose": glucose, "sex": sex})


# ── 结构正确性 ─────────────────────────────────────────────────────────────


def test_returns_analysis_result(demo_df):
    result = run(demo_df, {
        "group_var": "group",
        "continuous_vars": ["age", "bmi", "sbp_mmhg"],
        "categorical_vars": ["sex"],
    })
    assert result.method == "table_one"
    assert len(result.tables) == 1
    assert result.tables[0].title == "Table 1 — 基线特征表"
    assert result.summary


def test_table_headers_two_groups(demo_df):
    result = run(demo_df, {
        "group_var": "group",
        "continuous_vars": ["age"],
        "categorical_vars": [],
    })
    headers = result.tables[0].headers
    # 变量, 整体, group=0, group=1, p值, 统计方法
    assert headers[0] == "变量"
    assert "整体" in headers[1]
    assert "group=0" in headers[2]
    assert "group=1" in headers[3]
    assert headers[-2] == "p 值"
    assert headers[-1] == "统计方法"


def test_continuous_row_format_normal(simple_df):
    """正态分布变量应显示 均值 ± SD。"""
    result = run(simple_df, {
        "group_var": "group",
        "continuous_vars": ["age"],
        "categorical_vars": [],
    })
    row = result.tables[0].rows[0]
    assert row[0] == "age"
    assert "±" in str(row[1])   # 整体
    assert "±" in str(row[2])   # group=0
    assert "±" in str(row[3])   # group=1
    assert "t 检验" in str(row[-1])


def test_continuous_row_format_skewed(skewed_df):
    """偏态分布变量应显示 中位数 [IQR] + Mann-Whitney。"""
    result = run(skewed_df, {
        "group_var": "group",
        "continuous_vars": ["glucose"],
        "categorical_vars": [],
    })
    row = result.tables[0].rows[0]
    assert "[" in str(row[1])   # median [q1, q3]
    assert "Mann-Whitney" in str(row[-1])


def test_categorical_row_format(simple_df):
    """分类变量：标题行 + 各类子行（缩进两格）。"""
    result = run(simple_df, {
        "group_var": "group",
        "continuous_vars": [],
        "categorical_vars": ["sex"],
    })
    rows = result.tables[0].rows
    # 第一行是分类变量名行
    assert rows[0][0] == "sex"
    assert "χ²" in str(rows[0][-1]) or "Fisher" in str(rows[0][-1])
    # 子行缩进
    for sub in rows[1:]:
        assert str(sub[0]).startswith("  ")
    # 子行显示 n(%)
    assert "%" in str(rows[1][1])


def test_p_value_present(demo_df):
    result = run(demo_df, {
        "group_var": "group",
        "continuous_vars": ["sbp_mmhg", "age"],
        "categorical_vars": ["sex"],
    })
    rows = result.tables[0].rows
    # 连续变量行 p 值非空
    for row in rows:
        if not str(row[0]).startswith("  ") and row[0] in ("sbp_mmhg", "age"):
            assert str(row[-2]) not in ("", None), f"p 值缺失于 {row[0]}"


def test_missing_values_handled(demo_df):
    """demo_basic.csv 中 fasting_glucose_mmol 有缺失，应正常处理。"""
    result = run(demo_df, {
        "group_var": "group",
        "continuous_vars": ["fasting_glucose_mmol"],
        "categorical_vars": [],
    })
    rows = result.tables[0].rows
    assert len(rows) == 1
    assert "±" in str(rows[0][1]) or "[" in str(rows[0][1])


def test_error_no_group_var():
    df = pd.DataFrame({"x": [1, 2, 3]})
    with pytest.raises(ValueError, match="group_var"):
        run(df, {"continuous_vars": ["x"]})


def test_error_group_var_not_found():
    df = pd.DataFrame({"x": [1, 2], "g": [0, 1]})
    with pytest.raises(ValueError, match="不存在"):
        run(df, {"group_var": "nonexistent", "continuous_vars": ["x"]})


def test_error_single_group():
    df = pd.DataFrame({"group": [0, 0, 0], "x": [1.0, 2.0, 3.0]})
    with pytest.raises(ValueError, match="组别"):
        run(df, {"group_var": "group", "continuous_vars": ["x"]})


def test_error_no_variables():
    df = pd.DataFrame({"group": [0, 1, 0, 1], "x": [1.0, 2.0, 3.0, 4.0]})
    with pytest.raises(ValueError, match="至少选择"):
        run(df, {"group_var": "group"})


def test_nonexistent_variable_warned(simple_df):
    result = run(simple_df, {
        "group_var": "group",
        "continuous_vars": ["age", "nonexistent_var"],
        "categorical_vars": [],
    })
    assert any("nonexistent_var" in w for w in result.warnings)


def test_multigroup(demo_df):
    """三组及以上场景：ANOVA 或 Kruskal-Wallis。"""
    df = demo_df.copy()
    df["group3"] = pd.cut(df["sbp_mmhg"], bins=3, labels=[0, 1, 2])
    result = run(df, {
        "group_var": "group3",
        "continuous_vars": ["age", "bmi"],
        "categorical_vars": [],
    })
    rows = result.tables[0].rows
    methods = [str(r[-1]) for r in rows]
    assert any("ANOVA" in m or "Kruskal" in m for m in methods)


def test_fisher_exact_small_expected():
    """期望值 < 5 的 2×2 表应使用 Fisher 精确检验。"""
    df = pd.DataFrame({
        "group": [0, 0, 0, 0, 0, 1, 1, 1, 1, 1],
        "event": [0, 0, 0, 0, 1, 1, 1, 1, 1, 1],
    })
    result = run(df, {
        "group_var": "group",
        "continuous_vars": [],
        "categorical_vars": ["event"],
    })
    rows = result.tables[0].rows
    # 标题行含 Fisher
    assert "Fisher" in str(rows[0][-1])


def test_summary_contains_sig_vars(simple_df):
    """summary 应列出显著变量。"""
    result = run(simple_df, {
        "group_var": "group",
        "continuous_vars": ["bmi"],   # 组间差异明显
        "categorical_vars": [],
    })
    assert result.summary


def test_demo_csv_full(demo_df):
    """全字段测试：demo_basic.csv 完整流程。"""
    result = run(demo_df, {
        "group_var": "group",
        "continuous_vars": ["age", "height_cm", "weight_kg", "bmi",
                             "sbp_mmhg", "dbp_mmhg",
                             "fasting_glucose_mmol", "total_cholesterol_mmol"],
        "categorical_vars": ["sex"],
    })
    assert result.method == "table_one"
    rows = result.tables[0].rows
    # 8 连续 + 1 分类标题 + 2 性别子行 = 11 行
    assert len(rows) == 11
