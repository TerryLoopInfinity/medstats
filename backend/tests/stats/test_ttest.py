"""pytest 测试：stats/ttest.py 和 stats/hypothesis.py

使用 data/examples/demo_basic.csv（group 为分组变量）比较 age、sbp_mmhg、bmi。
"""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from app.stats.ttest import run as ttest_run
from app.stats.hypothesis import run as hypothesis_run

DEMO_CSV = Path(__file__).parents[2] / "data" / "examples" / "demo_basic.csv"


# ── fixtures ───────────────────────────────────────────────────────────────


@pytest.fixture
def demo_df() -> pd.DataFrame:
    return pd.read_csv(DEMO_CSV)


@pytest.fixture
def normal_df() -> pd.DataFrame:
    """两组正态分布数据，组间均值不同。"""
    rng = np.random.default_rng(0)
    n = 60
    return pd.DataFrame({
        "group": [0] * 30 + [1] * 30,
        "age":   rng.normal(50, 5, 30).tolist() + rng.normal(60, 5, 30).tolist(),
        "sbp":   rng.normal(120, 10, 30).tolist() + rng.normal(140, 10, 30).tolist(),
    })


@pytest.fixture
def skewed_df() -> pd.DataFrame:
    """偏态分布数据，触发 Mann-Whitney U。"""
    rng = np.random.default_rng(7)
    return pd.DataFrame({
        "group":   [0] * 40 + [1] * 40,
        "glucose": np.concatenate([
            np.exp(rng.normal(1.5, 0.5, 40)),
            np.exp(rng.normal(2.2, 0.5, 40)),
        ]).tolist(),
    })


@pytest.fixture
def multi_group_df() -> pd.DataFrame:
    """三组数据，触发 ANOVA / Kruskal-Wallis。"""
    rng = np.random.default_rng(42)
    return pd.DataFrame({
        "group": [0] * 30 + [1] * 30 + [2] * 30,
        "bmi": rng.normal(22, 3, 30).tolist()
               + rng.normal(25, 3, 30).tolist()
               + rng.normal(28, 3, 30).tolist(),
    })


# ═══════════════════════════════════════════════════════════════════════════════
# ttest.py 测试
# ═══════════════════════════════════════════════════════════════════════════════


class TestTtestBasic:
    def test_returns_analysis_result(self, demo_df):
        result = ttest_run(demo_df, {
            "group_var": "group",
            "compare_vars": ["age", "sbp_mmhg", "bmi"],
        })
        assert result.method == "ttest"
        assert len(result.tables) >= 1
        assert result.summary

    def test_table_headers_two_groups(self, demo_df):
        result = ttest_run(demo_df, {
            "group_var": "group",
            "compare_vars": ["age"],
        })
        headers = result.tables[0].headers
        assert headers[0] == "变量"
        assert any("group=0" in h for h in headers)
        assert any("group=1" in h for h in headers)
        assert "p 值" in headers
        assert "统计方法" in headers

    def test_one_row_per_variable(self, demo_df):
        result = ttest_run(demo_df, {
            "group_var": "group",
            "compare_vars": ["age", "sbp_mmhg", "bmi"],
        })
        assert len(result.tables[0].rows) == 3

    def test_charts_one_per_variable(self, demo_df):
        result = ttest_run(demo_df, {
            "group_var": "group",
            "compare_vars": ["age", "sbp_mmhg", "bmi"],
        })
        assert len(result.charts) == 3
        for chart in result.charts:
            assert chart.chart_type == "boxplot"
            assert "series" in chart.option

    def test_p_value_populated(self, demo_df):
        result = ttest_run(demo_df, {
            "group_var": "group",
            "compare_vars": ["age", "sbp_mmhg", "bmi"],
        })
        for row in result.tables[0].rows:
            p_str = str(row[-3])
            assert p_str != "" and p_str is not None, f"p 值缺失于 {row[0]}"

    def test_effect_size_populated(self, normal_df):
        result = ttest_run(normal_df, {
            "group_var": "group",
            "compare_vars": ["age"],
        })
        effect_col = result.tables[0].rows[0][-2]
        # Cohen's d 应包含 "d ="
        assert "d =" in str(effect_col) or effect_col == "—"


class TestTtestParametric:
    def test_normal_uses_ttest(self, normal_df):
        result = ttest_run(normal_df, {
            "group_var": "group",
            "compare_vars": ["age"],
        })
        method = str(result.tables[0].rows[0][-1])
        assert "t 检验" in method

    def test_mean_sd_format(self, normal_df):
        result = ttest_run(normal_df, {
            "group_var": "group",
            "compare_vars": ["age"],
        })
        # 第二列（group=0 描述统计）
        desc = str(result.tables[0].rows[0][1])
        assert "±" in desc


class TestTtestNonParametric:
    def test_skewed_uses_mannwhitney(self, skewed_df):
        result = ttest_run(skewed_df, {
            "group_var": "group",
            "compare_vars": ["glucose"],
        })
        method = str(result.tables[0].rows[0][-1])
        assert "Mann-Whitney" in method

    def test_median_iqr_format(self, skewed_df):
        result = ttest_run(skewed_df, {
            "group_var": "group",
            "compare_vars": ["glucose"],
        })
        desc = str(result.tables[0].rows[0][1])
        assert "[" in desc

    def test_rank_biserial_in_effect(self, skewed_df):
        result = ttest_run(skewed_df, {
            "group_var": "group",
            "compare_vars": ["glucose"],
        })
        effect = str(result.tables[0].rows[0][-2])
        assert "r =" in effect


class TestTtestMultiGroup:
    def test_multigroup_anova_or_kruskal(self, multi_group_df):
        result = ttest_run(multi_group_df, {
            "group_var": "group",
            "compare_vars": ["bmi"],
        })
        method = str(result.tables[0].rows[0][-1])
        assert "ANOVA" in method or "Kruskal" in method

    def test_posthoc_table_exists(self, multi_group_df):
        result = ttest_run(multi_group_df, {
            "group_var": "group",
            "compare_vars": ["bmi"],
        })
        assert len(result.tables) == 2
        assert "事后" in result.tables[1].title

    def test_posthoc_has_three_pairs(self, multi_group_df):
        """三组 → C(3,2) = 3 对比较。"""
        result = ttest_run(multi_group_df, {
            "group_var": "group",
            "compare_vars": ["bmi"],
        })
        ph_rows = result.tables[1].rows
        assert len(ph_rows) == 3


class TestTtestPaired:
    def test_paired_ttest_result(self, normal_df):
        result = ttest_run(normal_df, {
            "group_var": "group",
            "compare_vars": ["age"],
            "compare_type": "paired",
        })
        assert result.method == "ttest"
        row = result.tables[0].rows[0]
        # 差值列应含 ±
        assert "±" in str(row[1])

    def test_paired_headers(self, normal_df):
        result = ttest_run(normal_df, {
            "group_var": "group",
            "compare_vars": ["age"],
            "compare_type": "paired",
        })
        headers = result.tables[0].headers
        assert "差值" in headers[1]


class TestTtestErrors:
    def test_no_group_var(self, demo_df):
        with pytest.raises(ValueError, match="group_var"):
            ttest_run(demo_df, {"compare_vars": ["age"]})

    def test_group_var_not_found(self, demo_df):
        with pytest.raises(ValueError, match="不存在"):
            ttest_run(demo_df, {"group_var": "nonexistent", "compare_vars": ["age"]})

    def test_no_compare_vars(self, demo_df):
        with pytest.raises(ValueError, match="compare_vars"):
            ttest_run(demo_df, {"group_var": "group"})

    def test_nonexistent_var_skipped(self, demo_df):
        result = ttest_run(demo_df, {
            "group_var": "group",
            "compare_vars": ["age", "does_not_exist"],
        })
        assert any("does_not_exist" in w for w in result.warnings)
        assert len(result.tables[0].rows) == 1  # 仅 age

    def test_invalid_compare_type(self, demo_df):
        with pytest.raises(ValueError, match="compare_type"):
            ttest_run(demo_df, {
                "group_var": "group",
                "compare_vars": ["age"],
                "compare_type": "invalid",
            })


class TestTtestDemoCsv:
    """用 demo_basic.csv 以 group 为分组变量比较 age、sbp_mmhg、bmi。"""

    def test_demo_three_vars(self, demo_df):
        result = ttest_run(demo_df, {
            "group_var": "group",
            "compare_vars": ["age", "sbp_mmhg", "bmi"],
        })
        assert len(result.tables[0].rows) == 3
        assert len(result.charts) == 3

    def test_demo_summary_mentions_vars(self, demo_df):
        result = ttest_run(demo_df, {
            "group_var": "group",
            "compare_vars": ["age", "sbp_mmhg", "bmi"],
        })
        assert "3 个变量" in result.summary

    def test_demo_boxplot_option_structure(self, demo_df):
        result = ttest_run(demo_df, {
            "group_var": "group",
            "compare_vars": ["bmi"],
        })
        opt = result.charts[0].option
        assert "xAxis" in opt
        assert "yAxis" in opt
        assert any(s["type"] == "boxplot" for s in opt["series"])
        assert any(s["type"] == "scatter" for s in opt["series"])


# ═══════════════════════════════════════════════════════════════════════════════
# hypothesis.py 测试
# ═══════════════════════════════════════════════════════════════════════════════


class TestHypothesisNormality:
    def test_returns_result(self, demo_df):
        result = hypothesis_run(demo_df, {
            "test_type": "normality",
            "variables": ["age", "sbp_mmhg", "bmi"],
        })
        assert result.method == "hypothesis"
        assert len(result.tables) == 1

    def test_headers(self, demo_df):
        result = hypothesis_run(demo_df, {
            "test_type": "normality",
            "variables": ["age"],
        })
        h = result.tables[0].headers
        assert "S-W 统计量" in h
        assert "K-S 统计量" in h
        assert "结论" in h

    def test_conclusion_field(self, demo_df):
        result = hypothesis_run(demo_df, {
            "test_type": "normality",
            "variables": ["age"],
        })
        conclusion = str(result.tables[0].rows[0][-1])
        assert conclusion in ("正态分布", "非正态分布")

    def test_no_variables_error(self, demo_df):
        with pytest.raises(ValueError, match="variables"):
            hypothesis_run(demo_df, {"test_type": "normality"})


class TestHypothesisVariance:
    def test_variance_result(self, demo_df):
        result = hypothesis_run(demo_df, {
            "test_type": "variance",
            "variables": ["age", "sbp_mmhg", "bmi"],
            "group_var": "group",
        })
        assert len(result.tables[0].rows) == 3

    def test_levene_column(self, demo_df):
        result = hypothesis_run(demo_df, {
            "test_type": "variance",
            "variables": ["age"],
            "group_var": "group",
        })
        h = result.tables[0].headers
        assert "Levene 统计量" in h

    def test_no_group_var_error(self, demo_df):
        with pytest.raises(ValueError, match="group_var"):
            hypothesis_run(demo_df, {
                "test_type": "variance",
                "variables": ["age"],
            })


class TestHypothesisChi2:
    def test_chi2_basic(self, demo_df):
        result = hypothesis_run(demo_df, {
            "test_type": "chi2",
            "row_var": "group",
            "col_var": "sex",
        })
        assert result.method == "hypothesis"
        assert len(result.tables) == 2  # 列联表 + 检验结果

    def test_contingency_table_has_totals(self, demo_df):
        result = hypothesis_run(demo_df, {
            "test_type": "chi2",
            "row_var": "group",
            "col_var": "sex",
        })
        ct_rows = result.tables[0].rows
        last_row = ct_rows[-1]
        assert str(last_row[0]) == "合计"

    def test_fisher_small_expected(self):
        """期望值 < 5 的 2×2 表应用 Fisher 精确检验。"""
        df = pd.DataFrame({
            "a": [0, 0, 0, 0, 0, 1, 1, 1, 1, 1],
            "b": [0, 0, 0, 0, 1, 1, 1, 1, 1, 1],
        })
        result = hypothesis_run(df, {
            "test_type": "chi2",
            "row_var": "a",
            "col_var": "b",
        })
        assert "Fisher" in result.tables[1].rows[0][0]

    def test_missing_vars_error(self, demo_df):
        with pytest.raises(ValueError, match="row_var"):
            hypothesis_run(demo_df, {"test_type": "chi2"})


class TestHypothesisOneSample:
    def test_onesample_basic(self, demo_df):
        result = hypothesis_run(demo_df, {
            "test_type": "onesample",
            "variables": ["age"],
            "mu": 50.0,
        })
        assert result.method == "hypothesis"
        assert len(result.tables[0].rows) == 1

    def test_onesample_headers(self, demo_df):
        result = hypothesis_run(demo_df, {
            "test_type": "onesample",
            "variables": ["age"],
            "mu": 50.0,
        })
        h = result.tables[0].headers
        assert "假设均值 μ" in h
        assert "统计方法" in h

    def test_mu_in_result(self, demo_df):
        result = hypothesis_run(demo_df, {
            "test_type": "onesample",
            "variables": ["age"],
            "mu": 55.0,
        })
        row = result.tables[0].rows[0]
        # mu 列应为 55.0
        assert float(row[4]) == 55.0

    def test_invalid_test_type(self, demo_df):
        with pytest.raises(ValueError, match="未知检验类型"):
            hypothesis_run(demo_df, {"test_type": "bogus"})
