"""pytest tests for stats/linear_reg_adjusted.py

测试场景：
  - 因变量 sbp_mmhg
  - 暴露变量 bmi
  - 协变量 age、heart_rate、sex
  - 分层变量 group
  - 交互项 age
"""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from app.stats.linear_reg_adjusted import run

DEMO_CSV = Path(__file__).parents[2] / "data" / "examples" / "demo_basic.csv"


# ── fixtures ────────────────────────────────────────────────────────────────

@pytest.fixture
def demo_df() -> pd.DataFrame:
    return pd.read_csv(DEMO_CSV)


@pytest.fixture
def simple_df() -> pd.DataFrame:
    """带已知混杂关系的合成数据集。"""
    rng = np.random.default_rng(42)
    n = 200
    age = rng.normal(50, 10, n)
    exposure = 0.5 * age + rng.normal(0, 3, n)  # exposure 与 age 相关
    outcome = 2.0 * exposure + 1.5 * age + rng.normal(0, 5, n)
    group = (age > 50).astype(int)
    return pd.DataFrame({
        "outcome": outcome,
        "exposure": exposure,
        "age": age,
        "noise": rng.normal(0, 1, n),
        "group": group,
    })


# ── 基础功能 ─────────────────────────────────────────────────────────────────

class TestBasicRun:
    def test_returns_analysis_result(self, demo_df):
        result = run(demo_df, {
            "outcome": "sbp_mmhg",
            "exposure": "bmi",
            "covariates": ["age", "heart_rate", "sex"],
        })
        assert result.method == "linear_reg_adjusted"
        assert len(result.tables) >= 3
        assert result.summary

    def test_tables_have_required_titles(self, demo_df):
        result = run(demo_df, {
            "outcome": "sbp_mmhg",
            "exposure": "bmi",
            "covariates": ["age", "heart_rate", "sex"],
        })
        titles = [t.title for t in result.tables]
        assert any("逐步调整" in t for t in titles), "缺少模型对比表"
        assert any("系数" in t for t in titles), "缺少系数表"
        assert any("混杂" in t for t in titles), "缺少混杂评估表"

    def test_model_comparison_rows(self, demo_df):
        result = run(demo_df, {
            "outcome": "sbp_mmhg",
            "exposure": "bmi",
            "covariates": ["age", "heart_rate", "sex"],
            "mode": "both",
        })
        comp_table = next(t for t in result.tables if "逐步调整" in t.title)
        # mode=both: Model 1, Model 2, Model 3（至少 2 行）
        assert len(comp_table.rows) >= 2

    def test_comparison_table_headers(self, demo_df):
        result = run(demo_df, {
            "outcome": "sbp_mmhg",
            "exposure": "bmi",
            "covariates": ["age"],
        })
        comp_table = next(t for t in result.tables if "逐步调整" in t.title)
        h = comp_table.headers
        assert "β" in h
        assert "p 值" in h
        assert "R²" in h
        assert "β 变化 (%)" in h


class TestModes:
    def test_crude_only(self, demo_df):
        result = run(demo_df, {
            "outcome": "sbp_mmhg",
            "exposure": "bmi",
            "covariates": ["age"],
            "mode": "crude",
        })
        comp_table = next(t for t in result.tables if "逐步调整" in t.title)
        # crude: 只有 Model 1
        assert len(comp_table.rows) == 1
        assert "粗模型" in comp_table.rows[0][0]

    def test_adjusted_only(self, demo_df):
        result = run(demo_df, {
            "outcome": "sbp_mmhg",
            "exposure": "bmi",
            "covariates": ["age"],
            "mode": "adjusted",
        })
        comp_table = next(t for t in result.tables if "逐步调整" in t.title)
        # adjusted: 只有调整模型
        assert len(comp_table.rows) == 1

    def test_no_covariates_crude_adjusted_equal(self, demo_df):
        result = run(demo_df, {
            "outcome": "sbp_mmhg",
            "exposure": "bmi",
            "covariates": [],
        })
        comp_table = next(t for t in result.tables if "逐步调整" in t.title)
        # 无协变量时 crude == adjusted，β 变化 = 参照
        beta_col = 3  # headers: 模型, 协变量, n, β, ...
        betas = [row[beta_col] for row in comp_table.rows]
        assert betas[0] == betas[-1]


# ── 暴露系数变化追踪 ─────────────────────────────────────────────────────────

class TestConfoundingTracking:
    def test_pct_change_computed(self, simple_df):
        """已知 age 是混杂因素，调整后 exposure β 应有明显变化。"""
        result = run(simple_df, {
            "outcome": "outcome",
            "exposure": "exposure",
            "covariates": ["age"],
        })
        comp_table = next(t for t in result.tables if "逐步调整" in t.title)
        # Model 1 的 β 变化应为"参照"
        assert comp_table.rows[0][-1] == "参照"

    def test_confounding_table_rows_match_covariates(self, demo_df):
        result = run(demo_df, {
            "outcome": "sbp_mmhg",
            "exposure": "bmi",
            "covariates": ["age", "heart_rate", "sex"],
        })
        conf_table = next(t for t in result.tables if "混杂" in t.title)
        assert len(conf_table.rows) == 3

    def test_confounding_table_has_warning_flag(self, simple_df):
        """age 是强混杂，变化应 >10%，应有 ⚠ 标记。"""
        result = run(simple_df, {
            "outcome": "outcome",
            "exposure": "exposure",
            "covariates": ["age"],
        })
        conf_table = next(t for t in result.tables if "混杂" in t.title)
        pct_col = 4  # headers: 协变量, β, CI, p, β变化%, 判断
        pct_str = str(conf_table.rows[0][pct_col])
        # 因 age 是强混杂，变化应 >10%
        val = float(pct_str.replace("%", "").replace("⚠", "").strip())
        assert abs(val) > 10

    def test_crude_vs_adjusted_beta_direction(self, simple_df):
        """调整 age 后，exposure β 应更接近真实值 2.0。"""
        result = run(simple_df, {
            "outcome": "outcome",
            "exposure": "exposure",
            "covariates": ["age"],
            "mode": "both",
        })
        comp_table = next(t for t in result.tables if "逐步调整" in t.title)
        beta_idx = 3
        crude_beta = float(comp_table.rows[0][beta_idx])
        adj_beta = float(comp_table.rows[-1][beta_idx])
        # 调整后应更接近真值 2.0
        assert abs(adj_beta - 2.0) < abs(crude_beta - 2.0)


# ── 系数表 ───────────────────────────────────────────────────────────────────

class TestCoefficientTable:
    def test_exposure_marked_with_star(self, demo_df):
        result = run(demo_df, {
            "outcome": "sbp_mmhg",
            "exposure": "bmi",
            "covariates": ["age", "heart_rate"],
        })
        coef_table = next(t for t in result.tables if "系数" in t.title)
        first_col = [str(r[0]) for r in coef_table.rows]
        assert any("★" in c and "bmi" in c for c in first_col), "暴露变量应有 ★ 标记"

    def test_coef_table_includes_all_predictors(self, demo_df):
        result = run(demo_df, {
            "outcome": "sbp_mmhg",
            "exposure": "bmi",
            "covariates": ["age", "heart_rate"],
        })
        coef_table = next(t for t in result.tables if "系数" in t.title)
        row_names = " ".join(str(r[0]) for r in coef_table.rows)
        assert "bmi" in row_names
        assert "age" in row_names
        assert "heart_rate" in row_names
        assert "Intercept" in row_names

    def test_std_beta_not_shown_for_intercept(self, demo_df):
        result = run(demo_df, {
            "outcome": "sbp_mmhg",
            "exposure": "bmi",
            "covariates": ["age"],
        })
        coef_table = next(t for t in result.tables if "系数" in t.title)
        intercept_row = next(r for r in coef_table.rows if "Intercept" in str(r[0]))
        assert intercept_row[6] == "—"


# ── 分层分析 ─────────────────────────────────────────────────────────────────

class TestStratifiedAnalysis:
    def test_stratified_table_exists(self, demo_df):
        result = run(demo_df, {
            "outcome": "sbp_mmhg",
            "exposure": "bmi",
            "covariates": ["age"],
            "stratify_var": "group",
        })
        strat_tables = [t for t in result.tables if "分层" in t.title]
        assert len(strat_tables) == 1

    def test_stratified_rows_have_group_values(self, demo_df):
        result = run(demo_df, {
            "outcome": "sbp_mmhg",
            "exposure": "bmi",
            "covariates": ["age"],
            "stratify_var": "group",
        })
        strat_table = next(t for t in result.tables if "分层" in t.title)
        row_names = [str(r[0]) for r in strat_table.rows]
        assert any("group" in rn for rn in row_names)

    def test_stratified_forest_chart(self, demo_df):
        result = run(demo_df, {
            "outcome": "sbp_mmhg",
            "exposure": "bmi",
            "covariates": ["age"],
            "stratify_var": "group",
        })
        forest_charts = [c for c in result.charts if c.chart_type == "forest_plot"]
        assert len(forest_charts) >= 2  # 模型对比森林图 + 分层森林图

    def test_heterogeneity_warning_added(self, demo_df):
        result = run(demo_df, {
            "outcome": "sbp_mmhg",
            "exposure": "bmi",
            "covariates": ["age"],
            "stratify_var": "group",
        })
        assert any("Cochran Q" in w for w in result.warnings)

    def test_invalid_stratify_var_warned(self, demo_df):
        result = run(demo_df, {
            "outcome": "sbp_mmhg",
            "exposure": "bmi",
            "covariates": ["age"],
            "stratify_var": "nonexistent_var",
        })
        assert any("nonexistent_var" in w for w in result.warnings)
        strat_tables = [t for t in result.tables if "分层" in t.title]
        assert len(strat_tables) == 0  # 已跳过分层分析


# ── 交互项检验 ───────────────────────────────────────────────────────────────

class TestInteractionTest:
    def test_interaction_table_exists(self, demo_df):
        result = run(demo_df, {
            "outcome": "sbp_mmhg",
            "exposure": "bmi",
            "covariates": ["age", "heart_rate", "sex"],
            "interaction_var": "age",
        })
        int_tables = [t for t in result.tables if "交互" in t.title]
        assert len(int_tables) == 1

    def test_interaction_table_has_one_row(self, demo_df):
        result = run(demo_df, {
            "outcome": "sbp_mmhg",
            "exposure": "bmi",
            "covariates": ["age"],
            "interaction_var": "age",
        })
        int_table = next(t for t in result.tables if "交互" in t.title)
        assert len(int_table.rows) == 1

    def test_interaction_term_label(self, demo_df):
        result = run(demo_df, {
            "outcome": "sbp_mmhg",
            "exposure": "bmi",
            "covariates": ["age"],
            "interaction_var": "age",
        })
        int_table = next(t for t in result.tables if "交互" in t.title)
        term = str(int_table.rows[0][0])
        assert "bmi" in term and "age" in term

    def test_interaction_warning_in_warnings(self, demo_df):
        result = run(demo_df, {
            "outcome": "sbp_mmhg",
            "exposure": "bmi",
            "covariates": ["age"],
            "interaction_var": "age",
        })
        assert any("交互" in w for w in result.warnings)


# ── 图表 ────────────────────────────────────────────────────────────────────

class TestCharts:
    def test_forest_plot_chart_exists(self, demo_df):
        result = run(demo_df, {
            "outcome": "sbp_mmhg",
            "exposure": "bmi",
            "covariates": ["age"],
        })
        forest_charts = [c for c in result.charts if c.chart_type == "forest_plot"]
        assert len(forest_charts) >= 1

    def test_forest_plot_has_forest_data(self, demo_df):
        result = run(demo_df, {
            "outcome": "sbp_mmhg",
            "exposure": "bmi",
            "covariates": ["age"],
            "mode": "both",
        })
        forest = next(c for c in result.charts if c.chart_type == "forest_plot"
                      and "多模型" in c.title)
        fd = forest.option["forestData"]
        assert isinstance(fd, list)
        assert len(fd) >= 2  # mode=both: 至少 Model 1 + Model 3
        for point in fd:
            assert "label" in point
            assert "beta" in point
            assert "ci_lo" in point
            assert "ci_hi" in point

    def test_bar_chart_for_covariates(self, demo_df):
        result = run(demo_df, {
            "outcome": "sbp_mmhg",
            "exposure": "bmi",
            "covariates": ["age", "heart_rate"],
        })
        bar_charts = [c for c in result.charts if c.chart_type == "bar"]
        assert len(bar_charts) >= 1

    def test_no_covariates_no_bar_chart(self, demo_df):
        result = run(demo_df, {
            "outcome": "sbp_mmhg",
            "exposure": "bmi",
            "covariates": [],
        })
        bar_charts = [c for c in result.charts if c.chart_type == "bar"]
        assert len(bar_charts) == 0


# ── 全场景测试（demo_basic.csv 完整验证） ────────────────────────────────────

class TestFullScenarioDemoCsv:
    def test_full_run_sbp_bmi(self, demo_df):
        """完整场景：因变量 sbp_mmhg，暴露 bmi，协变量 age/heart_rate/sex，
        分层 group，交互 age。"""
        result = run(demo_df, {
            "outcome": "sbp_mmhg",
            "exposure": "bmi",
            "covariates": ["age", "heart_rate", "sex"],
            "mode": "both",
            "stratify_var": "group",
            "interaction_var": "age",
        })
        assert result.method == "linear_reg_adjusted"
        # 应有 5 张表：对比表、系数表、混杂表、分层表、交互表
        assert len(result.tables) == 5
        # 应有 3 个图：模型森林图、分层森林图、协变量条形图
        assert len(result.charts) == 3
        assert "sbp_mmhg" in result.summary
        assert "bmi" in result.summary

    def test_custom_model2_covariates(self, demo_df):
        result = run(demo_df, {
            "outcome": "sbp_mmhg",
            "exposure": "bmi",
            "covariates": ["age", "heart_rate", "sex"],
            "model2_covariates": ["age"],
            "model3_covariates": ["age", "heart_rate", "sex"],
        })
        comp_table = next(t for t in result.tables if "逐步调整" in t.title)
        # Model 1, Model 2 (age only), Model 3 (all)
        assert len(comp_table.rows) == 3

    def test_summary_contains_key_info(self, demo_df):
        result = run(demo_df, {
            "outcome": "sbp_mmhg",
            "exposure": "bmi",
            "covariates": ["age"],
            "mode": "both",
        })
        assert "粗效应" in result.summary
        assert "全调整" in result.summary
        assert "变化" in result.summary


# ── 错误处理 ─────────────────────────────────────────────────────────────────

class TestErrors:
    def test_missing_outcome(self, demo_df):
        with pytest.raises(ValueError, match="outcome"):
            run(demo_df, {"exposure": "bmi"})

    def test_missing_exposure(self, demo_df):
        with pytest.raises(ValueError, match="暴露变量"):
            run(demo_df, {"outcome": "sbp_mmhg"})

    def test_nonexistent_outcome(self, demo_df):
        with pytest.raises(ValueError, match="不存在"):
            run(demo_df, {"outcome": "no_col", "exposure": "bmi"})

    def test_nonexistent_exposure(self, demo_df):
        with pytest.raises(ValueError, match="不存在"):
            run(demo_df, {"outcome": "sbp_mmhg", "exposure": "no_col"})

    def test_exposure_equals_outcome(self, demo_df):
        with pytest.raises(ValueError, match="相同"):
            run(demo_df, {"outcome": "sbp_mmhg", "exposure": "sbp_mmhg"})

    def test_invalid_mode(self, demo_df):
        with pytest.raises(ValueError, match="mode"):
            run(demo_df, {"outcome": "sbp_mmhg", "exposure": "bmi", "mode": "invalid"})

    def test_nonexistent_covariate_warned(self, demo_df):
        result = run(demo_df, {
            "outcome": "sbp_mmhg",
            "exposure": "bmi",
            "covariates": ["age", "ghost_var"],
        })
        assert any("ghost_var" in w for w in result.warnings)
