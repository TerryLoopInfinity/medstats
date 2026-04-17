"""pytest tests for stats/logistic_reg_adjusted.py

测试场景：
  - 因变量 group（二分类）
  - 暴露变量 bmi
  - 协变量 age、sbp_mmhg（数值型）
  - 分类协变量 sex（含参考组）
  - 分层变量 sex（数值型编码后）
  - 交互项 age
"""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from app.stats.logistic_reg_adjusted import run

DEMO_CSV = Path(__file__).parents[2] / "data" / "examples" / "demo_basic.csv"


# ── fixtures ─────────────────────────────────────────────────────────────────

@pytest.fixture
def demo_df() -> pd.DataFrame:
    return pd.read_csv(DEMO_CSV)


@pytest.fixture
def synthetic_df() -> pd.DataFrame:
    """已知混杂关系的合成数据集：age 是 bmi→group 的混杂因素。"""
    rng = np.random.default_rng(42)
    n = 300
    age = rng.normal(50, 10, n)
    bmi = 0.3 * age + rng.normal(25, 3, n)
    log_odds = -10 + 0.15 * bmi + 0.08 * age
    prob = 1 / (1 + np.exp(-log_odds))
    group = rng.binomial(1, np.clip(prob, 0, 1), n)
    sex = (rng.random(n) > 0.5).astype(int)
    return pd.DataFrame({
        "group": group,
        "bmi": bmi,
        "age": age,
        "sex": sex,
        "noise": rng.normal(0, 1, n),
    })


# ── 基础功能 ─────────────────────────────────────────────────────────────────

class TestBasicRun:
    def test_returns_analysis_result(self, demo_df):
        result = run(demo_df, {
            "outcome": "group",
            "exposure": "bmi",
            "covariates": ["age", "sbp_mmhg"],
        })
        assert result.method == "logistic_reg_adjusted"
        assert len(result.tables) >= 3
        assert result.summary

    def test_tables_have_required_titles(self, demo_df):
        result = run(demo_df, {
            "outcome": "group",
            "exposure": "bmi",
            "covariates": ["age", "sbp_mmhg"],
        })
        titles = [t.title for t in result.tables]
        assert any("逐步调整" in t for t in titles)
        assert any("系数" in t for t in titles)
        assert any("混杂" in t for t in titles)
        assert any("AUC" in t for t in titles)

    def test_model_comparison_has_or_columns(self, demo_df):
        result = run(demo_df, {
            "outcome": "group",
            "exposure": "bmi",
            "covariates": ["age"],
            "mode": "both",
        })
        comp = next(t for t in result.tables if "逐步调整" in t.title)
        assert "OR" in comp.headers
        assert "AUC" in comp.headers
        assert "log(OR) 变化 (%)" in comp.headers

    def test_model1_pct_is_reference(self, demo_df):
        result = run(demo_df, {
            "outcome": "group",
            "exposure": "bmi",
            "covariates": ["age"],
            "mode": "both",
        })
        comp = next(t for t in result.tables if "逐步调整" in t.title)
        assert comp.rows[0][-1] == "参照"


# ── 分析模式 ─────────────────────────────────────────────────────────────────

class TestModes:
    def test_crude_only(self, demo_df):
        result = run(demo_df, {
            "outcome": "group",
            "exposure": "bmi",
            "covariates": ["age"],
            "mode": "crude",
        })
        comp = next(t for t in result.tables if "逐步调整" in t.title)
        assert len(comp.rows) == 1
        assert "粗模型" in comp.rows[0][0]

    def test_adjusted_only(self, demo_df):
        result = run(demo_df, {
            "outcome": "group",
            "exposure": "bmi",
            "covariates": ["age"],
            "mode": "adjusted",
        })
        comp = next(t for t in result.tables if "逐步调整" in t.title)
        assert len(comp.rows) == 1

    def test_both_mode_produces_model1_and_model3(self, demo_df):
        result = run(demo_df, {
            "outcome": "group",
            "exposure": "bmi",
            "covariates": ["age"],
            "mode": "both",
        })
        comp = next(t for t in result.tables if "逐步调整" in t.title)
        assert len(comp.rows) >= 2


# ── 暴露 OR 变化追踪 ─────────────────────────────────────────────────────────

class TestConfoundingTracking:
    def test_crude_vs_adjusted_or_changes(self, synthetic_df):
        """已知 age 是混杂，调整后 OR 应变化明显。"""
        result = run(synthetic_df, {
            "outcome": "group",
            "exposure": "bmi",
            "covariates": ["age"],
            "mode": "both",
        })
        comp = next(t for t in result.tables if "逐步调整" in t.title)
        # Model 3 的 log(OR) 变化列应包含百分比数字
        pct_str = str(comp.rows[-1][-1])
        assert "%" in pct_str

    def test_confounding_table_row_count(self, demo_df):
        result = run(demo_df, {
            "outcome": "group",
            "exposure": "bmi",
            "covariates": ["age", "sbp_mmhg"],
        })
        conf = next(t for t in result.tables if "混杂" in t.title)
        assert len(conf.rows) == 2

    def test_confounding_table_has_or_values(self, demo_df):
        result = run(demo_df, {
            "outcome": "group",
            "exposure": "bmi",
            "covariates": ["age"],
        })
        conf = next(t for t in result.tables if "混杂" in t.title)
        or_val_str = str(conf.rows[0][1])
        assert or_val_str != "—"
        float(or_val_str)  # 应能转为浮点

    def test_strong_confounder_flagged(self, synthetic_df):
        """age 是强混杂，应有 ⚠ 标记。"""
        result = run(synthetic_df, {
            "outcome": "group",
            "exposure": "bmi",
            "covariates": ["age"],
        })
        conf = next(t for t in result.tables if "混杂" in t.title)
        age_row = next(r for r in conf.rows if r[0] == "age")
        assert "⚠" in str(age_row[5]) or "混杂" in str(age_row[5])


# ── 分类协变量支持 ───────────────────────────────────────────────────────────

class TestCategoricalCovariates:
    def test_categorical_covariate_dummy_encoded(self, demo_df):
        result = run(demo_df, {
            "outcome": "group",
            "exposure": "bmi",
            "covariates": ["age", "sex"],
            "categorical_vars": ["sex"],
            "ref_categories": {"sex": "0"},
        })
        assert result.method == "logistic_reg_adjusted"
        coef = next(t for t in result.tables if "系数" in t.title)
        # dummy 编码后应有 sex_ 开头的行
        coef_names = [str(r[0]) for r in coef.rows]
        assert any("sex" in n for n in coef_names)

    def test_categorical_covariate_in_confounding_table(self, demo_df):
        result = run(demo_df, {
            "outcome": "group",
            "exposure": "bmi",
            "covariates": ["age", "sex"],
            "categorical_vars": ["sex"],
        })
        conf = next(t for t in result.tables if "混杂" in t.title)
        # 混杂评估表按原始变量名显示
        cov_names = [str(r[0]) for r in conf.rows]
        assert "sex" in cov_names


# ── 系数表 ───────────────────────────────────────────────────────────────────

class TestCoefficientTable:
    def test_exposure_marked_with_star(self, demo_df):
        result = run(demo_df, {
            "outcome": "group",
            "exposure": "bmi",
            "covariates": ["age"],
        })
        coef = next(t for t in result.tables if "系数" in t.title)
        first_col = [str(r[0]) for r in coef.rows]
        assert any("★" in c and "bmi" in c for c in first_col)

    def test_coef_table_has_or_column(self, demo_df):
        result = run(demo_df, {
            "outcome": "group",
            "exposure": "bmi",
            "covariates": ["age"],
        })
        coef = next(t for t in result.tables if "系数" in t.title)
        assert "OR" in coef.headers

    def test_coef_table_or_positive(self, demo_df):
        result = run(demo_df, {
            "outcome": "group",
            "exposure": "bmi",
            "covariates": ["age"],
        })
        coef = next(t for t in result.tables if "系数" in t.title)
        for row in coef.rows:
            or_str = str(row[5])
            if or_str not in ("—", ""):
                or_val = float(or_str)
                assert or_val > 0


# ── AUC 性能表 ───────────────────────────────────────────────────────────────

class TestAucTable:
    def test_auc_table_exists(self, demo_df):
        result = run(demo_df, {
            "outcome": "group",
            "exposure": "bmi",
            "covariates": ["age"],
        })
        auc_tables = [t for t in result.tables if "AUC" in t.title]
        assert len(auc_tables) == 1

    def test_auc_values_in_range(self, demo_df):
        result = run(demo_df, {
            "outcome": "group",
            "exposure": "bmi",
            "covariates": ["age"],
            "mode": "both",
        })
        auc_t = next(t for t in result.tables if "AUC" in t.title)
        for row in auc_t.rows:
            auc_str = str(row[1])
            if auc_str != "—":
                v = float(auc_str)
                assert 0 <= v <= 1


# ── 分层分析 ─────────────────────────────────────────────────────────────────

class TestStratifiedAnalysis:
    def test_stratified_table_exists(self, demo_df):
        result = run(demo_df, {
            "outcome": "group",
            "exposure": "bmi",
            "covariates": ["age"],
            "stratify_var": "sex",
        })
        strat = [t for t in result.tables if "分层" in t.title]
        assert len(strat) == 1

    def test_stratified_rows_contain_stratum_values(self, demo_df):
        result = run(demo_df, {
            "outcome": "group",
            "exposure": "bmi",
            "covariates": ["age"],
            "stratify_var": "sex",
        })
        strat = next(t for t in result.tables if "分层" in t.title)
        row_names = [str(r[0]) for r in strat.rows]
        assert any("sex" in rn for rn in row_names)

    def test_stratified_forest_chart_exists(self, demo_df):
        result = run(demo_df, {
            "outcome": "group",
            "exposure": "bmi",
            "covariates": ["age"],
            "stratify_var": "sex",
        })
        forest_charts = [c for c in result.charts if c.chart_type == "forest_plot"]
        assert len(forest_charts) >= 2

    def test_breslow_day_warning_added(self, demo_df):
        result = run(demo_df, {
            "outcome": "group",
            "exposure": "bmi",
            "covariates": ["age"],
            "stratify_var": "sex",
        })
        assert any("Breslow-Day" in w for w in result.warnings)

    def test_invalid_stratify_var_warned(self, demo_df):
        result = run(demo_df, {
            "outcome": "group",
            "exposure": "bmi",
            "covariates": ["age"],
            "stratify_var": "nonexistent_col",
        })
        assert any("nonexistent_col" in w for w in result.warnings)
        strat = [t for t in result.tables if "分层" in t.title]
        assert len(strat) == 0


# ── 交互项检验 ───────────────────────────────────────────────────────────────

class TestInteractionTest:
    def test_interaction_table_exists(self, demo_df):
        result = run(demo_df, {
            "outcome": "group",
            "exposure": "bmi",
            "covariates": ["age", "sbp_mmhg"],
            "interaction_var": "age",
        })
        int_tables = [t for t in result.tables if "交互" in t.title]
        assert len(int_tables) == 1

    def test_interaction_table_has_one_row(self, demo_df):
        result = run(demo_df, {
            "outcome": "group",
            "exposure": "bmi",
            "covariates": ["age"],
            "interaction_var": "age",
        })
        int_t = next(t for t in result.tables if "交互" in t.title)
        assert len(int_t.rows) == 1

    def test_interaction_term_label_contains_both_vars(self, demo_df):
        result = run(demo_df, {
            "outcome": "group",
            "exposure": "bmi",
            "covariates": ["age"],
            "interaction_var": "age",
        })
        int_t = next(t for t in result.tables if "交互" in t.title)
        term = str(int_t.rows[0][0])
        assert "bmi" in term and "age" in term

    def test_interaction_warning_added(self, demo_df):
        result = run(demo_df, {
            "outcome": "group",
            "exposure": "bmi",
            "covariates": ["age"],
            "interaction_var": "age",
        })
        assert any("交互" in w for w in result.warnings)

    def test_interaction_or_positive(self, demo_df):
        result = run(demo_df, {
            "outcome": "group",
            "exposure": "bmi",
            "covariates": ["age"],
            "interaction_var": "age",
        })
        int_t = next(t for t in result.tables if "交互" in t.title)
        or_val = float(int_t.rows[0][1])
        assert or_val > 0


# ── 图表 ─────────────────────────────────────────────────────────────────────

class TestCharts:
    def test_or_forest_plot_exists(self, demo_df):
        result = run(demo_df, {
            "outcome": "group",
            "exposure": "bmi",
            "covariates": ["age"],
        })
        forest = [c for c in result.charts if c.chart_type == "forest_plot"]
        assert len(forest) >= 1

    def test_forest_plot_null_line_is_one(self, demo_df):
        result = run(demo_df, {
            "outcome": "group",
            "exposure": "bmi",
            "covariates": ["age"],
            "mode": "both",
        })
        forest = next(c for c in result.charts if c.chart_type == "forest_plot")
        assert forest.option["nullLine"] == 1.0

    def test_forest_data_has_or_values(self, demo_df):
        result = run(demo_df, {
            "outcome": "group",
            "exposure": "bmi",
            "covariates": ["age"],
            "mode": "both",
        })
        forest = next(c for c in result.charts if c.chart_type == "forest_plot")
        fd = forest.option["forestData"]
        assert isinstance(fd, list)
        for pt in fd:
            assert "label" in pt and "beta" in pt

    def test_auc_bar_chart_exists(self, demo_df):
        result = run(demo_df, {
            "outcome": "group",
            "exposure": "bmi",
            "covariates": ["age"],
        })
        bar_charts = [c for c in result.charts if c.chart_type == "bar"]
        assert len(bar_charts) >= 1

    def test_covariate_bar_chart_exists(self, demo_df):
        result = run(demo_df, {
            "outcome": "group",
            "exposure": "bmi",
            "covariates": ["age", "sbp_mmhg"],
        })
        bar_charts = [c for c in result.charts if c.chart_type == "bar"]
        assert len(bar_charts) >= 2  # AUC + confounding


# ── 全场景测试（demo_basic.csv + 所有功能） ──────────────────────────────────

class TestFullScenario:
    def test_full_run_group_bmi(self, demo_df):
        """因变量 group，暴露 bmi，协变量 age/sbp_mmhg/sex，
        分层 sex，交互 age。"""
        result = run(demo_df, {
            "outcome": "group",
            "exposure": "bmi",
            "covariates": ["age", "sbp_mmhg", "sex"],
            "mode": "both",
            "stratify_var": "sex",
            "interaction_var": "age",
        })
        assert result.method == "logistic_reg_adjusted"
        # 应有：对比表、系数表、混杂表、AUC表、分层表、交互表 = 6张
        assert len(result.tables) == 6
        # 应有：OR森林图、分层森林图、AUC条形图、协变量条形图 = 4张
        assert len(result.charts) == 4
        assert "group" in result.summary
        assert "bmi" in result.summary

    def test_summary_contains_crude_and_adjusted_or(self, demo_df):
        result = run(demo_df, {
            "outcome": "group",
            "exposure": "bmi",
            "covariates": ["age"],
            "mode": "both",
        })
        assert "粗 OR" in result.summary
        assert "全调整后 OR" in result.summary
        assert "AUC" in result.summary

    def test_custom_model2_covariates(self, demo_df):
        result = run(demo_df, {
            "outcome": "group",
            "exposure": "bmi",
            "covariates": ["age", "sbp_mmhg"],
            "model2_covariates": ["age"],
            "mode": "both",
        })
        comp = next(t for t in result.tables if "逐步调整" in t.title)
        # Model 1 + Model 2 (age) + Model 3 (age+sbp)
        assert len(comp.rows) == 3


# ── 错误处理 ─────────────────────────────────────────────────────────────────

class TestErrors:
    def test_missing_outcome(self, demo_df):
        with pytest.raises(ValueError, match="outcome"):
            run(demo_df, {"exposure": "bmi"})

    def test_missing_exposure(self, demo_df):
        with pytest.raises(ValueError, match="暴露变量"):
            run(demo_df, {"outcome": "group"})

    def test_nonexistent_outcome(self, demo_df):
        with pytest.raises(ValueError, match="不存在"):
            run(demo_df, {"outcome": "no_col", "exposure": "bmi"})

    def test_nonexistent_exposure(self, demo_df):
        with pytest.raises(ValueError, match="不存在"):
            run(demo_df, {"outcome": "group", "exposure": "no_col"})

    def test_non_numeric_exposure(self, demo_df):
        df = demo_df.copy()
        df["label"] = df["sex"].map({0: "female", 1: "male"})
        with pytest.raises(ValueError, match="数值型"):
            run(df, {"outcome": "group", "exposure": "label"})

    def test_exposure_equals_outcome(self, demo_df):
        with pytest.raises(ValueError, match="相同"):
            run(demo_df, {"outcome": "group", "exposure": "group"})

    def test_invalid_mode(self, demo_df):
        with pytest.raises(ValueError, match="mode"):
            run(demo_df, {"outcome": "group", "exposure": "bmi", "mode": "invalid"})

    def test_nonexistent_covariate_warned(self, demo_df):
        result = run(demo_df, {
            "outcome": "group",
            "exposure": "bmi",
            "covariates": ["age", "ghost_col"],
        })
        assert any("ghost_col" in w for w in result.warnings)
