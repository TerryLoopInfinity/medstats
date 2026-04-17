"""pytest tests for stats/correlation.py and stats/linear_reg.py.

Correlation: age, sbp_mmhg, dbp_mmhg, bmi (demo_basic.csv)
Linear regression: outcome=sbp_mmhg, predictors=age, bmi, heart_rate
"""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from app.stats.correlation import run as corr_run
from app.stats.linear_reg import run as linreg_run

DEMO_CSV = Path(__file__).parents[2] / "data" / "examples" / "demo_basic.csv"


# ── fixtures ───────────────────────────────────────────────────────────────────


@pytest.fixture
def demo_df() -> pd.DataFrame:
    return pd.read_csv(DEMO_CSV)


@pytest.fixture
def normal_df() -> pd.DataFrame:
    rng = np.random.default_rng(1)
    x = rng.normal(50, 10, 100)
    y = 2.0 * x + rng.normal(0, 5, 100)
    return pd.DataFrame({"x": x, "y": y, "z": rng.normal(0, 1, 100)})


@pytest.fixture
def perfect_reg_df() -> pd.DataFrame:
    rng = np.random.default_rng(7)
    x1 = rng.normal(0, 1, 80)
    x2 = rng.normal(0, 1, 80)
    y = 3.0 + 2.0 * x1 - 1.5 * x2 + rng.normal(0, 0.5, 80)
    return pd.DataFrame({"y": y, "x1": x1, "x2": x2})


# ============================================================================
# correlation.py tests
# ============================================================================


class TestCorrelationBasic:
    def test_returns_analysis_result(self, demo_df):
        result = corr_run(demo_df, {"variables": ["age", "sbp_mmhg", "dbp_mmhg", "bmi"]})
        assert result.method == "correlation"
        assert len(result.tables) == 2
        assert result.summary

    def test_matrix_table_dimensions(self, demo_df):
        result = corr_run(demo_df, {"variables": ["age", "sbp_mmhg", "dbp_mmhg", "bmi"]})
        t = result.tables[0]
        assert len(t.headers) == 5  # variable name col + 4 variable cols
        assert len(t.rows) == 4

    def test_diagonal_is_one(self, demo_df):
        result = corr_run(demo_df, {"variables": ["age", "sbp_mmhg", "dbp_mmhg", "bmi"]})
        for i, row in enumerate(result.tables[0].rows):
            assert str(row[i + 1]) == "1.000"

    def test_pair_table_six_rows(self, demo_df):
        # C(4,2) = 6 pairs
        result = corr_run(demo_df, {"variables": ["age", "sbp_mmhg", "dbp_mmhg", "bmi"]})
        assert len(result.tables[1].rows) == 6

    def test_pair_table_headers(self, demo_df):
        result = corr_run(demo_df, {"variables": ["age", "sbp_mmhg", "dbp_mmhg", "bmi"]})
        h = result.tables[1].headers
        assert "p 值" in h
        assert "95% CI" in h
        assert "样本量 n" in h

    def test_heatmap_chart_exists(self, demo_df):
        result = corr_run(demo_df, {"variables": ["age", "sbp_mmhg", "dbp_mmhg", "bmi"]})
        heatmaps = [c for c in result.charts if c.chart_type == "heatmap"]
        assert len(heatmaps) == 1
        assert "visualMap" in heatmaps[0].option

    def test_scatter_charts_for_four_vars(self, demo_df):
        # 4 vars <= 5, should produce C(4,2)=6 scatter plots
        result = corr_run(demo_df, {"variables": ["age", "sbp_mmhg", "dbp_mmhg", "bmi"]})
        scatter_charts = [c for c in result.charts if c.chart_type == "scatter"]
        assert len(scatter_charts) == 6

    def test_no_scatter_for_six_vars(self, demo_df):
        result = corr_run(demo_df, {
            "variables": [
                "age", "sbp_mmhg", "dbp_mmhg", "bmi",
                "fasting_glucose_mmol", "total_cholesterol_mmol",
            ],
        })
        scatter_charts = [c for c in result.charts if c.chart_type == "scatter"]
        assert len(scatter_charts) == 0

    def test_significance_marks(self, normal_df):
        result = corr_run(normal_df, {"variables": ["x", "y"], "method": "pearson"})
        sig_col = result.tables[1].rows[0][6]
        assert sig_col in ("*", "**", "***")

    def test_ci_format_pearson(self, demo_df):
        result = corr_run(demo_df, {"variables": ["age", "sbp_mmhg"], "method": "pearson"})
        ci = str(result.tables[1].rows[0][4])
        assert ci.startswith("[") and "," in ci

    def test_heatmap_data_count(self, demo_df):
        result = corr_run(demo_df, {"variables": ["age", "sbp_mmhg", "dbp_mmhg", "bmi"]})
        heatmap = next(c for c in result.charts if c.chart_type == "heatmap")
        data = heatmap.option["series"][0]["data"]
        assert len(data) == 16  # 4x4

    def test_matrix_symmetry(self, demo_df):
        result = corr_run(demo_df, {"variables": ["age", "sbp_mmhg", "bmi"]})
        rows = result.tables[0].rows
        # rows[0][2] = r(age, sbp), rows[1][1] = r(sbp, age)
        assert rows[0][2] == rows[1][1]


class TestCorrelationMethods:
    def test_pearson(self, normal_df):
        result = corr_run(normal_df, {"variables": ["x", "y"], "method": "pearson"})
        assert "Pearson" in result.tables[0].title

    def test_spearman(self, normal_df):
        result = corr_run(normal_df, {"variables": ["x", "y"], "method": "spearman"})
        assert "Spearman" in result.tables[0].title

    def test_kendall_no_ci(self, normal_df):
        result = corr_run(normal_df, {"variables": ["x", "y"], "method": "kendall"})
        assert "Kendall" in result.tables[0].title
        # Kendall CI is not implemented, should show dash
        assert result.tables[1].rows[0][4] == "—"

    def test_auto_runs_without_error(self, normal_df):
        result = corr_run(normal_df, {"variables": ["x", "y"], "method": "auto"})
        assert result.method == "correlation"


class TestCorrelationErrors:
    def test_too_few_variables(self, demo_df):
        with pytest.raises(ValueError, match="至少需要"):
            corr_run(demo_df, {"variables": ["age"]})

    def test_empty_variables(self, demo_df):
        with pytest.raises(ValueError, match="至少需要"):
            corr_run(demo_df, {})

    def test_invalid_method(self, demo_df):
        with pytest.raises(ValueError, match="method"):
            corr_run(demo_df, {"variables": ["age", "sbp_mmhg"], "method": "invalid"})

    def test_nonexistent_var_warning(self, demo_df):
        result = corr_run(demo_df, {"variables": ["age", "sbp_mmhg", "ghost"]})
        assert any("ghost" in w for w in result.warnings)
        assert len(result.tables[1].rows) == 1  # only 1 valid pair


# ============================================================================
# linear_reg.py tests
# ============================================================================


class TestLinearRegBasic:
    def test_returns_analysis_result(self, demo_df):
        result = linreg_run(demo_df, {
            "outcome": "sbp_mmhg",
            "predictors": ["age", "bmi", "heart_rate"],
        })
        assert result.method == "linear_reg"
        assert len(result.tables) >= 2
        assert result.summary

    def test_coef_table_row_count(self, demo_df):
        result = linreg_run(demo_df, {
            "outcome": "sbp_mmhg",
            "predictors": ["age", "bmi", "heart_rate"],
            "mode": "multivariate",
        })
        coef_table = next(t for t in result.tables if "回归系数" in t.title)
        # intercept + 3 predictors
        assert len(coef_table.rows) == 4

    def test_coef_table_headers(self, demo_df):
        result = linreg_run(demo_df, {
            "outcome": "sbp_mmhg",
            "predictors": ["age", "bmi", "heart_rate"],
            "mode": "multivariate",
        })
        coef_table = next(t for t in result.tables if "回归系数" in t.title)
        h = coef_table.headers
        assert "β" in h
        assert "p 值" in h
        assert "标准化 β" in h

    def test_fit_metrics_present(self, demo_df):
        result = linreg_run(demo_df, {
            "outcome": "sbp_mmhg",
            "predictors": ["age", "bmi", "heart_rate"],
            "mode": "multivariate",
        })
        fit_table = next(t for t in result.tables if "拟合" in t.title)
        metric_names = [r[0] for r in fit_table.rows]
        for m in ["R²", "调整 R²", "AIC", "BIC"]:
            assert m in metric_names

    def test_r_squared_range(self, demo_df):
        result = linreg_run(demo_df, {
            "outcome": "sbp_mmhg",
            "predictors": ["age", "bmi", "heart_rate"],
            "mode": "multivariate",
        })
        fit_table = next(t for t in result.tables if "拟合" in t.title)
        r2 = float(next(r[1] for r in fit_table.rows if r[0] == "R²"))
        assert 0.0 <= r2 <= 1.0

    def test_residual_diagnostics_rows(self, demo_df):
        result = linreg_run(demo_df, {
            "outcome": "sbp_mmhg",
            "predictors": ["age", "bmi", "heart_rate"],
            "mode": "multivariate",
        })
        diag_table = next(t for t in result.tables if "残差" in t.title)
        names = [r[0] for r in diag_table.rows]
        assert any("正态性" in n for n in names)
        assert any("Breusch-Pagan" in n for n in names)
        assert any("Durbin-Watson" in n for n in names)

    def test_vif_table_three_rows(self, demo_df):
        result = linreg_run(demo_df, {
            "outcome": "sbp_mmhg",
            "predictors": ["age", "bmi", "heart_rate"],
            "mode": "multivariate",
        })
        vif_table = next(t for t in result.tables if "VIF" in t.title)
        assert len(vif_table.rows) == 3

    def test_three_charts(self, demo_df):
        result = linreg_run(demo_df, {
            "outcome": "sbp_mmhg",
            "predictors": ["age", "bmi", "heart_rate"],
            "mode": "multivariate",
        })
        assert len(result.charts) == 3
        titles = [c.title for c in result.charts]
        assert any("实际" in t for t in titles)
        assert any("残差" in t for t in titles)
        assert any("QQ" in t for t in titles)

    def test_chart_option_has_required_keys(self, demo_df):
        result = linreg_run(demo_df, {
            "outcome": "sbp_mmhg",
            "predictors": ["age", "bmi"],
            "mode": "multivariate",
        })
        for chart in result.charts:
            assert "series" in chart.option
            assert "xAxis" in chart.option
            assert "yAxis" in chart.option


class TestLinearRegUnivariate:
    def test_univariate_row_count(self, demo_df):
        result = linreg_run(demo_df, {
            "outcome": "sbp_mmhg",
            "predictors": ["age", "bmi", "heart_rate"],
            "mode": "univariate",
        })
        uni_table = next(t for t in result.tables if "单变量" in t.title)
        assert len(uni_table.rows) == 3

    def test_univariate_headers(self, demo_df):
        result = linreg_run(demo_df, {
            "outcome": "sbp_mmhg",
            "predictors": ["age"],
            "mode": "univariate",
        })
        h = result.tables[0].headers
        assert "β" in h
        assert "SE" in h
        assert "p 值" in h
        assert "R²" in h

    def test_both_mode_has_both_tables(self, demo_df):
        result = linreg_run(demo_df, {
            "outcome": "sbp_mmhg",
            "predictors": ["age", "bmi", "heart_rate"],
            "mode": "both",
        })
        titles = [t.title for t in result.tables]
        assert any("单变量" in t for t in titles)
        assert any("多变量" in t for t in titles)


class TestLinearRegCoefficients:
    def test_slope_direction(self, perfect_reg_df):
        result = linreg_run(perfect_reg_df, {
            "outcome": "y",
            "predictors": ["x1", "x2"],
            "mode": "multivariate",
        })
        coef_table = next(t for t in result.tables if "回归系数" in t.title)
        rows_dict = {r[0]: r for r in coef_table.rows}
        beta_x1 = float(rows_dict["x1"][1])
        beta_x2 = float(rows_dict["x2"][1])
        assert 1.5 < beta_x1 < 2.5
        assert -2.0 < beta_x2 < -1.0

    def test_intercept_no_std_beta(self, perfect_reg_df):
        result = linreg_run(perfect_reg_df, {
            "outcome": "y",
            "predictors": ["x1", "x2"],
            "mode": "multivariate",
        })
        coef_table = next(t for t in result.tables if "回归系数" in t.title)
        intercept_row = next(r for r in coef_table.rows if "Intercept" in r[0])
        assert intercept_row[6] == "—"


class TestLinearRegErrors:
    def test_no_outcome(self, demo_df):
        with pytest.raises(ValueError, match="outcome"):
            linreg_run(demo_df, {"predictors": ["age"]})

    def test_nonexistent_outcome(self, demo_df):
        with pytest.raises(ValueError, match="不存在"):
            linreg_run(demo_df, {"outcome": "no_such_col", "predictors": ["age"]})

    def test_no_predictors(self, demo_df):
        with pytest.raises(ValueError, match="predictors"):
            linreg_run(demo_df, {"outcome": "sbp_mmhg"})

    def test_outcome_same_as_predictor_warned(self, demo_df):
        result = linreg_run(demo_df, {
            "outcome": "sbp_mmhg",
            "predictors": ["sbp_mmhg", "age"],
        })
        assert any("sbp_mmhg" in w for w in result.warnings)

    def test_nonexistent_predictor_warned(self, demo_df):
        result = linreg_run(demo_df, {
            "outcome": "sbp_mmhg",
            "predictors": ["age", "ghost_var"],
        })
        assert any("ghost_var" in w for w in result.warnings)

    def test_invalid_mode(self, demo_df):
        with pytest.raises(ValueError, match="mode"):
            linreg_run(demo_df, {
                "outcome": "sbp_mmhg",
                "predictors": ["age"],
                "mode": "bad_mode",
            })


class TestLinearRegDemoCsv:
    def test_full_run_sbp(self, demo_df):
        result = linreg_run(demo_df, {
            "outcome": "sbp_mmhg",
            "predictors": ["age", "bmi", "heart_rate"],
        })
        assert result.method == "linear_reg"
        assert len(result.charts) == 3
        assert "sbp_mmhg" in result.summary

    def test_no_vif_for_single_predictor(self, demo_df):
        result = linreg_run(demo_df, {
            "outcome": "sbp_mmhg",
            "predictors": ["age"],
            "mode": "multivariate",
        })
        vif_tables = [t for t in result.tables if "VIF" in t.title]
        assert len(vif_tables) == 0
