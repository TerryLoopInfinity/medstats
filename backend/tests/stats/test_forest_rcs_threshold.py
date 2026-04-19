"""Tests for forest_plot, rcs, threshold modules."""
import pandas as pd
import pytest
from pathlib import Path

DATA_PATH = Path(__file__).parent.parent.parent / "data" / "examples" / "demo_basic.csv"


@pytest.fixture
def df():
    return pd.read_csv(DATA_PATH)


# ─── forest_plot tests ───────────────────────────────────────────────

class TestForestPlot:
    def test_logistic_basic(self, df):
        """Forest plot with logistic model, sex subgroup."""
        from app.stats import forest_plot
        result = forest_plot.run(df, {
            "model_type": "logistic",
            "outcome": "group",
            "exposure": "bmi",
            "covariates": ["age"],
            "subgroup_vars": ["sex"],
            "categorical_vars": ["sex"],
        })
        # Should have tables
        assert len(result.tables) >= 1
        # Overall effect table
        overall_table = next((t for t in result.tables if "整体" in t.title), None)
        assert overall_table is not None, "Missing overall effect table"
        assert len(overall_table.rows) >= 1
        # Subgroup table
        sg_table = next((t for t in result.tables if "亚组" in t.title), None)
        assert sg_table is not None, "Missing subgroup table"
        # Each subgroup has at least one non-header row
        non_header_rows = [r for r in sg_table.rows if r[1] != "—" and r[1] != ""]
        assert len(non_header_rows) >= 1, "No subgroup-level rows found"
        # P interaction should be present
        p_int_col = sg_table.headers.index("P interaction") if "P interaction" in sg_table.headers else -1
        assert p_int_col >= 0, "P interaction column missing"
        # Chart
        assert len(result.charts) >= 1
        forest_charts = [c for c in result.charts if c.chart_type == "forest_plot"]
        assert len(forest_charts) >= 1
        forest_data = forest_charts[0].option.get("forestData", [])
        assert len(forest_data) >= 1

    def test_linear_median_split_subgroup(self, df):
        """Forest plot with linear model, age median-split subgroup."""
        from app.stats import forest_plot
        result = forest_plot.run(df, {
            "model_type": "linear",
            "outcome": "sbp_mmhg",
            "exposure": "bmi",
            "covariates": ["age"],
            "subgroup_vars": ["age"],  # will be median-split since not in categorical_vars
            "categorical_vars": [],
        })
        assert result is not None
        sg_table = next((t for t in result.tables if "亚组" in t.title), None)
        assert sg_table is not None

    def test_missing_exposure_raises(self, df):
        from app.stats import forest_plot
        with pytest.raises(ValueError, match="nonexistent_col"):
            forest_plot.run(df, {
                "model_type": "logistic",
                "outcome": "group",
                "exposure": "nonexistent_col",
                "covariates": [],
                "subgroup_vars": ["sex"],
                "categorical_vars": [],
            })


# ─── rcs tests ───────────────────────────────────────────────────────

class TestRCS:
    def test_linear_4knots(self, df):
        """RCS with linear model, 4 knots."""
        from app.stats import rcs
        df_clean = df.dropna(subset=["sbp_mmhg", "bmi", "age"])
        result = rcs.run(df_clean, {
            "model_type": "linear",
            "outcome": "sbp_mmhg",
            "exposure": "bmi",
            "covariates": ["age"],
            "n_knots": 4,
        })
        # Non-linearity test table
        nl_table = next((t for t in result.tables if "非线性" in t.title or "检验" in t.title), None)
        assert nl_table is not None, "Missing non-linearity test table"
        # P for non-linearity should be present
        assert any("non-linearity" in str(r).lower() or "非线性" in str(r) for r in nl_table.rows)
        # Charts: at least curve chart
        curve_chart = next((c for c in result.charts if "RCS" in c.title or "曲线" in c.title), None)
        assert curve_chart is not None, "Missing RCS curve chart"
        # Curve chart should have data
        series = curve_chart.option.get("series", [])
        main_series = [s for s in series if s.get("type") == "line"]
        assert len(main_series) >= 1
        data_pts = main_series[0].get("data", [])
        assert len(data_pts) >= 10, f"Too few curve points: {len(data_pts)}"

    def test_3knots_vs_5knots(self, df):
        """Knot count can be changed."""
        from app.stats import rcs
        df_clean = df.dropna(subset=["sbp_mmhg", "bmi", "age"])
        for n in [3, 5]:
            result = rcs.run(df_clean, {
                "model_type": "linear",
                "outcome": "sbp_mmhg",
                "exposure": "bmi",
                "covariates": ["age"],
                "n_knots": n,
            })
            knot_table = next((t for t in result.tables if "节点" in t.title), None)
            assert knot_table is not None
            assert len(knot_table.rows) == n, f"Expected {n} knot rows, got {len(knot_table.rows)}"

    def test_logistic_rcs(self, df):
        """RCS with logistic model."""
        from app.stats import rcs
        df_clean = df.dropna(subset=["group", "bmi", "age"])
        result = rcs.run(df_clean, {
            "model_type": "logistic",
            "outcome": "group",
            "exposure": "bmi",
            "covariates": ["age"],
            "n_knots": 4,
        })
        assert result is not None
        assert len(result.charts) >= 1


# ─── threshold tests ─────────────────────────────────────────────────

class TestThreshold:
    def test_linear_basic(self, df):
        """Threshold analysis with linear model."""
        from app.stats import threshold
        df_clean = df.dropna(subset=["sbp_mmhg", "bmi", "age"])
        result = threshold.run(df_clean, {
            "model_type": "linear",
            "outcome": "sbp_mmhg",
            "exposure": "bmi",
            "covariates": ["age"],
            "n_steps": 20,       # small for speed
            "n_bootstrap": 10,   # small for speed
        })
        # Changepoint table
        cp_table = next((t for t in result.tables if "拐点" in t.title), None)
        assert cp_table is not None, "Missing changepoint table"
        # Extract best changepoint value
        cp_row = next((r for r in cp_table.rows if "最佳拐点" in str(r[0])), None)
        assert cp_row is not None, "Missing best changepoint row"
        cp_val = float(cp_row[1])
        # Changepoint should be within exposure range
        bmi_min = df_clean["bmi"].quantile(0.05)
        bmi_max = df_clean["bmi"].quantile(0.95)
        assert bmi_min <= cp_val <= bmi_max, f"Changepoint {cp_val} out of range [{bmi_min}, {bmi_max}]"
        # Effects table
        effects_table = next((t for t in result.tables if "效应" in t.title and "两侧" in t.title), None)
        assert effects_table is not None, "Missing effects table"
        assert len(effects_table.rows) == 2, "Should have left and right effect rows"
        # Log-likelihood table
        ll_table = next((t for t in result.tables if "对数似然" in t.title or "曲线" in t.title), None)
        assert ll_table is not None, "Missing LL table"
        assert len(ll_table.rows) >= 5, "Too few LL curve data points"
        # Charts
        threshold_chart = next((c for c in result.charts if "阈值" in c.title or "threshold" in c.title.lower()), None)
        assert threshold_chart is not None, "Missing threshold effect chart"
        ll_chart = next((c for c in result.charts if "对数似然" in c.title or "似然" in c.title), None)
        assert ll_chart is not None, "Missing LL curve chart"

    def test_logistic_basic(self, df):
        """Threshold analysis with logistic model."""
        from app.stats import threshold
        df_clean = df.dropna(subset=["group", "bmi", "age"])
        result = threshold.run(df_clean, {
            "model_type": "logistic",
            "outcome": "group",
            "exposure": "bmi",
            "covariates": ["age"],
            "n_steps": 15,
            "n_bootstrap": 5,
        })
        assert result is not None
        assert len(result.tables) >= 1
        assert len(result.charts) >= 1

    def test_changepoint_in_valid_range(self, df):
        """Custom search range."""
        from app.stats import threshold
        df_clean = df.dropna(subset=["sbp_mmhg", "bmi"])
        bmi_p20 = float(df_clean["bmi"].quantile(0.2))
        bmi_p80 = float(df_clean["bmi"].quantile(0.8))
        result = threshold.run(df_clean, {
            "model_type": "linear",
            "outcome": "sbp_mmhg",
            "exposure": "bmi",
            "covariates": [],
            "search_range": [bmi_p20, bmi_p80],
            "n_steps": 10,
            "n_bootstrap": 5,
        })
        cp_table = next((t for t in result.tables if "拐点" in t.title), None)
        assert cp_table is not None
        cp_val = float(next(r[1] for r in cp_table.rows if "最佳拐点" in str(r[0])))
        assert bmi_p20 <= cp_val <= bmi_p80
