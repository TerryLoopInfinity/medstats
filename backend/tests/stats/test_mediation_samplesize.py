"""Tests for mediation analysis and sample size calculation modules."""
import numpy as np
import pandas as pd
import pytest

from app.stats import mediation, sample_size


# ─────────────────────────────────────────────────────────────────────────────
# 工具函数
# ─────────────────────────────────────────────────────────────────────────────

def _make_mediation_data(n: int = 300, seed: int = 42) -> pd.DataFrame:
    """
    构造部分中介数据集:
    - X → M (a ≈ 0.5)
    - M → Y (b ≈ 0.4)
    - X → Y 直接效应 (c' ≈ 0.2)
    - 间接效应 ≈ 0.5 × 0.4 = 0.2，中介比例约 50%
    """
    rng = np.random.default_rng(seed)
    X = rng.standard_normal(n)
    M = 0.5 * X + rng.standard_normal(n)
    Y = 0.2 * X + 0.4 * M + rng.standard_normal(n)
    return pd.DataFrame({"X": X, "M": M, "Y": Y})


# ─────────────────────────────────────────────────────────────────────────────
# 中介分析测试
# ─────────────────────────────────────────────────────────────────────────────

class TestMediation:
    def test_basic_continuous(self):
        """连续型中介分析：间接效应显著，中介比例在合理范围。"""
        df = _make_mediation_data()
        params = {
            "exposure": "X",
            "mediator": "M",
            "outcome": "Y",
            "outcome_type": "continuous",
            "mediator_type": "continuous",
            "n_bootstrap": 500,  # 测试时用小 bootstrap 数
        }
        result = mediation.run(df, params)

        assert result.method == "mediation"
        assert len(result.tables) == 3
        assert len(result.charts) == 3

        # 路径系数表检查
        path_table = result.tables[0]
        assert path_table.title == "Baron & Kenny 路径系数"
        assert len(path_table.rows) == 4  # c, a, b, c'

        # 路径 a 应显著（约 0.5）
        a_row = next(r for r in path_table.rows if r[0] == "a")
        a_coef = float(a_row[2])
        assert 0.3 < a_coef < 0.7, f"路径 a 系数异常：{a_coef}"
        assert a_row[5] != "ns", "路径 a 应显著"

        # 路径 b 应显著（约 0.4）
        b_row = next(r for r in path_table.rows if r[0] == "b")
        b_coef = float(b_row[2])
        assert 0.2 < b_coef < 0.6, f"路径 b 系数异常：{b_coef}"
        assert b_row[5] != "ns", "路径 b 应显著"

    def test_indirect_effect_significant(self):
        """Bootstrap 置信区间应不包含 0（间接效应显著）。"""
        df = _make_mediation_data()
        params = {
            "exposure": "X",
            "mediator": "M",
            "outcome": "Y",
            "outcome_type": "continuous",
            "mediator_type": "continuous",
            "n_bootstrap": 500,
        }
        result = mediation.run(df, params)

        effect_table = result.tables[1]
        indirect_row = next(r for r in effect_table.rows if "间接效应" in str(r[0]))
        bc_ci_str = str(indirect_row[3])  # BC CI 列
        assert bc_ci_str != "—", "间接效应应有 BC CI"

        # 解析 CI
        ci_clean = bc_ci_str.strip("[]").split(",")
        lo, hi = float(ci_clean[0]), float(ci_clean[1])
        assert lo > 0 or hi < 0, f"间接效应 Bootstrap CI 应不包含 0：[{lo}, {hi}]"

    def test_mediation_proportion(self):
        """中介比例应在 20%~80% 之间（构造数据约 50%）。"""
        df = _make_mediation_data(n=500)
        params = {
            "exposure": "X",
            "mediator": "M",
            "outcome": "Y",
            "outcome_type": "continuous",
            "mediator_type": "continuous",
            "n_bootstrap": 200,
        }
        result = mediation.run(df, params)
        effect_table = result.tables[1]
        pct_row = next(r for r in effect_table.rows if "中介比例" in str(r[0]))
        pct_str = str(pct_row[1]).replace("%", "").strip()
        if pct_str != "N/A":
            pct = float(pct_str)
            assert 20.0 <= pct <= 80.0, f"中介比例不合理：{pct}%"

    def test_charts_structure(self):
        """图表数据结构检查：路径图、直方图、饼图。"""
        df = _make_mediation_data(n=200)
        params = {
            "exposure": "X",
            "mediator": "M",
            "outcome": "Y",
            "outcome_type": "continuous",
            "mediator_type": "continuous",
            "n_bootstrap": 200,
        }
        result = mediation.run(df, params)
        chart_types = [c.chart_type for c in result.charts]
        assert "mediation_path" in chart_types
        assert "bar" in chart_types
        assert "pie" in chart_types

        # 路径图数据完整性
        path_chart = next(c for c in result.charts if c.chart_type == "mediation_path")
        opt = path_chart.option
        assert "nodes" in opt
        assert "paths" in opt
        assert set(opt["paths"].keys()) == {"a", "b", "c", "c_prime"}

    def test_missing_param_raises(self):
        df = _make_mediation_data()
        with pytest.raises(ValueError, match="暴露变量"):
            mediation.run(df, {"outcome": "Y", "mediator": "M"})

    def test_same_exposure_mediator_raises(self):
        df = _make_mediation_data()
        with pytest.raises(ValueError, match="不能相同"):
            mediation.run(df, {"exposure": "X", "mediator": "X", "outcome": "Y"})

    def test_with_covariates(self):
        """含协变量时分析不崩溃，路径系数仍合理。"""
        rng = np.random.default_rng(0)
        n = 250
        C = rng.standard_normal(n)
        X = rng.standard_normal(n)
        M = 0.5 * X + 0.2 * C + rng.standard_normal(n)
        Y = 0.2 * X + 0.4 * M + 0.3 * C + rng.standard_normal(n)
        df = pd.DataFrame({"X": X, "M": M, "Y": Y, "C": C})

        result = mediation.run(df, {
            "exposure": "X",
            "mediator": "M",
            "outcome": "Y",
            "covariates": ["C"],
            "outcome_type": "continuous",
            "mediator_type": "continuous",
            "n_bootstrap": 200,
        })
        assert result.method == "mediation"
        assert len(result.tables) == 3


# ─────────────────────────────────────────────────────────────────────────────
# 样本量计算测试
# ─────────────────────────────────────────────────────────────────────────────

class TestSampleSize:
    def test_two_means_classic(self):
        """经典两组均值比较：delta=5, sd=10 → 约每组 64 人。"""
        result = sample_size.run({
            "calc_type": "two_means",
            "mean_diff": 5,
            "sd": 10,
            "alpha": 0.05,
            "power": 0.80,
            "sides": 2,
            "solve_for": "sample_size",
        })
        res_table = result.tables[0]
        n_row = next(r for r in res_table.rows if "每组" in str(r[0]))
        n = int(n_row[1])
        assert 60 <= n <= 70, f"每组样本量应约 64，实际 {n}"

    def test_two_proportions(self):
        """两组率比较：p1=0.3, p2=0.5 → 结果应合理。"""
        result = sample_size.run({
            "calc_type": "two_proportions",
            "p1": 0.3,
            "p2": 0.5,
            "alpha": 0.05,
            "power": 0.80,
            "sides": 2,
            "solve_for": "sample_size",
        })
        res_table = result.tables[0]
        n_row = next(r for r in res_table.rows if "每组" in str(r[0]))
        n = int(n_row[1])
        assert 50 <= n <= 150, f"每组样本量应在合理范围，实际 {n}"

    def test_cox_schoenfeld(self):
        """Cox/Schoenfeld 公式：HR=1.5, event_rate=0.3。"""
        result = sample_size.run({
            "calc_type": "cox",
            "hr": 1.5,
            "event_rate": 0.3,
            "r2": 0.0,
            "alpha": 0.05,
            "power": 0.80,
            "sides": 2,
            "solve_for": "sample_size",
        })
        assert result.method == "sample_size"
        res_table = result.tables[0]
        n_row = next(r for r in res_table.rows if "每组" in str(r[0]))
        n = int(n_row[1])
        assert n > 0, "样本量应大于 0"

    def test_reverse_power_calculation(self):
        """反向计算：给定 n=100，算 power，结果应在 (0, 1)。"""
        result = sample_size.run({
            "calc_type": "two_means",
            "mean_diff": 5,
            "sd": 10,
            "alpha": 0.05,
            "power": 0.80,
            "sides": 2,
            "solve_for": "power",
            "n": 100,
        })
        res_table = result.tables[0]
        pw_row = next(r for r in res_table.rows if "Power" in str(r[0]) and "实际" in str(r[0]))
        pw_str = str(pw_row[1]).split("（")[0]
        pw = float(pw_str)
        assert 0 < pw < 1, f"Power 应在 (0, 1)，实际 {pw}"

    def test_power_curve_monotone(self):
        """功效曲线应单调递增（在合理 n 范围内）。"""
        result = sample_size.run({
            "calc_type": "two_means",
            "mean_diff": 5,
            "sd": 10,
            "alpha": 0.05,
            "power": 0.80,
            "sides": 2,
            "solve_for": "sample_size",
        })
        assert len(result.charts) == 1
        chart = result.charts[0]
        assert chart.chart_type == "line"

        series_data = chart.option["series"][0]["data"]  # type: ignore[index]
        assert len(series_data) >= 10, "功效曲线数据点应足够多"

        powers = [float(pt[1]) for pt in series_data]
        # 检查整体单调趋势（允许微小波动，但不应大幅回落）
        assert powers[-1] > powers[0], "功效曲线应整体递增"
        # 检验约 80% 的相邻点满足非递减
        nondecreasing = sum(1 for i in range(1, len(powers)) if powers[i] >= powers[i - 1] - 0.01)
        assert nondecreasing / (len(powers) - 1) >= 0.85

    def test_correlation(self):
        """相关系数检验：r=0.5，双侧，alpha=0.05，power=0.80。"""
        result = sample_size.run({
            "calc_type": "correlation",
            "r": 0.5,
            "alpha": 0.05,
            "power": 0.80,
            "sides": 2,
            "solve_for": "sample_size",
        })
        res_table = result.tables[0]
        n_row = next(r for r in res_table.rows if "每组" in str(r[0]))
        n = int(n_row[1])
        assert 20 <= n <= 50, f"样本量约 29，实际 {n}"

    def test_logistic(self):
        """Logistic 回归样本量：p0=0.3, OR=2.0。"""
        result = sample_size.run({
            "calc_type": "logistic",
            "p0": 0.3,
            "or_value": 2.0,
            "r2": 0.0,
            "alpha": 0.05,
            "power": 0.80,
            "sides": 2,
            "solve_for": "sample_size",
        })
        assert result.method == "sample_size"
        assert len(result.tables) == 3
        assert len(result.charts) == 1

    def test_invalid_calc_type(self):
        with pytest.raises(ValueError, match="不支持"):
            sample_size.run({"calc_type": "unknown_method"})

    def test_invalid_alpha(self):
        with pytest.raises(ValueError, match="alpha"):
            sample_size.run({
                "calc_type": "two_means",
                "mean_diff": 5,
                "sd": 10,
                "alpha": 1.5,
            })
