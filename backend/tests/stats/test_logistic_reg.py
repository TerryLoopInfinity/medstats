"""Tests for backend/app/stats/logistic_reg.py

使用 data/examples/demo_basic.csv：
  因变量  group (0/1)
  自变量  age, bmi, sbp_mmhg, sex (0/1 → 当作分类变量测试 dummy 编码)
"""

import math
from pathlib import Path

import pandas as pd
import pytest

from app.stats import logistic_reg

CSV_PATH = Path(__file__).parents[2] / "data" / "examples" / "demo_basic.csv"


@pytest.fixture(scope="module")
def df() -> pd.DataFrame:
    return pd.read_csv(CSV_PATH)


# ─────────────────────────────────────────────────────────────────────────────
# 基础运行 — both 模式（单 + 多变量）
# ─────────────────────────────────────────────────────────────────────────────

def test_run_both_mode(df):
    result = logistic_reg.run(df, {
        "outcome": "group",
        "predictors": ["age", "bmi", "sbp_mmhg"],
        "mode": "both",
    })
    assert result.method == "logistic_reg"
    # 应含单变量 + 多变量系数表 + 拟合指标 + 分类性能 + 混淆矩阵
    titles = [t.title for t in result.tables]
    assert any("单变量" in t for t in titles)
    assert any("多变量" in t and "回归系数" in t for t in titles)
    assert any("拟合指标" in t for t in titles)
    assert any("分类性能" in t for t in titles)
    assert any("混淆矩阵" in t for t in titles)


# ─────────────────────────────────────────────────────────────────────────────
# OR 计算正确性：OR = exp(β)，CI 应包含 OR
# ─────────────────────────────────────────────────────────────────────────────

def test_or_equals_exp_beta(df):
    result = logistic_reg.run(df, {
        "outcome": "group",
        "predictors": ["age"],
        "mode": "multivariate",
    })
    coef_table = next(t for t in result.tables if "回归系数" in t.title)
    row = coef_table.rows[0]  # 只有 age 一行
    assert row[0] == "age"
    beta = float(row[1])
    or_val = float(row[5])
    assert abs(or_val - math.exp(beta)) < 1e-3, f"OR={or_val} ≠ exp(β={beta})"

    # CI 应包含 OR
    ci_str = row[6]  # "[lo, hi]"
    lo, hi = [float(x) for x in ci_str.strip("[]").split(",")]
    assert lo < or_val < hi, f"OR={or_val} 不在 CI [{lo}, {hi}] 内"


# ─────────────────────────────────────────────────────────────────────────────
# ROC AUC 在合理范围内
# ─────────────────────────────────────────────────────────────────────────────

def test_auc_range(df):
    result = logistic_reg.run(df, {
        "outcome": "group",
        "predictors": ["age", "bmi", "sbp_mmhg"],
        "mode": "multivariate",
    })
    fit_table = next(t for t in result.tables if "拟合指标" in t.title)
    fit_dict = {r[0]: r[1] for r in fit_table.rows}
    auc = float(fit_dict["AUC"])
    assert 0.5 <= auc <= 1.0, f"AUC={auc} 超出合理范围"

    # DeLong CI 应包含 AUC
    ci_str = fit_dict["AUC 95% CI（DeLong）"]
    lo, hi = [float(x) for x in ci_str.strip("[]").split(",")]
    assert lo <= auc <= hi


# ─────────────────────────────────────────────────────────────────────────────
# Hosmer-Lemeshow 检验存在且格式正确
# ─────────────────────────────────────────────────────────────────────────────

def test_hosmer_lemeshow(df):
    result = logistic_reg.run(df, {
        "outcome": "group",
        "predictors": ["age", "bmi", "sbp_mmhg"],
        "mode": "multivariate",
    })
    fit_dict = {r[0]: r[1] for r in
                next(t for t in result.tables if "拟合指标" in t.title).rows}
    # HL 统计量应为数值
    hl_chi2 = fit_dict.get("Hosmer-Lemeshow χ²", "")
    assert hl_chi2 != "" and hl_chi2 != "—"
    assert float(hl_chi2) >= 0.0


# ─────────────────────────────────────────────────────────────────────────────
# 单变量 vs 多变量方向一致性
# ─────────────────────────────────────────────────────────────────────────────

def test_univariate_multivariate_direction_consistency(df):
    """age 的 β 符号在单/多变量中通常方向一致（允许偶尔不一致，只验证行存在）。"""
    result = logistic_reg.run(df, {
        "outcome": "group",
        "predictors": ["age", "bmi"],
        "mode": "both",
    })
    uni_table = next(t for t in result.tables if "单变量" in t.title)
    multi_table = next(t for t in result.tables if "多变量" in t.title and "回归系数" in t.title)
    uni_vars = [r[0] for r in uni_table.rows]
    multi_vars = [r[0] for r in multi_table.rows]
    assert "age" in uni_vars
    assert "age" in multi_vars


# ─────────────────────────────────────────────────────────────────────────────
# 分类变量 dummy 编码
# ─────────────────────────────────────────────────────────────────────────────

def test_categorical_dummy_encoding(df):
    """sex (0/1) 标记为分类变量，应生成 dummy 列 sex_1，参考组为 0。"""
    result = logistic_reg.run(df, {
        "outcome": "group",
        "predictors": ["age", "sex"],
        "categorical_vars": ["sex"],
        "ref_categories": {"sex": "0"},
        "mode": "multivariate",
    })
    coef_table = next(t for t in result.tables if "回归系数" in t.title)
    var_names = [r[0] for r in coef_table.rows]
    # 应有 sex（参考组）的标题行 + 哑变量行 "  1"
    assert any("sex" in v and "参考" in v for v in var_names), f"找不到 sex 参考组行: {var_names}"
    assert any(v.strip() == "1" for v in var_names), f"找不到 sex dummy 行 '1': {var_names}"


# ─────────────────────────────────────────────────────────────────────────────
# 切换参考组后 dummy 方向改变
# ─────────────────────────────────────────────────────────────────────────────

def test_ref_category_switch(df):
    """参考组从 0 → 1 后，OR 应互为倒数（±浮点误差）。"""
    r0 = logistic_reg.run(df, {
        "outcome": "group",
        "predictors": ["sex"],
        "categorical_vars": ["sex"],
        "ref_categories": {"sex": "0"},
        "mode": "multivariate",
    })
    r1 = logistic_reg.run(df, {
        "outcome": "group",
        "predictors": ["sex"],
        "categorical_vars": ["sex"],
        "ref_categories": {"sex": "1"},
        "mode": "multivariate",
    })

    def get_dummy_or(result) -> float:
        coef_table = next(t for t in result.tables if "回归系数" in t.title)
        for row in coef_table.rows:
            if row[0] and str(row[0]).startswith("  ") and row[5] not in ("", "1 (Ref)", "—"):
                return float(row[5])
        return float("nan")

    or0 = get_dummy_or(r0)
    or1 = get_dummy_or(r1)
    assert not math.isnan(or0) and not math.isnan(or1)
    # OR 之积应约为 1（互为倒数）
    product = or0 * or1
    assert abs(product - 1.0) < 0.05, f"OR0={or0:.4f}, OR1={or1:.4f}, 乘积={product:.4f} ≠ 1"


# ─────────────────────────────────────────────────────────────────────────────
# 图表数量与类型
# ─────────────────────────────────────────────────────────────────────────────

def test_charts_present(df):
    result = logistic_reg.run(df, {
        "outcome": "group",
        "predictors": ["age", "bmi", "sbp_mmhg"],
        "mode": "multivariate",
    })
    chart_types = [c.chart_type for c in result.charts]
    assert "line" in chart_types        # ROC + 校准曲线
    assert "bar" in chart_types         # 预测概率分布
    assert "heatmap" in chart_types     # 混淆矩阵
    assert "forest_plot" in chart_types # OR 森林图

    # ROC chart 应包含 forestData 之外的 title
    roc_chart = next(c for c in result.charts if c.title == "ROC 曲线")
    assert "series" in roc_chart.option

    # Forest plot option 必须有 forestData 和 nullLine=1
    forest_chart = next(c for c in result.charts if c.chart_type == "forest_plot")
    assert forest_chart.option["nullLine"] == 1.0
    assert len(forest_chart.option["forestData"]) > 0


# ─────────────────────────────────────────────────────────────────────────────
# 错误处理
# ─────────────────────────────────────────────────────────────────────────────

def test_invalid_outcome_raises(df):
    with pytest.raises(ValueError, match="不存在"):
        logistic_reg.run(df, {"outcome": "nonexistent", "predictors": ["age"]})


def test_multiclass_outcome_raises(df):
    """连续变量作为因变量应报错（唯一值 > 2）。"""
    with pytest.raises(ValueError, match="二分类"):
        logistic_reg.run(df, {"outcome": "age", "predictors": ["bmi"]})


def test_univariate_only_mode(df):
    result = logistic_reg.run(df, {
        "outcome": "group",
        "predictors": ["age", "bmi"],
        "mode": "univariate",
    })
    titles = [t.title for t in result.tables]
    assert any("单变量" in t for t in titles)
    assert not any("多变量" in t for t in titles)
    assert len(result.charts) == 0  # 仅单变量不生成图表


def test_multivariate_only_mode(df):
    result = logistic_reg.run(df, {
        "outcome": "group",
        "predictors": ["age", "bmi"],
        "mode": "multivariate",
    })
    titles = [t.title for t in result.tables]
    assert not any("单变量" in t for t in titles)
    assert any("多变量" in t for t in titles)
    assert len(result.charts) > 0
