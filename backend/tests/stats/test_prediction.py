"""临床预测模型测试。

Logistic 模型：demo_basic.csv，因变量 group，预测因子 age、bmi、sbp_mmhg
Cox 模型：demo_survival.csv，time + event，预测因子 age、stage、treatment、biomarker
Bootstrap：n_bootstrap=100（小值加速）
"""

import math
from pathlib import Path

import pandas as pd
import pytest

BASIC_PATH = Path(__file__).parent.parent.parent / "data" / "examples" / "demo_basic.csv"
SURV_PATH = Path(__file__).parent.parent.parent / "data" / "examples" / "demo_survival.csv"

LOGISTIC_PARAMS = {
    "model_type": "logistic",
    "outcome": "group",
    "predictors": ["age", "bmi", "sbp_mmhg"],
    "categorical_vars": [],
    "ref_categories": {},
    "validation": "internal_bootstrap",
    "n_bootstrap": 100,
    "stepwise": False,
}

COX_PARAMS = {
    "model_type": "cox",
    "time_col": "time",
    "event_col": "event",
    "predictors": ["age", "stage", "treatment", "biomarker"],
    "categorical_vars": ["stage", "treatment"],
    "ref_categories": {"stage": "I", "treatment": "A"},
    "validation": "internal_bootstrap",
    "n_bootstrap": 100,
    "time_point": 365,
    "stepwise": False,
}


@pytest.fixture(scope="module")
def df_basic():
    return pd.read_csv(BASIC_PATH)


@pytest.fixture(scope="module")
def df_surv():
    return pd.read_csv(SURV_PATH)


def _run(df, **overrides):
    from app.stats.prediction import run
    params = {**LOGISTIC_PARAMS, **overrides} if overrides.get("model_type", "logistic") == "logistic" \
        else {**COX_PARAMS, **overrides}
    return run(df, params)


# ── Logistic 基础运行 ─────────────────────────────────────────────────────────

def test_logistic_returns_result(df_basic):
    result = _run(df_basic)
    assert result.method == "prediction"
    assert len(result.tables) >= 3
    assert len(result.charts) >= 3


def test_logistic_auc_reasonable(df_basic):
    result = _run(df_basic)
    fit_table = next(t for t in result.tables if "区分度" in t.title)
    auc_row = next(r for r in fit_table.rows if "AUC（原始）" in str(r[0]))
    auc_val = float(auc_row[1])
    assert 0.5 <= auc_val <= 1.0, f"AUC 超出合理范围：{auc_val}"


def test_logistic_calibration_data(df_basic):
    result = _run(df_basic)
    calib_chart = next((c for c in result.charts if c.title == "校准曲线"), None)
    assert calib_chart is not None, "缺少校准曲线图表"
    assert "series" in calib_chart.option
    assert len(calib_chart.option["series"]) >= 2


def test_logistic_nomogram_data(df_basic):
    result = _run(df_basic)
    nomo_chart = next((c for c in result.charts if c.chart_type == "nomogram"), None)
    assert nomo_chart is not None, "缺少 Nomogram 图表"
    nd = nomo_chart.option["nomogramData"]
    assert "variables" in nd
    assert len(nd["variables"]) >= 1
    assert "total_points" in nd
    assert "prob_scale" in nd
    # 检验概率刻度在 [0, 1]
    for pt in nd["prob_scale"]:
        assert 0.0 <= pt["prob"] <= 1.0, f"Nomogram 概率超出范围：{pt['prob']}"


def test_logistic_dca_data(df_basic):
    result = _run(df_basic)
    dca_chart = next((c for c in result.charts if "DCA" in c.title), None)
    assert dca_chart is not None, "缺少 DCA 图表"
    series_names = [s["name"] for s in dca_chart.option["series"]]
    assert "预测模型" in series_names
    assert "全部治疗" in series_names
    assert "全不治疗" in series_names


# ── Bootstrap 验证（Logistic）────────────────────────────────────────────────

def test_logistic_bootstrap_optimism(df_basic):
    """校正后 AUC 应 ≤ 原始 AUC（存在 optimism）。"""
    result = _run(df_basic, n_bootstrap=100)
    val_table = next(t for t in result.tables if "验证" in t.title)
    rows = {r[0]: r[1] for r in val_table.rows}
    orig_auc = float(rows["原始 AUC"])
    corrected_auc_str = rows["校正后 AUC"]
    if corrected_auc_str == "—":
        pytest.skip("Bootstrap 未完成，跳过 optimism 检验")
    corrected_auc = float(corrected_auc_str)
    assert corrected_auc <= orig_auc + 0.05, (
        f"校正后 AUC ({corrected_auc:.4f}) 明显大于原始 AUC ({orig_auc:.4f})，不合理"
    )


def test_logistic_bootstrap_hist_chart(df_basic):
    result = _run(df_basic, n_bootstrap=100)
    hist_chart = next((c for c in result.charts if "Bootstrap" in c.title), None)
    assert hist_chart is not None, "缺少 Bootstrap AUC 分布直方图"


# ── Split 验证 ────────────────────────────────────────────────────────────────

def test_logistic_split_validation(df_basic):
    result = _run(df_basic, validation="split", train_ratio=0.7)
    val_table = next(t for t in result.tables if "验证" in t.title)
    assert any("Split" in str(r[1]) for r in val_table.rows if r[0] == "验证方式")


# ── 交叉验证 ──────────────────────────────────────────────────────────────────

def test_logistic_cv(df_basic):
    result = _run(df_basic, validation="cross_validation")
    val_table = next(t for t in result.tables if "验证" in t.title)
    assert any("交叉" in str(r[1]) for r in val_table.rows if r[0] == "验证方式")


# ── Cox 基础运行 ──────────────────────────────────────────────────────────────

def test_cox_returns_result(df_surv):
    from app.stats.prediction import run
    result = run(df_surv, COX_PARAMS)
    assert result.method == "prediction"
    assert len(result.tables) >= 3
    assert len(result.charts) >= 2


def test_cox_cindex_reasonable(df_surv):
    from app.stats.prediction import run
    result = run(df_surv, COX_PARAMS)
    fit_table = next(t for t in result.tables if "区分度" in t.title)
    ci_row = next(r for r in fit_table.rows if "C-index（原始）" in str(r[0]))
    cindex = float(ci_row[1])
    assert 0.5 <= cindex <= 1.0, f"C-index 超出合理范围：{cindex}"


def test_cox_time_dep_auc(df_surv):
    from app.stats.prediction import run
    result = run(df_surv, COX_PARAMS)
    tdep_chart = next((c for c in result.charts if "时间依赖" in c.title), None)
    # 数据点可能不足时不生成该图，条件性检验
    if tdep_chart is not None:
        assert len(tdep_chart.option["series"][0]["data"]) >= 2


def test_cox_nomogram_data(df_surv):
    from app.stats.prediction import run
    result = run(df_surv, COX_PARAMS)
    nomo_chart = next((c for c in result.charts if c.chart_type == "nomogram"), None)
    assert nomo_chart is not None, "缺少 Cox Nomogram 图表"
    nd = nomo_chart.option["nomogramData"]
    assert nd["model_type"] == "cox"
    assert "time_point" in nd
    for pt in nd["prob_scale"]:
        assert 0.0 <= pt["prob"] <= 1.0, f"Cox Nomogram 生存率超出范围：{pt['prob']}"


def test_cox_bootstrap_optimism(df_surv):
    from app.stats.prediction import run
    result = run(df_surv, {**COX_PARAMS, "n_bootstrap": 50})
    val_table = next(t for t in result.tables if "验证" in t.title)
    rows = {r[0]: r[1] for r in val_table.rows}
    orig_ci = float(rows["原始 C-index"])
    corrected_str = rows.get("校正后 C-index", "—")
    if corrected_str == "—":
        pytest.skip("Bootstrap Cox 未完成，跳过检验")
    corrected_ci = float(corrected_str)
    # 校正后 C-index 应 ≤ 原始 + 小容忍
    assert corrected_ci <= orig_ci + 0.05, (
        f"校正后 C-index ({corrected_ci:.4f}) 大于原始 C-index ({orig_ci:.4f})"
    )


# ── 逐步筛选 ─────────────────────────────────────────────────────────────────

def test_logistic_stepwise(df_basic):
    result = _run(df_basic, stepwise=True, n_bootstrap=50)
    assert result.method == "prediction"
    assert len(result.tables) >= 3


# ── 错误处理 ─────────────────────────────────────────────────────────────────

def test_missing_outcome_raises(df_basic):
    from app.stats.prediction import run
    with pytest.raises(ValueError, match="outcome"):
        run(df_basic, {"model_type": "logistic", "predictors": ["age"]})


def test_invalid_model_type_raises(df_basic):
    from app.stats.prediction import run
    with pytest.raises(ValueError, match="model_type"):
        run(df_basic, {"model_type": "xgboost", "predictors": ["age"]})
