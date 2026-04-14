"""pytest 测试：stats/descriptive.py

使用 data/examples/demo_basic.csv 以及合成数据集覆盖主要逻辑。
"""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from app.stats.descriptive import run

DEMO_CSV = Path(__file__).parents[2] / "data" / "examples" / "demo_basic.csv"


# ── fixtures ──────────────────────────────────────────────────────


@pytest.fixture
def demo_df() -> pd.DataFrame:
    return pd.read_csv(DEMO_CSV)


@pytest.fixture
def normal_df() -> pd.DataFrame:
    rng = np.random.default_rng(0)
    return pd.DataFrame({"x": rng.normal(0, 1, 100), "y": rng.normal(5, 2, 100)})


@pytest.fixture
def skewed_df() -> pd.DataFrame:
    rng = np.random.default_rng(1)
    return pd.DataFrame({"glucose": np.exp(rng.normal(1.7, 0.4, 100))})


# ── 结构正确性 ─────────────────────────────────────────────────────


def test_returns_analysis_result(demo_df):
    result = run(demo_df, {})
    assert result.method == "descriptive"
    assert len(result.tables) == 2
    assert len(result.charts) > 0
    assert result.summary


def test_table_headers(demo_df):
    result = run(demo_df, {})
    desc_table = result.tables[0]
    assert "均值" in desc_table.headers
    assert "标准差" in desc_table.headers
    assert "中位数" in desc_table.headers
    assert "P25" in desc_table.headers
    assert "P75" in desc_table.headers

    norm_table = result.tables[1]
    assert "p 值" in norm_table.headers
    assert "结论" in norm_table.headers


def test_row_count_matches_variables(demo_df):
    vars_ = ["age", "bmi", "sbp_mmhg"]
    result = run(demo_df, {"variables": vars_})
    assert len(result.tables[0].rows) == 3
    assert len(result.tables[1].rows) == 3


def test_charts_per_variable(demo_df):
    vars_ = ["age", "height_cm"]
    result = run(demo_df, {"variables": vars_})
    # 每个变量 2 张图（直方图 + QQ 图）
    assert len(result.charts) == 4
    chart_types = {c.chart_type for c in result.charts}
    assert "bar" in chart_types
    assert "scatter" in chart_types


# ── 数值正确性 ─────────────────────────────────────────────────────


def test_descriptive_stats_values(normal_df):
    result = run(normal_df, {"variables": ["x"]})
    row = result.tables[0].rows[0]
    # row: [变量, n, missing, mean, std, median, q1, q3, min, max]
    assert row[1] == 100          # n
    assert row[2] == 0            # no missing
    # 均值接近 0
    assert abs(float(row[3])) < 0.5


def test_missing_count_reported(demo_df):
    result = run(demo_df, {"variables": ["fasting_glucose_mmol"]})
    row = result.tables[0].rows[0]
    missing = row[2]
    assert missing > 0


def test_normality_normal_data(normal_df):
    result = run(normal_df, {"variables": ["x"]})
    norm_row = result.tables[1].rows[0]
    # 正态数据 p > 0.05 → 结论含"正态"而非"非正态"
    conclusion = norm_row[-1]
    assert "非正态" not in conclusion
    assert "正态" in conclusion


def test_normality_skewed_data(skewed_df):
    result = run(skewed_df, {"variables": ["glucose"]})
    norm_row = result.tables[1].rows[0]
    conclusion = norm_row[-1]
    assert "非正态" in conclusion


# ── 边界情况 ───────────────────────────────────────────────────────


def test_auto_select_all_numeric(demo_df):
    """不指定 variables 时自动选全部数值列"""
    result = run(demo_df, {})
    n_numeric = len(demo_df.select_dtypes(include="number").columns)
    assert len(result.tables[0].rows) == n_numeric


def test_nonexistent_variable_warned(demo_df):
    result = run(demo_df, {"variables": ["age", "no_such_col"]})
    assert any("no_such_col" in w for w in result.warnings)
    assert len(result.tables[0].rows) == 1  # 只分析了 age


def test_non_numeric_variable_warned(demo_df):
    df = demo_df.copy()
    df["label"] = "A"
    result = run(df, {"variables": ["age", "label"]})
    assert any("非数值" in w for w in result.warnings)


def test_all_missing_skipped():
    df = pd.DataFrame({"x": [1.0, 2.0], "y": [float("nan"), float("nan")]})
    result = run(df, {"variables": ["x", "y"]})
    assert len(result.tables[0].rows) == 1
    assert any("全部缺失" in w for w in result.warnings)


def test_raises_on_no_numeric_data():
    df = pd.DataFrame({"group": ["A", "B", "C"]})
    with pytest.raises(ValueError, match="数值"):
        run(df, {})


def test_small_sample_normality():
    """n < 3 时不做检验"""
    df = pd.DataFrame({"x": [1.0, 2.0]})
    result = run(df, {"variables": ["x"]})
    assert "样本量不足" in result.tables[1].rows[0][-1]


# ── API 端点集成测试 ───────────────────────────────────────────────


@pytest.mark.asyncio
async def test_descriptive_endpoint(demo_df):
    """通过 HTTP 客户端调用 /api/analyze/descriptive"""
    import io

    from httpx import ASGITransport, AsyncClient

    from app.main import app
    from app.services.file_store import save_dataframe

    file_id = save_dataframe(demo_df)

    async with AsyncClient(transport=ASGITransport(app=app),
                           base_url="http://test") as client:
        resp = await client.post(
            "/api/analyze/descriptive",
            json={"file_id": file_id, "params": {"variables": ["age", "bmi"]}},
        )

    assert resp.status_code == 200
    data = resp.json()
    assert data["method"] == "descriptive"
    assert len(data["tables"]) == 2
    assert len(data["charts"]) == 4


@pytest.mark.asyncio
async def test_descriptive_endpoint_404():
    from httpx import ASGITransport, AsyncClient

    from app.main import app

    async with AsyncClient(transport=ASGITransport(app=app),
                           base_url="http://test") as client:
        resp = await client.post(
            "/api/analyze/descriptive",
            json={"file_id": "nonexistent_id", "params": {}},
        )
    assert resp.status_code == 404
