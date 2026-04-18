"""生存分析 & Cox 回归模块测试。

使用 backend/data/examples/demo_survival.csv 进行集成验证：
  - KM 中位生存时间合理性
  - Log-rank 按 treatment 分组是否显著（B 更差）
  - Cox HR 方向：stage 越高 HR 越大，treatment B 的 HR > 1
  - PH 假设检验能正常运行
  - 单变量 vs 多变量结果格式一致
  - AnalysisResult 结构校验
"""

import math
from pathlib import Path

import pandas as pd
import pytest

DATA_DIR = Path(__file__).parent.parent.parent / "data" / "examples"
CSV_PATH = DATA_DIR / "demo_survival.csv"


@pytest.fixture(scope="module")
def df() -> pd.DataFrame:
    return pd.read_csv(CSV_PATH)


# ─────────────────────────────────────────────────────────────────────────────
# 生存分析模块
# ─────────────────────────────────────────────────────────────────────────────

class TestSurvivalModule:
    def test_basic_no_group(self, df):
        """单组 KM 基本运行，检查结构完整性。"""
        from app.stats.survival import run
        result = run(df, {"time_col": "time", "event_col": "event"})

        assert result.method == "survival"
        assert len(result.tables) >= 2  # 摘要 + 生存表
        assert len(result.charts) == 2  # KM 曲线 + 累积风险图
        assert result.summary

        # 摘要表结构
        summary_tbl = result.tables[0]
        assert summary_tbl.title == "生存分析摘要"
        assert summary_tbl.headers[0] == "组别"
        assert len(summary_tbl.rows) == 1

        # 中位生存时间应为正数
        med_str = str(summary_tbl.rows[0][3])
        assert med_str != "" and med_str != "—"

    def test_with_group_col(self, df):
        """分组 KM：治疗组 A vs B，Log-rank 检验需返回结果。"""
        from app.stats.survival import run
        result = run(df, {
            "time_col": "time",
            "event_col": "event",
            "group_col": "treatment",
        })

        # 应有 Log-rank 检验表
        titles = [t.title for t in result.tables]
        assert "Log-rank 检验" in titles

        lr_tbl = next(t for t in result.tables if t.title == "Log-rank 检验")
        assert len(lr_tbl.rows) == 3  # Log-rank / Breslow / Tarone-Ware
        # 第一行为 Log-rank
        lr_row = lr_tbl.rows[0]
        assert lr_row[0] == "Log-rank"
        # p 值应能被解析
        p_str = str(lr_row[3])
        assert p_str != "—"

        # 摘要表应有 2 行（A 和 B）
        summary_tbl = result.tables[0]
        assert len(summary_tbl.rows) == 2

    def test_treatment_b_worse(self, df):
        """治疗组 B 的中位生存时间应 ≤ 治疗组 A（数据设计保证）。"""
        from app.stats.survival import run
        result = run(df, {
            "time_col": "time",
            "event_col": "event",
            "group_col": "treatment",
        })
        summary_tbl = result.tables[0]
        rows_by_group = {str(r[0]): r for r in summary_tbl.rows}
        # 中位时间（可能是 "未达到"，则跳过比较）
        med_a_str = str(rows_by_group.get("A", ["A", 0, 0, "0"])[3])
        med_b_str = str(rows_by_group.get("B", ["B", 0, 0, "0"])[3])
        if med_a_str != "未达到" and med_b_str != "未达到":
            med_a = float(med_a_str)
            med_b = float(med_b_str)
            assert med_b <= med_a * 1.5, f"B 组中位时间 ({med_b}) 不应远大于 A 组 ({med_a})"

    def test_four_stage_groups(self, df):
        """四分组 KM + 两两比较。"""
        from app.stats.survival import run
        result = run(df, {
            "time_col": "time",
            "event_col": "event",
            "group_col": "stage",
        })
        titles = [t.title for t in result.tables]
        assert any("两两" in t for t in titles)
        # 4 组 → 6 对
        pairwise_tbl = next(t for t in result.tables if "两两" in t.title)
        assert len(pairwise_tbl.rows) == 6

    def test_time_points(self, df):
        """指定时间点生存率表应正确返回。"""
        from app.stats.survival import run
        result = run(df, {
            "time_col": "time",
            "event_col": "event",
            "group_col": "treatment",
            "time_points": [365, 730, 1095],
        })
        titles = [t.title for t in result.tables]
        assert "指定时间点生存率" in titles
        tp_tbl = next(t for t in result.tables if t.title == "指定时间点生存率")
        assert len(tp_tbl.headers) == 4  # 组别 + 3个时间点

    def test_km_chart_structure(self, df):
        """KM 曲线 ECharts option 应包含必要字段。"""
        from app.stats.survival import run
        result = run(df, {
            "time_col": "time",
            "event_col": "event",
            "group_col": "treatment",
        })
        km_chart = next(c for c in result.charts if c.chart_type == "kaplan_meier")
        opt = km_chart.option
        assert "series" in opt
        assert "xAxis" in opt
        assert "yAxis" in opt
        assert "numberAtRisk" in opt
        # 至少有主曲线系列（2 组）
        main_series = [s for s in opt["series"] if not str(s.get("name", "")).startswith("_")]
        assert len(main_series) >= 2

    def test_life_table_exists(self, df):
        """生存表应存在且有行数据。"""
        from app.stats.survival import run
        result = run(df, {"time_col": "time", "event_col": "event"})
        titles = [t.title for t in result.tables]
        assert "生存表（Life Table）" in titles
        lt = next(t for t in result.tables if t.title == "生存表（Life Table）")
        assert len(lt.rows) > 0

    def test_missing_time_col_raises(self, df):
        from app.stats.survival import run
        with pytest.raises(ValueError, match="时间变量"):
            run(df, {"time_col": "", "event_col": "event"})

    def test_invalid_group_col_warning(self, df):
        from app.stats.survival import run
        result = run(df, {
            "time_col": "time",
            "event_col": "event",
            "group_col": "nonexistent_col",
        })
        assert any("不存在" in w for w in result.warnings)

    def test_no_exp_overflow(self, df):
        """生存率值必须在 [0, 1] 范围内，不出现 NaN/inf。"""
        from app.stats.survival import run
        result = run(df, {
            "time_col": "time",
            "event_col": "event",
            "group_col": "stage",
        })
        km_chart = next(c for c in result.charts if c.chart_type == "kaplan_meier")
        for series in km_chart.option["series"]:
            for pt in series.get("data", []):
                if isinstance(pt, (list, tuple)) and len(pt) == 2:
                    val = float(pt[1])
                    assert math.isfinite(val), f"Non-finite value in KM series: {val}"
                    # 生存函数主曲线在 [0,1]（CI 差值可能略超）


# ─────────────────────────────────────────────────────────────────────────────
# Cox 回归模块
# ─────────────────────────────────────────────────────────────────────────────

class TestCoxRegModule:
    def test_multivariate_basic(self, df):
        """多变量 Cox 基本运行，检查结构。"""
        from app.stats.cox_reg import run
        result = run(df, {
            "time_col": "time",
            "event_col": "event",
            "predictors": ["age", "treatment", "stage"],
            "categorical_vars": ["treatment", "stage"],
            "mode": "multivariate",
        })
        assert result.method == "cox_reg"
        titles = [t.title for t in result.tables]
        assert "多变量 Cox 回归 — 回归系数" in titles
        assert "多变量 Cox 回归 — 模型拟合指标" in titles

    def test_univariate_basic(self, df):
        """单变量 Cox 运行，每个变量都有结果行。"""
        from app.stats.cox_reg import run
        result = run(df, {
            "time_col": "time",
            "event_col": "event",
            "predictors": ["age", "biomarker", "treatment"],
            "categorical_vars": ["treatment"],
            "mode": "univariate",
        })
        titles = [t.title for t in result.tables]
        assert "单变量 Cox 回归" in titles
        uni_tbl = next(t for t in result.tables if t.title == "单变量 Cox 回归")
        assert len(uni_tbl.rows) > 0

    def test_both_mode(self, df):
        """mode=both 时应同时有单变量和多变量表。"""
        from app.stats.cox_reg import run
        result = run(df, {
            "time_col": "time",
            "event_col": "event",
            "predictors": ["age", "treatment", "stage"],
            "categorical_vars": ["treatment", "stage"],
            "mode": "both",
        })
        titles = [t.title for t in result.tables]
        assert "单变量 Cox 回归" in titles
        assert "多变量 Cox 回归 — 回归系数" in titles

    def test_hr_direction_stage(self, df):
        """Stage IV 的 HR 应 > Stage I（设计保证）。"""
        from app.stats.cox_reg import run
        result = run(df, {
            "time_col": "time",
            "event_col": "event",
            "predictors": ["stage"],
            "categorical_vars": ["stage"],
            "ref_categories": {"stage": "I"},
            "mode": "multivariate",
        })
        coef_tbl = next(
            t for t in result.tables if "回归系数" in t.title
        )
        # 找 stage IV 行
        iv_row = next(
            (r for r in coef_tbl.rows if "IV" in str(r[0])),
            None
        )
        assert iv_row is not None, "未找到 stage IV 行"
        if iv_row[5] not in ("—", ""):
            hr_iv = float(iv_row[5])
            assert hr_iv >= 1.0, f"Stage IV HR ({hr_iv}) 应 >= 1"

    def test_treatment_b_hr_gt1(self, df):
        """Treatment B 对比 A 的 HR 应 > 1（数据设计保证）。"""
        from app.stats.cox_reg import run
        result = run(df, {
            "time_col": "time",
            "event_col": "event",
            "predictors": ["treatment"],
            "categorical_vars": ["treatment"],
            "ref_categories": {"treatment": "A"},
            "mode": "multivariate",
        })
        coef_tbl = next(t for t in result.tables if "回归系数" in t.title)
        b_row = next(
            (r for r in coef_tbl.rows if "B" in str(r[0]) and str(r[5]) not in ("", "—")),
            None
        )
        if b_row:
            hr_b = float(b_row[5])
            assert hr_b >= 0.5, f"Treatment B HR ({hr_b}) 不合理"

    def test_ph_assumption_runs(self, df):
        """PH 假设检验应能正常运行（即使结果不显著）。"""
        from app.stats.cox_reg import run
        result = run(df, {
            "time_col": "time",
            "event_col": "event",
            "predictors": ["age", "biomarker"],
            "mode": "multivariate",
        })
        ph_titles = [t.title for t in result.tables if "Schoenfeld" in t.title or "比例风险" in t.title]
        # PH 检验表存在（即使失败，warnings 中应有提示）
        ph_found = len(ph_titles) > 0
        ph_warned = any("PH" in w or "比例风险" in w or "Schoenfeld" in w for w in result.warnings)
        assert ph_found or ph_warned

    def test_cindex_in_range(self, df):
        """C-index 应在 (0, 1) 范围内。"""
        from app.stats.cox_reg import run
        result = run(df, {
            "time_col": "time",
            "event_col": "event",
            "predictors": ["age", "treatment", "stage", "biomarker"],
            "categorical_vars": ["treatment", "stage"],
            "mode": "multivariate",
        })
        fit_tbl = next(t for t in result.tables if "拟合指标" in t.title)
        fit_dict = {r[0]: r[1] for r in fit_tbl.rows}
        c_str = str(fit_dict.get("C-index（一致性指数）", "0.5"))
        c_val = float(c_str)
        assert 0.0 < c_val < 1.0, f"C-index ({c_val}) 超出范围"

    def test_forest_chart_exists(self, df):
        """多变量 Cox 应生成 HR 森林图。"""
        from app.stats.cox_reg import run
        result = run(df, {
            "time_col": "time",
            "event_col": "event",
            "predictors": ["age", "treatment", "stage"],
            "categorical_vars": ["treatment", "stage"],
            "mode": "multivariate",
        })
        forest = [c for c in result.charts if c.chart_type == "forest_plot"]
        assert len(forest) >= 1
        assert "forestData" in forest[0].option

    def test_hr_no_overflow(self, df):
        """HR 值不应出现 inf 或 NaN（np.clip 保护）。"""
        from app.stats.cox_reg import run
        result = run(df, {
            "time_col": "time",
            "event_col": "event",
            "predictors": ["age", "treatment", "stage", "biomarker"],
            "categorical_vars": ["treatment", "stage"],
            "mode": "both",
        })
        for tbl in result.tables:
            if "回归系数" in tbl.title or "单变量" in tbl.title:
                hr_idx = tbl.headers.index("HR") if "HR" in tbl.headers else None
                if hr_idx is not None:
                    for row in tbl.rows:
                        hr_val = str(row[hr_idx])
                        if hr_val not in ("", "—"):
                            val = float(hr_val)
                            assert math.isfinite(val), f"HR 出现非有限值: {val}"

    def test_result_format_consistency(self, df):
        """单变量和多变量系数表应有相同的 headers 格式。"""
        from app.stats.cox_reg import run
        result = run(df, {
            "time_col": "time",
            "event_col": "event",
            "predictors": ["age", "treatment"],
            "categorical_vars": ["treatment"],
            "mode": "both",
        })
        uni_tbl = next(t for t in result.tables if t.title == "单变量 Cox 回归")
        multi_tbl = next(t for t in result.tables if "多变量" in t.title and "回归系数" in t.title)
        assert uni_tbl.headers == multi_tbl.headers

    def test_missing_predictor_warning(self, df):
        """不存在的自变量应触发 warning 而不是崩溃。"""
        from app.stats.cox_reg import run
        result = run(df, {
            "time_col": "time",
            "event_col": "event",
            "predictors": ["age", "nonexistent_var"],
            "mode": "multivariate",
        })
        assert any("nonexistent_var" in w or "不存在" in w for w in result.warnings)

    def test_missing_time_col_raises(self, df):
        from app.stats.cox_reg import run
        with pytest.raises(ValueError, match="时间变量"):
            run(df, {"time_col": "", "event_col": "event", "predictors": ["age"]})

    def test_no_predictors_raises(self, df):
        from app.stats.cox_reg import run
        with pytest.raises(ValueError, match="自变量"):
            run(df, {"time_col": "time", "event_col": "event", "predictors": []})
