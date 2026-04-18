"""生存分析模块 — Kaplan-Meier 估计与 Log-rank 检验。

支持：
  - 单组 / 多组 KM 曲线（生存函数 + 95% CI）
  - 中位生存时间 + 95% CI
  - 指定时间点的生存率（如 1年/3年/5年）
  - Log-rank / Breslow / Tarone-Ware 多检验
  - ≥3 组两两比较（Bonferroni 校正）
  - Life table（生存表）
  - KM 曲线 ECharts step-line + CI 阴影 + 删失标记
  - 累积风险函数图
  - Number at risk 表嵌入图表数据
"""

import logging
import warnings as _warnings_mod
from typing import Any

import numpy as np
import pandas as pd

from app.models.analysis import AnalysisResult, ChartResult, TableResult

logger = logging.getLogger(__name__)

# ECharts 调色板
_PALETTE = [
    "#5470c6", "#91cc75", "#ee6666", "#fac858",
    "#73c0de", "#3ba272", "#fc8452", "#9a60b4",
]


# ─────────────────────────────────────────────────────────────────────────────
# 主入口
# ─────────────────────────────────────────────────────────────────────────────

def run(df: pd.DataFrame, params: dict) -> AnalysisResult:
    """
    参数
    ----
    params["time_col"]      str           时间变量
    params["event_col"]     str           事件变量（0=删失, 1=事件）
    params["group_col"]     str | None    分组变量（可选）
    params["time_points"]   list[float]   指定时间点（可选），如 [365, 1095, 1825]
    """
    try:
        from lifelines import KaplanMeierFitter
        from lifelines.statistics import (
            logrank_test,
            multivariate_logrank_test,
        )
    except ImportError as exc:
        raise RuntimeError("lifelines 未安装，无法进行生存分析") from exc

    time_col: str = str(params.get("time_col", ""))
    event_col: str = str(params.get("event_col", ""))
    group_col: str | None = params.get("group_col") or None
    time_points: list[float] = [float(t) for t in (params.get("time_points") or [])]
    warnings: list[str] = []

    # ── 参数校验 ──────────────────────────────────────────────────────────────
    if not time_col:
        raise ValueError("请指定时间变量 (time_col)")
    if not event_col:
        raise ValueError("请指定事件变量 (event_col)")
    for col in [time_col, event_col]:
        if col not in df.columns:
            raise ValueError(f"列 '{col}' 不存在于数据集中")
    if group_col and group_col not in df.columns:
        warnings.append(f"分组变量 '{group_col}' 不存在，将忽略分组分析")
        group_col = None

    # ── 数据清洗 ──────────────────────────────────────────────────────────────
    needed = [time_col, event_col] + ([group_col] if group_col else [])
    df_work = df[needed].dropna().copy()
    df_work[time_col] = pd.to_numeric(df_work[time_col], errors="coerce")
    df_work[event_col] = pd.to_numeric(df_work[event_col], errors="coerce")
    df_work = df_work.dropna(subset=[time_col, event_col])

    n_orig, n_clean = len(df), len(df_work)
    if n_orig - n_clean > 0:
        warnings.append(f"删除 {n_orig - n_clean} 行含缺失值的数据，剩余 {n_clean} 行")

    neg_time = (df_work[time_col] < 0).sum()
    if neg_time > 0:
        warnings.append(f"发现 {neg_time} 行时间值 < 0，已剔除")
        df_work = df_work[df_work[time_col] >= 0]

    invalid_event = (~df_work[event_col].isin([0, 1])).sum()
    if invalid_event > 0:
        warnings.append(f"事件变量含 {invalid_event} 个非 0/1 值，已剔除")
        df_work = df_work[df_work[event_col].isin([0, 1])]

    if len(df_work) < 10:
        raise ValueError(
            f"有效数据不足 10 行（当前 {len(df_work)} 行），无法进行生存分析"
        )

    event_rate = df_work[event_col].mean()
    if event_rate < 0.05:
        warnings.append(f"事件发生率较低（{event_rate:.1%}），估计结果可能不稳定")

    T = df_work[time_col].values
    E = df_work[event_col].values.astype(int)

    tables: list[TableResult] = []
    charts: list[ChartResult] = []

    # ── 分组 ──────────────────────────────────────────────────────────────────
    logrank_rows: list[list[Any]] = []
    if group_col:
        groups = sorted(df_work[group_col].astype(str).unique())
        if len(groups) > 8:
            warnings.append(f"分组变量超过 8 组（{len(groups)} 组），仅展示前 8 组")
            groups = groups[:8]
        if len(groups) < 2:
            warnings.append("分组变量只有 1 个类别，将忽略分组分析")
            group_col = None
            groups = []
    else:
        groups = []

    # ── 各组 KM 估计 ──────────────────────────────────────────────────────────
    km_data: dict[str, Any] = {}  # group_label -> {kmf, T, E}

    if not group_col:
        from lifelines import KaplanMeierFitter
        kmf = KaplanMeierFitter(label="全体")
        with _warnings_mod.catch_warnings():
            _warnings_mod.simplefilter("ignore")
            kmf.fit(T, E)
        km_data["全体"] = {"kmf": kmf, "T": T, "E": E}
    else:
        from lifelines import KaplanMeierFitter
        for g in groups:
            mask = df_work[group_col].astype(str) == g
            Tg = df_work.loc[mask, time_col].values
            Eg = df_work.loc[mask, event_col].values.astype(int)
            kmf = KaplanMeierFitter(label=str(g))
            with _warnings_mod.catch_warnings():
                _warnings_mod.simplefilter("ignore")
                kmf.fit(Tg, Eg)
            km_data[str(g)] = {"kmf": kmf, "T": Tg, "E": Eg}

    # ── 摘要表 ────────────────────────────────────────────────────────────────
    summary_rows: list[list[Any]] = []
    for label, d in km_data.items():
        kmf = d["kmf"]
        n_total = len(d["T"])
        n_events = int(d["E"].sum())
        med = kmf.median_survival_time_
        med_str = f"{med:.1f}" if np.isfinite(med) else "未达到"
        ci_low, ci_high = _median_ci_from_kmf(kmf)
        ci_str = (
            f"({ci_low:.1f}, {ci_high:.1f})"
            if (ci_low is not None and ci_high is not None)
            else "—"
        )
        summary_rows.append([label, n_total, n_events, med_str, ci_str])

    tables.append(TableResult(
        title="生存分析摘要",
        headers=["组别", "总人数", "事件数", "中位生存时间", "95% CI"],
        rows=summary_rows,
    ))

    # ── 指定时间点生存率 ─────────────────────────────────────────────────────
    if time_points:
        tp_headers = [
            "组别"
        ] + [
            f"t={int(t) if float(t) == int(t) else t} 生存率"
            for t in time_points
        ]
        tp_rows: list[list[Any]] = []
        for label, d in km_data.items():
            kmf = d["kmf"]
            row: list[Any] = [label]
            for t in time_points:
                try:
                    sf_val = float(kmf.survival_function_at_times([t]).values[0])
                    sf_val = float(np.clip(sf_val, 0.0, 1.0))
                    ci_df = kmf.confidence_interval_survival_function_
                    t_idx = int(ci_df.index.searchsorted(t, side="right")) - 1
                    t_idx = max(0, min(t_idx, len(ci_df) - 1))
                    lo = float(np.clip(ci_df.iloc[t_idx, 0], 0.0, 1.0))
                    hi = float(np.clip(ci_df.iloc[t_idx, 1], 0.0, 1.0))
                    row.append(f"{sf_val:.1%} ({lo:.1%}~{hi:.1%})")
                except Exception:
                    row.append("—")
            tp_rows.append(row)
        tables.append(TableResult(
            title="指定时间点生存率",
            headers=tp_headers,
            rows=tp_rows,
        ))

    # ── Log-rank 检验 ─────────────────────────────────────────────────────────
    if group_col and len(groups) >= 2:
        logrank_rows = _run_logrank_tests(df_work, time_col, event_col, group_col, groups)
        tables.append(TableResult(
            title="Log-rank 检验",
            headers=["检验方法", "统计量", "df", "p 值", "显著性"],
            rows=logrank_rows,
        ))

        if len(groups) >= 3:
            pairwise_rows = _pairwise_logrank(
                df_work, time_col, event_col, group_col, groups
            )
            tables.append(TableResult(
                title=f"两两 Log-rank 比较（Bonferroni 校正，共 {len(pairwise_rows)} 对）",
                headers=["组对", "统计量", "原始 p", "校正 p", "显著性"],
                rows=pairwise_rows,
            ))

    # ── Life table ────────────────────────────────────────────────────────────
    life_table_rows = _build_life_table(km_data)
    tables.append(TableResult(
        title="生存表（Life Table）",
        headers=["组别", "时间区间", "风险集 n", "事件数", "删失数", "生存率", "95% CI"],
        rows=life_table_rows,
    ))

    # ── KM 曲线图 + 累积风险图 ────────────────────────────────────────────────
    charts.append(_build_km_chart(km_data, group_col))
    charts.append(_build_cumulative_hazard_chart(km_data, group_col))

    # ── 文字摘要 ──────────────────────────────────────────────────────────────
    n_total = len(df_work)
    n_events = int(E.sum())
    if group_col:
        summary = (
            f"生存分析，时间变量：{time_col}，事件变量：{event_col}，"
            f"分组：{group_col}（{len(groups)} 组），"
            f"总样本 n = {n_total}，事件 {n_events} 例（{n_events / n_total:.1%}）。"
        )
        if logrank_rows:
            lr_p = logrank_rows[0][3] if logrank_rows else "—"
            summary += f"Log-rank 检验 p = {lr_p}。"
    else:
        kmf0 = next(iter(km_data.values()))["kmf"]
        med = kmf0.median_survival_time_
        med_str = f"{med:.1f}" if np.isfinite(med) else "未达到"
        summary = (
            f"生存分析，时间变量：{time_col}，事件变量：{event_col}，"
            f"样本 n = {n_total}，事件 {n_events} 例（{n_events / n_total:.1%}），"
            f"中位生存时间 = {med_str}。"
        )

    return AnalysisResult(
        method="survival",
        tables=tables,
        charts=charts,
        summary=summary,
        warnings=warnings,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Log-rank / Breslow / Tarone-Ware 检验
# ─────────────────────────────────────────────────────────────────────────────

def _run_logrank_tests(
    df_work: pd.DataFrame,
    time_col: str,
    event_col: str,
    group_col: str,
    groups: list[str],
) -> list[list[Any]]:
    from lifelines.statistics import multivariate_logrank_test

    T_all = df_work[time_col].values
    E_all = df_work[event_col].values.astype(int)
    G_all = df_work[group_col].astype(str).values

    rows: list[list[Any]] = []
    for method_name, wt_param in [
        ("Log-rank", None),
        ("Breslow", "wilcoxon"),
        ("Tarone-Ware", "tarone-ware"),
    ]:
        try:
            result = multivariate_logrank_test(
                T_all, G_all, E_all,
                weightings=wt_param,
            )
            stat = float(np.clip(result.test_statistic, -1e15, 1e15))
            p = float(result.p_value)
            df_val = len(groups) - 1
            sig = "***" if p < 0.001 else ("**" if p < 0.01 else ("*" if p < 0.05 else "ns"))
            rows.append([method_name, f"{stat:.3f}", df_val, _fmt_p(p), sig])
        except Exception as exc:
            logger.warning("Log-rank test (%s) failed: %s", method_name, exc)
            rows.append([method_name, "—", "—", "—", "—"])
    return rows


def _pairwise_logrank(
    df_work: pd.DataFrame,
    time_col: str,
    event_col: str,
    group_col: str,
    groups: list[str],
) -> list[list[Any]]:
    from lifelines.statistics import logrank_test

    n_pairs = len(groups) * (len(groups) - 1) // 2
    pairs: list[list[Any]] = []
    for i, g1 in enumerate(groups):
        for g2 in groups[i + 1:]:
            m1 = df_work[group_col].astype(str) == g1
            m2 = df_work[group_col].astype(str) == g2
            T1 = df_work.loc[m1, time_col].values
            E1 = df_work.loc[m1, event_col].values.astype(int)
            T2 = df_work.loc[m2, time_col].values
            E2 = df_work.loc[m2, event_col].values.astype(int)
            try:
                res = logrank_test(T1, T2, E1, E2)
                stat = float(res.test_statistic)
                p_raw = float(res.p_value)
                p_adj = min(p_raw * n_pairs, 1.0)
                sig = (
                    "***" if p_adj < 0.001
                    else ("**" if p_adj < 0.01
                    else ("*" if p_adj < 0.05 else "ns"))
                )
                pairs.append([
                    f"{g1} vs {g2}",
                    f"{stat:.3f}",
                    _fmt_p(p_raw),
                    _fmt_p(p_adj),
                    sig,
                ])
            except Exception as exc:
                logger.warning("Pairwise logrank %s vs %s failed: %s", g1, g2, exc)
                pairs.append([f"{g1} vs {g2}", "—", "—", "—", "—"])
    return pairs


# ─────────────────────────────────────────────────────────────────────────────
# Life table
# ─────────────────────────────────────────────────────────────────────────────

def _build_life_table(km_data: dict) -> list[list[Any]]:
    rows: list[list[Any]] = []
    for label, d in km_data.items():
        kmf = d["kmf"]
        sf = kmf.survival_function_
        ci = kmf.confidence_interval_survival_function_
        T_arr = d["T"]
        E_arr = d["E"]

        times = sf.index.tolist()
        sf_vals = sf.iloc[:, 0].tolist()
        ci_lo = ci.iloc[:, 0].tolist()
        ci_hi = ci.iloc[:, 1].tolist()

        for idx in range(len(times)):
            t = times[idx]
            t_prev = times[idx - 1] if idx > 0 else 0.0
            at_risk = int((T_arr >= t).sum())
            events = int(((T_arr == t) & (E_arr == 1)).sum())
            censored = int(((T_arr == t) & (E_arr == 0)).sum())
            sv = float(np.clip(sf_vals[idx], 0.0, 1.0))
            lo = float(np.clip(ci_lo[idx], 0.0, 1.0))
            hi = float(np.clip(ci_hi[idx], 0.0, 1.0))
            rows.append([
                label,
                f"{t_prev:.0f}–{t:.0f}",
                at_risk,
                events,
                censored,
                f"{sv:.4f}",
                f"({lo:.4f}, {hi:.4f})",
            ])
    return rows


# ─────────────────────────────────────────────────────────────────────────────
# 中位生存时间 CI
# ─────────────────────────────────────────────────────────────────────────────

def _median_ci_from_kmf(kmf) -> tuple[float | None, float | None]:
    """从 CI 曲线中估计中位生存时间的 95% CI。"""
    try:
        ci = kmf.confidence_interval_survival_function_
        times = ci.index.values
        lo_col = ci.iloc[:, 0].values  # KM_lower
        hi_col = ci.iloc[:, 1].values  # KM_upper

        def _first_cross(vals: np.ndarray, threshold: float = 0.5):
            for i, v in enumerate(vals):
                if v <= threshold:
                    return float(times[i])
            return None

        ci_low = _first_cross(hi_col)   # upper CI 曲线穿越 0.5 → CI 下界
        ci_high = _first_cross(lo_col)  # lower CI 曲线穿越 0.5 → CI 上界
        return ci_low, ci_high
    except Exception:
        return None, None


# ─────────────────────────────────────────────────────────────────────────────
# ECharts KM 曲线（step-line + CI 阴影 + 删失点）
# ─────────────────────────────────────────────────────────────────────────────

def _build_km_chart(km_data: dict, group_col: str | None) -> ChartResult:
    series: list[dict] = []
    legend_data: list[str] = []

    for color_idx, (label, d) in enumerate(km_data.items()):
        color = _PALETTE[color_idx % len(_PALETTE)]
        kmf = d["kmf"]
        T_arr = d["T"]
        E_arr = d["E"]

        sf = kmf.survival_function_
        ci = kmf.confidence_interval_survival_function_
        times = [0.0] + sf.index.tolist()
        sf_vals = [1.0] + [float(np.clip(v, 0.0, 1.0)) for v in sf.iloc[:, 0].tolist()]
        ci_lo_vals = [1.0] + [float(np.clip(v, 0.0, 1.0)) for v in ci.iloc[:, 0].tolist()]
        ci_hi_vals = [1.0] + [float(np.clip(v, 0.0, 1.0)) for v in ci.iloc[:, 1].tolist()]

        # 主曲线
        series.append({
            "name": label,
            "type": "line",
            "step": "end",
            "data": [[t, round(s, 5)] for t, s in zip(times, sf_vals)],
            "lineStyle": {"color": color, "width": 2.5},
            "itemStyle": {"color": color},
            "symbol": "none",
            "z": 3,
        })
        legend_data.append(label)

        # CI 阴影：用 stack 方式（下界 + 差值叠加）
        stack_key = f"CI_{label}"
        series.append({
            "name": f"_{label}_ci_base",
            "type": "line",
            "step": "end",
            "data": [[t, round(lo, 5)] for t, lo in zip(times, ci_lo_vals)],
            "lineStyle": {"width": 0},
            "itemStyle": {"color": color},
            "symbol": "none",
            "stack": stack_key,
            "areaStyle": {"opacity": 0},
            "showInLegend": False,
            "legendHoverLink": False,
        })
        series.append({
            "name": f"_{label}_ci_band",
            "type": "line",
            "step": "end",
            "data": [
                [t, round(hi - lo, 5)]
                for t, lo, hi in zip(times, ci_lo_vals, ci_hi_vals)
            ],
            "lineStyle": {"width": 0},
            "itemStyle": {"color": color},
            "symbol": "none",
            "stack": stack_key,
            "areaStyle": {"color": color, "opacity": 0.15},
            "showInLegend": False,
            "legendHoverLink": False,
        })

        # 删失点（散点 "+" 形状）
        censored_pts: list[list[float]] = []
        for t_i, e_i in zip(T_arr, E_arr):
            if int(e_i) == 0:
                try:
                    sv = float(
                        kmf.survival_function_at_times([float(t_i)]).values[0]
                    )
                    sv = float(np.clip(sv, 0.0, 1.0))
                    censored_pts.append([float(t_i), round(sv, 5)])
                except Exception:
                    pass
        if censored_pts:
            series.append({
                "name": f"_{label}_censored",
                "type": "scatter",
                "data": censored_pts,
                "symbol": "path://M-5,0 L5,0 M0,-5 L0,5",
                "symbolSize": 9,
                "itemStyle": {"color": color, "opacity": 0.85},
                "showInLegend": False,
                "legendHoverLink": False,
                "z": 4,
            })

    nar_data = _build_number_at_risk(km_data)

    option: dict[str, Any] = {
        "title": {
            "text": "Kaplan-Meier 生存曲线",
            "left": "center",
            "textStyle": {"fontSize": 14},
        },
        "tooltip": {"trigger": "axis"},
        "legend": {"data": legend_data, "top": "28px"},
        "grid": {"left": "10%", "right": "5%", "top": "14%", "bottom": "18%"},
        "xAxis": {
            "type": "value",
            "name": "时间",
            "nameLocation": "middle",
            "nameGap": 28,
            "min": 0,
        },
        "yAxis": {
            "type": "value",
            "name": "生存概率 S(t)",
            "min": 0,
            "max": 1,
            "axisLabel": {"formatter": "{value}"},
        },
        "series": series,
        "numberAtRisk": nar_data,
    }
    return ChartResult(title="KM 生存曲线", chart_type="kaplan_meier", option=option)


def _build_number_at_risk(km_data: dict) -> list[dict]:
    """计算各时间点 number at risk，供前端展示在图表下方。"""
    all_times: set[float] = set()
    for d in km_data.values():
        all_times.update(float(t) for t in d["T"])
    sorted_times = sorted(all_times)
    if len(sorted_times) > 10:
        idxs = np.linspace(0, len(sorted_times) - 1, 10, dtype=int)
        tick_times = [sorted_times[i] for i in idxs]
    else:
        tick_times = sorted_times

    nar: list[dict] = []
    for label, d in km_data.items():
        T_arr = d["T"]
        counts = [int((T_arr >= t).sum()) for t in tick_times]
        nar.append({"group": label, "times": tick_times, "counts": counts})
    return nar


# ─────────────────────────────────────────────────────────────────────────────
# 累积风险函数图
# ─────────────────────────────────────────────────────────────────────────────

def _build_cumulative_hazard_chart(km_data: dict, group_col: str | None) -> ChartResult:
    series: list[dict] = []
    legend_data: list[str] = []

    for color_idx, (label, d) in enumerate(km_data.items()):
        color = _PALETTE[color_idx % len(_PALETTE)]
        kmf = d["kmf"]
        sf = kmf.survival_function_
        times = [0.0] + sf.index.tolist()
        sf_raw = [1.0] + sf.iloc[:, 0].tolist()
        ch_vals = []
        for sv in sf_raw:
            sv_c = float(np.clip(sv, 1e-10, 1.0))
            ch = -np.log(sv_c)
            ch_vals.append(round(float(np.clip(ch, 0.0, 1e6)), 5))

        series.append({
            "name": label,
            "type": "line",
            "step": "end",
            "data": [[t, h] for t, h in zip(times, ch_vals)],
            "lineStyle": {"color": color, "width": 2.5},
            "itemStyle": {"color": color},
            "symbol": "none",
        })
        legend_data.append(label)

    option: dict[str, Any] = {
        "title": {
            "text": "累积风险函数  H(t) = −log S(t)",
            "left": "center",
            "textStyle": {"fontSize": 14},
        },
        "tooltip": {"trigger": "axis"},
        "legend": {"data": legend_data, "top": "28px"},
        "grid": {"left": "10%", "right": "5%", "top": "14%", "bottom": "10%"},
        "xAxis": {
            "type": "value",
            "name": "时间",
            "nameLocation": "middle",
            "nameGap": 28,
            "min": 0,
        },
        "yAxis": {"type": "value", "name": "累积风险 H(t)", "min": 0},
        "series": series,
    }
    return ChartResult(title="累积风险函数", chart_type="line", option=option)


# ─────────────────────────────────────────────────────────────────────────────
# 格式化辅助
# ─────────────────────────────────────────────────────────────────────────────

def _fmt_p(p: float) -> str:
    if p < 0.001:
        return "< 0.001"
    return f"{p:.3f}"
