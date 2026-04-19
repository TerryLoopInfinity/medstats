"""临床预测模型模块。

支持：
  - Logistic 回归 / Cox 回归预测模型
  - 向后逐步变量筛选（AIC-based，可选）
  - 区分度：AUC + DeLong CI（logistic）；C-index + 时间依赖 AUC（Cox）
  - 校准度：校准曲线、Hosmer-Lemeshow（logistic）；GND 十分位（cox）
  - 内部验证：Bootstrap / Split / K-fold 交叉验证
  - Nomogram 数据（分值刻度，供前端 ECharts 自定义渲染）
  - 决策曲线分析（DCA）
"""

import logging
import math
import warnings as _warnings_mod
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats

from app.models.analysis import AnalysisResult, ChartResult, TableResult

logger = logging.getLogger(__name__)

_CLIP = 500.0
_PALETTE = [
    "#5470c6", "#ee6666", "#91cc75", "#fac858",
    "#73c0de", "#3ba272", "#fc8452", "#9a60b4",
]


# ─────────────────────────────────────────────────────────────────────────────
# 溢出安全 exp
# ─────────────────────────────────────────────────────────────────────────────

def _safe_exp(x: float) -> float:
    return float(np.exp(np.clip(float(x), -_CLIP, _CLIP)))


# ─────────────────────────────────────────────────────────────────────────────
# 主入口
# ─────────────────────────────────────────────────────────────────────────────

def run(df: pd.DataFrame, params: dict) -> AnalysisResult:
    """
    参数
    ----
    model_type        str           "logistic" | "cox"
    outcome           str           因变量（logistic）
    time_col          str           时间变量（cox）
    event_col         str           事件变量（cox）
    predictors        list[str]     预测因子
    categorical_vars  list[str]     分类变量
    ref_categories    dict[str,str] 参考组
    validation        str           "internal_bootstrap" | "split" | "cross_validation"
    n_bootstrap       int           Bootstrap 次数（默认 1000）
    train_ratio       float         训练集比例（split 时，默认 0.7）
    time_point        float|None    Cox 预测时间点
    stepwise          bool          是否向后逐步筛选
    """
    model_type: str = str(params.get("model_type", "logistic")).lower()
    predictors: list[str] = list(params.get("predictors") or [])
    categorical_vars: list[str] = list(params.get("categorical_vars") or [])
    ref_categories: dict[str, str] = dict(params.get("ref_categories") or {})
    validation: str = str(params.get("validation", "internal_bootstrap")).lower()
    n_bootstrap: int = int(params.get("n_bootstrap", 1000))
    train_ratio: float = float(params.get("train_ratio", 0.7))
    time_point: float | None = params.get("time_point")
    stepwise: bool = bool(params.get("stepwise", False))
    warnings: list[str] = []

    if model_type not in ("logistic", "cox"):
        raise ValueError("model_type 必须为 logistic 或 cox")
    if not predictors:
        raise ValueError("请至少选择一个预测因子 (predictors)")

    if model_type == "logistic":
        return _run_logistic(
            df, params, predictors, categorical_vars, ref_categories,
            validation, n_bootstrap, train_ratio, stepwise, warnings,
        )
    return _run_cox(
        df, params, predictors, categorical_vars, ref_categories,
        validation, n_bootstrap, train_ratio, time_point, stepwise, warnings,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Logistic 预测模型
# ─────────────────────────────────────────────────────────────────────────────

def _run_logistic(
    df: pd.DataFrame,
    params: dict,
    predictors: list[str],
    categorical_vars: list[str],
    ref_categories: dict[str, str],
    validation: str,
    n_bootstrap: int,
    train_ratio: float,
    stepwise: bool,
    warnings: list[str],
) -> AnalysisResult:
    import statsmodels.api as sm

    outcome: str = str(params.get("outcome", ""))
    if not outcome:
        raise ValueError("请指定因变量 (outcome)")
    if outcome not in df.columns:
        raise ValueError(f"因变量 '{outcome}' 不存在")

    y_encoded, pos_label, enc_warns = _encode_outcome(df[outcome])
    warnings.extend(enc_warns)

    valid_preds, _ = _filter_predictors(predictors, df, outcome, warnings)
    if not valid_preds:
        raise ValueError("没有有效的预测因子")

    valid_cat = [v for v in categorical_vars if v in valid_preds]
    df_work, dummy_map, dum_warns = _make_dummies(df, valid_preds, valid_cat, ref_categories)
    warnings.extend(dum_warns)

    all_cols, cont_vars, expand_warns = _expand_cols(valid_preds, dummy_map, df_work, warnings)
    warnings.extend(expand_warns)
    if not all_cols:
        raise ValueError("没有可用的预测变量")

    df_work = df_work.copy()
    df_work["__y__"] = y_encoded
    needed = ["__y__"] + all_cols
    df_clean = df_work[needed].dropna().reset_index(drop=True)
    y = df_clean["__y__"].values.astype(float)
    n = len(y)

    if n < len(all_cols) + 10:
        raise ValueError(f"有效样本量 ({n}) 不足以拟合模型")

    final_cols = all_cols
    if stepwise:
        final_cols, step_warns = _backward_aic_logistic(df_clean, y, all_cols)
        warnings.extend(step_warns)
        if not final_cols:
            warnings.append("逐步筛选移除了所有变量，恢复使用全部变量")
            final_cols = all_cols

    X = sm.add_constant(df_clean[final_cols].values.astype(float), prepend=True)
    with _warnings_mod.catch_warnings():
        _warnings_mod.simplefilter("ignore")
        model = sm.Logit(y, X).fit(disp=0, maxiter=300)

    y_pred = np.asarray(model.predict(X), dtype=float)
    auc, auc_lo, auc_hi = _delong_auc_ci(y, y_pred)

    coef_rows = _logistic_coef_rows(model, final_cols, dummy_map, valid_preds, cont_vars)

    hl_stat, hl_p, hl_df = _hosmer_lemeshow(y, y_pred)
    cal_slope, cal_intercept, cal_oe = _calibration_stats(y, y_pred)
    fit_rows: list[list] = [
        ["样本量 n", str(n)],
        ["事件数", str(int(np.sum(y)))],
        ["AUC（原始）", f"{auc:.4f}"],
        ["AUC 95% CI（DeLong）", f"[{auc_lo:.4f}, {auc_hi:.4f}]"],
        ["Hosmer-Lemeshow χ²", f"{hl_stat:.3f}"],
        [f"H-L p 值（df={hl_df}）", _fmt_p(hl_p)],
        ["校准斜率", f"{cal_slope:.4f}" if not math.isnan(cal_slope) else "—"],
        ["校准截距", f"{cal_intercept:.4f}" if not math.isnan(cal_intercept) else "—"],
        ["O/E 比", f"{cal_oe:.4f}" if not math.isnan(cal_oe) else "—"],
        ["AIC", f"{float(model.aic):.2f}"],
    ]

    val_result = _validate_logistic(df_clean, y, final_cols, validation, n_bootstrap, train_ratio)
    warnings.extend(val_result.get("warnings", []))

    val_rows: list[list] = [
        ["验证方式", val_result["method_label"]],
        ["原始 AUC", f"{auc:.4f}"],
        ["校正后 AUC", f"{val_result['corrected_auc']:.4f}" if not math.isnan(val_result['corrected_auc']) else "—"],
        ["Optimism（过拟合程度）", f"{val_result['optimism']:.4f}" if not math.isnan(val_result['optimism']) else "—"],
    ]
    if "cv_aucs" in val_result and val_result["cv_aucs"]:
        cv_mean = float(np.mean(val_result["cv_aucs"]))
        cv_sd = float(np.std(val_result["cv_aucs"]))
        val_rows.append(["CV 平均 AUC ± SD", f"{cv_mean:.4f} ± {cv_sd:.4f}"])

    nomo_data = _nomogram_logistic(model, final_cols, dummy_map, valid_preds, df_clean)
    dca = _dca_data(y, y_pred)

    charts: list[ChartResult] = []
    charts.append(_build_roc_chart(y, y_pred, auc, auc_lo, auc_hi))
    calib_chart = _build_calibration_chart_logistic(y, y_pred)
    if calib_chart:
        charts.append(calib_chart)
    if nomo_data:
        charts.append(ChartResult(
            title="列线图（Nomogram）",
            chart_type="nomogram",
            option={"nomogramData": nomo_data},
        ))
    charts.append(_build_dca_chart(dca))
    boot_vals = val_result.get("boot_aucs", [])
    if boot_vals:
        charts.append(_build_bootstrap_hist(
            boot_vals, auc, val_result["corrected_auc"]
        ))

    corrected_str = (f"{val_result['corrected_auc']:.4f}"
                     if not math.isnan(val_result['corrected_auc']) else "—")
    summary = (
        f"Logistic 临床预测模型，因变量：{outcome}（阳性：{pos_label}），"
        f"纳入 {len(final_cols)} 个预测变量，有效样本 n = {n}，"
        f"AUC = {auc:.4f}，校正后 AUC = {corrected_str}。"
    )
    if hl_p < 0.05:
        warnings.append(f"Hosmer-Lemeshow 检验 p = {_fmt_p(hl_p)}，校准度可能不佳")

    return AnalysisResult(
        method="prediction",
        tables=[
            TableResult(title="预测模型系数表",
                        headers=["变量", "β", "SE", "OR", "95% CI", "p 值"],
                        rows=coef_rows),
            TableResult(title="模型拟合与区分度",
                        headers=["指标", "值"],
                        rows=fit_rows),
            TableResult(title="内部验证结果",
                        headers=["指标", "值"],
                        rows=val_rows),
        ],
        charts=charts,
        summary=summary,
        warnings=warnings,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Cox 预测模型
# ─────────────────────────────────────────────────────────────────────────────

def _run_cox(
    df: pd.DataFrame,
    params: dict,
    predictors: list[str],
    categorical_vars: list[str],
    ref_categories: dict[str, str],
    validation: str,
    n_bootstrap: int,
    train_ratio: float,
    time_point: float | None,
    stepwise: bool,
    warnings: list[str],
) -> AnalysisResult:
    try:
        from lifelines import CoxPHFitter
    except ImportError as exc:
        raise RuntimeError("lifelines 未安装") from exc

    time_col: str = str(params.get("time_col", ""))
    event_col: str = str(params.get("event_col", ""))
    if not time_col:
        raise ValueError("Cox 模型请指定时间变量 (time_col)")
    if not event_col:
        raise ValueError("Cox 模型请指定事件变量 (event_col)")
    for col in [time_col, event_col]:
        if col not in df.columns:
            raise ValueError(f"列 '{col}' 不存在")

    valid_preds, _ = _filter_predictors(
        predictors, df, "", warnings, exclude=[time_col, event_col]
    )
    if not valid_preds:
        raise ValueError("没有有效的预测因子")

    valid_cat = [v for v in categorical_vars if v in valid_preds]
    df_work, dummy_map, dum_warns = _make_dummies_cox(df, valid_preds, valid_cat, ref_categories)
    warnings.extend(dum_warns)

    all_cols, _, _ = _expand_cols_cox(valid_preds, dummy_map, df_work, warnings)
    if not all_cols:
        raise ValueError("没有可用的预测变量")

    needed = [time_col, event_col] + all_cols
    df_work2 = df_work[needed].dropna()
    df_work2 = df_work2[pd.to_numeric(df_work2[time_col], errors="coerce") > 0]
    df_work2 = df_work2[df_work2[event_col].isin([0, 1])].reset_index(drop=True)

    n = len(df_work2)
    n_events = int(df_work2[event_col].sum())
    if n < 20:
        raise ValueError(f"有效样本量不足 ({n})，无法拟合 Cox 预测模型")
    if n_events < 10:
        raise ValueError(f"事件数不足 ({n_events})，无法拟合 Cox 预测模型")

    final_cols = all_cols
    if stepwise:
        final_cols, step_warns = _backward_aic_cox(df_work2, time_col, event_col, all_cols)
        warnings.extend(step_warns)
        if not final_cols:
            warnings.append("逐步筛选移除了所有变量，恢复全部变量")
            final_cols = all_cols

    fit_cols = [time_col, event_col] + final_cols
    df_fit = df_work2[fit_cols].dropna().reset_index(drop=True)
    cph = CoxPHFitter(penalizer=0.01)
    with _warnings_mod.catch_warnings():
        _warnings_mod.simplefilter("ignore")
        cph.fit(df_fit, duration_col=time_col, event_col=event_col)

    cindex = float(np.clip(float(cph.concordance_index_), 0.0, 1.0))

    if time_point is None:
        time_point = float(np.median(df_fit[time_col]))
        warnings.append(f"未指定预测时间点，使用时间中位数 {time_point:.1f}")

    bs0 = _baseline_survival_at(cph, df_fit, time_point)

    col_to_orig: dict[str, tuple[str, str]] = {}
    for v, cols in dummy_map.items():
        for c in cols:
            cat = c.split("__", 1)[-1]
            col_to_orig[c] = (v, cat)

    coef_rows = _cox_coef_rows(cph, final_cols, dummy_map, valid_preds, col_to_orig)

    risk_score = np.asarray(cph.predict_log_partial_hazard(df_fit[final_cols]), dtype=float)
    time_pts_for_auc = _compute_time_points(df_fit[time_col], df_fit[event_col], time_point)
    tdep_auc_list = _time_dep_auc(df_fit, time_col, event_col, risk_score, time_pts_for_auc)

    pred_surv = _cox_pred_surv(cph, df_fit[final_cols], time_point)
    cal_rows_cox = _cox_calibration(df_fit, time_col, event_col, pred_surv, time_point)

    val_result = _validate_cox(
        df_fit, time_col, event_col, final_cols, cph,
        validation, n_bootstrap, train_ratio, time_point,
    )
    warnings.extend(val_result.get("warnings", []))

    try:
        aic = float(cph.AIC_partial_)
    except Exception:
        aic = float("nan")
    try:
        llr_p = float(cph.log_likelihood_ratio_test().p_value)
    except Exception:
        llr_p = float("nan")

    fit_rows: list[list] = [
        ["样本量 n", str(n)],
        ["事件数", str(n_events)],
        ["C-index（原始）", f"{cindex:.4f}"],
        ["C-index（校正后）", f"{val_result['corrected_cindex']:.4f}" if not math.isnan(val_result['corrected_cindex']) else "—"],
        ["Optimism", f"{val_result['optimism']:.4f}" if not math.isnan(val_result['optimism']) else "—"],
        ["预测时间点", f"{time_point:.1f}"],
        ["基线生存率 S0(t)", f"{bs0:.4f}" if not math.isnan(bs0) else "—"],
        ["AIC", f"{aic:.2f}" if not math.isnan(aic) else "—"],
        ["LRT p 值", _fmt_p(llr_p) if not math.isnan(llr_p) else "—"],
    ]

    val_rows: list[list] = [
        ["验证方式", val_result["method_label"]],
        ["原始 C-index", f"{cindex:.4f}"],
        ["校正后 C-index", f"{val_result['corrected_cindex']:.4f}" if not math.isnan(val_result['corrected_cindex']) else "—"],
        ["Optimism", f"{val_result['optimism']:.4f}" if not math.isnan(val_result['optimism']) else "—"],
    ]

    nomo_data = _nomogram_cox(cph, final_cols, dummy_map, valid_preds, df_fit, time_point, bs0)
    y_event = df_fit[event_col].values.astype(float)
    dca = _dca_data(y_event, 1.0 - pred_surv)

    charts: list[ChartResult] = []
    if tdep_auc_list and len(tdep_auc_list) >= 2:
        charts.append(_build_time_dep_auc_chart(tdep_auc_list, cindex))
    charts.append(_build_cox_calibration_chart(cal_rows_cox, time_point))
    if nomo_data:
        charts.append(ChartResult(
            title="列线图（Nomogram）",
            chart_type="nomogram",
            option={"nomogramData": nomo_data},
        ))
    charts.append(_build_dca_chart(dca))
    boot_vals = val_result.get("boot_cindices", [])
    if boot_vals:
        charts.append(_build_bootstrap_hist(
            boot_vals, cindex, val_result["corrected_cindex"], label="C-index"
        ))

    corrected_ci_str = (f"{val_result['corrected_cindex']:.4f}"
                        if not math.isnan(val_result["corrected_cindex"]) else "—")
    summary = (
        f"Cox 临床预测模型，时间变量：{time_col}，事件变量：{event_col}，"
        f"纳入 {len(final_cols)} 个预测变量，有效样本 n = {n}，事件 {n_events} 例，"
        f"C-index = {cindex:.4f}，校正后 C-index = {corrected_ci_str}。"
    )

    return AnalysisResult(
        method="prediction",
        tables=[
            TableResult(title="预测模型系数表",
                        headers=["变量", "β", "SE", "HR", "95% CI", "p 值"],
                        rows=coef_rows),
            TableResult(title="模型拟合与区分度",
                        headers=["指标", "值"],
                        rows=fit_rows),
            TableResult(title="校准度（预测概率十分位）",
                        headers=["分组", "预测事件率", "实际事件率（KM）", "例数"],
                        rows=cal_rows_cox),
            TableResult(title="内部验证结果",
                        headers=["指标", "值"],
                        rows=val_rows),
        ],
        charts=charts,
        summary=summary,
        warnings=warnings,
    )


# ─────────────────────────────────────────────────────────────────────────────
# 数据预处理
# ─────────────────────────────────────────────────────────────────────────────

def _encode_outcome(col: pd.Series) -> tuple[pd.Series, str, list[str]]:
    warns: list[str] = []
    nn = col.dropna()
    if len(nn) == 0:
        raise ValueError("因变量全为缺失值")
    if pd.api.types.is_numeric_dtype(col):
        uv = sorted(float(v) for v in nn.unique())
        if len(uv) > 2:
            raise ValueError("因变量有超过两个唯一值，请确认为二分类变量")
        if len(uv) == 1:
            raise ValueError("因变量只有一个唯一值")
        if set(uv) == {0.0, 1.0}:
            return col.astype(float), "1", warns
        lo, hi = uv
        warns.append(f"因变量数值编码：{lo}→0，{hi}→1")
        return (
            col.map(lambda x: {lo: 0.0, hi: 1.0}.get(float(x)) if pd.notna(x) else float("nan")),
            str(hi), warns,
        )
    col_s = col.where(col.isna(), col.astype(str))
    uv_s = sorted(col_s.dropna().unique())
    if len(uv_s) > 2:
        raise ValueError("因变量有超过两个唯一值")
    if len(uv_s) == 1:
        raise ValueError("因变量只有一个唯一值")
    if set(uv_s) == {"False", "True"}:
        return col_s.map({"False": 0.0, "True": 1.0}), "True", warns
    neg, pos = uv_s[0], uv_s[1]
    warns.append(f"因变量编码：'{neg}'→0，'{pos}'→1")
    return col_s.map({neg: 0.0, pos: 1.0}), pos, warns


def _filter_predictors(
    predictors: list[str],
    df: pd.DataFrame,
    outcome: str,
    warnings: list[str],
    exclude: list[str] | None = None,
) -> tuple[list[str], list[str]]:
    exclude_set = set(exclude or []) | ({outcome} if outcome else set())
    valid, skipped = [], []
    for v in predictors:
        if v in exclude_set:
            warnings.append(f"'{v}' 与结局/时间变量重叠，已忽略")
            skipped.append(v)
        elif v not in df.columns:
            warnings.append(f"'{v}' 不存在，已忽略")
            skipped.append(v)
        else:
            valid.append(v)
    return valid, skipped


def _make_dummies(
    df: pd.DataFrame,
    predictors: list[str],
    categorical_vars: list[str],
    ref_categories: dict[str, str],
) -> tuple[pd.DataFrame, dict[str, list[str]], list[str]]:
    warns: list[str] = []
    dummy_map: dict[str, list[str]] = {}
    df_out = df.copy()
    for var in categorical_vars:
        if var not in df.columns:
            continue
        nn_mask = df_out[var].notna()
        col_str = df_out.loc[nn_mask, var].astype(str)
        cats = sorted(col_str.unique())
        if len(cats) < 2:
            warns.append(f"'{var}' 唯一值不足，跳过 dummy 编码")
            continue
        ref = str(ref_categories.get(var, cats[0]))
        if ref not in cats:
            ref = cats[0]
        ordered = [ref] + [c for c in cats if c != ref]
        cat_col = pd.Categorical(col_str, categories=ordered)
        dummies = pd.get_dummies(
            pd.Series(cat_col, index=col_str.index), prefix=var, drop_first=True, dtype=float
        )
        dummy_full = pd.DataFrame(np.nan, index=df_out.index, columns=dummies.columns)
        dummy_full.loc[nn_mask] = dummies.values
        for cn in dummy_full.columns:
            df_out[cn] = dummy_full[cn]
        dummy_map[var] = list(dummies.columns)
        warns.append(f"'{var}' dummy 编码，参考组：'{ref}'")
    return df_out, dummy_map, warns


def _make_dummies_cox(
    df: pd.DataFrame,
    predictors: list[str],
    categorical_vars: list[str],
    ref_categories: dict[str, str],
) -> tuple[pd.DataFrame, dict[str, list[str]], list[str]]:
    warns: list[str] = []
    dummy_map: dict[str, list[str]] = {}
    df_out = df.copy()
    for var in categorical_vars:
        if var not in df.columns or var not in predictors:
            continue
        df_out[var] = df_out[var].astype(str)
        cats = sorted(df_out[var].dropna().unique().tolist())
        if len(cats) < 2:
            warns.append(f"'{var}' 唯一值不足，跳过")
            continue
        ref = str(ref_categories.get(var, cats[0]))
        if ref not in cats:
            ref = cats[0]
        non_ref = [c for c in cats if c != ref]
        for c in non_ref:
            col_name = f"{var}__{c}"
            df_out[col_name] = (df_out[var] == c).astype(float)
            dummy_map.setdefault(var, []).append(col_name)
    return df_out, dummy_map, warns


def _expand_cols(
    predictors: list[str],
    dummy_map: dict[str, list[str]],
    df_work: pd.DataFrame,
    warnings: list[str],
) -> tuple[list[str], list[str], list[str]]:
    all_cols, cont_vars, warns = [], [], []
    for v in predictors:
        if v in dummy_map:
            all_cols.extend(dummy_map[v])
        elif pd.api.types.is_numeric_dtype(df_work[v]):
            all_cols.append(v)
            cont_vars.append(v)
        else:
            warns.append(f"'{v}' 非数值型且未标记分类变量，已忽略")
    return all_cols, cont_vars, warns


def _expand_cols_cox(
    predictors: list[str],
    dummy_map: dict[str, list[str]],
    df_work: pd.DataFrame,
    warnings: list[str],
) -> tuple[list[str], list[str], list[str]]:
    return _expand_cols(predictors, dummy_map, df_work, warnings)


# ─────────────────────────────────────────────────────────────────────────────
# 向后逐步筛选（AIC-based）
# ─────────────────────────────────────────────────────────────────────────────

def _backward_aic_logistic(
    df_clean: pd.DataFrame,
    y: np.ndarray,
    all_cols: list[str],
) -> tuple[list[str], list[str]]:
    import statsmodels.api as sm

    warns: list[str] = []
    current = list(all_cols)

    def _aic(cols: list[str]) -> float:
        X_ = sm.add_constant(df_clean[cols].values.astype(float), prepend=True)
        with _warnings_mod.catch_warnings():
            _warnings_mod.simplefilter("ignore")
            m_ = sm.Logit(y, X_).fit(disp=0, maxiter=200)
        return float(m_.aic)

    try:
        best_aic = _aic(current)
    except Exception:
        return current, warns

    improved = True
    while improved and len(current) > 1:
        improved = False
        best_drop = None
        for col in current:
            candidate = [c for c in current if c != col]
            try:
                aic_ = _aic(candidate)
                if aic_ < best_aic - 1e-4:
                    best_aic = aic_
                    best_drop = col
            except Exception:
                continue
        if best_drop is not None:
            current.remove(best_drop)
            warns.append(f"逐步筛选移除变量：'{best_drop}'（AIC → {best_aic:.2f}）")
            improved = True

    return current, warns


def _backward_aic_cox(
    df_fit: pd.DataFrame,
    time_col: str,
    event_col: str,
    all_cols: list[str],
) -> tuple[list[str], list[str]]:
    from lifelines import CoxPHFitter

    warns: list[str] = []
    current = list(all_cols)

    def _aic(cols: list[str]) -> float:
        sub = df_fit[[time_col, event_col] + cols].dropna()
        cph = CoxPHFitter(penalizer=0.01)
        with _warnings_mod.catch_warnings():
            _warnings_mod.simplefilter("ignore")
            cph.fit(sub, duration_col=time_col, event_col=event_col)
        try:
            return float(cph.AIC_partial_)
        except Exception:
            return float("inf")

    try:
        best_aic = _aic(current)
    except Exception:
        return current, warns

    improved = True
    while improved and len(current) > 1:
        improved = False
        best_drop = None
        for col in current:
            candidate = [c for c in current if c != col]
            try:
                aic_ = _aic(candidate)
                if aic_ < best_aic - 1e-4:
                    best_aic = aic_
                    best_drop = col
            except Exception:
                continue
        if best_drop is not None:
            current.remove(best_drop)
            warns.append(f"逐步筛选移除变量：'{best_drop}'（AIC → {best_aic:.2f}）")
            improved = True

    return current, warns


# ─────────────────────────────────────────────────────────────────────────────
# 系数表
# ─────────────────────────────────────────────────────────────────────────────

def _logistic_coef_rows(
    model,
    final_cols: list[str],
    dummy_map: dict[str, list[str]],
    orig_preds: list[str],
    cont_vars: list[str],
) -> list[list]:
    rows: list[list] = []
    ci_arr = np.asarray(model.conf_int())
    col_to_idx = {c: i + 1 for i, c in enumerate(final_cols)}
    processed: set[str] = set()

    for v in orig_preds:
        if v in dummy_map:
            if v in processed:
                continue
            processed.add(v)
            dummy_cols = [c for c in dummy_map[v] if c in col_to_idx]
            if not dummy_cols:
                continue
            rows.append([f"{v}（参考组）", "", "", "1 (Ref)", "", ""])
            for dc in dummy_cols:
                idx = col_to_idx[dc]
                beta = float(model.params[idx])
                se = float(model.bse[idx])
                p = float(model.pvalues[idx])
                or_v = _safe_exp(beta)
                lo = _safe_exp(float(ci_arr[idx, 0]))
                hi = _safe_exp(float(ci_arr[idx, 1]))
                rows.append([f"  {dc[len(v) + 1:]}", f"{beta:.4f}", f"{se:.4f}",
                              f"{or_v:.3f}", f"({lo:.3f}, {hi:.3f})", _fmt_p(p)])
        elif v in cont_vars and v in col_to_idx:
            idx = col_to_idx[v]
            beta = float(model.params[idx])
            se = float(model.bse[idx])
            p = float(model.pvalues[idx])
            or_v = _safe_exp(beta)
            lo = _safe_exp(float(ci_arr[idx, 0]))
            hi = _safe_exp(float(ci_arr[idx, 1]))
            rows.append([v, f"{beta:.4f}", f"{se:.4f}",
                         f"{or_v:.3f}", f"({lo:.3f}, {hi:.3f})", _fmt_p(p)])
    return rows


def _cox_coef_rows(
    cph,
    final_cols: list[str],
    dummy_map: dict[str, list[str]],
    orig_preds: list[str],
    col_to_orig: dict[str, tuple[str, str]],
) -> list[list]:
    rows: list[list] = []
    emitted: set[str] = set()
    summ = cph.summary

    for col in final_cols:
        if col not in summ.index:
            continue
        row = summ.loc[col]
        beta = float(row.get("coef", 0.0))
        se = float(row.get("se(coef)", 0.0))
        p = float(row.get("p", 1.0))
        beta_c = float(np.clip(beta, -_CLIP, _CLIP))
        hr = float(np.exp(beta_c))
        ci_lo_keys = [k for k in row.index if "lower" in k.lower()]
        ci_hi_keys = [k for k in row.index if "upper" in k.lower()]
        if ci_lo_keys and ci_hi_keys:
            lo_b = float(np.clip(float(row[ci_lo_keys[0]]), -_CLIP, _CLIP))
            hi_b = float(np.clip(float(row[ci_hi_keys[0]]), -_CLIP, _CLIP))
            hr_lo, hr_hi = float(np.exp(lo_b)), float(np.exp(hi_b))
        else:
            hr_lo = float(np.exp(float(np.clip(beta - 1.96 * se, -_CLIP, _CLIP))))
            hr_hi = float(np.exp(float(np.clip(beta + 1.96 * se, -_CLIP, _CLIP))))

        if col in col_to_orig:
            orig_v, cat = col_to_orig[col]
            if orig_v not in emitted:
                rows.append([orig_v, "", "", "", "", ""])
                emitted.add(orig_v)
            label = f"  {cat}"
        else:
            label = col

        rows.append([label, f"{beta:.4f}", f"{se:.4f}",
                     f"{hr:.3f}", f"({hr_lo:.3f}, {hr_hi:.3f})", _fmt_p(p)])
    return rows


# ─────────────────────────────────────────────────────────────────────────────
# 区分度
# ─────────────────────────────────────────────────────────────────────────────

def _delong_auc_ci(
    y_true: np.ndarray,
    y_score: np.ndarray,
    alpha: float = 0.05,
) -> tuple[float, float, float]:
    from sklearn.metrics import roc_auc_score

    n_pos = int(np.sum(y_true == 1))
    n_neg = int(np.sum(y_true == 0))
    if n_pos == 0 or n_neg == 0:
        return 0.5, 0.0, 1.0

    auc = float(roc_auc_score(y_true, y_score))
    pos_sc = y_score[y_true == 1]
    neg_sc = y_score[y_true == 0]
    v10 = np.array([float(np.mean(neg_sc < s) + 0.5 * np.mean(neg_sc == s)) for s in pos_sc])
    v01 = np.array([float(np.mean(pos_sc > s) + 0.5 * np.mean(pos_sc == s)) for s in neg_sc])
    var_auc = np.var(v10, ddof=1) / n_pos + np.var(v01, ddof=1) / n_neg
    se = math.sqrt(max(var_auc, 0.0))
    z = float(stats.norm.ppf(1.0 - alpha / 2.0))
    return auc, max(0.0, auc - z * se), min(1.0, auc + z * se)


def _compute_time_points(
    time_series: pd.Series, event_series: pd.Series, center: float
) -> list[float]:
    q25 = float(np.quantile(time_series, 0.25))
    q50 = float(np.quantile(time_series, 0.5))
    q75 = float(np.quantile(time_series, 0.75))
    pts = sorted({round(q25, 1), round(q50, 1), round(q75, 1), round(center, 1)})
    return [t for t in pts if t > 0]


def _time_dep_auc(
    df_fit: pd.DataFrame,
    time_col: str,
    event_col: str,
    risk_score: np.ndarray,
    time_pts: list[float],
) -> list[dict]:
    results = []
    t_vals = df_fit[time_col].values
    e_vals = df_fit[event_col].values
    for t in time_pts:
        cases_mask = (e_vals == 1) & (t_vals <= t)
        controls_mask = t_vals > t
        n_cases = int(np.sum(cases_mask))
        n_controls = int(np.sum(controls_mask))
        if n_cases < 3 or n_controls < 3:
            continue
        rs_cases = risk_score[cases_mask]
        rs_controls = risk_score[controls_mask]
        # Concordance P(risk_case > risk_control)
        greater = float(np.mean(rs_cases[:, None] > rs_controls[None, :]))
        ties = float(np.mean(rs_cases[:, None] == rs_controls[None, :]))
        auc_t = float(np.clip(greater + 0.5 * ties, 0, 1))
        results.append({"time": round(t, 1), "auc": round(auc_t, 4),
                         "n_cases": n_cases, "n_controls": n_controls})
    return results


def _baseline_survival_at(cph, df_fit: pd.DataFrame, time_point: float) -> float:
    try:
        bsf = cph.baseline_survival_
        times = bsf.index.values
        vals = bsf.iloc[:, 0].values
        if time_point <= times[0]:
            return float(vals[0])
        if time_point >= times[-1]:
            return float(vals[-1])
        idx = int(np.searchsorted(times, time_point, side="right")) - 1
        return float(vals[idx])
    except Exception:
        return float("nan")


def _cox_pred_surv(cph, X_df: pd.DataFrame, time_point: float) -> np.ndarray:
    try:
        with _warnings_mod.catch_warnings():
            _warnings_mod.simplefilter("ignore")
            sf = cph.predict_survival_function(X_df, times=[time_point])
        return np.clip(sf.values.flatten(), 0.0, 1.0)
    except Exception:
        return np.full(len(X_df), 0.5)


# ─────────────────────────────────────────────────────────────────────────────
# 校准度
# ─────────────────────────────────────────────────────────────────────────────

def _hosmer_lemeshow(
    y_true: np.ndarray, y_pred: np.ndarray, g: int = 10
) -> tuple[float, float, int]:
    n = len(y_true)
    g = min(g, max(4, n // 4))
    order = np.argsort(y_pred)
    ys, ps = y_true[order], y_pred[order]
    hl = 0.0
    for grp in np.array_split(np.arange(n), g):
        if not len(grp):
            continue
        o1, e1 = float(np.sum(ys[grp])), float(np.sum(ps[grp]))
        o0, e0 = len(grp) - o1, len(grp) - e1
        if e1 > 1e-9:
            hl += (o1 - e1) ** 2 / e1
        if e0 > 1e-9:
            hl += (o0 - e0) ** 2 / e0
    df = max(g - 2, 1)
    return hl, float(stats.chi2.sf(hl, df)), df


def _calibration_stats(
    y_true: np.ndarray, y_pred: np.ndarray
) -> tuple[float, float, float]:
    try:
        import statsmodels.api as sm
        lo = np.log(np.clip(y_pred, 1e-7, 1 - 1e-7) / (1 - np.clip(y_pred, 1e-7, 1 - 1e-7)))
        X = sm.add_constant(lo, prepend=True)
        with _warnings_mod.catch_warnings():
            _warnings_mod.simplefilter("ignore")
            m = sm.Logit(y_true, X).fit(disp=0, maxiter=200)
        slope = float(m.params[1])
        intercept = float(m.params[0])
    except Exception:
        slope, intercept = float("nan"), float("nan")
    mean_pred = float(np.mean(y_pred))
    oe = float(np.mean(y_true)) / mean_pred if mean_pred > 0 else float("nan")
    return slope, intercept, oe


def _cox_calibration(
    df_fit: pd.DataFrame,
    time_col: str,
    event_col: str,
    pred_surv: np.ndarray,
    time_point: float,
    g: int = 10,
) -> list[list]:
    try:
        from lifelines import KaplanMeierFitter

        pred_death = 1.0 - pred_surv
        df_cal = df_fit[[time_col, event_col]].copy().reset_index(drop=True)
        df_cal["pred_death"] = pred_death
        df_cal["decile"] = pd.qcut(df_cal["pred_death"], q=g, labels=False, duplicates="drop")

        rows: list[list] = []
        for grp_id in sorted(df_cal["decile"].dropna().unique()):
            sub = df_cal[df_cal["decile"] == grp_id]
            n_grp = len(sub)
            mean_pred = float(sub["pred_death"].mean())
            kmf = KaplanMeierFitter()
            with _warnings_mod.catch_warnings():
                _warnings_mod.simplefilter("ignore")
                kmf.fit(sub[time_col], sub[event_col])
            sf_at_t = kmf.predict(time_point)
            obs_death = float(1.0 - sf_at_t)
            rows.append([f"第{int(grp_id)+1}分位",
                         f"{mean_pred:.4f}", f"{obs_death:.4f}", str(n_grp)])
        return rows
    except Exception as exc:
        logger.warning("Cox 校准计算失败: %s", exc)
        return []


# ─────────────────────────────────────────────────────────────────────────────
# 内部验证
# ─────────────────────────────────────────────────────────────────────────────

def _validate_logistic(
    df_clean: pd.DataFrame,
    y: np.ndarray,
    cols: list[str],
    validation: str,
    n_bootstrap: int,
    train_ratio: float,
) -> dict[str, Any]:
    if validation == "internal_bootstrap":
        return _bootstrap_logistic(df_clean, y, cols, n_bootstrap)
    elif validation == "split":
        return _split_logistic(df_clean, y, cols, train_ratio)
    elif validation == "cross_validation":
        return _cv_logistic(df_clean, y, cols)
    return {"method_label": "无", "corrected_auc": float("nan"),
            "optimism": float("nan"), "warnings": []}


def _bootstrap_logistic(
    df_clean: pd.DataFrame, y: np.ndarray, cols: list[str], n_boot: int
) -> dict[str, Any]:
    import statsmodels.api as sm

    warns: list[str] = []
    n = len(y)
    rng = np.random.default_rng(42)

    X_orig = sm.add_constant(df_clean[cols].values.astype(float), prepend=True)
    with _warnings_mod.catch_warnings():
        _warnings_mod.simplefilter("ignore")
        try:
            m_orig = sm.Logit(y, X_orig).fit(disp=0, maxiter=300)
        except Exception as exc:
            return {"method_label": "Bootstrap 验证", "corrected_auc": float("nan"),
                    "optimism": float("nan"), "boot_aucs": [], "warnings": [str(exc)]}

    orig_pred = np.asarray(m_orig.predict(X_orig), dtype=float)
    orig_auc, _, _ = _delong_auc_ci(y, orig_pred)

    optimisms: list[float] = []
    boot_aucs: list[float] = []

    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        df_b = df_clean.iloc[idx].reset_index(drop=True)
        y_b = y[idx]
        try:
            X_b = sm.add_constant(df_b[cols].values.astype(float), prepend=True)
            with _warnings_mod.catch_warnings():
                _warnings_mod.simplefilter("ignore")
                m_b = sm.Logit(y_b, X_b).fit(disp=0, maxiter=200)
            pred_b = np.asarray(m_b.predict(X_b), dtype=float)
            auc_b, _, _ = _delong_auc_ci(y_b, pred_b)
            pred_test = np.asarray(m_b.predict(X_orig), dtype=float)
            auc_test, _, _ = _delong_auc_ci(y, pred_test)
            optimisms.append(auc_b - auc_test)
            boot_aucs.append(auc_b)
        except Exception:
            continue

    if not optimisms:
        warns.append("Bootstrap 验证未能完成任何迭代")
        return {"method_label": "Bootstrap 验证", "corrected_auc": orig_auc,
                "optimism": 0.0, "boot_aucs": [], "warnings": warns}

    mean_opt = float(np.mean(optimisms))
    corrected = float(np.clip(orig_auc - mean_opt, 0.0, 1.0))
    return {
        "method_label": f"Bootstrap 验证（n={len(optimisms)}）",
        "corrected_auc": corrected,
        "optimism": mean_opt,
        "boot_aucs": boot_aucs,
        "warnings": warns,
    }


def _split_logistic(
    df_clean: pd.DataFrame, y: np.ndarray, cols: list[str], train_ratio: float
) -> dict[str, Any]:
    import statsmodels.api as sm

    n = len(y)
    n_train = int(n * train_ratio)
    rng = np.random.default_rng(42)
    idx = rng.permutation(n)
    train_idx, test_idx = idx[:n_train], idx[n_train:]

    df_train = df_clean.iloc[train_idx].reset_index(drop=True)
    y_train = y[train_idx]
    df_test = df_clean.iloc[test_idx].reset_index(drop=True)
    y_test = y[test_idx]

    try:
        X_tr = sm.add_constant(df_train[cols].values.astype(float), prepend=True)
        X_te = sm.add_constant(df_test[cols].values.astype(float), prepend=True)
        with _warnings_mod.catch_warnings():
            _warnings_mod.simplefilter("ignore")
            m = sm.Logit(y_train, X_tr).fit(disp=0, maxiter=300)
        auc_train, _, _ = _delong_auc_ci(y_train, np.asarray(m.predict(X_tr)))
        auc_test, _, _ = _delong_auc_ci(y_test, np.asarray(m.predict(X_te)))
        return {
            "method_label": f"Split 验证（训练集 {train_ratio*100:.0f}%）",
            "corrected_auc": auc_test,
            "optimism": auc_train - auc_test,
            "warnings": [],
        }
    except Exception as exc:
        return {"method_label": "Split 验证", "corrected_auc": float("nan"),
                "optimism": float("nan"), "warnings": [f"Split 验证失败：{exc}"]}


def _cv_logistic(
    df_clean: pd.DataFrame, y: np.ndarray, cols: list[str], k: int = 5
) -> dict[str, Any]:
    import statsmodels.api as sm

    n = len(y)
    rng = np.random.default_rng(42)
    fold_idx = rng.permutation(n)
    folds = np.array_split(fold_idx, k)
    cv_aucs: list[float] = []
    warns: list[str] = []

    X_all = sm.add_constant(df_clean[cols].values.astype(float), prepend=True)
    orig_auc = float("nan")
    try:
        with _warnings_mod.catch_warnings():
            _warnings_mod.simplefilter("ignore")
            m_full = sm.Logit(y, X_all).fit(disp=0, maxiter=300)
        orig_pred = np.asarray(m_full.predict(X_all), dtype=float)
        orig_auc, _, _ = _delong_auc_ci(y, orig_pred)
    except Exception:
        pass

    for fold in folds:
        test_mask = np.zeros(n, dtype=bool)
        test_mask[fold] = True
        train_mask = ~test_mask
        try:
            X_tr = sm.add_constant(df_clean[cols].values[train_mask].astype(float), prepend=True)
            X_te = sm.add_constant(df_clean[cols].values[test_mask].astype(float), prepend=True)
            with _warnings_mod.catch_warnings():
                _warnings_mod.simplefilter("ignore")
                m_ = sm.Logit(y[train_mask], X_tr).fit(disp=0, maxiter=200)
            auc_, _, _ = _delong_auc_ci(y[test_mask], np.asarray(m_.predict(X_te)))
            cv_aucs.append(auc_)
        except Exception:
            continue

    if not cv_aucs:
        warns.append("交叉验证未完成任何折")
        return {"method_label": f"{k}-fold 交叉验证", "corrected_auc": float("nan"),
                "optimism": float("nan"), "cv_aucs": [], "warnings": warns}

    mean_cv = float(np.mean(cv_aucs))
    optimism = float(orig_auc - mean_cv) if not math.isnan(orig_auc) else float("nan")
    return {
        "method_label": f"{k}-fold 交叉验证",
        "corrected_auc": mean_cv,
        "optimism": optimism,
        "cv_aucs": cv_aucs,
        "warnings": warns,
    }


def _validate_cox(
    df_fit: pd.DataFrame,
    time_col: str,
    event_col: str,
    cols: list[str],
    cph_orig,
    validation: str,
    n_bootstrap: int,
    train_ratio: float,
    time_point: float,
) -> dict[str, Any]:
    if validation == "internal_bootstrap":
        return _bootstrap_cox(df_fit, time_col, event_col, cols, n_bootstrap, time_point)
    elif validation == "split":
        return _split_cox(df_fit, time_col, event_col, cols, train_ratio, time_point)
    elif validation == "cross_validation":
        return _cv_cox(df_fit, time_col, event_col, cols, time_point)
    return {"method_label": "无", "corrected_cindex": float("nan"),
            "optimism": float("nan"), "warnings": []}


def _bootstrap_cox(
    df_fit: pd.DataFrame,
    time_col: str,
    event_col: str,
    cols: list[str],
    n_boot: int,
    time_point: float,
) -> dict[str, Any]:
    from lifelines import CoxPHFitter
    from lifelines.utils import concordance_index

    warns: list[str] = []
    n = len(df_fit)
    rng = np.random.default_rng(42)

    cph_full = CoxPHFitter(penalizer=0.01)
    with _warnings_mod.catch_warnings():
        _warnings_mod.simplefilter("ignore")
        cph_full.fit(df_fit, duration_col=time_col, event_col=event_col)
    orig_cindex = float(np.clip(float(cph_full.concordance_index_), 0.0, 1.0))

    T_orig = df_fit[time_col].values
    E_orig = df_fit[event_col].values
    X_orig = df_fit[cols]

    optimisms: list[float] = []
    boot_cindices: list[float] = []

    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        df_b = df_fit.iloc[idx].reset_index(drop=True)
        try:
            cph_b = CoxPHFitter(penalizer=0.1)
            with _warnings_mod.catch_warnings():
                _warnings_mod.simplefilter("ignore")
                cph_b.fit(df_b, duration_col=time_col, event_col=event_col)
            ph_b = np.asarray(cph_b.predict_log_partial_hazard(df_b[cols]), dtype=float)
            ci_b = float(concordance_index(df_b[time_col], -ph_b, df_b[event_col]))
            ph_test = np.asarray(cph_b.predict_log_partial_hazard(X_orig), dtype=float)
            ci_test = float(concordance_index(T_orig, -ph_test, E_orig))
            optimisms.append(ci_b - ci_test)
            boot_cindices.append(ci_b)
        except Exception:
            continue

    if not optimisms:
        warns.append("Bootstrap Cox 验证未完成任何迭代")
        return {"method_label": "Bootstrap 验证", "corrected_cindex": orig_cindex,
                "optimism": 0.0, "boot_cindices": [], "warnings": warns}

    mean_opt = float(np.mean(optimisms))
    corrected = float(np.clip(orig_cindex - mean_opt, 0.0, 1.0))
    return {
        "method_label": f"Bootstrap 验证（n={len(optimisms)}）",
        "corrected_cindex": corrected,
        "optimism": mean_opt,
        "boot_cindices": boot_cindices,
        "warnings": warns,
    }


def _split_cox(
    df_fit: pd.DataFrame,
    time_col: str,
    event_col: str,
    cols: list[str],
    train_ratio: float,
    time_point: float,
) -> dict[str, Any]:
    from lifelines import CoxPHFitter
    from lifelines.utils import concordance_index

    n = len(df_fit)
    n_train = int(n * train_ratio)
    rng = np.random.default_rng(42)
    perm = rng.permutation(n)
    df_train = df_fit.iloc[perm[:n_train]].reset_index(drop=True)
    df_test = df_fit.iloc[perm[n_train:]].reset_index(drop=True)

    try:
        cph = CoxPHFitter(penalizer=0.01)
        with _warnings_mod.catch_warnings():
            _warnings_mod.simplefilter("ignore")
            cph.fit(df_train, duration_col=time_col, event_col=event_col)
        ci_train = float(np.clip(float(cph.concordance_index_), 0, 1))
        ph_test = np.asarray(cph.predict_log_partial_hazard(df_test[cols]), dtype=float)
        ci_test = float(concordance_index(
            df_test[time_col].values, -ph_test, df_test[event_col].values
        ))
        return {
            "method_label": f"Split 验证（训练集 {train_ratio*100:.0f}%）",
            "corrected_cindex": ci_test,
            "optimism": ci_train - ci_test,
            "warnings": [],
        }
    except Exception as exc:
        return {"method_label": "Split 验证", "corrected_cindex": float("nan"),
                "optimism": float("nan"), "warnings": [f"Split Cox 验证失败：{exc}"]}


def _cv_cox(
    df_fit: pd.DataFrame,
    time_col: str,
    event_col: str,
    cols: list[str],
    time_point: float,
    k: int = 5,
) -> dict[str, Any]:
    from lifelines import CoxPHFitter
    from lifelines.utils import concordance_index

    n = len(df_fit)
    rng = np.random.default_rng(42)
    perm = rng.permutation(n)
    folds = np.array_split(perm, k)
    cv_cindices: list[float] = []
    warns: list[str] = []

    cph_full = CoxPHFitter(penalizer=0.01)
    with _warnings_mod.catch_warnings():
        _warnings_mod.simplefilter("ignore")
        cph_full.fit(df_fit, duration_col=time_col, event_col=event_col)
    orig_ci = float(np.clip(float(cph_full.concordance_index_), 0, 1))

    for fold in folds:
        test_mask = np.zeros(n, dtype=bool)
        test_mask[fold] = True
        df_train = df_fit.iloc[~test_mask].reset_index(drop=True)
        df_test_fold = df_fit.iloc[test_mask].reset_index(drop=True)
        try:
            cph_ = CoxPHFitter(penalizer=0.1)
            with _warnings_mod.catch_warnings():
                _warnings_mod.simplefilter("ignore")
                cph_.fit(df_train, duration_col=time_col, event_col=event_col)
            ph_ = np.asarray(cph_.predict_log_partial_hazard(df_test_fold[cols]), dtype=float)
            ci_ = float(concordance_index(
                df_test_fold[time_col].values, -ph_, df_test_fold[event_col].values
            ))
            cv_cindices.append(ci_)
        except Exception:
            continue

    if not cv_cindices:
        return {"method_label": f"{k}-fold CV", "corrected_cindex": float("nan"),
                "optimism": float("nan"), "warnings": [f"{k}-fold CV 未完成"]}

    mean_ci = float(np.mean(cv_cindices))
    return {
        "method_label": f"{k}-fold 交叉验证",
        "corrected_cindex": mean_ci,
        "optimism": orig_ci - mean_ci,
        "boot_cindices": cv_cindices,
        "warnings": warns,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Nomogram 数据生成
# ─────────────────────────────────────────────────────────────────────────────

def _nomogram_logistic(
    model,
    final_cols: list[str],
    dummy_map: dict[str, list[str]],
    orig_preds: list[str],
    df_clean: pd.DataFrame,
) -> dict[str, Any] | None:
    try:
        params = np.asarray(model.params, dtype=float)
        intercept = float(params[0])
        col_to_beta = {c: float(params[i + 1]) for i, c in enumerate(final_cols)}

        # 计算每个预测因子的最大贡献范围（用于归一化）
        max_range = 0.0
        for col in final_cols:
            beta = col_to_beta[col]
            vals = df_clean[col].dropna().values.astype(float)
            if len(vals) == 0:
                continue
            rng_v = abs(beta) * (float(vals.max()) - float(vals.min()))
            if rng_v > max_range:
                max_range = rng_v

        if max_range < 1e-8:
            return None

        pts_per_logit = 100.0 / max_range

        variables: list[dict] = []
        sum_min_contrib = 0.0
        seen_cat: set[str] = set()

        for v in orig_preds:
            if v in dummy_map:
                if v in seen_cat:
                    continue
                seen_cat.add(v)
                dummy_cols = [c for c in dummy_map[v] if c in col_to_beta]
                if not dummy_cols:
                    continue
                # contributions：0 for reference, beta_dc for each dummy
                contribs = [0.0] + [col_to_beta[dc] for dc in dummy_cols]
                min_contrib = min(contribs)
                sum_min_contrib += min_contrib
                ticks = [{"label": "参考组", "pts": round((0.0 - min_contrib) * pts_per_logit, 1)}]
                for dc, contrib in zip(dummy_cols, contribs[1:]):
                    ticks.append({
                        "label": dc[len(v) + 1:],
                        "pts": round((contrib - min_contrib) * pts_per_logit, 1),
                    })
                variables.append({"name": v, "type": "categorical", "ticks": ticks})
            elif v in col_to_beta:
                beta = col_to_beta[v]
                vals = df_clean[v].dropna().values.astype(float)
                if len(vals) == 0:
                    continue
                x_min, x_max = float(vals.min()), float(vals.max())
                min_contrib = min(beta * x_min, beta * x_max)
                sum_min_contrib += min_contrib
                tick_xs = np.linspace(x_min, x_max, 6)
                ticks = [{
                    "label": f"{xv:.1f}",
                    "pts": round((beta * float(xv) - min_contrib) * pts_per_logit, 1),
                } for xv in tick_xs]
                variables.append({"name": v, "type": "continuous", "ticks": ticks})

        if not variables:
            return None

        total_max = float(sum(max(t["pts"] for t in var["ticks"]) for var in variables))
        n_tick_pts = min(11, int(total_max / 10) + 2)
        total_ticks = [round(float(x), 1) for x in np.linspace(0, total_max, n_tick_pts)]

        # 概率刻度：log-odds = total_pts / pts_per_logit + offset
        offset = intercept + sum_min_contrib
        prob_ticks = []
        for tp in np.linspace(0, total_max, 20):
            log_odds = tp / pts_per_logit + offset
            prob = float(1.0 / (1.0 + _safe_exp(-log_odds)))
            prob_ticks.append({"pts": round(float(tp), 1), "prob": round(prob, 4)})

        return {
            "model_type": "logistic",
            "variables": variables,
            "total_points": {"min": 0.0, "max": total_max, "ticks": total_ticks},
            "prob_scale": prob_ticks,
        }
    except Exception as exc:
        logger.warning("Nomogram logistic 生成失败: %s", exc)
        return None


def _nomogram_cox(
    cph,
    final_cols: list[str],
    dummy_map: dict[str, list[str]],
    orig_preds: list[str],
    df_fit: pd.DataFrame,
    time_point: float,
    bs0: float,
) -> dict[str, Any] | None:
    try:
        summ = cph.summary
        col_to_beta: dict[str, float] = {}
        for col in final_cols:
            if col in summ.index:
                col_to_beta[col] = float(summ.loc[col, "coef"])

        if not col_to_beta:
            return None

        max_range = 0.0
        for col, beta in col_to_beta.items():
            vals = df_fit[col].dropna().values.astype(float)
            if len(vals) == 0:
                continue
            rng_v = abs(beta) * (float(vals.max()) - float(vals.min()))
            if rng_v > max_range:
                max_range = rng_v

        if max_range < 1e-8:
            return None

        pts_per_loghr = 100.0 / max_range
        variables: list[dict] = []
        sum_min_contrib = 0.0
        seen_cat: set[str] = set()

        for v in orig_preds:
            if v in dummy_map:
                if v in seen_cat:
                    continue
                seen_cat.add(v)
                dummy_cols = [c for c in dummy_map[v] if c in col_to_beta]
                if not dummy_cols:
                    continue
                contribs = [0.0] + [col_to_beta[dc] for dc in dummy_cols]
                min_contrib = min(contribs)
                sum_min_contrib += min_contrib
                ticks = [{"label": "参考组", "pts": round((0.0 - min_contrib) * pts_per_loghr, 1)}]
                for dc, contrib in zip(dummy_cols, contribs[1:]):
                    ticks.append({
                        "label": dc[len(v) + 1:],
                        "pts": round((contrib - min_contrib) * pts_per_loghr, 1),
                    })
                variables.append({"name": v, "type": "categorical", "ticks": ticks})
            elif v in col_to_beta:
                beta = col_to_beta[v]
                vals = df_fit[v].dropna().values.astype(float)
                if len(vals) == 0:
                    continue
                x_min, x_max = float(vals.min()), float(vals.max())
                min_contrib = min(beta * x_min, beta * x_max)
                sum_min_contrib += min_contrib
                tick_xs = np.linspace(x_min, x_max, 6)
                ticks = [{
                    "label": f"{xv:.1f}",
                    "pts": round((beta * float(xv) - min_contrib) * pts_per_loghr, 1),
                } for xv in tick_xs]
                variables.append({"name": v, "type": "continuous", "ticks": ticks})

        if not variables:
            return None

        total_max = float(sum(max(t["pts"] for t in var["ticks"]) for var in variables))
        n_tick_pts = min(11, int(total_max / 10) + 2)
        total_ticks = [round(float(x), 1) for x in np.linspace(0, total_max, n_tick_pts)]

        bs0_safe = float(np.clip(bs0, 1e-6, 1 - 1e-6)) if not math.isnan(bs0) else 0.5
        offset = sum_min_contrib
        prob_ticks = []
        for tp in np.linspace(0, total_max, 20):
            lp = float(np.clip(tp / pts_per_loghr + offset, -_CLIP, _CLIP))
            surv = float(np.clip(bs0_safe ** float(np.exp(lp)), 0.0, 1.0))
            prob_ticks.append({"pts": round(float(tp), 1), "prob": round(surv, 4)})

        return {
            "model_type": "cox",
            "time_point": time_point,
            "variables": variables,
            "total_points": {"min": 0.0, "max": total_max, "ticks": total_ticks},
            "prob_scale": prob_ticks,
            "prob_label": f"{time_point:.0f} 天生存率",
        }
    except Exception as exc:
        logger.warning("Nomogram Cox 生成失败: %s", exc)
        return None


# ─────────────────────────────────────────────────────────────────────────────
# 决策曲线分析（DCA）
# ─────────────────────────────────────────────────────────────────────────────

def _dca_data(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, Any]:
    n = len(y_true)
    event_rate = float(np.mean(y_true))
    thresholds = np.linspace(0.01, 0.99, 99)

    model_nb, treat_all_nb, treat_none_nb = [], [], []
    for t in thresholds:
        y_cls = (y_pred >= t).astype(int)
        tp = int(np.sum((y_cls == 1) & (y_true == 1)))
        fp = int(np.sum((y_cls == 1) & (y_true == 0)))
        nb_model = float(tp / n - fp / n * t / (1.0 - t))
        nb_all = float(event_rate - (1.0 - event_rate) * t / (1.0 - t))
        t_val = round(float(t), 4)
        model_nb.append([t_val, round(nb_model, 5)])
        treat_all_nb.append([t_val, round(max(nb_all, -0.1), 5)])
        treat_none_nb.append([t_val, 0.0])

    return {"model": model_nb, "treat_all": treat_all_nb, "treat_none": treat_none_nb}


# ─────────────────────────────────────────────────────────────────────────────
# 图表构建
# ─────────────────────────────────────────────────────────────────────────────

def _build_roc_chart(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    auc: float,
    ci_lo: float,
    ci_hi: float,
) -> ChartResult:
    from sklearn.metrics import roc_curve

    fpr, tpr, thr = roc_curve(y_true, y_pred)
    youden_idx = int(np.argmax(tpr - fpr))
    best_fpr = float(fpr[youden_idx])
    best_tpr = float(tpr[youden_idx])
    best_thr = float(thr[youden_idx])

    if len(fpr) > 500:
        idx = np.linspace(0, len(fpr) - 1, 500, dtype=int)
        fpr, tpr = fpr[idx], tpr[idx]
    roc_pts = [[round(float(f), 5), round(float(t), 5)] for f, t in zip(fpr, tpr)]

    option: dict[str, Any] = {
        "title": {
            "text": "ROC 曲线",
            "subtext": f"AUC = {auc:.4f}（95% CI：{ci_lo:.4f}–{ci_hi:.4f}）",
            "left": "center",
        },
        "tooltip": {"trigger": "axis"},
        "legend": {"data": ["ROC 曲线", "随机参考线", "最佳截断点"], "top": "15%"},
        "grid": {"left": "12%", "right": "5%", "top": "25%", "bottom": "12%"},
        "xAxis": {"type": "value", "name": "1 - 特异性（FPR）", "nameLocation": "center",
                  "nameGap": 28, "min": 0, "max": 1},
        "yAxis": {"type": "value", "name": "敏感性（TPR）", "nameLocation": "center",
                  "nameGap": 36, "min": 0, "max": 1},
        "series": [
            {"name": "ROC 曲线", "type": "line", "data": roc_pts, "showSymbol": False,
             "lineStyle": {"color": "#5470c6", "width": 2},
             "areaStyle": {"color": "rgba(84,112,198,0.08)"}},
            {"name": "随机参考线", "type": "line", "data": [[0, 0], [1, 1]],
             "showSymbol": False, "lineStyle": {"type": "dashed", "color": "#aaa", "width": 1.5}},
            {"name": "最佳截断点", "type": "scatter",
             "data": [[round(best_fpr, 4), round(best_tpr, 4)]],
             "symbolSize": 12, "itemStyle": {"color": "#ee6666"},
             "label": {"show": True, "formatter": f"截断={best_thr:.3f}",
                       "position": "right", "fontSize": 11}},
        ],
    }
    return ChartResult(title="ROC 曲线", chart_type="line", option=option)


def _build_calibration_chart_logistic(
    y_true: np.ndarray, y_pred: np.ndarray
) -> ChartResult | None:
    try:
        from sklearn.calibration import calibration_curve

        fop, mpv = calibration_curve(y_true, y_pred, n_bins=10)
        calib_pts = [[round(float(m), 5), round(float(f), 5)] for m, f in zip(mpv, fop)]

        loess_pts: list[list] = []
        try:
            from statsmodels.nonparametric.smoothers_lowess import lowess
            sm_data = lowess(fop, mpv, frac=0.6)
            loess_pts = [[round(float(x), 5), round(float(y), 5)] for x, y in sm_data]
        except Exception:
            pass

        series: list[dict] = [
            {"name": "校准曲线（分组）", "type": "scatter",
             "data": calib_pts, "symbolSize": 10,
             "itemStyle": {"color": "#5470c6"}},
            {"name": "理想校准", "type": "line", "data": [[0, 0], [1, 1]],
             "showSymbol": False, "lineStyle": {"type": "dashed", "color": "#91cc75", "width": 2}},
        ]
        if loess_pts:
            series.append({
                "name": "Loess 平滑线", "type": "line", "data": loess_pts,
                "showSymbol": False, "lineStyle": {"color": "#ee6666", "width": 2},
            })

        option: dict[str, Any] = {
            "title": {"text": "校准曲线", "left": "center"},
            "tooltip": {"trigger": "axis"},
            "legend": {"data": ["校准曲线（分组）", "理想校准", "Loess 平滑线"], "top": "10%"},
            "grid": {"left": "12%", "right": "5%", "top": "20%", "bottom": "12%"},
            "xAxis": {"type": "value", "name": "预测概率", "nameLocation": "center",
                      "nameGap": 28, "min": 0, "max": 1},
            "yAxis": {"type": "value", "name": "实际概率", "nameLocation": "center",
                      "nameGap": 36, "min": 0, "max": 1},
            "series": series,
        }
        return ChartResult(title="校准曲线", chart_type="scatter", option=option)
    except Exception as exc:
        logger.warning("校准曲线失败: %s", exc)
        return None


def _build_cox_calibration_chart(
    cal_rows: list[list], time_point: float
) -> ChartResult:
    if not cal_rows:
        option: dict[str, Any] = {"title": {"text": "Cox 校准曲线（数据不足）"}}
        return ChartResult(title="Cox 校准曲线", chart_type="bar", option=option)

    categories = [r[0] for r in cal_rows]
    pred_vals = [float(r[1]) for r in cal_rows]
    obs_vals = [float(r[2]) for r in cal_rows]

    option = {
        "title": {"text": f"Cox 校准曲线（时间点 {time_point:.0f}）", "left": "center"},
        "tooltip": {"trigger": "axis"},
        "legend": {"data": ["预测事件率", "实际事件率（KM）"]},
        "xAxis": {"type": "category", "data": categories,
                  "axisLabel": {"rotate": 30, "fontSize": 10}},
        "yAxis": {"type": "value", "name": "事件率", "min": 0},
        "series": [
            {"name": "预测事件率", "type": "bar", "data": pred_vals,
             "itemStyle": {"color": "rgba(84,112,198,0.75)"}},
            {"name": "实际事件率（KM）", "type": "bar", "data": obs_vals,
             "itemStyle": {"color": "rgba(238,102,102,0.75)"}},
        ],
    }
    return ChartResult(title="Cox 校准曲线", chart_type="bar", option=option)


def _build_dca_chart(dca: dict[str, Any]) -> ChartResult:
    option: dict[str, Any] = {
        "title": {"text": "决策曲线分析（DCA）", "left": "center"},
        "tooltip": {"trigger": "axis"},
        "legend": {"data": ["预测模型", "全部治疗", "全不治疗"], "top": "10%"},
        "grid": {"left": "12%", "right": "5%", "top": "20%", "bottom": "12%"},
        "xAxis": {"type": "value", "name": "阈概率", "nameLocation": "center",
                  "nameGap": 28, "min": 0, "max": 1},
        "yAxis": {"type": "value", "name": "净获益", "nameLocation": "center", "nameGap": 40},
        "series": [
            {"name": "预测模型", "type": "line", "data": dca["model"],
             "showSymbol": False, "lineStyle": {"color": "#5470c6", "width": 2}},
            {"name": "全部治疗", "type": "line", "data": dca["treat_all"],
             "showSymbol": False, "lineStyle": {"color": "#91cc75", "width": 2, "type": "dashed"}},
            {"name": "全不治疗", "type": "line", "data": dca["treat_none"],
             "showSymbol": False, "lineStyle": {"color": "#aaa", "width": 1.5}},
        ],
    }
    return ChartResult(title="决策曲线分析（DCA）", chart_type="line", option=option)


def _build_time_dep_auc_chart(
    tdep_list: list[dict], cindex: float
) -> ChartResult:
    pts = [[d["time"], d["auc"]] for d in tdep_list]
    t_min = pts[0][0]
    t_max = pts[-1][0]
    option: dict[str, Any] = {
        "title": {
            "text": "时间依赖 AUC 曲线",
            "subtext": f"C-index = {cindex:.4f}",
            "left": "center",
        },
        "tooltip": {"trigger": "axis"},
        "legend": {"data": ["时间依赖 AUC", "参考线（0.5）"]},
        "xAxis": {"type": "value", "name": "时间", "nameLocation": "center", "nameGap": 28},
        "yAxis": {"type": "value", "name": "AUC", "min": 0, "max": 1},
        "series": [
            {"name": "时间依赖 AUC", "type": "line", "data": pts,
             "symbolSize": 10, "lineStyle": {"color": "#5470c6", "width": 2},
             "itemStyle": {"color": "#5470c6"},
             "label": {"show": True, "formatter": "{@[1]}", "fontSize": 10}},
            {"name": "参考线（0.5）", "type": "line",
             "data": [[t_min, 0.5], [t_max, 0.5]],
             "showSymbol": False, "lineStyle": {"type": "dashed", "color": "#aaa", "width": 1.5}},
        ],
    }
    return ChartResult(title="时间依赖 AUC 曲线", chart_type="line", option=option)


def _build_bootstrap_hist(
    values: list[float],
    orig: float,
    corrected: float,
    label: str = "AUC",
) -> ChartResult:
    if not values:
        return ChartResult(title=f"Bootstrap {label} 分布", chart_type="bar",
                           option={"title": {"text": f"Bootstrap {label} 分布（无数据）"}})

    vals_arr = np.array(values)
    edges = np.linspace(float(vals_arr.min()), float(vals_arr.max()), 21)
    counts, _ = np.histogram(vals_arr, bins=edges)
    centers = [(float(edges[i]) + float(edges[i + 1])) / 2 for i in range(len(edges) - 1)]

    option: dict[str, Any] = {
        "title": {"text": f"Bootstrap {label} 分布（n={len(values)}）", "left": "center"},
        "tooltip": {"trigger": "axis"},
        "xAxis": {"type": "category", "data": [f"{c:.4f}" for c in centers],
                  "name": label, "nameLocation": "center", "nameGap": 28,
                  "axisLabel": {"rotate": 30, "fontSize": 9}},
        "yAxis": {"type": "value", "name": "频数"},
        "series": [{
            "type": "bar", "data": counts.tolist(),
            "itemStyle": {"color": "rgba(84,112,198,0.7)"},
            "markLine": {
                "symbol": ["none", "none"],
                "data": [
                    {"xAxis": next((i for i, c in enumerate(centers) if c >= orig), len(centers) - 1),
                     "lineStyle": {"color": "#ee6666", "width": 2}},
                    {"xAxis": next((i for i, c in enumerate(centers) if c >= corrected), len(centers) - 1),
                     "lineStyle": {"color": "#91cc75", "width": 2, "type": "dashed"}},
                ],
                "label": {"show": False},
            },
        }],
    }
    return ChartResult(title=f"Bootstrap {label} 分布", chart_type="bar", option=option)


# ─────────────────────────────────────────────────────────────────────────────
# 辅助
# ─────────────────────────────────────────────────────────────────────────────

def _fmt_p(p: float) -> str:
    if math.isnan(p):
        return "—"
    if p < 0.001:
        return "< 0.001"
    return f"{p:.3f}"


def _parse_p(s: str) -> float:
    s = s.strip()
    if s.startswith("<"):
        return 0.0005
    try:
        return float(s)
    except Exception:
        return float("nan")
