"""限制性立方样条（RCS）分析模块。

支持：
  - 模型类型：logistic、linear、cox
  - Harrell 推荐节点位置（3/4/5 个节点）
  - 纯 Python 实现，不依赖 rpy2 / R
  - 非线性检验（Wald 检验）与总体关联检验
  - 效应曲线（OR / HR / β，相对于参考值）及 95% CI（Delta 法）
  - 图表：RCS 曲线（含 CI 带、节点标记、rug 散点）、暴露变量分布直方图
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


# ─────────────────────────────────────────────────────────────────────────────
# 溢出安全的 exp
# ─────────────────────────────────────────────────────────────────────────────

def _safe_exp(x: float) -> float:
    """np.clip 防止 exp 溢出。"""
    return float(np.exp(np.clip(float(x), -500.0, 500.0)))


def _safe_exp_arr(arr: np.ndarray) -> np.ndarray:
    """向量版安全 exp。"""
    return np.exp(np.clip(arr, -500.0, 500.0))


# ─────────────────────────────────────────────────────────────────────────────
# 节点百分位默认值（Harrell 推荐）
# ─────────────────────────────────────────────────────────────────────────────

_DEFAULT_KNOT_PERCENTILES: dict[int, list[float]] = {
    3: [10.0, 50.0, 90.0],
    4: [5.0, 35.0, 65.0, 95.0],
    5: [5.0, 27.5, 50.0, 72.5, 95.0],
}


# ─────────────────────────────────────────────────────────────────────────────
# Harrell RCS 基函数
# ─────────────────────────────────────────────────────────────────────────────

def _rcs_basis(x: np.ndarray, knots: np.ndarray) -> np.ndarray:
    """计算 RCS 样条基函数。

    参数
    ----
    x     : shape (n,)，暴露变量观测值（或单个网格点数组）
    knots : shape (k,)，节点位置，按升序排列

    返回
    ----
    shape (n, k-2) 的矩阵，每列为一个样条基函数的值。
    """
    k = len(knots)
    n = len(x)
    t_km1 = knots[-2]  # t_{k-1}
    t_k = knots[-1]    # t_k
    denom = t_k - t_km1  # 总是正数（节点升序且不重复）

    result = np.zeros((n, k - 2), dtype=float)
    for j in range(k - 2):
        tj = knots[j]
        # (x - t_j)_+^3
        term1 = np.maximum(x - tj, 0.0) ** 3
        # (x - t_{k-1})_+^3 * (t_k - t_j) / (t_k - t_{k-1})
        term2 = np.maximum(x - t_km1, 0.0) ** 3 * (t_k - tj) / denom
        # (x - t_k)_+^3 * (t_{k-1} - t_j) / (t_k - t_{k-1})
        term3 = np.maximum(x - t_k, 0.0) ** 3 * (t_km1 - tj) / denom
        result[:, j] = term1 - term2 + term3

    return result


# ─────────────────────────────────────────────────────────────────────────────
# 辅助函数
# ─────────────────────────────────────────────────────────────────────────────

def _fmt_p(p: float) -> str:
    if math.isnan(p):
        return "—"
    if p < 0.001:
        return "< 0.001"
    return f"{p:.3f}"


def _encode_binary(col: pd.Series) -> pd.Series:
    """将二分类变量编码为 0/1 浮点型（保留 NaN）。"""
    non_null = col.dropna()
    if len(non_null) == 0:
        raise ValueError(f"因变量 '{col.name}' 全为缺失值")

    if pd.api.types.is_numeric_dtype(col):
        unique_vals = sorted(float(v) for v in non_null.unique())
        if len(unique_vals) > 2:
            raise ValueError(
                f"因变量 '{col.name}' 有 {len(unique_vals)} 个唯一值，"
                "logistic 模型要求二分类变量"
            )
        if set(unique_vals) == {0.0, 1.0}:
            return col.astype(float)
        lo, hi = unique_vals[0], unique_vals[1]
        return col.map(
            lambda x: {lo: 0.0, hi: 1.0}.get(float(x)) if pd.notna(x) else float("nan")
        )

    col_str = col.where(col.isna(), col.astype(str))
    unique_str = sorted(col_str.dropna().unique())
    if len(unique_str) > 2:
        raise ValueError(
            f"因变量 '{col.name}' 有 {len(unique_str)} 个唯一类别，"
            "logistic 模型要求二分类变量"
        )
    neg_label, pos_label = unique_str[0], unique_str[1]
    return col_str.map({neg_label: 0.0, pos_label: 1.0})


# ─────────────────────────────────────────────────────────────────────────────
# 主入口
# ─────────────────────────────────────────────────────────────────────────────

def run(df: pd.DataFrame, params: dict) -> AnalysisResult:
    """执行 RCS 分析。

    参数（params 字典）
    -------------------
    model_type        : str            "logistic" | "cox" | "linear"
    outcome           : str            因变量（logistic/linear）
    time_col          : str            时间变量（cox）
    event_col         : str            事件变量（cox）
    exposure          : str            连续型暴露变量
    covariates        : list[str]      调整协变量
    n_knots           : int            节点数，3/4/5（默认 4）
    knot_percentiles  : list[float]    可选：手动指定百分位数
    ref_value         : float          参考值（默认：暴露变量中位数）
    n_curve_points    : int            曲线点数（默认 100）
    """
    warnings: list[str] = []

    # ── 读取参数 ──────────────────────────────────────────────────────────────
    model_type: str = str(params.get("model_type", "logistic")).lower()
    exposure: str = str(params.get("exposure", ""))
    outcome: str = str(params.get("outcome", ""))
    time_col: str = str(params.get("time_col", ""))
    event_col: str = str(params.get("event_col", ""))
    covariates: list[str] = list(params.get("covariates") or [])
    n_knots: int = int(params.get("n_knots", 4))
    knot_percentiles_override: list[float] | None = params.get("knot_percentiles")
    ref_value_param: float | None = params.get("ref_value")
    n_curve_points: int = int(params.get("n_curve_points", 100))

    # ── 参数校验 ──────────────────────────────────────────────────────────────
    if model_type not in ("logistic", "cox", "linear"):
        raise ValueError("model_type 必须为 logistic / cox / linear")

    if n_knots not in (3, 4, 5):
        raise ValueError("n_knots 必须为 3、4 或 5")

    if not exposure:
        raise ValueError("请指定暴露变量 (exposure)")
    if exposure not in df.columns:
        raise ValueError(f"暴露变量 '{exposure}' 不存在于数据集中")
    if not pd.api.types.is_numeric_dtype(df[exposure]):
        raise ValueError(f"暴露变量 '{exposure}' 必须为数值型连续变量")

    if model_type in ("logistic", "linear"):
        if not outcome:
            raise ValueError("logistic/linear 模型需要指定因变量 (outcome)")
        if outcome not in df.columns:
            raise ValueError(f"因变量 '{outcome}' 不存在于数据集中")
    else:  # cox
        if not time_col:
            raise ValueError("cox 模型需要指定时间变量 (time_col)")
        if not event_col:
            raise ValueError("cox 模型需要指定事件变量 (event_col)")
        for col, label in [(time_col, "时间变量"), (event_col, "事件变量")]:
            if col not in df.columns:
                raise ValueError(f"{label} '{col}' 不存在于数据集中")

    # ── 协变量校验：跳过不存在的 ──────────────────────────────────────────────
    valid_covs: list[str] = []
    for cov in covariates:
        if cov not in df.columns:
            warnings.append(f"协变量 '{cov}' 不存在于数据集中，已忽略")
        elif not pd.api.types.is_numeric_dtype(df[cov]):
            warnings.append(f"协变量 '{cov}' 非数值型，已忽略")
        else:
            valid_covs.append(cov)

    # ── 构建完整案例集 ────────────────────────────────────────────────────────
    if model_type == "cox":
        needed_cols = [exposure, time_col, event_col] + valid_covs
    else:
        needed_cols = [exposure, outcome] + valid_covs

    # 去重
    needed_cols = list(dict.fromkeys(needed_cols))

    df_work = df[needed_cols].copy()

    # logistic：编码因变量
    if model_type == "logistic":
        df_work[outcome] = _encode_binary(df_work[outcome])

    # cox：确保事件变量为 0/1 数值
    if model_type == "cox":
        if not pd.api.types.is_numeric_dtype(df_work[event_col]):
            df_work[event_col] = _encode_binary(df_work[event_col])
        else:
            df_work[event_col] = df_work[event_col].astype(float)
        df_work[time_col] = pd.to_numeric(df_work[time_col], errors="coerce")

    df_complete = df_work.dropna()
    n_total = len(df_work)
    n_complete = len(df_complete)

    if n_total > 0 and (n_total - n_complete) / n_total > 0.20:
        pct_dropped = (n_total - n_complete) / n_total * 100
        warnings.append(
            f"完整案例筛选后删除了 {n_total - n_complete} 行（{pct_dropped:.1f}%），"
            "超过 20%，请检查数据质量"
        )

    if n_complete < n_knots + 2:
        raise ValueError(
            f"有效样本量 ({n_complete}) 不足，无法拟合 {n_knots} 节点 RCS 模型"
        )

    # ── 节点位置 ──────────────────────────────────────────────────────────────
    x_vals = df_complete[exposure].values.astype(float)

    if knot_percentiles_override:
        if len(knot_percentiles_override) != n_knots:
            raise ValueError(
                f"knot_percentiles 长度 ({len(knot_percentiles_override)}) "
                f"与 n_knots ({n_knots}) 不一致"
            )
        knot_pcts = list(knot_percentiles_override)
    else:
        knot_pcts = _DEFAULT_KNOT_PERCENTILES[n_knots]

    knots = np.array([float(np.percentile(x_vals, p)) for p in knot_pcts])

    # 检查节点是否唯一（防止分母为零）
    if len(np.unique(knots)) < len(knots):
        raise ValueError(
            "节点位置存在重复值，请调整 n_knots 或 knot_percentiles，"
            "或检查暴露变量的分布"
        )

    n_spline = n_knots - 2  # 样条项数

    # ── 参考值 ────────────────────────────────────────────────────────────────
    ref_value: float = (
        float(ref_value_param) if ref_value_param is not None
        else float(np.median(x_vals))
    )

    # ── 构建设计矩阵 ──────────────────────────────────────────────────────────
    X_spline = _rcs_basis(x_vals, knots)  # (n, n_spline)
    X_cov = df_complete[valid_covs].values.astype(float) if valid_covs else np.empty((n_complete, 0))

    # 列顺序：[exposure, s_1...s_{k-2}, cov_1...cov_m]
    X_main = np.column_stack([x_vals.reshape(-1, 1), X_spline, X_cov])
    # exposure 在第 0 列，spline 在 1..n_spline，covariates 之后

    # 模型列名（便于调试）
    spline_col_names = [f"rcs_s{j+1}" for j in range(n_spline)]
    col_names = [exposure] + spline_col_names + valid_covs

    # ── 拟合模型 ──────────────────────────────────────────────────────────────
    if model_type == "logistic":
        result_dict = _fit_logistic(df_complete, outcome, X_main, col_names, warnings)
    elif model_type == "linear":
        result_dict = _fit_linear(df_complete, outcome, X_main, col_names, warnings)
    else:  # cox
        result_dict = _fit_cox(
            df_complete, time_col, event_col, X_main, col_names, valid_covs, warnings
        )

    params_vec: np.ndarray = result_dict["params"]      # full coef vector (with/without intercept)
    cov_matrix: np.ndarray = result_dict["cov_matrix"]  # full covariance matrix
    has_intercept: bool = result_dict["has_intercept"]

    # 定位 exposure 和 spline 系数在 params_vec 中的起始索引
    intercept_offset = 1 if has_intercept else 0
    # exposure 位于 params_vec[intercept_offset]
    # spline   位于 params_vec[intercept_offset+1 : intercept_offset+1+n_spline]
    exp_idx = intercept_offset
    spl_start = intercept_offset + 1
    spl_end = spl_start + n_spline

    beta_exposure = float(params_vec[exp_idx])
    beta_splines = params_vec[spl_start:spl_end]  # shape (n_spline,)

    # ── 非线性检验（Wald，仅样条项） ──────────────────────────────────────────
    beta_spl = beta_splines
    sigma_spl = cov_matrix[spl_start:spl_end, spl_start:spl_end]
    try:
        sigma_spl_inv = np.linalg.inv(sigma_spl)
        wald_nonlinear = float(beta_spl @ sigma_spl_inv @ beta_spl)
    except np.linalg.LinAlgError:
        wald_nonlinear = float("nan")
    df_nonlinear = n_spline
    p_nonlinear = float(stats.chi2.sf(wald_nonlinear, df_nonlinear)) if not math.isnan(wald_nonlinear) else float("nan")

    # ── 总体关联检验（Wald，exposure + 样条项） ───────────────────────────────
    joint_start = exp_idx
    joint_end = spl_end
    beta_joint = params_vec[joint_start:joint_end]  # shape (n_knots-1,)
    sigma_joint = cov_matrix[joint_start:joint_end, joint_start:joint_end]
    try:
        sigma_joint_inv = np.linalg.inv(sigma_joint)
        wald_overall = float(beta_joint @ sigma_joint_inv @ beta_joint)
    except np.linalg.LinAlgError:
        wald_overall = float("nan")
    df_overall = n_knots - 1
    p_overall = float(stats.chi2.sf(wald_overall, df_overall)) if not math.isnan(wald_overall) else float("nan")

    # ── 效应曲线 ──────────────────────────────────────────────────────────────
    x_grid = np.linspace(
        float(np.percentile(x_vals, 1)),
        float(np.percentile(x_vals, 99)),
        n_curve_points,
    )

    # 参考值处的样条基
    ref_arr = np.array([ref_value], dtype=float)
    s_ref = _rcs_basis(ref_arr, knots)[0]  # shape (n_spline,)

    # 网格处的样条基
    s_grid = _rcs_basis(x_grid, knots)  # shape (n_curve_points, n_spline)

    # delta（线性预测器差值）
    delta_arr = beta_exposure * (x_grid - ref_value) + s_grid @ beta_splines - s_ref @ beta_splines

    # Delta 法 SE
    # contrast vector c 的维度与 [exposure, spline...] 对应，长度为 n_knots-1
    # c = [(x_new - ref), (s_new_1 - s_ref_1), ..., (s_new_{k-2} - s_ref_{k-2})]
    # covariance 取 params_vec[exp_idx:spl_end] 对应的子矩阵
    sigma_es = cov_matrix[joint_start:joint_end, joint_start:joint_end]  # (n_knots-1, n_knots-1)

    se_delta = np.zeros(n_curve_points, dtype=float)
    for i in range(n_curve_points):
        c_vec = np.concatenate([
            [(x_grid[i] - ref_value)],
            s_grid[i] - s_ref,
        ])  # shape (n_knots-1,)
        var_i = float(c_vec @ sigma_es @ c_vec)
        se_delta[i] = math.sqrt(max(var_i, 0.0))

    z95 = float(stats.norm.ppf(0.975))

    if model_type in ("logistic", "cox"):
        effect_arr = _safe_exp_arr(delta_arr)
        ci_lo_arr = _safe_exp_arr(delta_arr - z95 * se_delta)
        ci_hi_arr = _safe_exp_arr(delta_arr + z95 * se_delta)
        ref_line_y = 1.0
        y_label = "OR" if model_type == "logistic" else "HR"
    else:  # linear
        effect_arr = delta_arr.copy()
        ci_lo_arr = delta_arr - z95 * se_delta
        ci_hi_arr = delta_arr + z95 * se_delta
        ref_line_y = 0.0
        y_label = "β"

    # ── 输出表格 1：非线性检验 ────────────────────────────────────────────────
    table_test = TableResult(
        title="RCS 非线性检验",
        headers=["检验", "统计量", "自由度", "p 值"],
        rows=[
            [
                "P for non-linearity (Wald)",
                f"{wald_nonlinear:.3f}" if not math.isnan(wald_nonlinear) else "—",
                str(df_nonlinear),
                _fmt_p(p_nonlinear),
            ],
            [
                "P for overall association",
                f"{wald_overall:.3f}" if not math.isnan(wald_overall) else "—",
                str(df_overall),
                _fmt_p(p_overall),
            ],
        ],
    )

    # ── 输出表格 2：节点信息 ──────────────────────────────────────────────────
    knot_rows: list[list[Any]] = [
        [str(j + 1), f"{knot_pcts[j]:.1f}%", f"{knots[j]:.4f}"]
        for j in range(n_knots)
    ]
    table_knots = TableResult(
        title="节点信息",
        headers=["节点编号", "位置（百分位数）", "暴露变量值"],
        rows=knot_rows,
    )

    # ── 输出表格 3：模型系数 ──────────────────────────────────────────────────
    table_coef = _build_coef_table(
        result_dict=result_dict,
        col_names=col_names,
        exposure=exposure,
        spline_col_names=spline_col_names,
        model_type=model_type,
        has_intercept=has_intercept,
    )

    # ── 图表 1：RCS 曲线 ──────────────────────────────────────────────────────
    chart_rcs = _build_rcs_chart(
        x_grid=x_grid,
        effect_arr=effect_arr,
        ci_lo_arr=ci_lo_arr,
        ci_hi_arr=ci_hi_arr,
        x_raw=x_vals,
        knots=knots,
        ref_value=ref_value,
        ref_line_y=ref_line_y,
        exposure=exposure,
        y_label=y_label,
        p_nonlinear=p_nonlinear,
        p_overall=p_overall,
    )

    # ── 图表 2：暴露变量分布直方图 ────────────────────────────────────────────
    chart_hist = _build_exposure_hist(x_vals, exposure)

    # ── 摘要 ──────────────────────────────────────────────────────────────────
    summary = (
        f"RCS 曲线分析：{model_type} 模型，暴露变量 {exposure}"
        f"（参考值 {ref_value:.2f}），节点数 {n_knots}，"
        f"P for non-linearity = {_fmt_p(p_nonlinear)}，"
        f"P for overall association = {_fmt_p(p_overall)}。"
    )

    return AnalysisResult(
        method="rcs",
        tables=[table_test, table_knots, table_coef],
        charts=[chart_rcs, chart_hist],
        summary=summary,
        warnings=warnings,
    )


# ─────────────────────────────────────────────────────────────────────────────
# 模型拟合函数
# ─────────────────────────────────────────────────────────────────────────────

def _fit_logistic(
    df_complete: pd.DataFrame,
    outcome: str,
    X_main: np.ndarray,
    col_names: list[str],
    warnings: list[str],
) -> dict[str, Any]:
    """拟合 logistic 模型，返回标准化结果字典。"""
    import statsmodels.api as sm

    y = df_complete[outcome].values.astype(float)
    X = sm.add_constant(X_main, prepend=True)

    try:
        with _warnings_mod.catch_warnings():
            _warnings_mod.simplefilter("ignore")
            model = sm.Logit(y, X).fit(disp=0, maxiter=300)
    except Exception as exc:
        raise ValueError(f"Logistic 模型拟合失败：{exc}") from exc

    params = np.asarray(model.params, dtype=float)
    cov_matrix = np.asarray(model.cov_params(), dtype=float)
    bse = np.asarray(model.bse, dtype=float)
    pvalues = np.asarray(model.pvalues, dtype=float)
    ci_arr = np.asarray(model.conf_int(), dtype=float)

    return {
        "params": params,
        "cov_matrix": cov_matrix,
        "bse": bse,
        "pvalues": pvalues,
        "ci_arr": ci_arr,
        "has_intercept": True,
        "col_names_with_intercept": ["截距（Intercept）"] + col_names,
    }


def _fit_linear(
    df_complete: pd.DataFrame,
    outcome: str,
    X_main: np.ndarray,
    col_names: list[str],
    warnings: list[str],
) -> dict[str, Any]:
    """拟合 OLS 线性回归，返回标准化结果字典。"""
    import statsmodels.api as sm

    y = df_complete[outcome].values.astype(float)
    X = sm.add_constant(X_main, prepend=True)

    try:
        with _warnings_mod.catch_warnings():
            _warnings_mod.simplefilter("ignore")
            model = sm.OLS(y, X).fit()
    except Exception as exc:
        raise ValueError(f"线性回归模型拟合失败：{exc}") from exc

    params = np.asarray(model.params, dtype=float)
    cov_matrix = np.asarray(model.cov_params(), dtype=float)
    bse = np.asarray(model.bse, dtype=float)
    pvalues = np.asarray(model.pvalues, dtype=float)
    ci_arr = np.asarray(model.conf_int(), dtype=float)

    return {
        "params": params,
        "cov_matrix": cov_matrix,
        "bse": bse,
        "pvalues": pvalues,
        "ci_arr": ci_arr,
        "has_intercept": True,
        "col_names_with_intercept": ["截距（Intercept）"] + col_names,
    }


def _fit_cox(
    df_complete: pd.DataFrame,
    time_col: str,
    event_col: str,
    X_main: np.ndarray,
    col_names: list[str],
    valid_covs: list[str],
    warnings: list[str],
) -> dict[str, Any]:
    """拟合 Cox 比例风险模型，返回标准化结果字典。"""
    try:
        from lifelines import CoxPHFitter
    except ImportError as exc:
        raise ValueError("Cox 模型需要 lifelines 库，请安装：pip install lifelines") from exc

    # 构建 lifelines 期望的 DataFrame 格式
    df_fit = pd.DataFrame(X_main, columns=col_names)
    df_fit[time_col] = df_complete[time_col].values.astype(float)
    df_fit[event_col] = df_complete[event_col].values.astype(float)

    try:
        with _warnings_mod.catch_warnings():
            _warnings_mod.simplefilter("ignore")
            cph = CoxPHFitter(penalizer=0.01)
            cph.fit(df_fit, duration_col=time_col, event_col=event_col)
    except Exception as exc:
        raise ValueError(f"Cox 模型拟合失败：{exc}") from exc

    # lifelines 的系数顺序与 col_names 对应（无截距）
    params = np.array([float(cph.params_[c]) for c in col_names], dtype=float)

    # 协方差矩阵：lifelines >= 0.27 提供 variance_matrix_
    try:
        var_matrix = cph.variance_matrix_
        cov_matrix = np.array(
            [[float(var_matrix.loc[r, c]) for c in col_names] for r in col_names],
            dtype=float,
        )
    except Exception as exc:
        logger.warning("无法获取 Cox 协方差矩阵，使用 SE^2 对角近似：%s", exc)
        se_arr = np.array([float(cph.standard_errors_[c]) for c in col_names], dtype=float)
        cov_matrix = np.diag(se_arr ** 2)

    se_arr = np.sqrt(np.diag(cov_matrix))
    pvalues = np.array(
        [2.0 * (1.0 - float(stats.norm.cdf(abs(p / s)))) if s > 0 else float("nan")
         for p, s in zip(params, se_arr)],
        dtype=float,
    )

    z95 = float(stats.norm.ppf(0.975))
    ci_lo = params - z95 * se_arr
    ci_hi = params + z95 * se_arr
    ci_arr = np.column_stack([ci_lo, ci_hi])

    return {
        "params": params,
        "cov_matrix": cov_matrix,
        "bse": se_arr,
        "pvalues": pvalues,
        "ci_arr": ci_arr,
        "has_intercept": False,
        "col_names_with_intercept": col_names,  # 无截距，直接用 col_names
    }


# ─────────────────────────────────────────────────────────────────────────────
# 系数表构建
# ─────────────────────────────────────────────────────────────────────────────

def _build_coef_table(
    result_dict: dict[str, Any],
    col_names: list[str],
    exposure: str,
    spline_col_names: list[str],
    model_type: str,
    has_intercept: bool,
) -> TableResult:
    """构建模型系数表（含 OR/HR/β 及 95% CI）。"""
    params = result_dict["params"]
    bse = result_dict["bse"]
    pvalues = result_dict["pvalues"]
    ci_arr = result_dict["ci_arr"]
    col_names_full = result_dict["col_names_with_intercept"]

    rows: list[list[Any]] = []

    for i, name in enumerate(col_names_full):
        beta = float(params[i])
        se = float(bse[i])
        p = float(pvalues[i])
        ci_lo = float(ci_arr[i, 0])
        ci_hi = float(ci_arr[i, 1])

        if model_type in ("logistic", "cox"):
            effect_val = _safe_exp(beta)
            ci_lo_eff = _safe_exp(ci_lo)
            ci_hi_eff = _safe_exp(ci_hi)
            effect_label = f"{effect_val:.4f}"
            ci_label = f"[{ci_lo_eff:.4f}, {ci_hi_eff:.4f}]"
        else:  # linear
            effect_val = beta
            effect_label = f"{beta:.4f}"
            ci_label = f"[{ci_lo:.4f}, {ci_hi:.4f}]"

        rows.append([
            name,
            f"{beta:.4f}",
            f"{se:.4f}",
            _fmt_p(p),
            effect_label,
            ci_label,
        ])

    effect_col_name = "OR" if model_type == "logistic" else ("HR" if model_type == "cox" else "β（效应值）")
    return TableResult(
        title="模型系数",
        headers=["变量", "β", "SE", "p 值", effect_col_name, "95% CI"],
        rows=rows,
    )


# ─────────────────────────────────────────────────────────────────────────────
# 图表：RCS 曲线
# ─────────────────────────────────────────────────────────────────────────────

def _build_rcs_chart(
    x_grid: np.ndarray,
    effect_arr: np.ndarray,
    ci_lo_arr: np.ndarray,
    ci_hi_arr: np.ndarray,
    x_raw: np.ndarray,
    knots: np.ndarray,
    ref_value: float,
    ref_line_y: float,
    exposure: str,
    y_label: str,
    p_nonlinear: float,
    p_overall: float,
) -> ChartResult:
    """构建 RCS 曲线 ECharts option。"""
    n_pts = len(x_grid)

    # 主曲线数据
    main_data = [[float(x_grid[i]), float(effect_arr[i])] for i in range(n_pts)]
    ci_hi_data = [[float(x_grid[i]), float(ci_hi_arr[i])] for i in range(n_pts)]
    ci_lo_data = [[float(x_grid[i]), float(ci_lo_arr[i])] for i in range(n_pts)]

    # Rug 散点（最多 200 个原始值）
    rug_x = x_raw
    if len(rug_x) > 200:
        rng = np.random.default_rng(42)
        idx = rng.choice(len(rug_x), size=200, replace=False)
        rug_x = rug_x[idx]
    rug_data = [[float(v), ref_line_y] for v in rug_x]

    # 节点 markLine
    mark_lines = [
        {
            "xAxis": float(k),
            "lineStyle": {"type": "dashed", "color": "#aaaaaa", "width": 1},
            "label": {"show": True, "formatter": f"{k:.2f}", "position": "end", "fontSize": 10},
        }
        for k in knots
    ]

    # 参考线（水平）
    ref_mark = {
        "yAxis": ref_line_y,
        "lineStyle": {"type": "dashed", "color": "#ee6666", "width": 1.5},
        "label": {
            "show": True,
            "formatter": f"Ref={ref_value:.2f}",
            "position": "insideEndTop",
            "fontSize": 10,
        },
    }

    p_nl_str = _fmt_p(p_nonlinear)
    p_ov_str = _fmt_p(p_overall)

    option: dict[str, Any] = {
        "title": {
            "text": f"RCS 曲线（{exposure}）",
            "left": "center",
        },
        "tooltip": {"trigger": "axis"},
        "legend": {
            "data": ["效应曲线", "95% CI 上界", "95% CI 下界"],
            "top": "8%",
            "right": "5%",
        },
        "grid": {"left": "12%", "right": "5%", "top": "22%", "bottom": "12%"},
        "xAxis": {
            "type": "value",
            "name": exposure,
            "nameLocation": "center",
            "nameGap": 28,
            "scale": True,
        },
        "yAxis": {
            "type": "value",
            "name": f"{y_label} (vs ref)",
            "nameLocation": "center",
            "nameGap": 40,
            "scale": True,
        },
        "series": [
            # 主曲线
            {
                "name": "效应曲线",
                "type": "line",
                "data": main_data,
                "showSymbol": False,
                "lineStyle": {"color": "#5470c6", "width": 2.5},
                "itemStyle": {"color": "#5470c6"},
                "z": 3,
                "markLine": {
                    "silent": True,
                    "symbol": "none",
                    "data": [ref_mark] + mark_lines,
                },
            },
            # CI 上界（不可见折线，用于 stack 基准）
            {
                "name": "95% CI 上界",
                "type": "line",
                "data": ci_hi_data,
                "showSymbol": False,
                "lineStyle": {"color": "transparent", "width": 0},
                "itemStyle": {"color": "transparent"},
                "stack": "ci_band",
                "z": 1,
            },
            # CI 下界（带 areaStyle，填充到上界）
            {
                "name": "95% CI 下界",
                "type": "line",
                "data": ci_lo_data,
                "showSymbol": False,
                "lineStyle": {"color": "rgba(84,112,198,0.3)", "width": 0},
                "areaStyle": {
                    "color": "rgba(84,112,198,0.15)",
                    "origin": "auto",
                },
                "z": 2,
            },
            # Rug 散点
            {
                "name": "原始数据",
                "type": "scatter",
                "data": rug_data,
                "symbol": "line",
                "symbolSize": [1, 8],
                "itemStyle": {"color": "rgba(150,150,150,0.5)"},
                "z": 0,
            },
        ],
        "graphic": [
            {
                "type": "text",
                "right": "8%",
                "top": "22%",
                "style": {
                    "text": f"P non-linearity = {p_nl_str}\nP overall = {p_ov_str}",
                    "font": "13px sans-serif",
                    "fill": "#444",
                    "textAlign": "right",
                },
            }
        ],
    }

    return ChartResult(title="RCS 曲线", chart_type="line", option=option)


# ─────────────────────────────────────────────────────────────────────────────
# 图表：暴露变量分布直方图
# ─────────────────────────────────────────────────────────────────────────────

def _build_exposure_hist(x_vals: np.ndarray, exposure: str) -> ChartResult:
    """构建暴露变量 20-bin 分布直方图。"""
    n_bins = 20
    counts, bin_edges = np.histogram(x_vals, bins=n_bins)

    bin_centers = [(bin_edges[i] + bin_edges[i + 1]) / 2.0 for i in range(n_bins)]
    categories = [f"{c:.2f}" for c in bin_centers]

    option: dict[str, Any] = {
        "title": {
            "text": f"暴露变量分布（{exposure}）",
            "left": "center",
            "top": "4%",
        },
        "tooltip": {"trigger": "axis"},
        "grid": {"left": "10%", "right": "5%", "top": "18%", "bottom": "14%"},
        "xAxis": {
            "type": "category",
            "data": categories,
            "name": exposure,
            "nameLocation": "center",
            "nameGap": 28,
            "axisLabel": {"rotate": 30, "fontSize": 10},
        },
        "yAxis": {
            "type": "value",
            "name": "频数",
        },
        "series": [
            {
                "name": "频数",
                "type": "bar",
                "data": counts.tolist(),
                "itemStyle": {"color": "rgba(84,112,198,0.75)"},
                "barCategoryGap": "5%",
            }
        ],
    }

    return ChartResult(title="暴露变量分布", chart_type="bar", option=option)
