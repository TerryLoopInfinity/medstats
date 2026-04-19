"""亚组分析与森林图模块。

支持：
  - logistic / cox / linear 三种模型
  - 整体效应估计（全样本拟合）
  - 按亚组变量分层分析（分类变量取唯一水平，连续变量中位数切分）
  - 交互作用 p 值（LRT）
  - Cochran Q 异质性检验 + I²
  - 标准化 forestData 格式，供前端 ForestPlot 组件渲染
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

_HR_CLIP = 500.0  # exp 溢出保护


# ─────────────────────────────────────────────────────────────────────────────
# 溢出安全 exp
# ─────────────────────────────────────────────────────────────────────────────

def _safe_exp(x: float) -> float:
    """np.clip 防止 exp 溢出。"""
    return float(np.exp(np.clip(float(x), -_HR_CLIP, _HR_CLIP)))


# ─────────────────────────────────────────────────────────────────────────────
# 格式辅助
# ─────────────────────────────────────────────────────────────────────────────

def _fmt_p(p: float) -> str:
    if math.isnan(p) or math.isinf(p):
        return "—"
    if p < 0.001:
        return "< 0.001"
    return f"{p:.3f}"


def _fmt_effect(val: float, lo: float, hi: float) -> str:
    return f"{val:.3f} ({lo:.3f}–{hi:.3f})"


def _nan_to_none(v: Any) -> Any:
    """将 float NaN/inf 转为 None，以保证 JSON 安全。"""
    if isinstance(v, float) and (math.isnan(v) or math.isinf(v)):
        return None
    return v


# ─────────────────────────────────────────────────────────────────────────────
# 因变量编码（logistic，同 logistic_reg.py 逻辑）
# ─────────────────────────────────────────────────────────────────────────────

def _encode_outcome(col: pd.Series) -> tuple[pd.Series, str, list[str]]:
    enc_warnings: list[str] = []
    non_null = col.dropna()
    if len(non_null) == 0:
        raise ValueError("因变量全为缺失值")

    if pd.api.types.is_numeric_dtype(col):
        unique_vals = sorted(float(v) for v in non_null.unique())
        if len(unique_vals) > 2:
            raise ValueError(
                f"因变量为数值型但有 {len(unique_vals)} 个唯一值，"
                "logistic 回归要求二分类变量"
            )
        if len(unique_vals) == 1:
            raise ValueError("因变量只有一个唯一值，无法进行 logistic 回归")
        if set(unique_vals) == {0.0, 1.0}:
            return col.astype(float), "1", enc_warnings
        lo, hi = unique_vals[0], unique_vals[1]
        enc_warnings.append(f"因变量数值编码：{lo} → 0，{hi} → 1（阳性类别：{hi}）")
        return (
            col.map(lambda x: {lo: 0.0, hi: 1.0}.get(float(x)) if pd.notna(x) else float("nan")),
            str(hi),
            enc_warnings,
        )

    col_str = col.where(col.isna(), col.astype(str))
    unique_str = sorted(col_str.dropna().unique())
    if len(unique_str) > 2:
        raise ValueError(
            f"因变量有 {len(unique_str)} 个唯一类别，logistic 回归要求二分类变量"
        )
    if len(unique_str) == 1:
        raise ValueError("因变量只有一个唯一值，无法进行 logistic 回归")
    if set(unique_str) == {"False", "True"}:
        return col_str.map({"False": 0.0, "True": 1.0}), "True", enc_warnings
    neg_label, pos_label = unique_str[0], unique_str[1]
    enc_warnings.append(
        f"因变量字符串编码：'{neg_label}' → 0，'{pos_label}' → 1（阳性类别：'{pos_label}'）"
    )
    return col_str.map({neg_label: 0.0, pos_label: 1.0}), pos_label, enc_warnings


# ─────────────────────────────────────────────────────────────────────────────
# Dummy 编码辅助（参考 logistic_reg.py 模式）
# ─────────────────────────────────────────────────────────────────────────────

def _make_dummies(
    df: pd.DataFrame,
    predictors: list[str],
    categorical_vars: list[str],
    ref_categories: dict[str, str] | None = None,
) -> tuple[pd.DataFrame, dict[str, list[str]], list[str]]:
    """返回 (df_with_dummies, dummy_map, warnings)。"""
    ref_categories = ref_categories or {}
    dummy_warnings: list[str] = []
    dummy_map: dict[str, list[str]] = {}
    df_out = df.copy()

    for var in categorical_vars:
        if var not in df.columns or var not in predictors:
            continue

        non_null_mask = df_out[var].notna()
        col_str_vals = df_out.loc[non_null_mask, var].astype(str)
        unique_str = sorted(col_str_vals.unique())

        if len(unique_str) < 2:
            dummy_warnings.append(f"分类变量 '{var}' 唯一值不足，已忽略 dummy 编码")
            continue

        ref = str(ref_categories.get(var, unique_str[0]))
        if ref not in unique_str:
            dummy_warnings.append(
                f"变量 '{var}' 参考组 '{ref}' 不存在，已改用：'{unique_str[0]}'"
            )
            ref = unique_str[0]

        ordered = [ref] + sorted(v for v in unique_str if v != ref)
        col_cat = pd.Categorical(col_str_vals, categories=ordered)
        dummies_nonnull = pd.get_dummies(
            pd.Series(col_cat, index=col_str_vals.index),
            prefix=var, drop_first=True, dtype=float,
        )

        dummy_full = pd.DataFrame(
            np.nan, index=df_out.index, columns=dummies_nonnull.columns
        )
        dummy_full.loc[non_null_mask] = dummies_nonnull.values

        for col_name in dummy_full.columns:
            df_out[col_name] = dummy_full[col_name]

        dummy_map[var] = list(dummies_nonnull.columns)
        dummy_warnings.append(
            f"分类变量 '{var}' dummy 编码完成，参考组：'{ref}'，"
            f"哑变量：{list(dummies_nonnull.columns)}"
        )

    return df_out, dummy_map, dummy_warnings


# ─────────────────────────────────────────────────────────────────────────────
# 模型拟合助手
# ─────────────────────────────────────────────────────────────────────────────

class _FitResult:
    """统一的模型拟合结果容器。"""
    def __init__(
        self,
        beta: float,
        se: float,
        p: float,
        effect: float,
        ci_lo: float,
        ci_hi: float,
        n: int,
        n_events: int,
        llf: float | None = None,
    ):
        self.beta = beta
        self.se = se
        self.p = p
        self.effect = effect
        self.ci_lo = ci_lo
        self.ci_hi = ci_hi
        self.n = n
        self.n_events = n_events
        self.llf = llf  # 用于 LRT 交互检验


def _fit_logistic(
    df_sub: pd.DataFrame,
    outcome_col: str,
    exposure_col: str,
    covariate_cols: list[str],
) -> _FitResult:
    """在 df_sub 上拟合 logistic 回归，返回 exposure 的估计。"""
    import statsmodels.api as sm

    all_xcols = [exposure_col] + covariate_cols
    needed = [outcome_col] + all_xcols
    df_fit = df_sub[needed].dropna()
    n = len(df_fit)
    if n < len(all_xcols) + 3:
        raise ValueError(f"样本量不足（n={n}）")

    y = df_fit[outcome_col].values.astype(float)
    if len(np.unique(y[~np.isnan(y)])) < 2:
        raise ValueError("亚组内因变量只有一个唯一值")

    X = sm.add_constant(df_fit[all_xcols].values.astype(float), prepend=True)
    with _warnings_mod.catch_warnings():
        _warnings_mod.simplefilter("ignore")
        res = sm.Logit(y, X).fit(disp=0, maxiter=300)

    # exposure 在 all_xcols 中的位置 = 1（constant=0）
    beta = float(res.params[1])
    se = float(res.bse[1])
    p = float(res.pvalues[1])
    ci = np.asarray(res.conf_int())
    effect = _safe_exp(beta)
    ci_lo = _safe_exp(float(ci[1, 0]))
    ci_hi = _safe_exp(float(ci[1, 1]))
    n_events = int(y.sum())
    return _FitResult(beta, se, p, effect, ci_lo, ci_hi, n, n_events, llf=float(res.llf))


def _fit_cox(
    df_sub: pd.DataFrame,
    time_col: str,
    event_col: str,
    exposure_col: str,
    covariate_cols: list[str],
) -> _FitResult:
    """在 df_sub 上拟合 Cox 回归，返回 exposure 的估计。"""
    from lifelines import CoxPHFitter

    all_xcols = [exposure_col] + covariate_cols
    needed = [time_col, event_col] + all_xcols
    df_fit = df_sub[needed].dropna()
    df_fit = df_fit[df_fit[time_col] > 0]
    df_fit = df_fit[df_fit[event_col].isin([0, 1])]
    n = len(df_fit)
    n_events = int(df_fit[event_col].sum())
    if n < len(all_xcols) + 5 or n_events < 3:
        raise ValueError(f"样本量或事件数不足（n={n}, events={n_events}）")

    cph = CoxPHFitter(penalizer=0.01)
    with _warnings_mod.catch_warnings():
        _warnings_mod.simplefilter("ignore")
        cph.fit(df_fit, duration_col=time_col, event_col=event_col)

    summ = cph.summary
    if exposure_col not in summ.index:
        raise ValueError(f"exposure '{exposure_col}' 不在 Cox 系数表中")

    row = summ.loc[exposure_col]
    beta = float(row.get("coef", 0.0))
    se = float(row.get("se(coef)", row.get("se", 0.0)))
    p = float(row.get("p", row.get("p-value", 1.0)))
    beta_c = float(np.clip(beta, -_HR_CLIP, _HR_CLIP))
    effect = float(np.exp(beta_c))

    ci_lo_keys = [k for k in row.index if "lower" in k.lower()]
    ci_hi_keys = [k for k in row.index if "upper" in k.lower()]
    if ci_lo_keys and ci_hi_keys:
        ci_lo = float(np.exp(np.clip(float(row[ci_lo_keys[0]]), -_HR_CLIP, _HR_CLIP)))
        ci_hi = float(np.exp(np.clip(float(row[ci_hi_keys[0]]), -_HR_CLIP, _HR_CLIP)))
    else:
        ci_lo = _safe_exp(beta - 1.96 * se)
        ci_hi = _safe_exp(beta + 1.96 * se)

    try:
        llf = float(cph.log_likelihood_)
    except Exception:
        llf = None

    return _FitResult(beta, se, p, effect, ci_lo, ci_hi, n, n_events, llf=llf)


def _fit_linear(
    df_sub: pd.DataFrame,
    outcome_col: str,
    exposure_col: str,
    covariate_cols: list[str],
) -> _FitResult:
    """在 df_sub 上拟合 OLS，返回 exposure 的估计（β 直接报告）。"""
    import statsmodels.api as sm

    all_xcols = [exposure_col] + covariate_cols
    needed = [outcome_col] + all_xcols
    df_fit = df_sub[needed].dropna()
    n = len(df_fit)
    if n < len(all_xcols) + 3:
        raise ValueError(f"样本量不足（n={n}）")

    y = df_fit[outcome_col].values.astype(float)
    X = sm.add_constant(df_fit[all_xcols].values.astype(float), prepend=True)
    with _warnings_mod.catch_warnings():
        _warnings_mod.simplefilter("ignore")
        res = sm.OLS(y, X).fit()

    beta = float(res.params[1])
    se = float(res.bse[1])
    p = float(res.pvalues[1])
    ci = np.asarray(res.conf_int())
    effect = beta
    ci_lo = float(ci[1, 0])
    ci_hi = float(ci[1, 1])
    return _FitResult(beta, se, p, effect, ci_lo, ci_hi, n, 0, llf=float(res.llf))


def _fit_model(
    df_sub: pd.DataFrame,
    model_type: str,
    outcome_col: str,
    exposure_col: str,
    covariate_cols: list[str],
    time_col: str = "",
    event_col: str = "",
) -> _FitResult:
    """统一分发到各模型拟合函数。"""
    if model_type == "logistic":
        return _fit_logistic(df_sub, outcome_col, exposure_col, covariate_cols)
    elif model_type == "cox":
        return _fit_cox(df_sub, time_col, event_col, exposure_col, covariate_cols)
    elif model_type == "linear":
        return _fit_linear(df_sub, outcome_col, exposure_col, covariate_cols)
    else:
        raise ValueError(f"不支持的模型类型：{model_type}")


# ─────────────────────────────────────────────────────────────────────────────
# 交互作用 LRT p 值
# ─────────────────────────────────────────────────────────────────────────────

def _interaction_lrt_p(
    df_full: pd.DataFrame,
    model_type: str,
    outcome_col: str,
    exposure_col: str,
    covariate_cols: list[str],
    sv: str,
    sv_levels: list[str],
    is_categorical: bool,
    time_col: str = "",
    event_col: str = "",
) -> float:
    """
    LRT 交互检验：模型含 exposure × sv 交互项 vs 不含。
    返回最小交互 p（对多水平分类变量）或直接的交互 p（连续/二元）。
    """
    import statsmodels.api as sm

    # 对于 cox，LRT 用 log-likelihood 近似；
    # 实现：先按线性模型做交互 LRT（logistic/linear），cox 用 Wald 近似。

    try:
        if not is_categorical:
            # 连续 → 中位数二元化
            median_val = float(df_full[sv].median())
            bin_col = f"__sv_bin_{sv}__"
            df_aug = df_full.copy()
            df_aug[bin_col] = (df_aug[sv] >= median_val).astype(float)
            inter_col = f"__inter_{sv}__"
            df_aug[inter_col] = df_aug[exposure_col] * df_aug[bin_col]
            sv_dummy_cols = [bin_col]
            inter_cols = [inter_col]
        else:
            # 分类变量：创建 dummy
            df_aug = df_full.copy()
            ref = sv_levels[0]
            non_ref = sv_levels[1:]
            sv_dummy_cols = []
            inter_cols = []
            for cat in non_ref:
                d_col = f"__sv_d_{sv}_{cat}__"
                df_aug[d_col] = (df_aug[sv].astype(str) == cat).astype(float)
                i_col = f"__inter_{sv}_{cat}__"
                df_aug[i_col] = df_aug[exposure_col] * df_aug[d_col]
                sv_dummy_cols.append(d_col)
                inter_cols.append(i_col)

        base_xcols = [exposure_col] + covariate_cols + sv_dummy_cols
        full_xcols = base_xcols + inter_cols

        if model_type == "cox":
            # For Cox, use Wald test on the interaction coefficient
            from lifelines import CoxPHFitter
            needed = [time_col, event_col] + full_xcols
            df_fit = df_aug[needed].dropna()
            df_fit = df_fit[df_fit[time_col] > 0]
            df_fit = df_fit[df_fit[event_col].isin([0, 1])]
            if len(df_fit) < len(full_xcols) + 5:
                return float("nan")
            cph = CoxPHFitter(penalizer=0.01)
            with _warnings_mod.catch_warnings():
                _warnings_mod.simplefilter("ignore")
                cph.fit(df_fit, duration_col=time_col, event_col=event_col)
            summ = cph.summary
            p_vals = []
            for ic in inter_cols:
                if ic in summ.index:
                    p_vals.append(float(summ.loc[ic, "p"]))
            return min(p_vals) if p_vals else float("nan")

        # logistic / linear — LRT
        if model_type == "logistic":
            needed_base = [outcome_col] + base_xcols
            needed_full = [outcome_col] + full_xcols
            df_base_fit = df_aug[needed_base].dropna()
            df_full_fit = df_aug[needed_full].dropna()
            # 取相同行
            common_idx = df_base_fit.index.intersection(df_full_fit.index)
            df_base_fit = df_base_fit.loc[common_idx]
            df_full_fit = df_full_fit.loc[common_idx]
            if len(common_idx) < len(full_xcols) + 3:
                return float("nan")
            y = df_base_fit[outcome_col].values.astype(float)
            Xb = sm.add_constant(df_base_fit[base_xcols].values.astype(float), prepend=True)
            Xf = sm.add_constant(df_full_fit[full_xcols].values.astype(float), prepend=True)
            with _warnings_mod.catch_warnings():
                _warnings_mod.simplefilter("ignore")
                res_base = sm.Logit(y, Xb).fit(disp=0, maxiter=300)
                res_full = sm.Logit(y, Xf).fit(disp=0, maxiter=300)
            lr_stat = 2.0 * (float(res_full.llf) - float(res_base.llf))
            df_lrt = len(inter_cols)
            return float(stats.chi2.sf(lr_stat, df_lrt))

        else:  # linear
            needed_base = [outcome_col] + base_xcols
            needed_full = [outcome_col] + full_xcols
            df_base_fit = df_aug[needed_base].dropna()
            df_full_fit = df_aug[needed_full].dropna()
            common_idx = df_base_fit.index.intersection(df_full_fit.index)
            df_base_fit = df_base_fit.loc[common_idx]
            df_full_fit = df_full_fit.loc[common_idx]
            if len(common_idx) < len(full_xcols) + 3:
                return float("nan")
            y = df_base_fit[outcome_col].values.astype(float)
            Xb = sm.add_constant(df_base_fit[base_xcols].values.astype(float), prepend=True)
            Xf = sm.add_constant(df_full_fit[full_xcols].values.astype(float), prepend=True)
            with _warnings_mod.catch_warnings():
                _warnings_mod.simplefilter("ignore")
                res_base = sm.OLS(y, Xb).fit()
                res_full = sm.OLS(y, Xf).fit()
            lr_stat = 2.0 * (float(res_full.llf) - float(res_base.llf))
            df_lrt = len(inter_cols)
            return float(stats.chi2.sf(lr_stat, df_lrt))

    except Exception as exc:
        logger.debug("interaction LRT 失败 (sv=%s): %s", sv, exc)
        return float("nan")


# ─────────────────────────────────────────────────────────────────────────────
# Cochran Q 检验
# ─────────────────────────────────────────────────────────────────────────────

def _cochran_q(betas: list[float], ses: list[float]) -> tuple[float, float, float]:
    """
    返回 (Q, p_Q, I²)。
    betas/ses: 各水平的 β 和 SE。
    """
    if len(betas) < 2:
        return float("nan"), float("nan"), float("nan")

    b = np.array(betas, dtype=float)
    s = np.array(ses, dtype=float)
    # 排除 nan/0 SE
    valid = (~np.isnan(b)) & (~np.isnan(s)) & (s > 0)
    if valid.sum() < 2:
        return float("nan"), float("nan"), float("nan")

    b = b[valid]
    s = s[valid]
    w = 1.0 / (s ** 2)
    beta_w = float(np.sum(b * w) / np.sum(w))
    Q = float(np.sum(w * (b - beta_w) ** 2))
    df = len(b) - 1
    p_Q = float(stats.chi2.sf(Q, df))
    i2 = float(max(0.0, (Q - df) / Q)) if Q > 0 else 0.0
    return Q, p_Q, i2


# ─────────────────────────────────────────────────────────────────────────────
# 主入口
# ─────────────────────────────────────────────────────────────────────────────

def run(df: pd.DataFrame, params: dict) -> AnalysisResult:
    """
    参数
    ----
    params["model_type"]       str          "logistic" | "cox" | "linear"
    params["outcome"]          str          因变量（logistic / linear）
    params["time_col"]         str          时间变量（cox）
    params["event_col"]        str          事件变量（cox，0=删失 1=事件）
    params["exposure"]         str          暴露变量（关键预测变量）
    params["covariates"]       list[str]    调整协变量
    params["subgroup_vars"]    list[str]    亚组分层变量
    params["categorical_vars"] list[str]    分类变量（做 dummy 编码）
    """
    model_type: str = str(params.get("model_type", "logistic")).lower()
    outcome: str = str(params.get("outcome", ""))
    time_col: str = str(params.get("time_col", ""))
    event_col: str = str(params.get("event_col", ""))
    exposure: str = str(params.get("exposure", ""))
    covariates: list[str] = list(params.get("covariates") or [])
    subgroup_vars: list[str] = list(params.get("subgroup_vars") or [])
    categorical_vars: list[str] = list(params.get("categorical_vars") or [])
    warnings: list[str] = []

    # ── 参数校验 ──────────────────────────────────────────────────────────────
    if model_type not in ("logistic", "cox", "linear"):
        raise ValueError("model_type 必须为 logistic / cox / linear")
    if not exposure:
        raise ValueError("请指定暴露变量 (exposure)")
    if exposure not in df.columns:
        raise ValueError(f"暴露变量 '{exposure}' 不存在于数据集中")
    if not subgroup_vars:
        raise ValueError("请至少指定一个亚组变量 (subgroup_vars)")

    if model_type == "logistic":
        if not outcome:
            raise ValueError("logistic 模型请指定因变量 (outcome)")
        if outcome not in df.columns:
            raise ValueError(f"因变量 '{outcome}' 不存在于数据集中")
        # 编码因变量
        y_encoded, positive_label, enc_warnings = _encode_outcome(df[outcome])
        warnings.extend(enc_warnings)
        df = df.copy()
        df["__outcome__"] = y_encoded
        outcome_col_work = "__outcome__"
    elif model_type == "cox":
        if not time_col:
            raise ValueError("cox 模型请指定时间变量 (time_col)")
        if not event_col:
            raise ValueError("cox 模型请指定事件变量 (event_col)")
        if time_col not in df.columns:
            raise ValueError(f"时间变量 '{time_col}' 不存在于数据集中")
        if event_col not in df.columns:
            raise ValueError(f"事件变量 '{event_col}' 不存在于数据集中")
        outcome_col_work = ""
    else:  # linear
        if not outcome:
            raise ValueError("linear 模型请指定因变量 (outcome)")
        if outcome not in df.columns:
            raise ValueError(f"因变量 '{outcome}' 不存在于数据集中")
        outcome_col_work = outcome

    # 排除 covariates 中与 exposure 相同的变量
    covariates = [c for c in covariates if c != exposure]
    # 排除不在数据集中的协变量
    valid_covariates: list[str] = []
    for c in covariates:
        if c not in df.columns:
            warnings.append(f"协变量 '{c}' 不存在于数据集中，已忽略")
        else:
            valid_covariates.append(c)
    covariates = valid_covariates

    # Dummy 编码（对协变量中的分类变量）
    cov_cat_vars = [v for v in categorical_vars if v in covariates]
    df_work, dummy_map, dum_warnings = _make_dummies(df, covariates, cov_cat_vars)
    warnings.extend(dum_warnings)

    # 展开协变量列（含 dummy）
    expanded_covariates: list[str] = []
    for c in covariates:
        if c in dummy_map:
            expanded_covariates.extend(dummy_map[c])
        elif pd.api.types.is_numeric_dtype(df_work[c]):
            expanded_covariates.append(c)
        else:
            warnings.append(f"协变量 '{c}' 非数值型且未标记为分类变量，已忽略")

    n_total = len(df_work)

    # ── 整体效应 ──────────────────────────────────────────────────────────────
    overall_fit: _FitResult | None = None
    try:
        overall_fit = _fit_model(
            df_work, model_type,
            outcome_col=outcome_col_work,
            exposure_col=exposure,
            covariate_cols=expanded_covariates,
            time_col=time_col,
            event_col=event_col,
        )
    except Exception as exc:
        warnings.append(f"整体效应拟合失败：{exc}")

    # 效应标签
    if model_type == "logistic":
        effect_label = "OR"
    elif model_type == "cox":
        effect_label = "HR"
    else:
        effect_label = "β"

    null_line = 0.0 if model_type == "linear" else 1.0
    x_label = f"{effect_label} (95% CI)"

    # ── 亚组分析主循环 ────────────────────────────────────────────────────────

    # forestData 全局列表（含 overall + 所有亚组）
    forest_data: list[dict[str, Any]] = []

    # 加入整体效应行
    if overall_fit is not None:
        forest_data.append({
            "label": "总体",
            "effect": _nan_to_none(overall_fit.effect),
            "ci_lo": _nan_to_none(overall_fit.ci_lo),
            "ci_hi": _nan_to_none(overall_fit.ci_hi),
            "p": _nan_to_none(overall_fit.p),
            "n": overall_fit.n,
            "isOverall": True,
            "effectLabel": _fmt_effect(overall_fit.effect, overall_fit.ci_lo, overall_fit.ci_hi),
        })

    # 亚组汇总表行
    subgroup_rows: list[list[Any]] = []

    for sv in subgroup_vars:
        if sv not in df_work.columns:
            warnings.append(f"亚组变量 '{sv}' 不存在于数据集中，已跳过")
            continue

        is_categorical = sv in categorical_vars

        # 确定水平
        if is_categorical:
            raw_levels = sorted(df_work[sv].dropna().astype(str).unique().tolist())
            level_labels = raw_levels
            level_filter_vals = raw_levels  # 用字符串过滤
        else:
            # 连续变量：中位数切分
            median_val = float(df_work[sv].dropna().median())
            level_labels = [f"< {median_val:.4g}", f"≥ {median_val:.4g}"]
            level_filter_vals = ["low", "high"]  # 内部标记，实际用 mask

        # 交互作用 p 值
        p_interaction = _interaction_lrt_p(
            df_full=df_work,
            model_type=model_type,
            outcome_col=outcome_col_work,
            exposure_col=exposure,
            covariate_cols=expanded_covariates,
            sv=sv,
            sv_levels=raw_levels if is_categorical else level_labels,
            is_categorical=is_categorical,
            time_col=time_col,
            event_col=event_col,
        )

        # 收集各水平 β / SE，用于 Cochran Q
        betas_for_q: list[float] = []
        ses_for_q: list[float] = []
        level_results: list[dict[str, Any]] = []  # 存储水平级别结果

        total_n_sv = int(df_work[sv].notna().sum())
        total_events_sv = 0
        if model_type == "cox" and event_col in df_work.columns:
            total_events_sv = int(
                df_work.loc[df_work[sv].notna(), event_col]
                .astype(float).sum()
            )
        elif model_type == "logistic" and outcome_col_work in df_work.columns:
            total_events_sv = int(
                df_work.loc[df_work[sv].notna(), outcome_col_work]
                .astype(float).sum()
            )

        for i, lbl in enumerate(level_labels):
            # 按水平过滤
            if is_categorical:
                mask = df_work[sv].astype(str) == level_filter_vals[i]
            else:
                if level_filter_vals[i] == "low":
                    mask = df_work[sv] < median_val
                else:
                    mask = df_work[sv] >= median_val

            df_lv = df_work[mask].copy()

            try:
                fit = _fit_model(
                    df_lv, model_type,
                    outcome_col=outcome_col_work,
                    exposure_col=exposure,
                    covariate_cols=expanded_covariates,
                    time_col=time_col,
                    event_col=event_col,
                )
                betas_for_q.append(fit.beta)
                ses_for_q.append(fit.se)
                level_results.append({
                    "label": lbl,
                    "n": fit.n,
                    "events": fit.n_events,
                    "effect": fit.effect,
                    "ci_lo": fit.ci_lo,
                    "ci_hi": fit.ci_hi,
                    "p": fit.p,
                    "beta": fit.beta,
                    "se": fit.se,
                    "failed": False,
                })
            except Exception as exc:
                warnings.append(f"亚组 '{sv}={lbl}' 拟合失败：{exc}")
                betas_for_q.append(float("nan"))
                ses_for_q.append(float("nan"))
                level_results.append({
                    "label": lbl,
                    "n": int(mask.sum()),
                    "events": 0,
                    "effect": float("nan"),
                    "ci_lo": float("nan"),
                    "ci_hi": float("nan"),
                    "p": float("nan"),
                    "beta": float("nan"),
                    "se": float("nan"),
                    "failed": True,
                })

        # Cochran Q
        valid_betas = [b for b in betas_for_q if not math.isnan(b)]
        valid_ses = [s for b, s in zip(betas_for_q, ses_for_q) if not math.isnan(b)]
        Q_stat, p_Q, i2 = _cochran_q(valid_betas, valid_ses)

        # ── 汇总表：亚组变量标题行 ─────────────────────────────────────────────
        subgroup_rows.append([
            sv,
            "—",
            str(total_n_sv),
            str(total_events_sv),
            "—",
            "—",
            "—",
            _fmt_p(p_interaction),
            _fmt_p(p_Q),
        ])

        # forestData 亚组标题行
        forest_data.append({"label": sv, "isHeader": True})

        for lr in level_results:
            eff = lr["effect"]
            lo = lr["ci_lo"]
            hi = lr["ci_hi"]
            p = lr["p"]
            failed = lr["failed"]

            eff_label = (
                _fmt_effect(eff, lo, hi)
                if not failed and not math.isnan(eff)
                else "—"
            )

            # 汇总表水平行
            subgroup_rows.append([
                f"  {sv}",
                lr["label"],
                str(lr["n"]),
                str(lr["events"]),
                f"{eff:.3f}" if not failed and not math.isnan(eff) else "—",
                f"({lo:.3f}, {hi:.3f})" if not failed and not math.isnan(lo) else "—",
                _fmt_p(p) if not failed else "—",
                "",  # p_interaction 只在变量标题行显示
                "",
            ])

            # forestData 水平行
            forest_data.append({
                "label": lr["label"],
                "effect": _nan_to_none(eff),
                "ci_lo": _nan_to_none(lo),
                "ci_hi": _nan_to_none(hi),
                "p": _nan_to_none(p),
                "n": lr["n"],
                "events": lr["events"],
                "isOverall": False,
                "effectLabel": eff_label,
                "p_interaction": _nan_to_none(p_interaction),
            })

    # ── 构建输出表格 ──────────────────────────────────────────────────────────
    tables: list[TableResult] = []

    # 整体效应表
    if overall_fit is not None:
        tables.append(TableResult(
            title="整体效应",
            headers=["效应指标", "效应值", "95% CI", "p 值"],
            rows=[[
                effect_label,
                f"{overall_fit.effect:.3f}",
                f"({overall_fit.ci_lo:.3f}, {overall_fit.ci_hi:.3f})",
                _fmt_p(overall_fit.p),
            ]],
        ))

    # 亚组分析汇总表
    if subgroup_rows:
        tables.append(TableResult(
            title="亚组分析汇总表",
            headers=[
                "亚组变量", "水平", "n", "事件/阳性数",
                "效应值", "95% CI", "p 值",
                "P interaction", "Cochran Q p",
            ],
            rows=subgroup_rows,
        ))

    # ── 构建图表 ──────────────────────────────────────────────────────────────
    charts: list[ChartResult] = []
    if forest_data:
        option: dict[str, Any] = {
            "forestData": forest_data,
            "nullLine": null_line,
            "xLabel": x_label,
            "title": "亚组分析 & 森林图",
            "modelType": model_type,
        }
        charts.append(ChartResult(
            title="亚组分析森林图",
            chart_type="forest_plot",
            option=option,
        ))

    # ── 摘要 ──────────────────────────────────────────────────────────────────
    n_used = overall_fit.n if overall_fit is not None else n_total
    if overall_fit is not None:
        summary = (
            f"亚组分析：{model_type} 模型，暴露变量 {exposure}，"
            f"共 {len(subgroup_vars)} 个亚组变量，有效样本量 n = {n_used}。"
            f"整体效应：{effect_label} = {overall_fit.effect:.3f} "
            f"({overall_fit.ci_lo:.3f}–{overall_fit.ci_hi:.3f})，"
            f"p = {_fmt_p(overall_fit.p)}。"
        )
    else:
        summary = (
            f"亚组分析：{model_type} 模型，暴露变量 {exposure}，"
            f"共 {len(subgroup_vars)} 个亚组变量，n = {n_used}。整体效应拟合失败，请检查数据。"
        )

    return AnalysisResult(
        method="forest_plot",
        tables=tables,
        charts=charts,
        summary=summary,
        warnings=warnings,
    )
