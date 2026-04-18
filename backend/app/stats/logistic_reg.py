"""Logistic 回归分析模块。

支持：
  - 因变量自动编码（0/1、True/False、两个字符串类别）
  - 分类变量 dummy 编码，可指定参考组
  - 单变量分析：每个自变量单独与因变量做 logistic 回归
  - 多变量分析：强制进入法
  - OR（优势比）= exp(β) 及 95% CI，标准化 OR（连续变量按 1 SD）
  - 模型拟合指标：Hosmer-Lemeshow、McFadden/Cox-Snell/Nagelkerke R²、AIC/BIC、LRT
  - 分类性能：ROC/AUC（DeLong CI）、Youden 最佳截断点、
    敏感性/特异性/PPV/NPV/准确率、混淆矩阵
  - 共线性诊断（VIF）
  - 图表：ROC 曲线、预测概率分布直方图、OR 森林图、校准曲线、混淆矩阵热力图
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
    """np.clip 防止 exp 溢出 (math range error)。"""
    return float(np.exp(np.clip(float(x), -500.0, 500.0)))


# ─────────────────────────────────────────────────────────────────────────────
# 主入口
# ─────────────────────────────────────────────────────────────────────────────

def run(df: pd.DataFrame, params: dict) -> AnalysisResult:
    """
    参数
    ----
    params["outcome"]           str           因变量（二分类）
    params["predictors"]        list[str]     自变量列表
    params["categorical_vars"]  list[str]     需要 dummy 编码的分类变量
    params["ref_categories"]    dict[str,str] 参考组 {变量名: 参考类别}
    params["mode"]              str           "both" | "univariate" | "multivariate"
    """
    outcome: str = str(params.get("outcome", ""))
    predictors: list[str] = list(params.get("predictors") or [])
    categorical_vars: list[str] = list(params.get("categorical_vars") or [])
    ref_categories: dict[str, str] = dict(params.get("ref_categories") or {})
    mode: str = str(params.get("mode", "both")).lower()
    warnings: list[str] = []

    # ── 参数校验 ──────────────────────────────────────────────────────────────
    if not outcome:
        raise ValueError("请指定因变量 (outcome)")
    if outcome not in df.columns:
        raise ValueError(f"因变量 '{outcome}' 不存在于数据集中")
    if not predictors:
        raise ValueError("请至少选择一个自变量 (predictors)")
    if mode not in ("both", "univariate", "multivariate"):
        raise ValueError("mode 必须为 both / univariate / multivariate")

    # ── 编码因变量 ────────────────────────────────────────────────────────────
    y_encoded, positive_label, enc_warnings = _encode_outcome(df[outcome])
    warnings.extend(enc_warnings)

    # ── 过滤有效自变量 ────────────────────────────────────────────────────────
    valid_preds: list[str] = []
    for v in predictors:
        if v == outcome:
            warnings.append(f"自变量 '{v}' 与因变量相同，已忽略")
            continue
        if v not in df.columns:
            warnings.append(f"自变量 '{v}' 不存在，已忽略")
            continue
        valid_preds.append(v)

    if not valid_preds:
        raise ValueError("没有有效的自变量，请检查变量选择")

    valid_cat_vars = [v for v in categorical_vars if v in valid_preds]

    # ── dummy 编码 ────────────────────────────────────────────────────────────
    df_work, dummy_map, dummy_warnings = _make_dummies(
        df, valid_preds, valid_cat_vars, ref_categories
    )
    warnings.extend(dummy_warnings)

    # 展开所有设计矩阵列；非数值非分类变量跳过
    all_expanded: list[str] = []
    cont_vars: list[str] = []
    skipped: list[str] = []
    for v in valid_preds:
        if v in dummy_map:
            all_expanded.extend(dummy_map[v])
        else:
            if not pd.api.types.is_numeric_dtype(df[v]):
                warnings.append(f"自变量 '{v}' 非数值型且未标记为分类变量，已忽略")
                skipped.append(v)
                continue
            all_expanded.append(v)
            cont_vars.append(v)

    valid_preds = [v for v in valid_preds if v not in skipped]
    if not all_expanded:
        raise ValueError("没有有效的自变量（数值型或已 dummy 编码），请检查配置")

    # ── 工作数据集（加入编码后的 y） ───────────────────────────────────────────
    df_work = df_work.copy()
    df_work["__y__"] = y_encoded

    tables: list[TableResult] = []
    charts: list[ChartResult] = []
    multi_result: dict[str, Any] | None = None

    # ── 单变量分析 ────────────────────────────────────────────────────────────
    if mode in ("both", "univariate"):
        uni_rows, uni_warnings = _univariate_analysis(
            df_work, valid_preds, dummy_map, cont_vars
        )
        warnings.extend(uni_warnings)
        tables.append(TableResult(
            title="单变量 Logistic 回归",
            headers=["变量", "β", "SE", "Wald z", "p 值", "OR", "95% CI", "标准化 OR"],
            rows=uni_rows,
        ))

    # ── 多变量分析 ────────────────────────────────────────────────────────────
    if mode in ("both", "multivariate"):
        needed = ["__y__"] + all_expanded
        df_multi = df_work[needed].dropna()
        n_multi = len(df_multi)
        if n_multi < len(all_expanded) + 2:
            raise ValueError(
                f"有效样本量 ({n_multi}) 不足以拟合模型（参数数：{len(all_expanded)}）"
            )

        y_multi = df_multi["__y__"].values.astype(float)
        try:
            multi_result = _multivariate_analysis(
                df_multi, y_multi, all_expanded, dummy_map, valid_preds, cont_vars
            )
        except ValueError:
            raise
        except Exception as exc:
            raise ValueError(f"多变量 logistic 回归失败：{exc}") from exc

        warnings.extend(multi_result.get("sep_warnings", []))
        tables.extend([
            TableResult(
                title="多变量 Logistic 回归 — 回归系数",
                headers=["变量", "β", "SE", "Wald z", "p 值", "OR", "95% CI", "标准化 OR"],
                rows=multi_result["coef_rows"],
            ),
            TableResult(
                title="多变量 Logistic 回归 — 模型拟合指标",
                headers=["指标", "值"],
                rows=multi_result["fit_rows"],
            ),
            TableResult(
                title="多变量 Logistic 回归 — 分类性能",
                headers=["指标", "值"],
                rows=multi_result["perf_rows"],
            ),
            TableResult(
                title=f"混淆矩阵（截断点 = {multi_result['best_threshold']:.3f}）",
                headers=["", "预测阴性 (0)", "预测阳性 (1)"],
                rows=multi_result["cm_rows"],
            ),
        ])
        if len(all_expanded) > 1 and multi_result["vif_rows"]:
            tables.append(TableResult(
                title="共线性诊断（VIF）",
                headers=["自变量", "VIF", "判断"],
                rows=multi_result["vif_rows"],
            ))

        charts.extend(_build_all_charts(
            y_multi,
            multi_result["y_pred_proba"],
            multi_result["roc_tuple"],
            multi_result["coef_data_for_forest"],
            multi_result["best_threshold"],
        ))

    # ── 摘要 ──────────────────────────────────────────────────────────────────
    n_used = df_work[["__y__"] + all_expanded].dropna().shape[0]
    if mode in ("both", "multivariate") and multi_result:
        fit_dict = {r[0]: r[1] for r in multi_result["fit_rows"]}
        auc_str = fit_dict.get("AUC", "—")
        summary = (
            f"多变量 Logistic 回归，因变量：{outcome}（阳性类别：{positive_label}），"
            f"纳入 {len(valid_preds)} 个自变量，有效样本量 n = {n_used}，AUC = {auc_str}。"
        )
        sig_vars = [
            r[0] for r in multi_result["coef_rows"]
            if r[0] and not str(r[0]).startswith("  ")
            and "参考" not in str(r[0])
            and len(r) > 4
            and _parse_p(str(r[4])) < 0.05
        ]
        if sig_vars:
            summary += f"统计显著变量（p < 0.05）：{', '.join(sig_vars)}。"
    else:
        summary = (
            f"单变量 Logistic 回归，因变量：{outcome}（阳性类别：{positive_label}），"
            f"共分析 {len(valid_preds)} 个自变量，有效样本量 n = {n_used}。"
        )

    return AnalysisResult(
        method="logistic_reg",
        tables=tables,
        charts=charts,
        summary=summary,
        warnings=warnings,
    )


# ─────────────────────────────────────────────────────────────────────────────
# 因变量编码
# ─────────────────────────────────────────────────────────────────────────────

def _encode_outcome(col: pd.Series) -> tuple[pd.Series, str, list[str]]:
    """将因变量编码为 0/1 浮点型 Series（保留 NaN 和原始索引）。"""
    enc_warnings: list[str] = []
    non_null = col.dropna()
    if len(non_null) == 0:
        raise ValueError("因变量全为缺失值")

    # ── 数值型 ──
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

    # ── 字符串 / 布尔型 ──
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
# 分类变量 dummy 编码
# ─────────────────────────────────────────────────────────────────────────────

def _make_dummies(
    df: pd.DataFrame,
    predictors: list[str],
    categorical_vars: list[str],
    ref_categories: dict[str, str],
) -> tuple[pd.DataFrame, dict[str, list[str]], list[str]]:
    """返回 (df_with_dummies, dummy_map, warnings)。"""
    dummy_warnings: list[str] = []
    dummy_map: dict[str, list[str]] = {}
    df_out = df.copy()

    for var in categorical_vars:
        if var not in df.columns:
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

        # 构造全索引 DataFrame，原始缺失行保留为 NaN
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
# 单变量分析
# ─────────────────────────────────────────────────────────────────────────────

def _univariate_analysis(
    df_work: pd.DataFrame,
    predictors: list[str],
    dummy_map: dict[str, list[str]],
    cont_vars: list[str],
) -> tuple[list[list[Any]], list[str]]:
    import statsmodels.api as sm

    rows: list[list[Any]] = []
    extra_warnings: list[str] = []

    for pred in predictors:
        if pred in dummy_map:
            dummy_cols = dummy_map[pred]
            sub = df_work[["__y__"] + dummy_cols].dropna()
            if len(sub) < len(dummy_cols) + 2:
                extra_warnings.append(f"'{pred}' 有效样本量不足，已跳过单变量分析")
                continue

            y = sub["__y__"].values.astype(float)
            X = sm.add_constant(sub[dummy_cols].values.astype(float), prepend=True)

            with _warnings_mod.catch_warnings():
                _warnings_mod.simplefilter("ignore")
                try:
                    res = sm.Logit(y, X).fit(disp=0, maxiter=200)
                except Exception as exc:
                    extra_warnings.append(f"'{pred}' 单变量分析失败：{exc}")
                    continue

            ci_arr = np.asarray(res.conf_int())
            rows.append([f"{pred}（参考组见注）", "", "", "", "", "1 (Ref)", "", "—"])
            for i, dc in enumerate(dummy_cols):
                idx = i + 1
                beta = float(res.params[idx])
                se = float(res.bse[idx])
                wald = beta / se if se > 0 else float("nan")
                p = float(res.pvalues[idx])
                or_val = _safe_exp(beta)
                or_lo = _safe_exp(float(ci_arr[idx, 0]))
                or_hi = _safe_exp(float(ci_arr[idx, 1]))
                level = dc[len(pred) + 1:]
                rows.append([
                    f"  {level}",
                    f"{beta:.4f}", f"{se:.4f}", f"{wald:.3f}",
                    _fmt_p(p), f"{or_val:.4f}",
                    f"[{or_lo:.4f}, {or_hi:.4f}]", "—",
                ])
        else:
            if pred not in cont_vars:
                continue
            sub = df_work[["__y__", pred]].dropna()
            if len(sub) < 4:
                extra_warnings.append(f"'{pred}' 有效样本量不足（n={len(sub)}），已跳过")
                continue

            y = sub["__y__"].values.astype(float)
            x = sub[pred].values.astype(float)
            X = sm.add_constant(x, prepend=True)

            with _warnings_mod.catch_warnings():
                _warnings_mod.simplefilter("ignore")
                try:
                    res = sm.Logit(y, X).fit(disp=0, maxiter=200)
                except Exception as exc:
                    extra_warnings.append(f"'{pred}' 单变量分析失败：{exc}")
                    continue

            ci_arr = np.asarray(res.conf_int())
            beta = float(res.params[1])
            se = float(res.bse[1])
            wald = beta / se if se > 0 else float("nan")
            p = float(res.pvalues[1])
            or_val = _safe_exp(beta)
            or_lo = _safe_exp(float(ci_arr[1, 0]))
            or_hi = _safe_exp(float(ci_arr[1, 1]))
            sd = float(np.std(x, ddof=1)) or 1.0
            or_std = _safe_exp(beta * sd)
            rows.append([
                pred,
                f"{beta:.4f}", f"{se:.4f}", f"{wald:.3f}",
                _fmt_p(p), f"{or_val:.4f}",
                f"[{or_lo:.4f}, {or_hi:.4f}]", f"{or_std:.4f}",
            ])

    return rows, extra_warnings


# ─────────────────────────────────────────────────────────────────────────────
# 多变量分析
# ─────────────────────────────────────────────────────────────────────────────

def _multivariate_analysis(
    df_clean: pd.DataFrame,
    y: np.ndarray,
    all_cols: list[str],
    dummy_map: dict[str, list[str]],
    orig_preds: list[str],
    cont_vars: list[str],
) -> dict[str, Any]:
    import statsmodels.api as sm
    from sklearn.metrics import roc_curve, confusion_matrix

    X_raw = df_clean[all_cols].values.astype(float)
    X = sm.add_constant(X_raw, prepend=True)
    n = len(y)

    with _warnings_mod.catch_warnings():
        _warnings_mod.simplefilter("ignore")
        model_fit = sm.Logit(y, X).fit(disp=0, maxiter=300)

    y_pred_proba = np.asarray(model_fit.predict(X), dtype=float)
    ci_arr = np.asarray(model_fit.conf_int())

    col_sds = {
        c: float(np.std(df_clean[c].values.astype(float), ddof=1)) or 1.0
        for c in all_cols
    }
    col_to_idx = {c: i + 1 for i, c in enumerate(all_cols)}

    # ── 回归系数表 ────────────────────────────────────────────────────────────
    coef_rows: list[list[Any]] = []
    coef_data_for_forest: list[dict[str, Any]] = []
    processed_cat: set[str] = set()

    for orig in orig_preds:
        if orig in dummy_map:
            if orig not in processed_cat:
                coef_rows.append([f"{orig}（参考组）", "", "", "", "", "1 (Ref)", "", "—"])
                coef_data_for_forest.append(
                    {"label": f"{orig}（参考）", "is_header": True,
                     "or_val": None, "ci_lo": None, "ci_hi": None, "p": "—"}
                )
                processed_cat.add(orig)
            for dc in dummy_map[orig]:
                idx = col_to_idx[dc]
                beta = float(model_fit.params[idx])
                se = float(model_fit.bse[idx])
                wald = beta / se if se > 0 else float("nan")
                p = float(model_fit.pvalues[idx])
                or_val = _safe_exp(beta)
                or_lo = _safe_exp(float(ci_arr[idx, 0]))
                or_hi = _safe_exp(float(ci_arr[idx, 1]))
                level = dc[len(orig) + 1:]
                coef_rows.append([
                    f"  {level}",
                    f"{beta:.4f}", f"{se:.4f}", f"{wald:.3f}",
                    _fmt_p(p), f"{or_val:.4f}",
                    f"[{or_lo:.4f}, {or_hi:.4f}]", "—",
                ])
                coef_data_for_forest.append(
                    {"label": f"  {level}", "or_val": or_val,
                     "ci_lo": or_lo, "ci_hi": or_hi, "p": _fmt_p(p)}
                )
        else:
            if orig not in cont_vars:
                continue
            idx = col_to_idx[orig]
            beta = float(model_fit.params[idx])
            se = float(model_fit.bse[idx])
            wald = beta / se if se > 0 else float("nan")
            p = float(model_fit.pvalues[idx])
            or_val = _safe_exp(beta)
            or_lo = _safe_exp(float(ci_arr[idx, 0]))
            or_hi = _safe_exp(float(ci_arr[idx, 1]))
            or_std = _safe_exp(beta * col_sds[orig])
            coef_rows.append([
                orig,
                f"{beta:.4f}", f"{se:.4f}", f"{wald:.3f}",
                _fmt_p(p), f"{or_val:.4f}",
                f"[{or_lo:.4f}, {or_hi:.4f}]", f"{or_std:.4f}",
            ])
            coef_data_for_forest.append(
                {"label": orig, "or_val": or_val,
                 "ci_lo": or_lo, "ci_hi": or_hi, "p": _fmt_p(p)}
            )

    # ── 模型拟合指标 ──────────────────────────────────────────────────────────
    llf = float(model_fit.llf)
    llnull = float(model_fit.llnull)
    k = len(all_cols)

    mcfadden = 1.0 - llf / llnull if llnull != 0 else float("nan")
    cox_snell = 1.0 - _safe_exp(2.0 / n * (llnull - llf))
    nag_max = 1.0 - _safe_exp(2.0 * llnull / n)
    nagelkerke = cox_snell / nag_max if nag_max > 0 else float("nan")
    lr_stat = 2.0 * (llf - llnull)
    lr_p = float(stats.chi2.sf(lr_stat, k))

    hl_stat, hl_p, hl_df = _hosmer_lemeshow(y, y_pred_proba)
    auc, auc_lo, auc_hi = _delong_auc_ci(y, y_pred_proba)

    fit_rows: list[list[Any]] = [
        ["样本量 n", str(n)],
        ["AIC", f"{float(model_fit.aic):.2f}"],
        ["BIC", f"{float(model_fit.bic):.2f}"],
        ["对数似然值", f"{llf:.4f}"],
        ["空模型对数似然", f"{llnull:.4f}"],
        ["似然比检验 χ²", f"{lr_stat:.3f}"],
        ["似然比检验 p 值", _fmt_p(lr_p)],
        ["McFadden R²", f"{mcfadden:.4f}" if not math.isnan(mcfadden) else "—"],
        ["Cox-Snell R²", f"{cox_snell:.4f}"],
        ["Nagelkerke R²", f"{nagelkerke:.4f}" if not math.isnan(nagelkerke) else "—"],
        ["Hosmer-Lemeshow χ²", f"{hl_stat:.3f}"],
        [f"Hosmer-Lemeshow p（df={hl_df}）", _fmt_p(hl_p)],
        ["AUC", f"{auc:.4f}"],
        ["AUC 95% CI（DeLong）", f"[{auc_lo:.4f}, {auc_hi:.4f}]"],
    ]

    # ── 分类性能 ──────────────────────────────────────────────────────────────
    fpr_arr, tpr_arr, thr_arr = roc_curve(y, y_pred_proba)
    youden_idx = int(np.argmax(tpr_arr - fpr_arr))
    best_thr = float(thr_arr[youden_idx])
    best_fpr = float(fpr_arr[youden_idx])
    best_tpr = float(tpr_arr[youden_idx])

    y_cls = (y_pred_proba >= best_thr).astype(int)
    cm = confusion_matrix(y, y_cls)
    tn, fp, fn, tp = cm.ravel()
    sens = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    spec = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    ppv = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    npv = tn / (tn + fn) if (tn + fn) > 0 else 0.0
    acc = (tp + tn) / n

    perf_rows: list[list[Any]] = [
        ["最佳截断点（Youden 指数）", f"{best_thr:.4f}"],
        ["敏感性（Sensitivity）", f"{sens:.4f}"],
        ["特异性（Specificity）", f"{spec:.4f}"],
        ["阳性预测值（PPV）", f"{ppv:.4f}"],
        ["阴性预测值（NPV）", f"{npv:.4f}"],
        ["准确率（Accuracy）", f"{acc:.4f}"],
        ["AUC", f"{auc:.4f}"],
        ["AUC 95% CI（DeLong）", f"[{auc_lo:.4f}, {auc_hi:.4f}]"],
    ]

    cm_rows: list[list[Any]] = [
        ["实际阴性 (0)", str(int(tn)), str(int(fp))],
        ["实际阳性 (1)", str(int(fn)), str(int(tp))],
    ]

    # ── 完全分离检测 ──────────────────────────────────────────────────────────
    sep_warnings: list[str] = []
    for i, col in enumerate(all_cols):
        b = float(model_fit.params[i + 1])
        s = float(model_fit.bse[i + 1])
        if abs(b) > 10 or s > 100:
            sep_warnings.append(
                f"变量 '{col}' 可能存在完全分离，OR 估计不可靠"
                f"（β = {b:.2f}，SE = {s:.2f}）"
            )

    # ── VIF ──────────────────────────────────────────────────────────────────
    vif_rows: list[list[Any]] = []
    if len(all_cols) > 1:
        try:
            from statsmodels.stats.outliers_influence import variance_inflation_factor
            for ki, col in enumerate(all_cols):
                vif_val = float(variance_inflation_factor(X, ki + 1))
                if math.isnan(vif_val) or math.isinf(vif_val):
                    judge = "计算异常"
                elif vif_val < 5:
                    judge = "无共线性"
                elif vif_val < 10:
                    judge = "轻度共线性"
                else:
                    judge = "严重共线性"
                vif_rows.append([col, f"{vif_val:.3f}", judge])
        except Exception as exc:
            logger.warning("VIF 计算失败: %s", exc)
            vif_rows = [[c, "—", "计算失败"] for c in all_cols]

    return {
        "coef_rows": coef_rows,
        "fit_rows": fit_rows,
        "perf_rows": perf_rows,
        "cm_rows": cm_rows,
        "vif_rows": vif_rows,
        "y_pred_proba": y_pred_proba,
        "best_threshold": best_thr,
        "roc_tuple": (fpr_arr, tpr_arr, thr_arr, auc, auc_lo, auc_hi, best_fpr, best_tpr, best_thr),
        "coef_data_for_forest": coef_data_for_forest,
        "sep_warnings": sep_warnings,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Hosmer-Lemeshow 拟合优度检验
# ─────────────────────────────────────────────────────────────────────────────

def _hosmer_lemeshow(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    g: int = 10,
) -> tuple[float, float, int]:
    """返回 (HL 统计量, p 值, df)。"""
    n = len(y_true)
    g = min(g, max(4, n // 4))

    order = np.argsort(y_pred)
    y_s = y_true[order]
    p_s = y_pred[order]

    hl = 0.0
    for grp in np.array_split(np.arange(n), g):
        if len(grp) == 0:
            continue
        o1 = float(np.sum(y_s[grp]))
        e1 = float(np.sum(p_s[grp]))
        o0 = len(grp) - o1
        e0 = len(grp) - e1
        if e1 > 1e-9:
            hl += (o1 - e1) ** 2 / e1
        if e0 > 1e-9:
            hl += (o0 - e0) ** 2 / e0

    df = g - 2
    p = float(stats.chi2.sf(hl, df))
    return hl, p, df


# ─────────────────────────────────────────────────────────────────────────────
# DeLong AUC 置信区间
# ─────────────────────────────────────────────────────────────────────────────

def _delong_auc_ci(
    y_true: np.ndarray,
    y_score: np.ndarray,
    alpha: float = 0.05,
) -> tuple[float, float, float]:
    """返回 (AUC, CI 下限, CI 上限)，DeLong 方法。"""
    from sklearn.metrics import roc_auc_score

    n_pos = int(np.sum(y_true == 1))
    n_neg = int(np.sum(y_true == 0))
    if n_pos == 0 or n_neg == 0:
        auc = 0.5
        return auc, 0.0, 1.0

    auc = float(roc_auc_score(y_true, y_score))
    pos_sc = y_score[y_true == 1]
    neg_sc = y_score[y_true == 0]

    # Structural components（向量化 pairwise 计算）
    v10 = np.array([
        float(np.mean(neg_sc < s) + 0.5 * np.mean(neg_sc == s))
        for s in pos_sc
    ])
    v01 = np.array([
        float(np.mean(pos_sc > s) + 0.5 * np.mean(pos_sc == s))
        for s in neg_sc
    ])

    var_auc = np.var(v10, ddof=1) / n_pos + np.var(v01, ddof=1) / n_neg
    se = math.sqrt(max(var_auc, 0.0))
    z = float(stats.norm.ppf(1.0 - alpha / 2.0))
    return auc, max(0.0, auc - z * se), min(1.0, auc + z * se)


# ─────────────────────────────────────────────────────────────────────────────
# 图表构建
# ─────────────────────────────────────────────────────────────────────────────

def _build_all_charts(
    y_true: np.ndarray,
    y_pred_proba: np.ndarray,
    roc_tuple: tuple,
    coef_data_for_forest: list[dict],
    best_threshold: float,
) -> list[ChartResult]:
    fpr, tpr, _, auc, auc_lo, auc_hi, best_fpr, best_tpr, best_thr = roc_tuple

    charts: list[ChartResult] = [
        _build_roc_chart(fpr, tpr, auc, auc_lo, auc_hi, best_fpr, best_tpr, best_thr),
        _build_prob_hist_chart(y_true, y_pred_proba),
        _build_cm_heatmap_chart(y_true, y_pred_proba, best_threshold),
    ]
    forest = _build_forest_chart(coef_data_for_forest)
    if forest:
        charts.append(forest)
    calib = _build_calibration_chart(y_true, y_pred_proba)
    if calib:
        charts.append(calib)
    return charts


def _build_roc_chart(
    fpr: np.ndarray,
    tpr: np.ndarray,
    auc: float,
    ci_lo: float,
    ci_hi: float,
    best_fpr: float,
    best_tpr: float,
    best_thr: float,
) -> ChartResult:
    n_pts = len(fpr)
    if n_pts > 600:
        idx = np.linspace(0, n_pts - 1, 600, dtype=int)
        fpr, tpr = fpr[idx], tpr[idx]
    roc_pts = [[float(fpr[i]), float(tpr[i])] for i in range(len(fpr))]

    option: dict[str, Any] = {
        "title": {
            "text": "ROC 曲线",
            "subtext": f"AUC = {auc:.3f}（95% CI：{ci_lo:.3f}–{ci_hi:.3f}）",
            "left": "center",
        },
        "tooltip": {"trigger": "axis"},
        "legend": {"data": ["ROC 曲线", "随机参考线", "最佳截断点"], "top": "16%"},
        "grid": {"left": "12%", "right": "5%", "top": "26%", "bottom": "12%"},
        "xAxis": {
            "type": "value", "name": "1 - 特异性（FPR）",
            "nameLocation": "center", "nameGap": 28,
            "min": 0, "max": 1,
        },
        "yAxis": {
            "type": "value", "name": "敏感性（TPR）",
            "nameLocation": "center", "nameGap": 36,
            "min": 0, "max": 1,
        },
        "series": [
            {
                "name": "ROC 曲线", "type": "line",
                "data": roc_pts, "showSymbol": False,
                "lineStyle": {"color": "#5470c6", "width": 2},
                "areaStyle": {"color": "rgba(84,112,198,0.08)"},
            },
            {
                "name": "随机参考线", "type": "line",
                "data": [[0, 0], [1, 1]], "showSymbol": False,
                "lineStyle": {"type": "dashed", "color": "#aaa", "width": 1.5},
            },
            {
                "name": "最佳截断点", "type": "scatter",
                "data": [[float(best_fpr), float(best_tpr)]],
                "symbolSize": 12, "itemStyle": {"color": "#ee6666"},
                "label": {
                    "show": True,
                    "formatter": f"截断={best_thr:.3f}",
                    "position": "right", "fontSize": 11,
                },
            },
        ],
    }
    return ChartResult(title="ROC 曲线", chart_type="line", option=option)


def _build_prob_hist_chart(
    y_true: np.ndarray,
    y_pred_proba: np.ndarray,
) -> ChartResult:
    edges = np.linspace(0, 1, 21)
    centers = (edges[:-1] + edges[1:]) / 2
    cnt0, _ = np.histogram(y_pred_proba[y_true == 0], bins=edges)
    cnt1, _ = np.histogram(y_pred_proba[y_true == 1], bins=edges)

    option: dict[str, Any] = {
        "title": {"text": "预测概率分布", "left": "center", "top": "4%"},
        "tooltip": {"trigger": "axis"},
        "legend": {"data": ["阴性 (0)", "阳性 (1)"], "right": "5%", "top": "4%"},
        "grid": {"left": "10%", "right": "5%", "top": "18%", "bottom": "14%"},
        "xAxis": {
            "type": "category",
            "data": [f"{c:.2f}" for c in centers],
            "name": "预测概率", "nameLocation": "center", "nameGap": 28,
            "axisLabel": {"rotate": 30, "fontSize": 10},
        },
        "yAxis": {"type": "value", "name": "频数"},
        "series": [
            {
                "name": "阴性 (0)", "type": "bar",
                "data": cnt0.tolist(),
                "itemStyle": {"color": "rgba(84,112,198,0.75)"},
                "barGap": "0%", "barCategoryGap": "10%",
            },
            {
                "name": "阳性 (1)", "type": "bar",
                "data": cnt1.tolist(),
                "itemStyle": {"color": "rgba(238,102,102,0.75)"},
                "barGap": "0%",
            },
        ],
    }
    return ChartResult(title="预测概率分布", chart_type="bar", option=option)


def _build_cm_heatmap_chart(
    y_true: np.ndarray,
    y_pred_proba: np.ndarray,
    threshold: float,
) -> ChartResult:
    from sklearn.metrics import confusion_matrix as _cm

    y_cls = (y_pred_proba >= threshold).astype(int)
    cm = _cm(y_true, y_cls)
    tn, fp, fn, tp = cm.ravel()
    max_val = max(int(tn), int(fp), int(fn), int(tp), 1)

    option: dict[str, Any] = {
        "title": {
            "text": f"混淆矩阵（截断点 = {threshold:.3f}）",
            "left": "center",
        },
        "tooltip": {"position": "top"},
        "grid": {"left": "22%", "right": "5%", "top": "20%", "bottom": "22%"},
        "xAxis": {
            "type": "category",
            "data": ["预测阴性 (0)", "预测阳性 (1)"],
            "splitArea": {"show": True},
        },
        "yAxis": {
            "type": "category",
            "data": ["实际阴性 (0)", "实际阳性 (1)"],
            "splitArea": {"show": True},
        },
        "visualMap": {
            "min": 0, "max": max_val,
            "calculable": True, "orient": "horizontal",
            "left": "center", "bottom": "2%",
            "inRange": {"color": ["#f0f4ff", "#5470c6"]},
        },
        "series": [{
            "name": "混淆矩阵", "type": "heatmap",
            "data": [[0, 0, int(tn)], [1, 0, int(fp)],
                     [0, 1, int(fn)], [1, 1, int(tp)]],
            "label": {"show": True, "fontSize": 16, "fontWeight": "bold"},
        }],
    }
    return ChartResult(title="混淆矩阵", chart_type="heatmap", option=option)


def _build_forest_chart(
    coef_data: list[dict[str, Any]],
) -> ChartResult | None:
    """OR 森林图，复用前端 ForestPlot 组件（nullLine=1）。"""
    forest_data = []
    for item in coef_data:
        or_val = item.get("or_val")
        if item.get("is_header") or or_val is None:
            forest_data.append({
                "label": item["label"],
                "beta": float("nan"),
                "ci_lo": float("nan"),
                "ci_hi": float("nan"),
                "p": item.get("p", "—"),
                "is_note": True,
            })
        else:
            forest_data.append({
                "label": item["label"],
                "beta": float(or_val),
                "ci_lo": float(item.get("ci_lo", float("nan"))),
                "ci_hi": float(item.get("ci_hi", float("nan"))),
                "p": item.get("p", "—"),
            })

    if not any(not d.get("is_note") for d in forest_data):
        return None

    option: dict[str, Any] = {
        "forestData": forest_data,
        "nullLine": 1.0,
        "xLabel": "OR（95% CI）",
        "title": "多变量 Logistic 回归 — OR 森林图",
    }
    return ChartResult(title="OR 森林图", chart_type="forest_plot", option=option)


def _build_calibration_chart(
    y_true: np.ndarray,
    y_pred_proba: np.ndarray,
) -> ChartResult | None:
    try:
        from sklearn.calibration import calibration_curve
        fop, mpv = calibration_curve(y_true, y_pred_proba, n_bins=10)
    except Exception as exc:
        logger.warning("校准曲线计算失败: %s", exc)
        return None

    option: dict[str, Any] = {
        "title": {"text": "校准曲线（Calibration Plot）", "left": "center", "top": "4%"},
        "tooltip": {"trigger": "axis"},
        "legend": {"data": ["校准曲线", "完美校准"], "right": "5%", "top": "4%"},
        "grid": {"left": "12%", "right": "5%", "top": "18%", "bottom": "12%"},
        "xAxis": {
            "type": "value", "name": "预测概率（平均值）",
            "nameLocation": "center", "nameGap": 28,
            "min": 0, "max": 1,
        },
        "yAxis": {
            "type": "value", "name": "实际阳性率",
            "nameLocation": "center", "nameGap": 36,
            "min": 0, "max": 1,
        },
        "series": [
            {
                "name": "校准曲线", "type": "line",
                "data": [[float(m), float(f)] for m, f in zip(mpv, fop)],
                "symbolSize": 8,
                "lineStyle": {"color": "#5470c6", "width": 2},
                "itemStyle": {"color": "#5470c6"},
            },
            {
                "name": "完美校准", "type": "line",
                "data": [[0, 0], [1, 1]], "showSymbol": False,
                "lineStyle": {"type": "dashed", "color": "#ee6666", "width": 2},
            },
        ],
    }
    return ChartResult(title="校准曲线", chart_type="line", option=option)


# ─────────────────────────────────────────────────────────────────────────────
# 辅助函数
# ─────────────────────────────────────────────────────────────────────────────

def _fmt_p(p: float) -> str:
    if math.isnan(p):
        return "—"
    if p < 0.001:
        return "< 0.001"
    return f"{p:.3f}"


def _parse_p(p_str: str) -> float:
    if p_str == "< 0.001":
        return 0.0005
    try:
        return float(p_str)
    except (ValueError, TypeError):
        return float("nan")
