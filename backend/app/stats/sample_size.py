"""在线样本量计算 — 基于精确公式的样本量与功效计算"""
import logging
from typing import Any

import numpy as np
from scipy import stats

from app.models.analysis import AnalysisResult, ChartResult, TableResult

logger = logging.getLogger(__name__)

_EXP_CLIP = 500  # 防止 log/exp 溢出的裁剪值


# ─────────────────────────────────────────────────────────────────────────────
# 公共辅助
# ─────────────────────────────────────────────────────────────────────────────

def _z_alpha(alpha: float, sides: int) -> float:
    """单/双侧检验的 z_{alpha}（或 z_{alpha/2}）分位数。"""
    return float(stats.norm.ppf(1 - alpha / sides))


def _z_beta(power: float) -> float:
    return float(stats.norm.ppf(power))


def _power_from_lambda(lam: float, za: float, sides: int) -> float:
    """根据非中心参数 lambda 计算检验功效。"""
    if sides == 2:
        pw = stats.norm.cdf(lam - za) + stats.norm.cdf(-lam - za)
    else:
        pw = stats.norm.cdf(lam - za)
    return float(np.clip(pw, 0.0, 1.0))


# ─────────────────────────────────────────────────────────────────────────────
# 各计算类型
# ─────────────────────────────────────────────────────────────────────────────

def _two_means(
    params: dict, alpha: float, power: float | None, sides: int,
    solve_for: str = "n", n_per_group: float | None = None,
) -> tuple[float, float, float] | float:
    """两组均值比较（独立样本 t 检验）。"""
    mean_diff = float(params.get("mean_diff", 5.0))
    sd_default = float(params.get("sd", 10.0))
    sd1 = float(params.get("sd1", sd_default))
    sd2 = float(params.get("sd2", sd_default))
    ratio = float(params.get("ratio", 1.0))  # n2/n1

    if abs(mean_diff) < 1e-10:
        raise ValueError("均值差（mean_diff）不能为 0")
    if sd1 <= 0 or sd2 <= 0:
        raise ValueError("标准差必须大于 0")
    if ratio <= 0:
        raise ValueError("样本量比例（ratio）必须大于 0")

    za = _z_alpha(alpha, sides)
    variance_sum = sd1**2 + sd2**2 / ratio  # 对 n1 的方差贡献

    if solve_for == "n":
        assert power is not None
        zb = _z_beta(power)
        n1 = (za + zb) ** 2 * variance_sum / mean_diff**2
        n1 = float(np.ceil(n1))
        n2 = float(np.ceil(n1 * ratio))
        lam = abs(mean_diff) * np.sqrt(n1) / np.sqrt(variance_sum)
        actual_power = _power_from_lambda(lam, za, sides)
        return n1, n1 + n2, actual_power
    else:
        n1 = float(n_per_group)  # type: ignore[arg-type]
        lam = abs(mean_diff) * np.sqrt(n1) / np.sqrt(variance_sum)
        return _power_from_lambda(lam, za, sides)


def _two_proportions(
    params: dict, alpha: float, power: float | None, sides: int,
    solve_for: str = "n", n_per_group: float | None = None,
) -> tuple[float, float, float] | float:
    """两组率比较（正态近似法）。"""
    p1 = float(params.get("p1", 0.3))
    p2 = float(params.get("p2", 0.5))
    ratio = float(params.get("ratio", 1.0))

    if not (0 < p1 < 1) or not (0 < p2 < 1):
        raise ValueError("p1 和 p2 必须在 (0, 1) 之间")
    if abs(p1 - p2) < 1e-6:
        raise ValueError("p1 和 p2 差异过小")

    delta = abs(p1 - p2)
    p_bar = (p1 + p2) / 2  # 等权平均，适用于 ratio=1
    sigma0 = np.sqrt(2 * p_bar * (1 - p_bar))
    sigma1 = np.sqrt(p1 * (1 - p1) + p2 * (1 - p2))

    za = _z_alpha(alpha, sides)

    if solve_for == "n":
        assert power is not None
        zb = _z_beta(power)
        n1 = (za * sigma0 + zb * sigma1) ** 2 / delta**2
        n1 = float(np.ceil(n1))
        n2 = float(np.ceil(n1 * ratio))
        lam = (np.sqrt(n1) * delta - za * sigma0) / sigma1
        actual_power = _power_from_lambda(max(lam, 0), za, sides)
        return n1, n1 + n2, actual_power
    else:
        n1 = float(n_per_group)  # type: ignore[arg-type]
        lam = (np.sqrt(n1) * delta - za * sigma0) / sigma1
        return _power_from_lambda(max(lam, 0), za, sides)


def _paired_means(
    params: dict, alpha: float, power: float | None, sides: int,
    solve_for: str = "n", n_per_group: float | None = None,
) -> tuple[float, float, float] | float:
    """配对均值比较（配对 t 检验）。"""
    mean_diff = float(params.get("mean_diff", 5.0))
    sd_diff = float(params.get("sd_diff", 10.0))

    if abs(mean_diff) < 1e-10:
        raise ValueError("配对差值均值（mean_diff）不能为 0")
    if sd_diff <= 0:
        raise ValueError("配对差值标准差（sd_diff）必须大于 0")

    za = _z_alpha(alpha, sides)

    if solve_for == "n":
        assert power is not None
        zb = _z_beta(power)
        n = (za + zb) ** 2 * sd_diff**2 / mean_diff**2
        n = float(np.ceil(n))
        lam = abs(mean_diff) * np.sqrt(n) / sd_diff
        actual_power = _power_from_lambda(lam, za, sides)
        return n, n, actual_power
    else:
        n = float(n_per_group)  # type: ignore[arg-type]
        lam = abs(mean_diff) * np.sqrt(n) / sd_diff
        return _power_from_lambda(lam, za, sides)


def _one_mean(
    params: dict, alpha: float, power: float | None, sides: int,
    solve_for: str = "n", n_per_group: float | None = None,
) -> tuple[float, float, float] | float:
    """单样本均值检验。"""
    mean_diff = float(params.get("mean_diff", 5.0))
    sd = float(params.get("sd", 10.0))

    if abs(mean_diff) < 1e-10:
        raise ValueError("与假设值的差（mean_diff）不能为 0")
    if sd <= 0:
        raise ValueError("标准差（sd）必须大于 0")

    za = _z_alpha(alpha, sides)

    if solve_for == "n":
        assert power is not None
        zb = _z_beta(power)
        n = (za + zb) ** 2 * sd**2 / mean_diff**2
        n = float(np.ceil(n))
        lam = abs(mean_diff) * np.sqrt(n) / sd
        actual_power = _power_from_lambda(lam, za, sides)
        return n, n, actual_power
    else:
        n = float(n_per_group)  # type: ignore[arg-type]
        lam = abs(mean_diff) * np.sqrt(n) / sd
        return _power_from_lambda(lam, za, sides)


def _one_proportion(
    params: dict, alpha: float, power: float | None, sides: int,
    solve_for: str = "n", n_per_group: float | None = None,
) -> tuple[float, float, float] | float:
    """单样本率检验（正态近似）。"""
    p0 = float(params.get("p0", 0.5))
    p1 = float(params.get("p1", 0.7))

    if not (0 < p0 < 1) or not (0 < p1 < 1):
        raise ValueError("p0 和 p1 必须在 (0, 1) 之间")
    if abs(p1 - p0) < 1e-6:
        raise ValueError("p1 和 p0 差异过小")

    delta = abs(p1 - p0)
    sigma0 = np.sqrt(p0 * (1 - p0))
    sigma1 = np.sqrt(p1 * (1 - p1))

    za = _z_alpha(alpha, sides)

    if solve_for == "n":
        assert power is not None
        zb = _z_beta(power)
        n = (za * sigma0 + zb * sigma1) ** 2 / delta**2
        n = float(np.ceil(n))
        lam = (np.sqrt(n) * delta - za * sigma0) / sigma1
        actual_power = _power_from_lambda(max(lam, 0), za, sides)
        return n, n, actual_power
    else:
        n = float(n_per_group)  # type: ignore[arg-type]
        lam = (np.sqrt(n) * delta - za * sigma0) / sigma1
        return _power_from_lambda(max(lam, 0), za, sides)


def _correlation(
    params: dict, alpha: float, power: float | None, sides: int,
    solve_for: str = "n", n_per_group: float | None = None,
) -> tuple[float, float, float] | float:
    """相关系数检验（Fisher's z 变换法）。"""
    r = float(params.get("r", 0.5))
    if not (-1 < r < 1) or abs(r) < 1e-6:
        raise ValueError("相关系数 r 必须在 (-1, 1) 之间且不为 0")

    z_r = float(0.5 * np.log((1 + abs(r)) / (1 - abs(r))))  # Fisher z
    za = _z_alpha(alpha, sides)

    if solve_for == "n":
        assert power is not None
        zb = _z_beta(power)
        n = (za + zb) ** 2 / z_r**2 + 3
        n = float(np.ceil(n))
        lam = z_r * np.sqrt(n - 3)
        actual_power = _power_from_lambda(lam, za, sides)
        return n, n, actual_power
    else:
        n = float(n_per_group)  # type: ignore[arg-type]
        lam = z_r * np.sqrt(max(n - 3, 0))
        return _power_from_lambda(lam, za, sides)


def _logistic(
    params: dict, alpha: float, power: float | None, sides: int,
    solve_for: str = "n", n_per_group: float | None = None,
) -> tuple[float, float, float] | float:
    """Logistic 回归样本量（Hsieh 1998 公式）。

    n = (z_alpha + z_beta)^2 / (p*(1-p) * beta^2 * (1-R^2))
    其中 beta = log(OR), p = 基线事件率, R^2 = 其他协变量解释的方差。
    """
    p0 = float(params.get("p0", 0.3))
    or_value = float(params.get("or_value", 2.0))
    r2 = float(params.get("r2", 0.0))

    if not (0 < p0 < 1):
        raise ValueError("基线事件率 p0 必须在 (0, 1) 之间")
    if or_value <= 0:
        raise ValueError("OR 值必须大于 0")
    if not (0 <= r2 < 1):
        raise ValueError("R² 必须在 [0, 1) 之间")

    beta_log = float(np.log(np.clip(or_value, 1e-6, 1e6)))
    denom = p0 * (1 - p0) * beta_log**2 * (1 - r2)
    if denom < 1e-12:
        raise ValueError("分母趋近于 0，请检查输入参数（OR 太接近 1 或 p0 极端）")

    za = _z_alpha(alpha, sides)

    if solve_for == "n":
        assert power is not None
        zb = _z_beta(power)
        n = (za + zb) ** 2 / denom
        n = float(np.ceil(n))
        lam = np.sqrt(n * denom)
        actual_power = _power_from_lambda(lam, za, sides)
        return n, n, actual_power
    else:
        n = float(n_per_group)  # type: ignore[arg-type]
        lam = np.sqrt(n * denom)
        return _power_from_lambda(lam, za, sides)


def _cox(
    params: dict, alpha: float, power: float | None, sides: int,
    solve_for: str = "n", n_per_group: float | None = None,
) -> tuple[float, float, float] | float:
    """Cox 回归 / 生存分析样本量（Schoenfeld 公式）。

    所需事件数 d = (z_alpha + z_beta)^2 / (log(HR))^2 / (1-R^2)
    总样本量 n = d / event_rate
    """
    hr = float(params.get("hr", 1.5))
    event_rate = float(params.get("event_rate", 0.3))
    r2 = float(params.get("r2", 0.0))

    if hr <= 0:
        raise ValueError("风险比（HR）必须大于 0")
    if abs(hr - 1.0) < 1e-4:
        raise ValueError("HR 太接近 1，效应量过小")
    if not (0 < event_rate <= 1):
        raise ValueError("事件率（event_rate）必须在 (0, 1] 之间")
    if not (0 <= r2 < 1):
        raise ValueError("R² 必须在 [0, 1) 之间")

    log_hr = float(np.log(np.clip(hr, 1e-6, 1e6)))
    za = _z_alpha(alpha, sides)

    if solve_for == "n":
        assert power is not None
        zb = _z_beta(power)
        events = (za + zb) ** 2 / (log_hr**2 * (1 - r2))
        n = float(np.ceil(events / event_rate))
        actual_events = n * event_rate
        lam = abs(log_hr) * np.sqrt(actual_events * (1 - r2))
        actual_power = _power_from_lambda(lam, za, sides)
        return n, n, actual_power
    else:
        n = float(n_per_group)  # type: ignore[arg-type]
        events = n * event_rate
        lam = abs(log_hr) * np.sqrt(events * (1 - r2))
        return _power_from_lambda(lam, za, sides)


# ─────────────────────────────────────────────────────────────────────────────
# 计算类型注册表
# ─────────────────────────────────────────────────────────────────────────────

_CALC_FNS = {
    "two_means": _two_means,
    "two_proportions": _two_proportions,
    "paired_means": _paired_means,
    "one_mean": _one_mean,
    "one_proportion": _one_proportion,
    "correlation": _correlation,
    "logistic": _logistic,
    "cox": _cox,
}

_CALC_LABELS = {
    "two_means": "两组均值比较（独立 t 检验）",
    "two_proportions": "两组率比较",
    "paired_means": "配对均值比较（配对 t 检验）",
    "one_mean": "单样本均值检验",
    "one_proportion": "单样本率检验",
    "correlation": "相关系数检验",
    "logistic": "Logistic 回归",
    "cox": "Cox 回归 / 生存分析",
}

_PARAM_DESCS: dict[str, list[tuple[str, str, str]]] = {
    "two_means": [
        ("mean_diff", "均值差", "两组均值之差（μ₁−μ₂）"),
        ("sd", "标准差（sd）", "两组共同标准差（也可分别设 sd1、sd2）"),
        ("ratio", "样本量比（n₂/n₁）", "两组样本量之比，默认 1:1"),
    ],
    "two_proportions": [
        ("p1", "组 1 的率（p₁）", "第一组发生率"),
        ("p2", "组 2 的率（p₂）", "第二组发生率"),
        ("ratio", "样本量比（n₂/n₁）", "两组样本量之比，默认 1:1"),
    ],
    "paired_means": [
        ("mean_diff", "配对差值均值", "配对差值的期望均值"),
        ("sd_diff", "配对差值标准差", "配对差值的标准差"),
    ],
    "one_mean": [
        ("mean_diff", "与假设值的差（δ）", "预期均值与假设均值的差值"),
        ("sd", "标准差（σ）", "总体标准差"),
    ],
    "one_proportion": [
        ("p0", "零假设率（p₀）", "原假设下的总体率"),
        ("p1", "预期率（p₁）", "备择假设下的总体率"),
    ],
    "correlation": [
        ("r", "预期相关系数（r）", "需检验的相关系数大小"),
    ],
    "logistic": [
        ("p0", "基线事件率（p₀）", "暴露 = 0 时的结局发生率"),
        ("or_value", "预期 OR 值", "暴露变量的优势比"),
        ("r2", "其他自变量的 R²", "暴露变量被其他协变量解释的方差比例（0~1）"),
    ],
    "cox": [
        ("hr", "预期风险比（HR）", "暴露变量的风险比"),
        ("event_rate", "事件率", "随访期间的总事件发生率（0~1）"),
        ("r2", "其他自变量的 R²", "暴露变量被其他协变量解释的方差比例（0~1）"),
    ],
}

_FORMULA_ROWS: dict[str, list[list[str]]] = {
    "two_means": [
        ["n₁ = (z_{α/2} + z_β)² × (σ₁² + σ₂²/k) / Δ²", "k = n₂/n₁（样本量比），Δ = μ₁−μ₂"],
    ],
    "two_proportions": [
        ["n = (z_{α/2}σ₀ + z_β σ₁)² / Δ²", "σ₀=√(2p̄(1-p̄)), σ₁=√(p₁(1-p₁)+p₂(1-p₂)), p̄=(p₁+p₂)/2"],
    ],
    "paired_means": [
        ["n = (z_{α/2} + z_β)² × σ_d² / Δ²", "σ_d = 配对差值标准差，Δ = 配对差值均值"],
    ],
    "one_mean": [
        ["n = (z_{α/2} + z_β)² × σ² / δ²", "σ = 总体标准差，δ = 与假设值的差"],
    ],
    "one_proportion": [
        ["n = (z_{α/2}√(p₀(1-p₀)) + z_β√(p₁(1-p₁)))² / (p₁-p₀)²", "正态近似法"],
    ],
    "correlation": [
        ["n = (z_{α/2} + z_β)² / ζ_r² + 3", "ζ_r = 0.5×ln((1+|r|)/(1-|r|))（Fisher's z 变换）"],
    ],
    "logistic": [
        ["n = (z_{α/2} + z_β)² / [p(1-p) × β² × (1-ρ²)]", "β = ln(OR), ρ² = 协变量 R²（Hsieh 1998）"],
    ],
    "cox": [
        ["d = (z_{α/2} + z_β)² / [(ln HR)² × (1-ρ²)]", "d = 所需事件数（Schoenfeld 公式）"],
        ["n = d / 事件率", "总样本量 = 事件数 / 期望事件率"],
    ],
}


# ─────────────────────────────────────────────────────────────────────────────
# 主函数
# ─────────────────────────────────────────────────────────────────────────────

def run(params: dict[str, Any]) -> AnalysisResult:
    calc_type: str = str(params.get("calc_type", "two_means"))
    alpha: float = float(params.get("alpha", 0.05))
    power: float = float(params.get("power", 0.80))
    sides: int = int(params.get("sides", 2))
    solve_for: str = str(params.get("solve_for", "sample_size"))
    n_given: int = int(params.get("n", 100))

    if not (0 < alpha < 1):
        raise ValueError("显著性水平 alpha 必须在 (0, 1) 之间")
    if solve_for == "sample_size" and not (0 < power < 1):
        raise ValueError("目标检验效能 power 必须在 (0, 1) 之间")
    if sides not in (1, 2):
        raise ValueError("sides 必须为 1（单侧）或 2（双侧）")
    if calc_type not in _CALC_FNS:
        raise ValueError(
            f"不支持的计算类型 '{calc_type}'。支持：{list(_CALC_FNS.keys())}"
        )

    calc_fn = _CALC_FNS[calc_type]

    # ── 计算 ──────────────────────────────────────────────────────────
    if solve_for == "sample_size":
        result = calc_fn(params, alpha, power, sides, solve_for="n")
        assert isinstance(result, tuple)
        n_per_group, n_total, actual_power = result
    else:
        # 给定样本量，计算 power
        raw_power = calc_fn(
            params, alpha, None, sides,
            solve_for="power", n_per_group=float(n_given),
        )
        assert isinstance(raw_power, float)
        actual_power = raw_power
        n_per_group = float(n_given)
        # 对于多组/多人计算类型，total n 的估算
        ratio = float(params.get("ratio", 1.0))
        if calc_type in ("two_means", "two_proportions"):
            n_total = n_given * (1 + ratio)
        else:
            n_total = float(n_given)

    n_per_group_int = int(np.ceil(n_per_group))
    n_total_int = int(np.ceil(n_total))

    # ── 功效曲线 ─────────────────────────────────────────────────────
    if solve_for == "sample_size":
        n_max = max(n_per_group_int * 3, 100)
        target_power_line = power
        target_n_mark = n_per_group_int
    else:
        n_max = max(n_given * 3, 100)
        target_power_line = actual_power
        target_n_mark = n_given

    step = max(1, n_max // 80)
    n_range = list(range(2, n_max + step, step))

    power_curve: list[list[float]] = []
    for ni in n_range:
        try:
            pw = calc_fn(params, alpha, None, sides, solve_for="power", n_per_group=float(ni))
            if isinstance(pw, float) and 0 <= pw <= 1:
                power_curve.append([float(ni), round(pw, 4)])
        except Exception:
            pass

    # ── 表格 ─────────────────────────────────────────────────────────
    label = _CALC_LABELS.get(calc_type, calc_type)
    sides_label = "双侧" if sides == 2 else "单侧"

    result_table = TableResult(
        title="样本量计算结果",
        headers=["指标", "值"],
        rows=[
            ["计算类型", label],
            ["显著性水平 (α)", str(alpha)],
            ["检验方向", sides_label],
            ["每组样本量（n per group）", str(n_per_group_int)],
            ["总样本量（N total）", str(n_total_int)],
            ["实际检验效能（Power）", f"{actual_power:.4f}（{actual_power * 100:.1f}%）"],
            [
                "目标效能" if solve_for == "sample_size" else "给定样本量",
                f"{power:.2f}" if solve_for == "sample_size" else str(n_given),
            ],
        ],
    )

    param_rows: list[list[str]] = []
    for key, name, desc in _PARAM_DESCS.get(calc_type, []):
        val = params.get(key, "—")
        param_rows.append([name, str(val), desc])

    param_table = TableResult(
        title="输入参数",
        headers=["参数", "输入值", "说明"],
        rows=param_rows,
    )

    formula_table = TableResult(
        title="计算公式",
        headers=["公式", "说明"],
        rows=_FORMULA_ROWS.get(calc_type, [["—", "—"]]),
    )

    # ── 图表 — 功效曲线 ─────────────────────────────────────────────
    power_chart = ChartResult(
        title="功效曲线",
        chart_type="line",
        option={
            "title": {"text": "检验功效曲线（Power Curve）", "left": "center"},
            "tooltip": {"trigger": "axis"},
            "xAxis": {
                "type": "value",
                "name": "每组样本量 (n)",
                "nameLocation": "center",
                "nameGap": 35,
                "min": 0,
            },
            "yAxis": {
                "type": "value",
                "name": "检验效能（Power）",
                "min": 0,
                "max": 1,
                "axisLabel": {"formatter": "{value}"},
            },
            "series": [
                {
                    "type": "line",
                    "name": "Power",
                    "data": [[p[0], p[1]] for p in power_curve],
                    "smooth": True,
                    "symbol": "none",
                    "lineStyle": {"color": "#5470c6", "width": 2.5},
                    "areaStyle": {
                        "color": {
                            "type": "linear",
                            "x": 0, "y": 0, "x2": 0, "y2": 1,
                            "colorStops": [
                                {"offset": 0, "color": "rgba(84,112,198,0.2)"},
                                {"offset": 1, "color": "rgba(84,112,198,0.01)"},
                            ],
                        }
                    },
                    "markLine": {
                        "symbol": ["none", "none"],
                        "silent": True,
                        "data": [
                            {
                                "yAxis": target_power_line,
                                "lineStyle": {"color": "#ee6666", "type": "dashed", "width": 1.5},
                                "label": {
                                    "formatter": f"Power={target_power_line:.2f}",
                                    "position": "insideStartTop",
                                },
                            },
                            {
                                "xAxis": target_n_mark,
                                "lineStyle": {"color": "#91cc75", "type": "dashed", "width": 1.5},
                                "label": {
                                    "formatter": f"n={target_n_mark}",
                                    "position": "insideEndTop",
                                },
                            },
                        ],
                    },
                }
            ],
            "legend": {"bottom": 0},
            "grid": {"bottom": 50, "top": 60, "left": 70, "right": 60},
        },
    )

    # ── 摘要 ─────────────────────────────────────────────────────────
    if solve_for == "sample_size":
        summary = (
            f"样本量计算（{label}）：α={alpha}，{sides_label}检验，"
            f"目标 Power={power:.2f}，每组需要 {n_per_group_int} 例，"
            f"共 {n_total_int} 例，实际 Power={actual_power:.4f}。"
        )
    else:
        summary = (
            f"功效计算（{label}）：α={alpha}，{sides_label}检验，"
            f"每组 n={n_given} 例，"
            f"实际检验效能={actual_power:.4f}（{actual_power * 100:.1f}%）。"
        )

    return AnalysisResult(
        method="sample_size",
        tables=[result_table, param_table, formula_table],
        charts=[power_chart],
        summary=summary,
        warnings=[],
    )
