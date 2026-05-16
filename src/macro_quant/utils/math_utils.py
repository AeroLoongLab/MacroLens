from __future__ import annotations

import math
from collections.abc import Mapping

import numpy as np


def clamp(value: float, low: float = -1.0, high: float = 1.0) -> float:
    if value is None or math.isnan(float(value)):
        return 0.0
    return float(max(low, min(high, value)))


def clamp_score(value: float) -> float:
    return clamp(value, -100.0, 100.0)


def normalize(value: float, neutral: float, scale: float) -> float:
    if scale == 0:
        return 0.0
    return clamp((float(value) - neutral) / scale)


def normalize_0_100(value: float, neutral: float, scale: float) -> float:
    return max(0.0, min(100.0, 50.0 + normalize(value, neutral, scale) * 50.0))


def softmax(scores: Mapping[str, float], temperature: float = 1.0) -> dict[str, float]:
    if not scores:
        return {}
    temp = max(temperature, 1e-6)
    keys = list(scores)
    values = np.array([float(scores[key]) / temp for key in keys], dtype=float)
    values = values - np.nanmax(values)
    exp = np.exp(values)
    total = float(exp.sum())
    if total <= 0 or math.isnan(total):
        equal = 1.0 / len(keys)
        return {key: equal for key in keys}
    return {key: float(value / total) for key, value in zip(keys, exp, strict=True)}


def safe_pct_change(current: float, previous: float) -> float:
    if previous == 0 or math.isnan(float(previous)):
        return 0.0
    return float(current / previous - 1.0)


def annualized_return(period_returns: np.ndarray, periods_per_year: int) -> float:
    returns = np.asarray(period_returns, dtype=float)
    returns = returns[~np.isnan(returns)]
    if returns.size == 0:
        return 0.0
    compounded = float(np.prod(1.0 + returns))
    years = returns.size / periods_per_year
    if years <= 0 or compounded <= 0:
        return 0.0
    return compounded ** (1.0 / years) - 1.0


def max_drawdown(period_returns: np.ndarray) -> float:
    returns = np.asarray(period_returns, dtype=float)
    if returns.size == 0:
        return 0.0
    curve = np.cumprod(1.0 + np.nan_to_num(returns))
    peak = np.maximum.accumulate(curve)
    drawdown = curve / peak - 1.0
    return float(np.min(drawdown))
