from __future__ import annotations

import numpy as np
import pandas as pd

from macro_quant.utils.math_utils import annualized_return, max_drawdown


def performance_metrics(returns: pd.Series, periods_per_year: int = 12) -> dict[str, float]:
    clean = returns.dropna().astype(float)
    if clean.empty:
        return _empty_metrics()
    ann_return = annualized_return(clean.to_numpy(), periods_per_year)
    ann_vol = float(clean.std(ddof=0) * np.sqrt(periods_per_year))
    downside = clean[clean < 0]
    downside_vol = float(downside.std(ddof=0) * np.sqrt(periods_per_year)) if not downside.empty else 0.0
    mdd = max_drawdown(clean.to_numpy())
    var_95 = float(clean.quantile(0.05))
    cvar_95 = float(clean[clean <= var_95].mean()) if (clean <= var_95).any() else var_95
    return {
        "CAGR": ann_return,
        "Volatility": ann_vol,
        "Sharpe": ann_return / ann_vol if ann_vol > 0 else 0.0,
        "Sortino": ann_return / downside_vol if downside_vol > 0 else 0.0,
        "Max Drawdown": mdd,
        "Calmar": ann_return / abs(mdd) if mdd < 0 else 0.0,
        "VaR": var_95,
        "CVaR": cvar_95,
        "Win Rate": float((clean > 0).mean()),
        "Monthly Win Rate": float((clean > 0).mean()),
        "Quarterly Win Rate": _quarterly_win_rate(clean),
        "Max Consecutive Loss Months": float(_max_consecutive_losses(clean)),
    }


def _quarterly_win_rate(monthly_returns: pd.Series) -> float:
    if monthly_returns.empty:
        return 0.0
    quarterly = (1.0 + monthly_returns).resample("QE").prod() - 1.0
    if quarterly.empty:
        return 0.0
    return float((quarterly > 0).mean())


def _max_consecutive_losses(returns: pd.Series) -> int:
    longest = 0
    current = 0
    for value in returns:
        if value < 0:
            current += 1
            longest = max(longest, current)
        else:
            current = 0
    return longest


def _empty_metrics() -> dict[str, float]:
    keys = [
        "CAGR",
        "Volatility",
        "Sharpe",
        "Sortino",
        "Max Drawdown",
        "Calmar",
        "VaR",
        "CVaR",
        "Win Rate",
        "Monthly Win Rate",
        "Quarterly Win Rate",
        "Max Consecutive Loss Months",
    ]
    return {key: 0.0 for key in keys}
