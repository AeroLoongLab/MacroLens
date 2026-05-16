from __future__ import annotations

import numpy as np
import pandas as pd

from macro_quant.utils.math_utils import clamp


def build_price_matrix(market_prices: pd.DataFrame) -> pd.DataFrame:
    if market_prices.empty:
        return pd.DataFrame()
    frame = market_prices.copy()
    frame["date"] = pd.to_datetime(frame["date"])
    pivot = (
        frame.sort_values("date")
        .pivot_table(index="date", columns="ticker", values="adj_close", aggfunc="last")
        .sort_index()
    )
    return pivot.ffill()


def rsi(series: pd.Series, window: int = 14) -> float:
    series = series.dropna()
    if len(series) < window + 1:
        return 50.0
    delta = series.diff()
    gain = delta.clip(lower=0).rolling(window).mean()
    loss = (-delta.clip(upper=0)).rolling(window).mean()
    rs = gain / loss.replace(0, np.nan)
    value = 100.0 - 100.0 / (1.0 + rs.iloc[-1])
    if np.isnan(value):
        return 50.0
    return float(value)


def max_drawdown_from_prices(series: pd.Series, window: int = 252) -> float:
    series = series.dropna().tail(window)
    if series.empty:
        return 0.0
    running_max = series.cummax()
    drawdown = series / running_max - 1.0
    return float(drawdown.min())


def build_market_features(market_prices: pd.DataFrame) -> dict[str, float]:
    prices = build_price_matrix(market_prices)
    if prices.empty:
        return {}
    features: dict[str, float] = {}
    returns = prices.pct_change()
    ticker_aliases = {
        "GLD": "gold",
        "SPY": "sp500",
        "QQQ": "qqq",
        "SHY": "short_bond",
        "IEF": "intermediate_bond",
        "TLT": "long_bond",
        "UUP": "usd",
        "DXY": "dxy",
        "CL=F": "oil",
        "VIX": "vix",
        "MOVE": "move",
        "HYG": "hyg",
        "LQD": "lqd",
    }
    for ticker, alias in ticker_aliases.items():
        if ticker not in prices:
            continue
        series = prices[ticker].dropna()
        if series.empty:
            continue
        latest = float(series.iloc[-1])
        for window, suffix in [(1, "1d"), (21, "21d"), (63, "63d"), (126, "126d")]:
            if len(series) > window:
                features[f"{alias}_return_{suffix}"] = float(series.iloc[-1] / series.iloc[-window - 1] - 1.0)
        if len(series) >= 50:
            features[f"{alias}_sma_50"] = float(series.tail(50).mean())
        if len(series) >= 200:
            sma_200 = float(series.tail(200).mean())
            features[f"{alias}_sma_200"] = sma_200
            features[f"{alias}_distance_200d"] = float(latest / sma_200 - 1.0) if sma_200 else 0.0
        features[f"{alias}_rsi_14"] = rsi(series)
        features[f"{alias}_volatility_21d"] = float(returns[ticker].tail(21).std() * np.sqrt(252))
        features[f"{alias}_max_drawdown_252d"] = max_drawdown_from_prices(series)

    features["sp500_momentum"] = clamp(features.get("sp500_return_63d", 0.0) / 0.15)
    features["qqq_momentum"] = clamp(features.get("qqq_return_63d", 0.0) / 0.18)
    features["gold_momentum"] = clamp(features.get("gold_return_63d", 0.0) / 0.15)
    features["oil_momentum"] = clamp(features.get("oil_return_63d", 0.0) / 0.25)
    features["long_bond_return"] = clamp(features.get("long_bond_return_63d", 0.0) / 0.15)
    usd_momentum = features.get("usd_return_63d", features.get("dxy_return_63d", 0.0))
    features["usd_momentum"] = clamp(usd_momentum / 0.06)
    features["usd_strength_score"] = max(0.0, features["usd_momentum"])
    features["earnings_momentum"] = max(0.0, features["qqq_momentum"])
    features["risk_appetite_score"] = clamp(features.get("sp500_momentum", 0.0) - features.get("vix_level", 0.0) * 0.2)
    features["valuation_pressure"] = max(0.0, clamp(features.get("qqq_distance_200d", 0.0) / 0.25))
    features["gold_crowding_score"] = max(
        0.0,
        clamp(
            0.5 * features.get("gold_momentum", 0.0) + 0.5 * clamp((features.get("gold_rsi_14", 50.0) - 60.0) / 25.0)
        ),
    )
    features["volatility_score"] = max(0.0, clamp((features.get("vix", 0.0) - 18.0) / 20.0))

    if {"SPY", "IEF"}.issubset(returns.columns):
        features["stock_bond_correlation_63d"] = _rolling_corr(returns["SPY"], returns["IEF"])
    if {"GLD", "TLT"}.issubset(returns.columns):
        features["gold_bond_correlation_63d"] = _rolling_corr(returns["GLD"], returns["TLT"])
    if "VIX" in prices:
        features["vix_level"] = clamp((float(prices["VIX"].iloc[-1]) - 18.0) / 20.0)
    if "MOVE" in prices:
        features["move_level"] = clamp((float(prices["MOVE"].iloc[-1]) - 100.0) / 70.0)
    return features


def _rolling_corr(left: pd.Series, right: pd.Series, window: int = 63) -> float:
    corr = left.tail(window).corr(right.tail(window))
    if pd.isna(corr):
        return 0.0
    return float(corr)
