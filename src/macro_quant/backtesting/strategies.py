from __future__ import annotations

import pandas as pd

ASSET_TO_TICKER = {
    "Gold": "GLD",
    "Equity": "SPY",
    "Short_Bond": "SHY",
    "Intermediate_Bond": "IEF",
    "Long_Bond": "TLT",
}


def weights_to_tickers(asset_weights: dict[str, float]) -> dict[str, float]:
    return {
        ASSET_TO_TICKER[asset]: weight
        for asset, weight in asset_weights.items()
        if asset in ASSET_TO_TICKER and weight != 0
    }


def benchmark_returns(monthly_returns: pd.DataFrame) -> pd.DataFrame:
    required = ["GLD", "SPY", "SHY", "IEF", "TLT"]
    available = [ticker for ticker in required if ticker in monthly_returns.columns]
    frame = pd.DataFrame(index=monthly_returns.index)
    frame["SPY only"] = monthly_returns.get("SPY", pd.Series(0.0, index=monthly_returns.index))
    frame["60/40"] = (
        monthly_returns.get("SPY", pd.Series(0.0, index=monthly_returns.index)) * 0.60
        + monthly_returns.get("IEF", pd.Series(0.0, index=monthly_returns.index)) * 0.40
    )
    if available:
        frame["Equal Weight"] = monthly_returns[available].mean(axis=1)
    else:
        frame["Equal Weight"] = 0.0
    return frame
