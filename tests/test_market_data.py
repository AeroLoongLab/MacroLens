from __future__ import annotations

import sys
import types

import pandas as pd

from macro_quant.data.market_data import fetch_market_prices


def test_live_market_prices_map_yfinance_aliases_to_canonical_tickers(monkeypatch) -> None:
    captured: dict[str, list[str]] = {}

    def fake_download(**kwargs):
        captured["tickers"] = list(kwargs["tickers"])
        dates = pd.to_datetime(["2026-05-14", "2026-05-15"])
        columns = pd.MultiIndex.from_product(
            [["^VIX", "DX-Y.NYB", "^MOVE"], ["Open", "High", "Low", "Close", "Adj Close", "Volume"]],
            names=["Ticker", "Price"],
        )
        return pd.DataFrame(
            [
                [17.8, 18.1, 17.2, 17.3, 17.3, 0, 98.4, 98.9, 98.4, 98.9, 98.9, 0, 70.2, 70.2, 69.6, 69.6, 69.6, 0],
                [18.0, 19.2, 17.8, 18.4, 18.4, 0, 98.9, 99.3, 98.9, 99.3, 99.3, 0, 69.6, 79.8, 69.6, 79.8, 79.8, 0],
            ],
            index=dates,
            columns=columns,
        )

    monkeypatch.setitem(sys.modules, "yfinance", types.SimpleNamespace(download=fake_download))
    frame = fetch_market_prices(
        {
            "market": {"volatility": ["VIX", "MOVE"], "usd": ["DXY"]},
            "provider_aliases": {"yfinance": {"VIX": "^VIX", "DXY": "DX-Y.NYB", "MOVE": "^MOVE"}},
        },
        live=True,
    )

    assert captured["tickers"] == ["DX-Y.NYB", "^MOVE", "^VIX"]
    assert set(frame["ticker"]) == {"DXY", "MOVE", "VIX"}
    assert set(frame["source"]) == {"yfinance"}
