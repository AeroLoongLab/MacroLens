from __future__ import annotations

from collections.abc import Mapping
from datetime import UTC, date, datetime
from typing import Any

import pandas as pd

from macro_quant.data.fixtures import make_market_fixture
from macro_quant.utils.logging import get_logger

logger = get_logger(__name__)


def _utc_now() -> datetime:
    return datetime.now(UTC).replace(tzinfo=None)


def configured_market_tickers(ticker_config: dict[str, Any]) -> list[str]:
    tickers: list[str] = []
    for values in ticker_config.get("market", {}).values():
        if isinstance(values, list):
            tickers.extend(str(value) for value in values)
    return sorted(set(tickers))


def yfinance_symbol_map(ticker_config: dict[str, Any]) -> dict[str, str]:
    aliases = ticker_config.get("provider_aliases", {}).get("yfinance", {})
    if not isinstance(aliases, Mapping):
        aliases = {}
    return {ticker: str(aliases.get(ticker, ticker)) for ticker in configured_market_tickers(ticker_config)}


def fetch_market_prices(
    ticker_config: dict[str, Any],
    start: str | date = "2010-01-01",
    end: str | date | None = None,
    live: bool = False,
) -> pd.DataFrame:
    tickers = configured_market_tickers(ticker_config)
    if not live:
        return make_market_fixture(tickers, start=start, end=end)
    try:
        import yfinance as yf

        provider_symbols = yfinance_symbol_map(ticker_config)
        provider_to_ticker = {provider: ticker for ticker, provider in provider_symbols.items()}
        download_symbols = sorted(provider_to_ticker)
        download = yf.download(
            tickers=download_symbols,
            start=str(start),
            end=str(end or date.today()),
            progress=False,
            auto_adjust=False,
            group_by="ticker",
            threads=True,
        )
        if download.empty:
            raise RuntimeError("yfinance returned an empty frame")
        rows: list[dict[str, object]] = []
        for provider_symbol in download_symbols:
            ticker = provider_to_ticker[provider_symbol]
            if isinstance(download.columns, pd.MultiIndex):
                if provider_symbol not in download.columns.get_level_values(0):
                    continue
                frame = download[provider_symbol].copy()
            else:
                frame = download.copy()
            frame = frame.rename(
                columns={
                    "Open": "open",
                    "High": "high",
                    "Low": "low",
                    "Close": "close",
                    "Adj Close": "adj_close",
                    "Volume": "volume",
                }
            )
            for dt, row in frame.dropna(subset=["close"]).iterrows():
                rows.append(
                    {
                        "date": pd.to_datetime(dt).date(),
                        "ticker": ticker,
                        "open": float(row.get("open", row["close"])),
                        "high": float(row.get("high", row["close"])),
                        "low": float(row.get("low", row["close"])),
                        "close": float(row["close"]),
                        "adj_close": float(row.get("adj_close", row["close"])),
                        "volume": float(row.get("volume", 0.0) or 0.0),
                        "source": "yfinance",
                        "created_at": _utc_now(),
                    }
                )
        if not rows:
            raise RuntimeError("No usable yfinance rows were parsed")
        return pd.DataFrame(rows)
    except Exception as exc:  # pragma: no cover - live fallback depends on network
        logger.warning("Live market fetch failed; falling back to fixtures: %s", exc)
        return make_market_fixture(tickers, start=start, end=end)
