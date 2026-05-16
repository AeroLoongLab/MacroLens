from __future__ import annotations

from datetime import UTC, date, datetime, timedelta

import numpy as np
import pandas as pd


def _utc_now() -> datetime:
    return datetime.now(UTC).replace(tzinfo=None)


def _business_dates(start: str | date, end: str | date) -> pd.DatetimeIndex:
    return pd.bdate_range(pd.to_datetime(start), pd.to_datetime(end))


def make_market_fixture(
    tickers: list[str], start: str | date = "2010-01-01", end: str | date | None = None
) -> pd.DataFrame:
    end = end or date.today()
    dates = _business_dates(start, end)
    ticker_params = {
        "SPY": (110.0, 0.075, 0.18),
        "QQQ": (45.0, 0.105, 0.22),
        "DIA": (100.0, 0.060, 0.16),
        "GLD": (108.0, 0.060, 0.17),
        "IAU": (11.0, 0.060, 0.17),
        "GC=F": (1120.0, 0.060, 0.18),
        "SHY": (82.0, 0.020, 0.025),
        "IEF": (88.0, 0.030, 0.075),
        "TLT": (92.0, 0.035, 0.145),
        "UUP": (22.0, 0.015, 0.08),
        "DXY": (80.0, 0.010, 0.075),
        "CL=F": (78.0, 0.020, 0.32),
        "BNO": (31.0, 0.020, 0.28),
        "VIX": (18.0, 0.000, 0.55),
        "MOVE": (90.0, 0.000, 0.35),
        "HYG": (78.0, 0.045, 0.095),
        "LQD": (103.0, 0.035, 0.085),
        "NVDA": (3.0, 0.280, 0.45),
        "MSFT": (28.0, 0.140, 0.24),
        "AMZN": (7.0, 0.150, 0.30),
        "GOOGL": (15.0, 0.130, 0.25),
        "META": (24.0, 0.140, 0.33),
        "AAPL": (8.0, 0.145, 0.25),
        "AVGO": (20.0, 0.180, 0.32),
    }
    rows: list[dict[str, object]] = []
    for index, ticker in enumerate(sorted(set(tickers))):
        start_price, annual_return, annual_vol = ticker_params.get(ticker, (50.0, 0.05, 0.16))
        rng = np.random.default_rng(10_000 + index)
        daily_mu = annual_return / 252.0
        daily_sigma = annual_vol / np.sqrt(252.0)
        shocks = rng.normal(daily_mu, daily_sigma, len(dates))
        if ticker in {"VIX", "MOVE"}:
            level = start_price + np.cumsum(rng.normal(0.0, annual_vol * 1.8, len(dates)))
            close = np.clip(level + 6 * np.sin(np.linspace(0, 45, len(dates))), 8.0, None)
        else:
            close = start_price * np.exp(np.cumsum(shocks))
        open_ = close * (1.0 + rng.normal(0.0, 0.002, len(dates)))
        high = np.maximum(open_, close) * (1.0 + np.abs(rng.normal(0.002, 0.003, len(dates))))
        low = np.minimum(open_, close) * (1.0 - np.abs(rng.normal(0.002, 0.003, len(dates))))
        volume = rng.integers(400_000, 20_000_000, len(dates))
        for dt, opn, hi, lo, cls, vol in zip(dates, open_, high, low, close, volume, strict=True):
            rows.append(
                {
                    "date": dt.date(),
                    "ticker": ticker,
                    "open": float(opn),
                    "high": float(hi),
                    "low": float(lo),
                    "close": float(cls),
                    "adj_close": float(cls),
                    "volume": float(vol),
                    "source": "fixture",
                    "created_at": _utc_now(),
                }
            )
    return pd.DataFrame(rows)


def make_macro_fixture(
    indicators: list[str], start: str | date = "2010-01-01", end: str | date | None = None
) -> pd.DataFrame:
    end = end or date.today()
    dates = _business_dates(start, end)
    base = {
        "CPIAUCSL": (218.0, 0.00010, 0.00035),
        "CPILFESL": (220.0, 0.00009, 0.00025),
        "PCEPI": (92.0, 0.00008, 0.00020),
        "PCEPILFE": (93.0, 0.00008, 0.00018),
        "UNRATE": (9.5, -0.0009, 0.015),
        "PAYEMS": (130_000.0, 4.5, 40.0),
        "FEDFUNDS": (0.15, 0.00025, 0.018),
        "DGS10": (3.4, 0.00005, 0.025),
        "DGS2": (0.9, 0.00028, 0.025),
        "DGS30": (4.2, 0.00003, 0.022),
        "DFII10": (0.6, 0.00006, 0.015),
        "T10YIE": (2.0, 0.00002, 0.010),
        "GFDEBTN": (13_000_000.0, 4100.0, 20_000.0),
        "A091RC1Q027SBEA": (190_000.0, 82.0, 850.0),
        "FYFSGDA188S": (-8.5, 0.0005, 0.018),
        "NAPM": (54.0, -0.0002, 0.070),
        "IPMANSICS": (91.0, 0.002, 0.035),
        "RSAFS": (360_000.0, 22.0, 500.0),
        "INDPRO": (92.0, 0.002, 0.040),
        "BAA10Y": (2.9, 0.00001, 0.010),
        "BAMLH0A0HYM2": (5.2, 0.00002, 0.025),
    }
    rows: list[dict[str, object]] = []
    for index, indicator in enumerate(sorted(set(indicators))):
        start_value, drift, vol = base.get(indicator, (50.0, 0.0, 0.1))
        rng = np.random.default_rng(20_000 + index)
        values = start_value + np.cumsum(rng.normal(drift, vol, len(dates)))
        if indicator == "UNRATE":
            values = np.clip(values, 3.3, 11.0)
        if indicator in {"FEDFUNDS", "DGS10", "DGS2", "DGS30", "DFII10", "T10YIE", "BAA10Y"}:
            values = np.clip(values + 1.4 * np.sin(np.linspace(0, 18, len(dates))), -1.0, 8.0)
        if indicator == "NAPM":
            values = np.clip(values + 4.0 * np.sin(np.linspace(0, 26, len(dates))), 35.0, 65.0)
        if indicator == "IPMANSICS":
            values = np.clip(values, 60.0, None)
        for dt, value in zip(dates, values, strict=True):
            rows.append(
                {
                    "date": dt.date(),
                    "indicator": indicator,
                    "value": float(value),
                    "source": "fixture",
                    "created_at": _utc_now(),
                }
            )
    return pd.DataFrame(rows)


def make_news_fixture(
    narrative_config: dict[str, object],
    as_of: str | date | None = None,
    days: int = 7,
) -> pd.DataFrame:
    as_of_date = pd.to_datetime(as_of or date.today()).date()
    narratives = narrative_config.get("narratives", {})
    rows: list[dict[str, object]] = []
    for idx, (narrative, spec) in enumerate(narratives.items()):
        spec_dict = spec if isinstance(spec, dict) else {}
        query = str(spec_dict.get("query", narrative))
        for offset in range(days):
            published = datetime.combine(as_of_date - timedelta(days=offset), datetime.min.time()).replace(
                hour=9 + (idx % 8)
            )
            article_id = f"fixture-{narrative}-{published.date()}"
            sentiment = ((idx % 5) - 2) / 5
            relevance = 0.55 + (idx % 4) * 0.1
            rows.append(
                {
                    "id": article_id,
                    "published_at": published,
                    "source": "FixtureWire",
                    "title": f"{query}: fixture update {published.date()}",
                    "url": f"https://example.com/{article_id}",
                    "topic": query,
                    "summary": f"Deterministic fixture article for {narrative}.",
                    "sentiment": float(sentiment),
                    "relevance_score": float(min(relevance, 1.0)),
                    "narrative_category": narrative,
                    "narrative_score": float(35 + idx * 3 + max(0, days - offset)),
                    "created_at": _utc_now(),
                }
            )
    return pd.DataFrame(rows)
