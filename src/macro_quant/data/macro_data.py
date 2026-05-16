from __future__ import annotations

import os
from datetime import UTC, date, datetime
from typing import Any

import pandas as pd
import requests

from macro_quant.data.fixtures import make_macro_fixture
from macro_quant.utils.logging import get_logger

logger = get_logger(__name__)


def _utc_now() -> datetime:
    return datetime.now(UTC).replace(tzinfo=None)


def configured_fred_indicators(indicator_config: dict[str, Any]) -> list[str]:
    return sorted(str(key) for key in indicator_config.get("fred", {}).keys())


def fetch_macro_indicators(
    indicator_config: dict[str, Any],
    start: str | date = "2010-01-01",
    end: str | date | None = None,
    live: bool = False,
) -> pd.DataFrame:
    indicators = configured_fred_indicators(indicator_config)
    if not live:
        return make_macro_fixture(indicators, start=start, end=end)
    api_key = os.getenv("FRED_API_KEY", "")
    if not api_key:
        logger.warning("FRED_API_KEY is missing; falling back to macro fixtures")
        return make_macro_fixture(indicators, start=start, end=end)
    rows: list[dict[str, object]] = []
    try:
        for indicator in indicators:
            response = requests.get(
                "https://api.stlouisfed.org/fred/series/observations",
                params={
                    "series_id": indicator,
                    "api_key": api_key,
                    "file_type": "json",
                    "observation_start": str(start),
                    "observation_end": str(end or date.today()),
                },
                timeout=20,
            )
            response.raise_for_status()
            payload = response.json()
            for observation in payload.get("observations", []):
                value = observation.get("value")
                if value in {None, "."}:
                    continue
                rows.append(
                    {
                        "date": pd.to_datetime(observation["date"]).date(),
                        "indicator": indicator,
                        "value": float(value),
                        "source": "fred",
                        "created_at": _utc_now(),
                    }
                )
        if not rows:
            raise RuntimeError("FRED returned no usable observations")
        return pd.DataFrame(rows)
    except Exception as exc:  # pragma: no cover - live fallback depends on network
        logger.warning("Live FRED fetch failed; falling back to fixtures: %s", exc)
        return make_macro_fixture(indicators, start=start, end=end)
