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
    failed_indicators: list[str] = []
    for indicator in indicators:
        try:
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
        except Exception as exc:  # pragma: no cover - live fallback depends on network
            failed_indicators.append(indicator)
            logger.warning(
                "Live FRED fetch failed for %s; falling back to fixture for this indicator: %s",
                indicator,
                _safe_error_summary(exc),
            )
    frames: list[pd.DataFrame] = []
    if rows:
        frames.append(pd.DataFrame(rows))
    if failed_indicators:
        frames.append(make_macro_fixture(failed_indicators, start=start, end=end))
    if frames:
        return pd.concat(frames, ignore_index=True)
    logger.warning("FRED returned no usable observations; falling back to macro fixtures")
    return make_macro_fixture(indicators, start=start, end=end)


def _safe_error_summary(exc: Exception) -> str:
    response = getattr(exc, "response", None)
    status_code = getattr(response, "status_code", None)
    if status_code is not None:
        return f"{type(exc).__name__}(status_code={status_code})"
    return type(exc).__name__
