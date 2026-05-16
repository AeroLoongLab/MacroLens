from __future__ import annotations

import hashlib
from datetime import UTC, date, datetime, timedelta
from email.utils import parsedate_to_datetime
from typing import Any
from urllib.parse import quote_plus
from xml.etree import ElementTree

import pandas as pd
import requests

from macro_quant.data.fixtures import make_news_fixture
from macro_quant.utils.logging import get_logger

logger = get_logger(__name__)


def _utc_now() -> datetime:
    return datetime.now(UTC).replace(tzinfo=None)


def fetch_news_articles(
    narrative_config: dict[str, Any],
    as_of: str | date | None = None,
    live: bool = False,
) -> pd.DataFrame:
    if not live:
        return make_news_fixture(narrative_config, as_of=as_of)
    rows: list[dict[str, object]] = []
    narratives = narrative_config.get("narratives", {})
    try:
        for narrative, spec in narratives.items():
            spec_dict = spec if isinstance(spec, dict) else {}
            query = str(spec_dict.get("query", narrative))
            url = "https://news.google.com/rss/search?q=" f"{quote_plus(query)}&hl=en-US&gl=US&ceid=US:en"
            response = requests.get(url, timeout=20)
            response.raise_for_status()
            root = ElementTree.fromstring(response.content)
            for item in root.findall(".//item")[:20]:
                title = item.findtext("title") or ""
                link = item.findtext("link") or ""
                source = item.findtext("source") or "Google News"
                published_text = item.findtext("pubDate")
                published = _parse_published_at(published_text)
                article_id = hashlib.sha1(link.encode("utf-8")).hexdigest()
                rows.append(
                    {
                        "id": article_id,
                        "published_at": published,
                        "source": source,
                        "title": title,
                        "url": link,
                        "topic": query,
                        "summary": title,
                        "sentiment": _keyword_sentiment(title),
                        "relevance_score": 0.7,
                        "narrative_category": narrative,
                        "narrative_score": 45.0,
                        "created_at": _utc_now(),
                    }
                )
        if not rows:
            raise RuntimeError("Google News RSS returned no usable rows")
        return pd.DataFrame(rows)
    except Exception as exc:  # pragma: no cover - live fallback depends on network
        logger.warning("Live news fetch failed; falling back to fixtures: %s", exc)
        return make_news_fixture(narrative_config, as_of=as_of)


def _parse_published_at(value: str | None) -> datetime:
    if not value:
        return _utc_now()
    try:
        return parsedate_to_datetime(value).replace(tzinfo=None)
    except Exception:
        return _utc_now() - timedelta(hours=1)


def _keyword_sentiment(text: str) -> float:
    lowered = text.lower()
    positive = ["growth", "beat", "strong", "cooling", "resilient", "demand"]
    negative = ["risk", "stress", "miss", "hot", "deficit", "shock", "recession"]
    score = sum(word in lowered for word in positive) - sum(word in lowered for word in negative)
    return max(-1.0, min(1.0, score / 3.0))
