from datetime import date

import pandas as pd

from macro_quant.data.fixtures import make_market_fixture
from macro_quant.data.storage import DuckDBStore


def test_market_price_upsert_is_idempotent(tmp_path) -> None:
    store = DuckDBStore(tmp_path / "test.duckdb")
    store.init_schema()
    frame = make_market_fixture(["GLD"], start=date(2026, 1, 1), end=date(2026, 1, 9))
    first = store.upsert_market_prices(frame)
    second = store.upsert_market_prices(frame)
    stored = store.read_table("market_prices")
    assert first == second == len(frame)
    assert len(stored) == len(frame)


def test_news_article_upsert_deduplicates_input_keys(tmp_path) -> None:
    store = DuckDBStore(tmp_path / "test.duckdb")
    store.init_schema()
    frame = pd.DataFrame(
        [
            {
                "id": "same-link",
                "published_at": pd.Timestamp("2026-05-16T10:00:00"),
                "source": "Google News",
                "title": "First title",
                "url": "https://example.com/a",
                "topic": "gold",
                "summary": "First title",
                "sentiment": 0.0,
                "relevance_score": 0.7,
                "narrative_category": "gold",
                "narrative_score": 45.0,
                "created_at": pd.Timestamp("2026-05-16T10:00:00"),
            },
            {
                "id": "same-link",
                "published_at": pd.Timestamp("2026-05-16T10:05:00"),
                "source": "Google News",
                "title": "Second title",
                "url": "https://example.com/a",
                "topic": "safe haven",
                "summary": "Second title",
                "sentiment": 0.1,
                "relevance_score": 0.7,
                "narrative_category": "gold",
                "narrative_score": 45.0,
                "created_at": pd.Timestamp("2026-05-16T10:05:00"),
            },
        ]
    )

    inserted = store.upsert_news_articles(frame)
    stored = store.read_table("news_articles")

    assert inserted == 1
    assert len(stored) == 1
    assert stored.iloc[0]["title"] == "Second title"
