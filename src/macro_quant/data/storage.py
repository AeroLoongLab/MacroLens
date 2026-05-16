from __future__ import annotations

from collections.abc import Iterable, Iterator
from contextlib import contextmanager
from pathlib import Path

import duckdb
import pandas as pd


class DuckDBStore:
    def __init__(self, db_path: Path | str) -> None:
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)

    def connect(self) -> duckdb.DuckDBPyConnection:
        return duckdb.connect(str(self.db_path))

    @contextmanager
    def connection(self) -> Iterator[duckdb.DuckDBPyConnection]:
        con = self.connect()
        try:
            yield con
        finally:
            con.close()

    def init_schema(self) -> None:
        with self.connection() as con:
            con.execute("""
                CREATE TABLE IF NOT EXISTS market_prices (
                    date DATE,
                    ticker TEXT,
                    open DOUBLE,
                    high DOUBLE,
                    low DOUBLE,
                    close DOUBLE,
                    adj_close DOUBLE,
                    volume DOUBLE,
                    source TEXT,
                    created_at TIMESTAMP
                )
                """)
            con.execute("""
                CREATE TABLE IF NOT EXISTS macro_indicators (
                    date DATE,
                    indicator TEXT,
                    value DOUBLE,
                    source TEXT,
                    created_at TIMESTAMP
                )
                """)
            con.execute("""
                CREATE TABLE IF NOT EXISTS news_articles (
                    id TEXT PRIMARY KEY,
                    published_at TIMESTAMP,
                    source TEXT,
                    title TEXT,
                    url TEXT,
                    topic TEXT,
                    summary TEXT,
                    sentiment DOUBLE,
                    relevance_score DOUBLE,
                    narrative_category TEXT,
                    narrative_score DOUBLE,
                    created_at TIMESTAMP
                )
                """)
            con.execute("""
                CREATE TABLE IF NOT EXISTS regime_probabilities (
                    date DATE,
                    regime TEXT,
                    probability DOUBLE,
                    model_version TEXT,
                    created_at TIMESTAMP
                )
                """)
            con.execute("""
                CREATE TABLE IF NOT EXISTS asset_scores (
                    date DATE,
                    asset TEXT,
                    direction_score DOUBLE,
                    risk_score DOUBLE,
                    valuation_score DOUBLE,
                    macro_fit_score DOUBLE,
                    final_score DOUBLE,
                    model_version TEXT,
                    created_at TIMESTAMP
                )
                """)
            con.execute("""
                CREATE TABLE IF NOT EXISTS portfolio_recommendations (
                    date DATE,
                    portfolio_type TEXT,
                    asset TEXT,
                    weight DOUBLE,
                    reason TEXT,
                    model_version TEXT,
                    created_at TIMESTAMP
                )
                """)

    def upsert_market_prices(self, frame: pd.DataFrame) -> int:
        columns = [
            "date",
            "ticker",
            "open",
            "high",
            "low",
            "close",
            "adj_close",
            "volume",
            "source",
            "created_at",
        ]
        return self._replace_rows("market_prices", frame, columns, ["date", "ticker", "source"])

    def upsert_macro_indicators(self, frame: pd.DataFrame) -> int:
        columns = ["date", "indicator", "value", "source", "created_at"]
        return self._replace_rows("macro_indicators", frame, columns, ["date", "indicator", "source"])

    def upsert_news_articles(self, frame: pd.DataFrame) -> int:
        columns = [
            "id",
            "published_at",
            "source",
            "title",
            "url",
            "topic",
            "summary",
            "sentiment",
            "relevance_score",
            "narrative_category",
            "narrative_score",
            "created_at",
        ]
        return self._replace_rows("news_articles", frame, columns, ["id"])

    def upsert_regime_probabilities(self, frame: pd.DataFrame) -> int:
        columns = ["date", "regime", "probability", "model_version", "created_at"]
        return self._replace_rows("regime_probabilities", frame, columns, ["date", "regime", "model_version"])

    def upsert_asset_scores(self, frame: pd.DataFrame) -> int:
        columns = [
            "date",
            "asset",
            "direction_score",
            "risk_score",
            "valuation_score",
            "macro_fit_score",
            "final_score",
            "model_version",
            "created_at",
        ]
        return self._replace_rows("asset_scores", frame, columns, ["date", "asset", "model_version"])

    def upsert_portfolio_recommendations(self, frame: pd.DataFrame) -> int:
        columns = [
            "date",
            "portfolio_type",
            "asset",
            "weight",
            "reason",
            "model_version",
            "created_at",
        ]
        return self._replace_rows(
            "portfolio_recommendations",
            frame,
            columns,
            ["date", "portfolio_type", "asset", "model_version"],
        )

    def read_table(self, table: str) -> pd.DataFrame:
        with self.connection() as con:
            return con.execute(f"SELECT * FROM {table}").fetchdf()

    def latest_date(self, table: str, date_column: str = "date") -> pd.Timestamp | None:
        with self.connection() as con:
            result = con.execute(f"SELECT max({date_column}) FROM {table}").fetchone()[0]
        return pd.to_datetime(result) if result is not None else None

    def _replace_rows(
        self,
        table: str,
        frame: pd.DataFrame,
        columns: list[str],
        key_columns: Iterable[str],
    ) -> int:
        if frame.empty:
            return 0
        clean = frame.copy()
        for column in columns:
            if column not in clean.columns:
                clean[column] = None
        clean = clean[columns]
        temp_name = f"tmp_{table}"
        key_expr = ", ".join(key_columns)
        with self.connection() as con:
            con.register(temp_name, clean)
            con.execute(f"DELETE FROM {table} WHERE ({key_expr}) IN (SELECT {key_expr} FROM {temp_name})")
            con.execute(f"INSERT INTO {table} SELECT * FROM {temp_name}")
            con.unregister(temp_name)
        return int(len(clean))
