from datetime import date

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
