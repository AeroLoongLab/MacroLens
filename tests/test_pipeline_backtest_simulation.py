from datetime import date
from pathlib import Path

from macro_quant.backtesting.backtest_engine import BacktestEngine
from macro_quant.data.macro_data import fetch_macro_indicators
from macro_quant.data.market_data import fetch_market_prices
from macro_quant.models.simulation_model import MonteCarloSimulation
from macro_quant.pipeline import run_daily_update
from macro_quant.utils.config import load_config

ROOT = Path(__file__).resolve().parents[1]


def test_offline_daily_update_generates_report(tmp_path) -> None:
    result = run_daily_update(
        live=False,
        as_of=date(2026, 5, 15),
        db_path=tmp_path / "macro.duckdb",
        project_root=ROOT,
        output_root=tmp_path,
        start="2024-01-01",
    )
    assert result.markdown_report.exists()
    assert result.markdown_report_zh.exists()
    assert result.html_report_zh.exists()
    assert result.markdown_report.name.endswith("_macrolens_report.md")
    assert result.markdown_report_zh.name.endswith("_macrolens_report_zh.md")
    text = result.markdown_report.read_text(encoding="utf-8")
    text_zh = result.markdown_report_zh.read_text(encoding="utf-8")
    assert "MacroLens Daily Report" in text
    assert "Regime Probabilities" in text
    assert "宏观透镜日报" in text_zh
    assert "Regime 概率" in text_zh
    assert "不构成投资建议" in text
    assert "不构成投资建议" in text_zh


def test_backtest_and_simulation_smoke(tmp_path) -> None:
    config = load_config(ROOT / "configs")
    market = fetch_market_prices(config.tickers, start="2023-01-01", end=date(2026, 5, 15), live=False)
    macro = fetch_macro_indicators(config.indicators, start="2023-01-01", end=date(2026, 5, 15), live=False)
    result = BacktestEngine(config).run(market, macro, start="2023-01-01", end=date(2026, 5, 15))
    assert "Model Portfolio" in result.metrics.index
    daily = run_daily_update(
        live=False,
        as_of=date(2026, 5, 15),
        db_path=tmp_path / "sim.duckdb",
        project_root=ROOT,
        output_root=tmp_path,
        start="2024-01-01",
    )
    simulation = MonteCarloSimulation(seed=7).run(daily.asset_scores, horizon="6m", paths=100)
    assert len(simulation.terminal_returns) == 100
    assert {"asset", "mean", "VaR_95", "CVaR_95"}.issubset(simulation.summary.columns)
