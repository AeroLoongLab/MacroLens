from __future__ import annotations

import argparse
import sys
from datetime import UTC, date, datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from macro_quant.backtesting.backtest_engine import BacktestEngine  # noqa: E402
from macro_quant.data.macro_data import fetch_macro_indicators  # noqa: E402
from macro_quant.data.market_data import fetch_market_prices  # noqa: E402
from macro_quant.utils.config import ensure_runtime_dirs, load_config, load_settings  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="Run MacroLens backtest.")
    parser.add_argument("--start", default="2010-01-01")
    parser.add_argument("--end", default="today")
    args = parser.parse_args()
    settings = load_settings(project_root=ROOT)
    ensure_runtime_dirs(settings)
    config = load_config(settings.config_dir)
    end = date.today() if args.end == "today" else date.fromisoformat(args.end)
    market = fetch_market_prices(config.tickers, start=args.start, end=end, live=False)
    macro = fetch_macro_indicators(config.indicators, start=args.start, end=end, live=False)
    result = BacktestEngine(config).run(market, macro, start=args.start, end=end)
    report_path = settings.reports_dir / "backtests" / f"{datetime.now(UTC).strftime('%Y%m%d_%H%M%S')}_backtest.md"
    report_path.write_text(_render_report(result.metrics), encoding="utf-8")
    print(result.metrics.to_string(float_format=lambda value: f"{value:.4f}"))
    print(f"Backtest report: {report_path}")


def _render_report(metrics) -> str:
    return "\n".join(
        [
            "# MacroLens Backtest",
            "",
            "## Metrics",
            "",
            metrics.to_markdown(floatfmt=".4f"),
            "",
            "本系统用于研究和辅助决策，不构成投资建议。",
            "",
        ]
    )


if __name__ == "__main__":
    main()
