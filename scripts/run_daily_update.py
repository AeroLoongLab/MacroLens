from __future__ import annotations

import argparse
import sys
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from macro_quant.pipeline import run_daily_update  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="Run MacroLens daily update.")
    parser.add_argument("--live", action="store_true", help="Use live data sources with fixture fallback.")
    parser.add_argument("--date", default=None, help="As-of date in YYYY-MM-DD format.")
    args = parser.parse_args()
    as_of = date.fromisoformat(args.date) if args.date else date.today()
    result = run_daily_update(live=args.live, as_of=as_of, project_root=ROOT)
    print(f"Markdown report: {result.markdown_report}")
    print(f"HTML report: {result.html_report}")
    print(f"Chinese Markdown report: {result.markdown_report_zh}")
    print(f"Chinese HTML report: {result.html_report_zh}")


if __name__ == "__main__":
    main()
