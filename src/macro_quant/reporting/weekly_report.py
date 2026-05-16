from __future__ import annotations

from datetime import date
from pathlib import Path


def generate_weekly_placeholder(as_of: date, reports_dir: Path) -> Path:
    weekly_dir = reports_dir / "weekly"
    weekly_dir.mkdir(parents=True, exist_ok=True)
    path = weekly_dir / f"{as_of.strftime('%Y-W%W')}_weekly_report.md"
    if not path.exists():
        path.write_text(
            "# MacroLens Weekly Report\n\nWeekly attribution is reserved for the next scheduled run.\n",
            encoding="utf-8",
        )
    return path
