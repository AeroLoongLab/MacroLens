from __future__ import annotations

import argparse
import sys
from datetime import UTC, datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from macro_quant.models.simulation_model import MonteCarloSimulation  # noqa: E402
from macro_quant.pipeline import run_daily_update  # noqa: E402
from macro_quant.utils.config import ensure_runtime_dirs, load_settings  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="Run MacroLens Monte Carlo simulation.")
    parser.add_argument("--horizon", default="6m", choices=["1m", "3m", "6m", "12m"])
    parser.add_argument("--paths", type=int, default=10_000)
    parser.add_argument("--live", action="store_true", help="Refresh live data before simulation.")
    args = parser.parse_args()
    settings = load_settings(project_root=ROOT)
    ensure_runtime_dirs(settings)
    daily = run_daily_update(live=args.live, project_root=ROOT)
    result = MonteCarloSimulation().run(daily.asset_scores, horizon=args.horizon, paths=args.paths)
    report_path = settings.reports_dir / "simulations" / f"{datetime.now(UTC).strftime('%Y%m%d_%H%M%S')}_simulation.md"
    report_path.write_text(_render_report(result.summary, args.horizon, args.paths), encoding="utf-8")
    print(result.summary.to_string(index=False, float_format=lambda value: f"{value:.4f}"))
    print(f"Simulation report: {report_path}")


def _render_report(summary, horizon: str, paths: int) -> str:
    return "\n".join(
        [
            f"# Monte Carlo Simulation - {horizon}",
            "",
            f"Paths: {paths}",
            "",
            summary.to_markdown(index=False, floatfmt=".4f"),
            "",
            "本系统用于研究和辅助决策，不构成投资建议。",
            "",
        ]
    )


if __name__ == "__main__":
    main()
