from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from macro_quant.models.asset_score_model import AssetScore


@dataclass(frozen=True)
class SimulationResult:
    horizon: str
    paths: int
    summary: pd.DataFrame
    terminal_returns: pd.DataFrame


class MonteCarloSimulation:
    periods_by_horizon = {"1m": 21, "3m": 63, "6m": 126, "12m": 252}
    asset_order = ["Gold", "Equity", "Short_Bond", "Intermediate_Bond", "Long_Bond"]

    def __init__(self, seed: int = 42) -> None:
        self.seed = seed

    def run(
        self,
        asset_scores: list[AssetScore],
        horizon: str = "6m",
        paths: int = 10_000,
    ) -> SimulationResult:
        if horizon not in self.periods_by_horizon:
            raise ValueError(f"Unsupported horizon {horizon}; expected one of {sorted(self.periods_by_horizon)}")
        periods = self.periods_by_horizon[horizon]
        score_map = {score.asset: score.final_score for score in asset_scores}
        annual_mu = np.array([0.02 + score_map.get(asset, 0.0) / 100.0 * 0.10 for asset in self.asset_order])
        annual_vol = np.array([0.17, 0.19, 0.025, 0.075, 0.145])
        corr = np.array(
            [
                [1.00, 0.10, -0.05, 0.05, 0.05],
                [0.10, 1.00, -0.10, -0.20, -0.25],
                [-0.05, -0.10, 1.00, 0.45, 0.25],
                [0.05, -0.20, 0.45, 1.00, 0.65],
                [0.05, -0.25, 0.25, 0.65, 1.00],
            ]
        )
        cov = np.outer(annual_vol, annual_vol) * corr / 252.0
        daily_mu = annual_mu / 252.0
        rng = np.random.default_rng(self.seed)
        shocks = rng.multivariate_normal(daily_mu, cov, size=(paths, periods))
        terminal = (1.0 + shocks).prod(axis=1) - 1.0
        terminal_frame = pd.DataFrame(terminal, columns=self.asset_order)
        summary = _summary_frame(terminal_frame)
        return SimulationResult(
            horizon=horizon,
            paths=paths,
            summary=summary,
            terminal_returns=terminal_frame,
        )


def _summary_frame(terminal_returns: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, float | str]] = []
    for asset in terminal_returns.columns:
        series = terminal_returns[asset]
        var_95 = float(series.quantile(0.05))
        cvar_95 = float(series[series <= var_95].mean())
        rows.append(
            {
                "asset": asset,
                "mean": float(series.mean()),
                "median": float(series.median()),
                "p05": var_95,
                "p95": float(series.quantile(0.95)),
                "VaR_95": var_95,
                "CVaR_95": cvar_95,
            }
        )
    return pd.DataFrame(rows)
