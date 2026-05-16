from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from datetime import UTC, date, datetime

import pandas as pd

from macro_quant.models.asset_score_model import AssetScore
from macro_quant.models.regime_model import RegimeResult


def _utc_now() -> datetime:
    return datetime.now(UTC).replace(tzinfo=None)


@dataclass(frozen=True)
class PortfolioRecommendation:
    portfolio_type: str
    asset: str
    weight: float
    reason: str
    selected: bool


class PortfolioModel:
    def __init__(self, portfolio_config: dict[str, object], model_version: str) -> None:
        self.portfolio_config = portfolio_config
        self.version = model_version

    def recommend(
        self,
        regime_results: Iterable[RegimeResult],
        asset_scores: Iterable[AssetScore],
        features: dict[str, float],
    ) -> list[PortfolioRecommendation]:
        selected_type = self.select_portfolio_type(regime_results, asset_scores, features)
        recommendations: list[PortfolioRecommendation] = []
        for portfolio_type, spec in self.portfolio_config.get("portfolio_types", {}).items():
            base_weights = spec.get("base_weights", {}) if isinstance(spec, dict) else {}
            reason = self._reason(portfolio_type, selected_type, features)
            for asset, weight in base_weights.items():
                recommendations.append(
                    PortfolioRecommendation(
                        portfolio_type=str(portfolio_type),
                        asset=str(asset),
                        weight=float(weight),
                        reason=reason,
                        selected=str(portfolio_type) == selected_type,
                    )
                )
        return recommendations

    def select_portfolio_type(
        self,
        regime_results: Iterable[RegimeResult],
        asset_scores: Iterable[AssetScore],
        features: dict[str, float],
    ) -> str:
        regimes = {result.regime: result.probability for result in regime_results}
        scores = {score.asset: score.final_score for score in asset_scores}
        if (
            regimes.get("R4_fiscal_credit_shock", 0.0) + regimes.get("R6_liquidity_cascade", 0.0) > 0.38
            or features.get("phase_transition_risk_index", 0.0) > 65
        ):
            return "crisis_defense"
        if regimes.get("R3_ai_productivity_bull", 0.0) > 0.28 and scores.get("Equity", 0.0) > 20:
            return "aggressive"
        if features.get("macro_stress_index", 0.0) > 60 or scores.get("Equity", 0.0) < -20:
            return "conservative"
        return "balanced"

    def to_frame(self, recommendations: list[PortfolioRecommendation], as_of: date) -> pd.DataFrame:
        return pd.DataFrame(
            [
                {
                    "date": as_of,
                    "portfolio_type": rec.portfolio_type,
                    "asset": rec.asset,
                    "weight": rec.weight,
                    "reason": rec.reason + (" [selected]" if rec.selected else ""),
                    "model_version": self.version,
                    "created_at": _utc_now(),
                }
                for rec in recommendations
            ]
        )

    @staticmethod
    def selected(recommendations: list[PortfolioRecommendation]) -> list[PortfolioRecommendation]:
        return [rec for rec in recommendations if rec.selected]

    @staticmethod
    def _reason(portfolio_type: str, selected_type: str, features: dict[str, float]) -> str:
        if portfolio_type != selected_type:
            return "Generated as comparison allocation"
        return (
            f"Selected by regime/score rules; macro stress={features.get('macro_stress_index', 0.0):.1f}, "
            f"phase risk={features.get('phase_transition_risk_index', 0.0):.1f}"
        )
