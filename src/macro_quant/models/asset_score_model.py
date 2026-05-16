from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, date, datetime

import pandas as pd

from macro_quant.utils.math_utils import clamp_score


def _utc_now() -> datetime:
    return datetime.now(UTC).replace(tzinfo=None)


@dataclass(frozen=True)
class AssetScore:
    asset: str
    direction_score: float
    risk_score: float
    valuation_score: float
    macro_fit_score: float
    final_score: float
    components: dict[str, float]


class AssetScoreModel:
    def __init__(self, model_config: dict[str, object]) -> None:
        self.model_config = model_config
        self.version = str(model_config.get("model_version", "v0.unknown"))
        self.weights: dict[str, dict[str, float]] = model_config.get("asset_weights", {})  # type: ignore[assignment]

    def score(self, features: dict[str, float]) -> list[AssetScore]:
        scores: list[AssetScore] = []
        for asset, weights in self.weights.items():
            components = {
                feature: float(weight) * float(features.get(feature, 0.0)) * 100.0
                for feature, weight in weights.items()
            }
            final = clamp_score(sum(components.values()))
            scores.append(
                AssetScore(
                    asset=asset,
                    direction_score=_direction_score(asset, features),
                    risk_score=_risk_score(asset, features),
                    valuation_score=_valuation_score(asset, features),
                    macro_fit_score=clamp_score(final + _macro_bias(asset, features)),
                    final_score=final,
                    components=components,
                )
            )
        return scores

    def to_frame(self, scores: list[AssetScore], as_of: date) -> pd.DataFrame:
        return pd.DataFrame(
            [
                {
                    "date": as_of,
                    "asset": score.asset,
                    "direction_score": score.direction_score,
                    "risk_score": score.risk_score,
                    "valuation_score": score.valuation_score,
                    "macro_fit_score": score.macro_fit_score,
                    "final_score": score.final_score,
                    "model_version": self.version,
                    "created_at": _utc_now(),
                }
                for score in scores
            ]
        )


def _direction_score(asset: str, features: dict[str, float]) -> float:
    mapping = {
        "Gold": "gold_momentum",
        "Equity": "sp500_momentum",
        "Short_Bond": "yield_level",
        "Intermediate_Bond": "rate_cut_expectation",
        "Long_Bond": "long_bond_return",
    }
    return clamp_score(float(features.get(mapping.get(asset, ""), 0.0)) * 100.0)


def _risk_score(asset: str, features: dict[str, float]) -> float:
    mapping = {
        "Gold": "gold_crowding_score",
        "Equity": "volatility_score",
        "Short_Bond": "reinvestment_risk",
        "Intermediate_Bond": "inflation_risk",
        "Long_Bond": "long_bond_fragility",
    }
    return max(0.0, min(100.0, float(features.get(mapping.get(asset, ""), 0.0)) * 100.0))


def _valuation_score(asset: str, features: dict[str, float]) -> float:
    mapping = {
        "Gold": "gold_crowding_score",
        "Equity": "valuation_pressure",
        "Short_Bond": "yield_level",
        "Intermediate_Bond": "yield_level",
        "Long_Bond": "term_premium",
    }
    value = float(features.get(mapping.get(asset, ""), 0.0))
    if asset in {"Gold", "Equity"}:
        value = -value
    return clamp_score(value * 100.0)


def _macro_bias(asset: str, features: dict[str, float]) -> float:
    if asset == "Gold":
        return 20.0 * features.get("fiscal_credit_risk", 0.0)
    if asset == "Equity":
        return 20.0 * features.get("liquidity_score", 0.0)
    if asset == "Long_Bond":
        return -20.0 * features.get("fiscal_credit_risk", 0.0)
    return 0.0
