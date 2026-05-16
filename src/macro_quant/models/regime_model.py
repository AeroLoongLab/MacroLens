from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, date, datetime

import pandas as pd

from macro_quant.utils.math_utils import softmax


def _utc_now() -> datetime:
    return datetime.now(UTC).replace(tzinfo=None)


@dataclass(frozen=True)
class RegimeResult:
    regime: str
    probability: float
    raw_score: float
    contributions: dict[str, float]


class RegimeModel:
    def __init__(self, model_config: dict[str, object]) -> None:
        self.model_config = model_config
        self.version = str(model_config.get("model_version", "v0.unknown"))
        self.temperature = float(model_config.get("regime_temperature", 1.0))
        self.weights: dict[str, dict[str, float]] = model_config.get("regime_weights", {})  # type: ignore[assignment]

    def predict(self, features: dict[str, float]) -> list[RegimeResult]:
        raw_scores: dict[str, float] = {}
        contributions_by_regime: dict[str, dict[str, float]] = {}
        for regime, weights in self.weights.items():
            contributions = {
                feature: float(weight) * float(features.get(feature, 0.0)) for feature, weight in weights.items()
            }
            raw_scores[regime] = float(sum(contributions.values()))
            contributions_by_regime[regime] = contributions
        probabilities = softmax(raw_scores, temperature=self.temperature)
        return [
            RegimeResult(
                regime=regime,
                probability=probabilities.get(regime, 0.0),
                raw_score=raw_scores.get(regime, 0.0),
                contributions=contributions_by_regime.get(regime, {}),
            )
            for regime in sorted(raw_scores)
        ]

    def to_frame(self, results: list[RegimeResult], as_of: date) -> pd.DataFrame:
        return pd.DataFrame(
            [
                {
                    "date": as_of,
                    "regime": result.regime,
                    "probability": result.probability,
                    "model_version": self.version,
                    "created_at": _utc_now(),
                }
                for result in results
            ]
        )
