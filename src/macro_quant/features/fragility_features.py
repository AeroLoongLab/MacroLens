from __future__ import annotations

from macro_quant.utils.math_utils import clamp


def build_fragility_features(features: dict[str, float]) -> dict[str, float]:
    macro_stress = _score_0_100(
        [
            features.get("inflation_risk", 0.0),
            features.get("oil_momentum", 0.0),
            features.get("credit_spread", 0.0),
            features.get("vix_level", 0.0),
            features.get("move_level", 0.0),
            features.get("usd_momentum", 0.0),
            features.get("treasury_supply", 0.0),
            features.get("fiscal_interest_pressure", 0.0),
        ]
    )
    fiscal_risk = _score_0_100(
        [
            features.get("treasury_supply", 0.0),
            features.get("fiscal_interest_pressure", 0.0),
            features.get("term_premium", 0.0),
            -features.get("long_bond_return", 0.0),
            features.get("gold_momentum", 0.0),
            features.get("fiscal_credit_risk_score", 0.0),
        ]
    )
    gold_crowding = _score_0_100(
        [
            features.get("gold_momentum", 0.0),
            features.get("gold_crowding_score", 0.0),
            features.get("gold_narrative_score", 0.0),
        ]
    )
    long_bond_fragility = _score_0_100(
        [
            features.get("term_premium", 0.0),
            features.get("inflation_risk", 0.0),
            features.get("fiscal_credit_risk_score", 0.0),
            -features.get("long_bond_return", 0.0),
            features.get("move_level", 0.0),
            features.get("usd_momentum", 0.0),
        ]
    )
    phase_transition = _score_0_100(
        [
            features.get("vix_level", 0.0),
            features.get("move_level", 0.0),
            features.get("credit_spread", 0.0),
            abs(features.get("stock_bond_correlation_63d", 0.0)),
            abs(features.get("gold_bond_correlation_63d", 0.0)),
            features.get("liquidity_pressure_score", 0.0),
        ]
    )
    ai_roi_pressure = _score_0_100(
        [
            features.get("ai_capex_risk_score", 0.0),
            features.get("valuation_pressure", 0.0),
            -features.get("ai_revenue_realization_score", 0.0),
        ]
    )
    return {
        "macro_stress_index": macro_stress,
        "fiscal_credibility_risk_index": fiscal_risk,
        "gold_crowding_index": gold_crowding,
        "long_bond_fragility_index": long_bond_fragility,
        "phase_transition_risk_index": phase_transition,
        "ai_roi_pressure_index": ai_roi_pressure,
        "fiscal_credit_risk": fiscal_risk / 100.0,
        "gold_crowding_score": gold_crowding / 100.0,
        "long_bond_fragility": long_bond_fragility / 100.0,
        "phase_transition_risk": phase_transition / 100.0,
        "liquidity_score": clamp(1.0 - phase_transition / 100.0, 0.0, 1.0),
    }


def _score_0_100(values: list[float]) -> float:
    if not values:
        return 0.0
    positive = [max(0.0, clamp(value)) for value in values]
    return float(max(0.0, min(100.0, 100.0 * sum(positive) / len(positive))))
