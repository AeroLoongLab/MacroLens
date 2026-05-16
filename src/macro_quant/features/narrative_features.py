from __future__ import annotations

from datetime import datetime
from typing import Any

import pandas as pd

from macro_quant.utils.math_utils import clamp


def classify_news(news: pd.DataFrame, narrative_config: dict[str, Any]) -> pd.DataFrame:
    if news.empty:
        return news
    frame = news.copy()
    narratives = narrative_config.get("narratives", {})
    for index, row in frame.iterrows():
        current = str(row.get("narrative_category") or "")
        if current in narratives:
            continue
        text = f"{row.get('title', '')} {row.get('summary', '')}".lower()
        best = "N10_liquidity_pressure"
        best_hits = -1
        for narrative, spec in narratives.items():
            keywords = spec.get("keywords", []) if isinstance(spec, dict) else []
            hits = sum(str(keyword).lower() in text for keyword in keywords)
            if hits > best_hits:
                best = narrative
                best_hits = hits
        frame.at[index, "narrative_category"] = best
    return frame


def build_narrative_features(
    news: pd.DataFrame, narrative_config: dict[str, Any], as_of: datetime | None = None
) -> dict[str, float]:
    if news.empty:
        return _empty_narrative_scores(narrative_config)
    frame = classify_news(news, narrative_config)
    frame["published_at"] = pd.to_datetime(frame["published_at"])
    as_of_ts = pd.to_datetime(as_of or frame["published_at"].max())
    source_quality = narrative_config.get("source_quality", {})
    features = _empty_narrative_scores(narrative_config)
    for narrative, group in frame.groupby("narrative_category"):
        if narrative not in narrative_config.get("narratives", {}):
            continue
        score = 0.0
        for _, row in group.iterrows():
            age_days = max(0.0, (as_of_ts - row["published_at"]).total_seconds() / 86400.0)
            recency = max(0.2, 1.0 - age_days / 14.0)
            source = float(source_quality.get(str(row.get("source", "")), 0.55))
            sentiment = float(row.get("sentiment", 0.0) or 0.0)
            relevance = float(row.get("relevance_score", 0.5) or 0.5)
            base = float(row.get("narrative_score", 35.0) or 35.0) / 100.0
            score += (0.35 + base + source * 0.15 + relevance * 0.25 + abs(sentiment) * 0.15) * recency
        normalized = min(100.0, score * 10.0)
        short_name = narrative.split("_", 1)[1] if "_" in narrative else narrative
        features[f"{short_name}_score"] = normalized / 100.0
        features[f"{narrative}_raw_score"] = normalized
    features["gold_narrative_score"] = features.get("gold_safe_haven_score", 0.0)
    features["geopolitical_risk"] = features.get("geopolitical_energy_shock_score", 0.0)
    features["ai_capex_pressure"] = features.get("ai_capex_risk_score", 0.0)
    features["ai_roi_quality"] = clamp(
        features.get("ai_revenue_realization_score", 0.0) - 0.5 * features.get("ai_capex_risk_score", 0.0),
        0.0,
        1.0,
    )
    features["central_bank_buying_proxy"] = features.get("gold_safe_haven_score", 0.0)
    return features


def _empty_narrative_scores(narrative_config: dict[str, Any]) -> dict[str, float]:
    features: dict[str, float] = {}
    for narrative in narrative_config.get("narratives", {}):
        short_name = narrative.split("_", 1)[1] if "_" in narrative else narrative
        features[f"{short_name}_score"] = 0.0
        features[f"{narrative}_raw_score"] = 0.0
    return features
