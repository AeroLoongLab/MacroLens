from datetime import date
from pathlib import Path

from macro_quant.data.fixtures import make_macro_fixture, make_market_fixture, make_news_fixture
from macro_quant.features.fragility_features import build_fragility_features
from macro_quant.features.macro_features import build_macro_features
from macro_quant.features.narrative_features import build_narrative_features
from macro_quant.features.technical_features import build_market_features
from macro_quant.models.asset_score_model import AssetScoreModel
from macro_quant.models.portfolio_model import PortfolioModel
from macro_quant.models.regime_model import RegimeModel
from macro_quant.utils.config import load_config

ROOT = Path(__file__).resolve().parents[1]


def test_regime_probabilities_and_asset_scores_are_bounded() -> None:
    config = load_config(ROOT / "configs")
    tickers = ["GLD", "SPY", "QQQ", "SHY", "IEF", "TLT", "UUP", "CL=F", "VIX", "MOVE"]
    indicators = list(config.indicators["fred"].keys())
    market = make_market_fixture(tickers, start=date(2024, 1, 1), end=date(2026, 5, 15))
    macro = make_macro_fixture(indicators, start=date(2024, 1, 1), end=date(2026, 5, 15))
    news = make_news_fixture(config.narrative_topics, as_of=date(2026, 5, 15))
    features = {}
    features.update(build_market_features(market))
    features.update(build_macro_features(macro))
    features.update(build_narrative_features(news, config.narrative_topics))
    features.update(build_fragility_features(features))
    regimes = RegimeModel(config.model_weights).predict(features)
    assert abs(sum(result.probability for result in regimes) - 1.0) < 1e-9
    scores = AssetScoreModel(config.model_weights).score(features)
    assert {score.asset for score in scores} == {
        "Gold",
        "Equity",
        "Short_Bond",
        "Intermediate_Bond",
        "Long_Bond",
    }
    assert all(-100 <= score.final_score <= 100 for score in scores)


def test_selected_portfolio_weights_sum_to_one_and_respect_ranges() -> None:
    config = load_config(ROOT / "configs")
    features = {
        "macro_stress_index": 40.0,
        "phase_transition_risk_index": 30.0,
        "liquidity_score": 0.6,
    }
    regimes = RegimeModel(config.model_weights).predict(features)
    scores = AssetScoreModel(config.model_weights).score(features)
    model = PortfolioModel(config.portfolio_rules, config.model_version)
    recommendations = model.recommend(regimes, scores, features)
    selected = model.selected(recommendations)
    assert round(sum(rec.weight for rec in selected), 10) == 1.0
    selected_type = selected[0].portfolio_type
    ranges = config.portfolio_rules["portfolio_types"][selected_type]["ranges"]
    for rec in selected:
        low, high = ranges[rec.asset]
        assert low <= rec.weight <= high
