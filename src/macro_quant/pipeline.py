from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from pathlib import Path

import pandas as pd

from macro_quant.data.macro_data import fetch_macro_indicators
from macro_quant.data.market_data import fetch_market_prices
from macro_quant.data.news_data import fetch_news_articles
from macro_quant.data.storage import DuckDBStore
from macro_quant.features.fragility_features import build_fragility_features
from macro_quant.features.macro_features import build_macro_features
from macro_quant.features.narrative_features import build_narrative_features
from macro_quant.features.technical_features import build_market_features
from macro_quant.models.asset_score_model import AssetScore, AssetScoreModel
from macro_quant.models.portfolio_model import PortfolioModel, PortfolioRecommendation
from macro_quant.models.regime_model import RegimeModel, RegimeResult
from macro_quant.reporting.daily_report import DailyReportPayload, generate_daily_report
from macro_quant.reporting.weekly_report import generate_weekly_placeholder
from macro_quant.utils.config import ConfigBundle, ensure_runtime_dirs, load_config, load_settings
from macro_quant.utils.logging import get_logger

logger = get_logger(__name__)


@dataclass(frozen=True)
class DailyRunResult:
    as_of: date
    markdown_report: Path
    html_report: Path
    markdown_report_zh: Path
    html_report_zh: Path
    regime_results: list[RegimeResult]
    asset_scores: list[AssetScore]
    portfolio_recommendations: list[PortfolioRecommendation]
    features: dict[str, float]


def run_daily_update(
    *,
    live: bool = False,
    as_of: date | None = None,
    db_path: Path | None = None,
    project_root: Path | None = None,
    output_root: Path | None = None,
    start: str | date = "2010-01-01",
) -> DailyRunResult:
    as_of = as_of or date.today()
    settings = load_settings(project_root=project_root, db_path=db_path)
    if output_root is not None:
        settings.reports_dir = Path(output_root) / "reports"  # type: ignore[misc]
    ensure_runtime_dirs(settings)
    config = load_config(settings.config_dir)
    store = DuckDBStore(settings.db_path)
    store.init_schema()
    previous_summary, previous_summary_zh = _previous_summaries(store, as_of)

    logger.info("Fetching market, macro, and news data; live=%s", live)
    market = fetch_market_prices(config.tickers, start=start, end=as_of, live=live)
    macro = fetch_macro_indicators(config.indicators, start=start, end=as_of, live=live)
    news = fetch_news_articles(config.narrative_topics, as_of=as_of, live=live)

    store.upsert_market_prices(market)
    store.upsert_macro_indicators(macro)
    store.upsert_news_articles(news)

    features = build_feature_snapshot(market, macro, news, config)
    regime_model = RegimeModel(config.model_weights)
    asset_model = AssetScoreModel(config.model_weights)
    portfolio_model = PortfolioModel(config.portfolio_rules, config.model_version)
    regimes = regime_model.predict(features)
    scores = asset_model.score(features)
    recommendations = portfolio_model.recommend(regimes, scores, features)

    store.upsert_regime_probabilities(regime_model.to_frame(regimes, as_of))
    store.upsert_asset_scores(asset_model.to_frame(scores, as_of))
    store.upsert_portfolio_recommendations(portfolio_model.to_frame(recommendations, as_of))
    generate_weekly_placeholder(as_of, settings.reports_dir)

    narrative_summary = {
        key: value
        for key, value in features.items()
        if key.endswith("_score")
        and any(
            prefix in key
            for prefix in [
                "inflation",
                "fiscal",
                "gold",
                "ai",
                "recession",
                "liquidity",
                "geopolitical",
            ]
        )
    }
    payload = DailyReportPayload(
        as_of=as_of,
        model_version=config.model_version,
        data_timestamp=str(max(pd.to_datetime(market["date"]).max(), pd.to_datetime(macro["date"]).max())),
        regime_results=regimes,
        asset_scores=scores,
        portfolio_recommendations=recommendations,
        features=features,
        narrative_summary=narrative_summary,
        previous_summary=previous_summary,
        previous_summary_zh=previous_summary_zh,
    )
    report_files = generate_daily_report(payload, settings.reports_dir)
    logger.info(
        "Generated reports: %s, %s, %s and %s",
        report_files.markdown,
        report_files.html,
        report_files.markdown_zh,
        report_files.html_zh,
    )
    return DailyRunResult(
        as_of=as_of,
        markdown_report=report_files.markdown,
        html_report=report_files.html,
        markdown_report_zh=report_files.markdown_zh,
        html_report_zh=report_files.html_zh,
        regime_results=regimes,
        asset_scores=scores,
        portfolio_recommendations=recommendations,
        features=features,
    )


def build_feature_snapshot(
    market: pd.DataFrame,
    macro: pd.DataFrame,
    news: pd.DataFrame,
    config: ConfigBundle,
) -> dict[str, float]:
    features: dict[str, float] = {}
    features.update(build_market_features(market))
    features.update(build_macro_features(macro))
    features.update(build_narrative_features(news, config.narrative_topics))
    features.update(build_fragility_features(features))
    features["volatility_score"] = max(0.0, features.get("vix_level", 0.0))
    features["risk_appetite_score"] = max(
        -1.0,
        min(1.0, features.get("sp500_momentum", 0.0) - 0.5 * features.get("vix_level", 0.0)),
    )
    features["ai_roi_quality"] = max(
        0.0,
        min(
            1.0,
            features.get("ai_revenue_realization_score", 0.0)
            - 0.4 * features.get("ai_capex_risk_score", 0.0)
            + 0.3 * max(0.0, features.get("qqq_momentum", 0.0)),
        ),
    )
    features["ai_capex_pressure"] = max(
        features.get("ai_capex_risk_score", 0.0),
        features.get("ai_roi_pressure_index", 0.0) / 100.0,
    )
    return features


def _previous_summaries(store: DuckDBStore, as_of: date) -> tuple[str, str]:
    try:
        scores = store.read_table("asset_scores")
        regimes = store.read_table("regime_probabilities")
    except Exception:
        return "No previous report in database.", "数据库中没有上一份报告。"
    if scores.empty or regimes.empty:
        return "No previous report in database.", "数据库中没有上一份报告。"
    scores["date"] = pd.to_datetime(scores["date"]).dt.date
    regimes["date"] = pd.to_datetime(regimes["date"]).dt.date
    previous_dates = sorted(set(scores.loc[scores["date"] < as_of, "date"]))
    if not previous_dates:
        return "No previous report in database.", "数据库中没有上一份报告。"
    previous = previous_dates[-1]
    score_rows = scores[scores["date"] == previous]
    regime_rows = regimes[regimes["date"] == previous]
    top_regime = regime_rows.sort_values("probability", ascending=False).head(1)
    top_text = (
        f"Previous top regime was `{top_regime.iloc[0]['regime']}` at {top_regime.iloc[0]['probability']:.1%}."
        if not top_regime.empty
        else "Previous top regime is unknown."
    )
    top_text_zh = (
        f"上一轮最高概率状态为 `{top_regime.iloc[0]['regime']}`，概率 {top_regime.iloc[0]['probability']:.1%}。"
        if not top_regime.empty
        else "上一轮最高概率状态未知。"
    )
    score_text = ", ".join(f"{row.asset}: {row.final_score:.1f}" for row in score_rows.itertuples(index=False))
    score_text_zh = "，".join(f"{row.asset}: {row.final_score:.1f}" for row in score_rows.itertuples(index=False))
    return f"{top_text} Previous asset scores: {score_text}.", f"{top_text_zh} 上一轮资产评分：{score_text_zh}。"
