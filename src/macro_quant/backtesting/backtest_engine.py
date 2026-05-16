from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from pathlib import Path

import pandas as pd

from macro_quant.backtesting.metrics import performance_metrics
from macro_quant.backtesting.strategies import benchmark_returns, weights_to_tickers
from macro_quant.data.fixtures import make_news_fixture
from macro_quant.features.fragility_features import build_fragility_features
from macro_quant.features.macro_features import build_macro_features
from macro_quant.features.narrative_features import build_narrative_features
from macro_quant.features.technical_features import build_market_features, build_price_matrix
from macro_quant.models.asset_score_model import AssetScoreModel
from macro_quant.models.portfolio_model import PortfolioModel
from macro_quant.models.regime_model import RegimeModel


@dataclass(frozen=True)
class BacktestResult:
    monthly_returns: pd.DataFrame
    metrics: pd.DataFrame
    turnover: float
    report_path: Path | None = None


class BacktestEngine:
    def __init__(self, config: object) -> None:
        self.config = config
        self.regime_model = RegimeModel(config.model_weights)
        self.asset_model = AssetScoreModel(config.model_weights)
        self.portfolio_model = PortfolioModel(config.portfolio_rules, config.model_version)

    def run(
        self,
        market_prices: pd.DataFrame,
        macro_indicators: pd.DataFrame,
        start: str | date = "2010-01-01",
        end: str | date | None = None,
    ) -> BacktestResult:
        prices = build_price_matrix(market_prices)
        if prices.empty:
            raise ValueError("Cannot backtest without market prices")
        prices = prices.loc[pd.to_datetime(start) :]
        if end and str(end) != "today":
            prices = prices.loc[: pd.to_datetime(end)]
        monthly_prices = prices.resample("ME").last().ffill()
        monthly_returns = monthly_prices.pct_change().dropna(how="all")
        strategy_returns: list[tuple[pd.Timestamp, float]] = []
        weights_history: list[dict[str, float]] = []
        rebalance_dates = list(monthly_returns.index)
        for idx, rebalance_date in enumerate(rebalance_dates[:-1]):
            if idx < 12:
                continue
            market_slice = market_prices[pd.to_datetime(market_prices["date"]) <= rebalance_date]
            macro_slice = macro_indicators[pd.to_datetime(macro_indicators["date"]) <= rebalance_date]
            narrative = make_news_fixture(self.config.narrative_topics, as_of=rebalance_date.date(), days=3)
            features = {}
            features.update(build_market_features(market_slice))
            features.update(build_macro_features(macro_slice))
            features.update(build_narrative_features(narrative, self.config.narrative_topics))
            features.update(build_fragility_features(features))
            regimes = self.regime_model.predict(features)
            asset_scores = self.asset_model.score(features)
            recommendations = self.portfolio_model.recommend(regimes, asset_scores, features)
            selected = self.portfolio_model.selected(recommendations)
            asset_weights = {rec.asset: rec.weight for rec in selected}
            ticker_weights = weights_to_tickers(asset_weights)
            next_date = rebalance_dates[idx + 1]
            next_returns = monthly_returns.loc[next_date]
            strategy_return = sum(
                weight * float(next_returns.get(ticker, 0.0)) for ticker, weight in ticker_weights.items()
            )
            strategy_returns.append((next_date, strategy_return))
            weights_history.append(ticker_weights)
        strategy = pd.Series(
            data=[value for _, value in strategy_returns],
            index=[dt for dt, _ in strategy_returns],
            name="Model Portfolio",
        )
        benchmarks = benchmark_returns(monthly_returns).reindex(strategy.index).fillna(0.0)
        returns_frame = pd.concat([strategy, benchmarks], axis=1).dropna()
        metrics = pd.DataFrame(
            {column: performance_metrics(returns_frame[column]) for column in returns_frame.columns}
        ).T
        metrics["Turnover"] = _average_turnover(weights_history)
        if "Model Portfolio" in metrics.index and "SPY only" in metrics.index:
            metrics.loc["Model Portfolio", "Excess vs SPY"] = (
                metrics.loc["Model Portfolio", "CAGR"] - metrics.loc["SPY only", "CAGR"]
            )
        if "Model Portfolio" in metrics.index and "60/40" in metrics.index:
            metrics.loc["Model Portfolio", "Excess vs 60/40"] = (
                metrics.loc["Model Portfolio", "CAGR"] - metrics.loc["60/40", "CAGR"]
            )
        return BacktestResult(
            monthly_returns=returns_frame,
            metrics=metrics,
            turnover=(float(metrics.loc["Model Portfolio", "Turnover"]) if "Model Portfolio" in metrics.index else 0.0),
        )


def _average_turnover(weights_history: list[dict[str, float]]) -> float:
    if len(weights_history) <= 1:
        return 0.0
    turnovers = []
    for previous, current in zip(weights_history[:-1], weights_history[1:], strict=True):
        tickers = set(previous) | set(current)
        turnovers.append(sum(abs(current.get(ticker, 0.0) - previous.get(ticker, 0.0)) for ticker in tickers) / 2.0)
    return float(sum(turnovers) / len(turnovers))
