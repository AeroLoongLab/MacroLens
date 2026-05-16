# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

MacroLens is an offline-first macro complex system quant research platform for gold, U.S. equities, and U.S. Treasuries. It runs without API keys by default using deterministic fixtures; live data is opt-in via `--live`.

## Commands

```bash
# Setup
python3 -m venv .venv && source .venv/bin/activate && pip install -e ".[dev]"

# Daily update (offline)
python scripts/run_daily_update.py

# Daily update with live data
python scripts/run_daily_update.py --live

# Backtest
python scripts/run_backtest.py --start 2010-01-01 --end today

# Simulation
python scripts/run_simulation.py --horizon 6m --paths 10000

# Dashboard
python scripts/run_dashboard.py

# Run tests
pytest

# Run single test
pytest tests/test_config.py

# Lint
ruff check src/

# Format
black src/
```

## Architecture

The system follows a pipeline pattern: `pipeline.py` orchestrates data fetch → feature building → model inference → storage → reporting.

```
macro_quant/
├── pipeline.py              # Main entry: run_daily_update(), build_feature_snapshot()
├── data/                   # Data ingestion (fixtures + live fallback)
│   ├── fixtures.py         # Deterministic test data
│   ├── market_data.py     # Price data (yfinance when live)
│   ├── macro_data.py      # FRED indicators (live with API key)
│   ├── news_data.py       # RSS feed parsing
│   └── storage.py          # DuckDB persistence (idempotent upsert)
├── features/               # Feature engineering
│   ├── technical_features.py  # Market technical features
│   ├── macro_features.py      # Macro indicator features
│   ├── narrative_features.py  # News sentiment scoring
│   └── fragility_features.py  # Cross-asset fragility indices
├── models/                 # Inference models
│   ├── regime_model.py     # 6-regime classifier (R1-R6)
│   ├── asset_score_model.py # Asset scoring (Gold, Equity, Bond variants)
│   ├── portfolio_model.py  # Portfolio allocation recommendation
│   └── simulation_model.py # Monte Carlo simulation
├── backtesting/            # Backtest engine and metrics
├── reporting/              # Report generation (daily/weekly, EN/ZH)
└── utils/
    └── config.py           # ConfigBundle (settings + yaml configs)
```

### Key Data Flow

1. **Data ingestion**: `DuckDBStore` persists all data; tables include `market_prices`, `macro_indicators`, `news_articles`, `regime_probabilities`, `asset_scores`, `portfolio_recommendations`
2. **Feature building**: `build_feature_snapshot()` aggregates market + macro + narrative features, then computes fragility features from the aggregate
3. **Model inference**: `RegimeModel` → `AssetScoreModel` → `PortfolioModel` chain
4. **Reporting**: `DailyReportPayload` drives `generate_daily_report()` which outputs EN/ZH markdown + HTML

### DuckDB Storage

`DuckDBStore` uses `INSERT OR REPLACE` semantics via `_replace_rows()` — unique keys are `(date, ticker, source)` for market prices, `(date, indicator, source)` for macro indicators, etc. This makes runs idempotent.

### Configuration

YAML files under `configs/` define `model_weights.yaml` (regime and asset feature weights), `portfolio_rules.yaml` (allocation rules), `indicators.yaml` (FRED indicators), `tickers.yaml` (market tickers), and `narrative_topics.yaml` (news topics). `ConfigBundle` in `src/macro_quant/utils/config.py` loads all of them.

### Live Mode

When `--live` is passed, `fetch_market_prices()` attempts yfinance, `fetch_macro_indicators()` attempts FRED API. Failures degrade gracefully to fixtures/logged warnings. `FRED_API_KEY` is optional — without it, macro indicators use fixture data.

### Regime Model

Six regimes: R1_soft_landing_reinflation, R2_stagflation_pressure, R3_ai_productivity_bull, R4_fiscal_credit_shock, R5_recessionary_rate_cut, R6_liquidity_cascade. Prediction uses weighted feature sums with softmax temperature.

### Portfolio Model

Four portfolio types: `aggressive`, `balanced`, `conservative`, `crisis_defense`. Selection is rule-based on regime probabilities and feature thresholds (e.g., `crisis_defense` triggers when R4+R6 probability > 0.38 or `phase_transition_risk_index > 65`).

### Fragility Features

`build_fragility_features()` computes: `macro_stress_index`, `fiscal_credibility_risk_index`, `gold_crowding_index`, `long_bond_fragility_index`, `phase_transition_risk_index`, `ai_roi_pressure_index`, plus normalized versions (0-1 scale) used by asset scoring. Also computes `liquidity_score = clamp(1 - phase_transition_risk_index/100, 0, 1)`.

## Risk Disclaimer

本系统用于研究和辅助决策，不构成投资建议。模型输出依赖历史数据、公开数据和假设权重。宏观市场存在非线性、突发政策、地缘冲突和流动性风险。任何配置建议都需要结合个人风险承受能力。