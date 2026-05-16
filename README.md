# MacroLens / 宏观透镜

Offline-first macro complex system quant research platform for gold, U.S. equities, and U.S. Treasuries.

The MVP is designed to run without API keys using deterministic fixtures. Live data is opt-in through `--live`.

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
cp .env.example .env
```

Optional live FRED ingestion requires `FRED_API_KEY` in `.env`.

## Commands

```bash
python scripts/run_daily_update.py
python scripts/run_daily_update.py --live
python scripts/run_backtest.py --start 2010-01-01 --end today
python scripts/run_simulation.py --horizon 6m --paths 10000
python scripts/run_dashboard.py
```

## Default Behavior

- Offline fixtures cover market prices, macro indicators, and news narratives.
- Live failures degrade to fixture/unknown values and are logged.
- Live market data keeps stable internal ticker names while using provider-specific Yahoo symbols where needed
  (for example `VIX -> ^VIX`, `DXY -> DX-Y.NYB`, `MOVE -> ^MOVE`).
- FRED no longer provides the legacy `NAPM` ISM PMI series used by the model fixtures. In live mode, `pmi_raw`
  is a PMI-like proxy derived from official FRED manufacturing output data (`IPMANSICS`, with `INDPRO` fallback),
  not the raw ISM Manufacturing PMI release.
- DuckDB storage is idempotent for repeated runs.
- Daily reports are generated in both English and Chinese:
  - `reports/daily/YYYY-MM-DD_macrolens_report.md`
  - `reports/daily/YYYY-MM-DD_macrolens_report.html`
  - `reports/daily/YYYY-MM-DD_macrolens_report_zh.md`
  - `reports/daily/YYYY-MM-DD_macrolens_report_zh.html`
- Reports include model version, data timestamp, update timestamp, and investment risk disclaimer.

## Risk Disclaimer

本系统用于研究和辅助决策，不构成投资建议。模型输出依赖历史数据、公开数据和假设权重。宏观市场存在非线性、突发政策、地缘冲突和流动性风险。任何配置建议都需要结合个人风险承受能力。
