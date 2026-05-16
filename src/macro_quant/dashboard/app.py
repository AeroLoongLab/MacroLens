from __future__ import annotations

# ruff: noqa: E402,I001

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "src"))

import pandas as pd
import plotly.express as px
import streamlit as st

from macro_quant.data.storage import DuckDBStore
from macro_quant.pipeline import run_daily_update
from macro_quant.utils.config import ensure_runtime_dirs, load_settings

st.set_page_config(page_title="MacroLens", page_icon="ML", layout="wide")


@st.cache_data(ttl=30)
def load_tables(db_path: str) -> dict[str, pd.DataFrame]:
    store = DuckDBStore(Path(db_path))
    tables = {}
    for table in [
        "market_prices",
        "macro_indicators",
        "news_articles",
        "regime_probabilities",
        "asset_scores",
        "portfolio_recommendations",
    ]:
        try:
            tables[table] = store.read_table(table)
        except Exception:
            tables[table] = pd.DataFrame()
    return tables


def _latest_by_date(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty or "date" not in frame:
        return frame
    clean = frame.copy()
    clean["date"] = pd.to_datetime(clean["date"])
    latest = clean["date"].max()
    return clean[clean["date"] == latest]


settings = load_settings(project_root=ROOT)
ensure_runtime_dirs(settings)
st.title("MacroLens / 宏观透镜")
st.caption("Offline-first macro regime, asset scoring, narrative, backtest, and simulation monitor.")

with st.sidebar:
    st.header("Controls")
    if st.button("Run offline daily update", use_container_width=True):
        run_daily_update(live=False, project_root=ROOT)
        st.cache_data.clear()
        st.success("Offline update completed.")
    st.write(f"Database: `{settings.db_path}`")

tables = load_tables(str(settings.db_path))
if all(frame.empty for frame in tables.values()):
    st.warning("No data found. Run `python scripts/run_daily_update.py` or click the sidebar update button.")
    st.stop()

tabs = st.tabs(
    [
        "Overview",
        "Regime",
        "Asset Scores",
        "Narrative Monitor",
        "Macro Indicators",
        "Backtest",
        "Simulation",
        "Portfolio",
    ]
)

with tabs[0]:
    regime = _latest_by_date(tables["regime_probabilities"])
    scores = _latest_by_date(tables["asset_scores"])
    c1, c2, c3 = st.columns(3)
    with c1:
        if not regime.empty:
            top = regime.sort_values("probability", ascending=False).iloc[0]
            st.metric("Top Regime", top["regime"], f"{top['probability']:.1%}")
    with c2:
        if not scores.empty:
            equity = scores.loc[scores["asset"] == "Equity", "final_score"]
            st.metric("Equity Score", f"{float(equity.iloc[0]) if not equity.empty else 0:.1f}")
    with c3:
        if not scores.empty:
            gold = scores.loc[scores["asset"] == "Gold", "final_score"]
            st.metric("Gold Score", f"{float(gold.iloc[0]) if not gold.empty else 0:.1f}")
    if not regime.empty:
        st.plotly_chart(
            px.bar(regime, x="regime", y="probability", title="Latest Regime Probability"),
            use_container_width=True,
        )

with tabs[1]:
    regime = tables["regime_probabilities"]
    if regime.empty:
        st.info("No regime probabilities yet.")
    else:
        regime["date"] = pd.to_datetime(regime["date"])
        st.plotly_chart(
            px.area(regime, x="date", y="probability", color="regime", title="Regime Probabilities"),
            use_container_width=True,
        )
        st.dataframe(
            regime.sort_values(["date", "probability"], ascending=[False, False]),
            use_container_width=True,
        )

with tabs[2]:
    scores = tables["asset_scores"]
    if scores.empty:
        st.info("No asset scores yet.")
    else:
        scores["date"] = pd.to_datetime(scores["date"])
        latest_scores = _latest_by_date(scores)
        st.plotly_chart(
            px.bar(latest_scores, x="asset", y="final_score", title="Latest Asset Scores"),
            use_container_width=True,
        )
        st.plotly_chart(
            px.line(scores, x="date", y="final_score", color="asset", title="Asset Score History"),
            use_container_width=True,
        )
        st.dataframe(latest_scores, use_container_width=True)

with tabs[3]:
    news = tables["news_articles"]
    if news.empty:
        st.info("No narrative articles yet.")
    else:
        news["published_at"] = pd.to_datetime(news["published_at"])
        narrative = (
            news.groupby("narrative_category", as_index=False)
            .agg(article_count=("id", "count"), narrative_score=("narrative_score", "mean"))
            .sort_values("narrative_score", ascending=False)
        )
        st.plotly_chart(
            px.bar(narrative, x="narrative_category", y="narrative_score", title="Narrative Strength"),
            use_container_width=True,
        )
        st.dataframe(news.sort_values("published_at", ascending=False), use_container_width=True)

with tabs[4]:
    macro = tables["macro_indicators"]
    if macro.empty:
        st.info("No macro indicators yet.")
    else:
        macro["date"] = pd.to_datetime(macro["date"])
        indicators = sorted(macro["indicator"].unique())
        selected = st.multiselect("Indicators", indicators, default=indicators[:5])
        subset = macro[macro["indicator"].isin(selected)]
        st.plotly_chart(
            px.line(subset, x="date", y="value", color="indicator", title="Macro Indicators"),
            use_container_width=True,
        )

with tabs[5]:
    st.info("Run `python scripts/run_backtest.py --start 2010-01-01 --end today` to refresh backtest reports.")
    report_dir = settings.reports_dir / "backtests"
    reports = sorted(report_dir.glob("*_backtest.md"), reverse=True)
    if reports:
        st.markdown(reports[0].read_text(encoding="utf-8"))

with tabs[6]:
    st.info("Run `python scripts/run_simulation.py --horizon 6m --paths 10000` to refresh simulation reports.")
    report_dir = settings.reports_dir / "simulations"
    reports = sorted(report_dir.glob("*_simulation.md"), reverse=True)
    if reports:
        st.markdown(reports[0].read_text(encoding="utf-8"))

with tabs[7]:
    portfolio = tables["portfolio_recommendations"]
    if portfolio.empty:
        st.info("No portfolio recommendations yet.")
    else:
        portfolio["date"] = pd.to_datetime(portfolio["date"])
        latest = _latest_by_date(portfolio)
        selected = latest[latest["reason"].str.contains("selected", case=False, na=False)]
        display = selected if not selected.empty else latest
        st.plotly_chart(
            px.pie(display, names="asset", values="weight", title="Selected Allocation"),
            use_container_width=True,
        )
        st.dataframe(latest.sort_values(["portfolio_type", "asset"]), use_container_width=True)
