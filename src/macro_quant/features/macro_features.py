from __future__ import annotations

import pandas as pd

from macro_quant.utils.math_utils import clamp, normalize


def build_macro_features(macro_indicators: pd.DataFrame) -> dict[str, float]:
    if macro_indicators.empty:
        return {}
    frame = macro_indicators.copy()
    frame["date"] = pd.to_datetime(frame["date"])
    pivot = (
        frame.sort_values("date")
        .pivot_table(index="date", columns="indicator", values="value", aggfunc="last")
        .sort_index()
        .ffill()
    )
    features: dict[str, float] = {}
    features["cpi_yoy_raw"] = _yoy(pivot, "CPIAUCSL")
    features["core_cpi_yoy_raw"] = _yoy(pivot, "CPILFESL")
    features["cpi_yoy"] = normalize(features["cpi_yoy_raw"], 2.0, 4.0)
    features["core_cpi_yoy"] = normalize(features["core_cpi_yoy_raw"], 2.0, 4.0)
    features["inflation_risk"] = max(0.0, (features["cpi_yoy"] + features["core_cpi_yoy"]) / 2.0)
    features["pmi_raw"] = _pmi_raw(pivot)
    features["pmi"] = normalize(features["pmi_raw"], 50.0, 10.0)
    unrate = pivot.get("UNRATE")
    if unrate is not None and len(unrate.dropna()) > 252:
        features["unemployment_rate_change"] = clamp((unrate.iloc[-1] - unrate.iloc[-253]) / 2.0)
    else:
        features["unemployment_rate_change"] = 0.0
    dgs10 = _latest(pivot, "DGS10", 4.0)
    dgs2 = _latest(pivot, "DGS2", 4.0)
    dgs30 = _latest(pivot, "DGS30", dgs10)
    fed = _latest(pivot, "FEDFUNDS", dgs2)
    real_yield = _latest(pivot, "DFII10", 1.0)
    breakeven = _latest(pivot, "T10YIE", 2.0)
    features["yield_10y"] = dgs10
    features["yield_2y"] = dgs2
    features["yield_30y"] = dgs30
    features["yield_curve_2y10y"] = dgs10 - dgs2
    features["term_premium"] = normalize(dgs30 - dgs10, 0.20, 1.0)
    features["fed_cut_probability"] = clamp((fed - dgs2) / 2.0)
    features["rate_cut_expectation"] = max(0.0, features["fed_cut_probability"])
    features["real_yield_score"] = normalize(real_yield, 1.0, 2.0)
    features["breakeven_inflation"] = normalize(breakeven, 2.2, 1.2)
    baa = _latest(pivot, "BAA10Y", 2.5)
    hy = _latest(pivot, "BAMLH0A0HYM2", baa + 2.0)
    features["credit_spread"] = max(normalize(baa, 2.2, 2.5), normalize(hy, 4.0, 5.0))
    deficit = abs(_latest(pivot, "FYFSGDA188S", -5.0))
    debt = _latest(pivot, "GFDEBTN", 30_000_000.0)
    interest = _latest(pivot, "A091RC1Q027SBEA", 800_000.0)
    features["treasury_supply"] = normalize(deficit, 5.0, 6.0)
    features["fiscal_deficit_pressure"] = features["treasury_supply"]
    features["fiscal_interest_pressure"] = normalize(interest, 650_000.0, 650_000.0)
    features["federal_debt_pressure"] = normalize(debt, 28_000_000.0, 12_000_000.0)
    features["yield_level"] = max(0.0, normalize(fed, 1.5, 4.0))
    features["cash_optional_value"] = max(0.0, normalize(fed, 2.0, 4.0))
    features["reinvestment_risk"] = max(0.0, features["rate_cut_expectation"])
    features["recession_risk"] = max(
        0.0,
        clamp(0.4 * features["unemployment_rate_change"] - 0.3 * features["pmi"] + 0.3 * features["credit_spread"]),
    )
    features["stability_score"] = max(0.0, 1.0 - abs(features["credit_spread"]))
    return features


def _latest(pivot: pd.DataFrame, column: str, default: float) -> float:
    if column not in pivot or pivot[column].dropna().empty:
        return default
    return float(pivot[column].dropna().iloc[-1])


def _pmi_raw(pivot: pd.DataFrame) -> float:
    if "NAPM" in pivot and not pivot["NAPM"].dropna().empty:
        return _latest(pivot, "NAPM", 50.0)
    for column in ["IPMANSICS", "INDPRO"]:
        proxy = _pmi_proxy_from_output(pivot, column)
        if proxy is not None:
            return proxy
    return 50.0


def _pmi_proxy_from_output(pivot: pd.DataFrame, column: str) -> float | None:
    if column not in pivot:
        return None
    series = pivot[column].dropna()
    if len(series) <= 252:
        return None
    latest = float(series.iloc[-1])
    three_month_previous = float(series.iloc[-64]) if len(series) > 63 else latest
    year_previous = float(series.iloc[-253])
    if latest == 0 or three_month_previous == 0 or year_previous == 0:
        return None
    three_month_change = latest / three_month_previous - 1.0
    yoy_change = latest / year_previous - 1.0
    return clamp(50.0 + 250.0 * three_month_change + 100.0 * yoy_change, 35.0, 65.0)


def _yoy(pivot: pd.DataFrame, column: str) -> float:
    if column not in pivot:
        return 0.0
    series = pivot[column].dropna()
    if len(series) <= 252:
        return 0.0
    previous = float(series.iloc[-253])
    if previous == 0:
        return 0.0
    return float(series.iloc[-1] / previous - 1.0) * 100.0
