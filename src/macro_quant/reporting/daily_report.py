from __future__ import annotations

import html
from dataclasses import dataclass
from datetime import UTC, date, datetime
from pathlib import Path

from macro_quant.models.asset_score_model import AssetScore
from macro_quant.models.portfolio_model import PortfolioRecommendation
from macro_quant.models.regime_model import RegimeResult

RISK_DISCLAIMER = (
    "本系统用于研究和辅助决策，不构成投资建议。模型输出依赖历史数据、公开数据和假设权重。"
    "宏观市场存在非线性、突发政策、地缘冲突和流动性风险。任何配置建议都需要结合个人风险承受能力。"
)

REGIME_LABELS_ZH = {
    "R1_soft_landing_reinflation": "R1 软着陆再通胀",
    "R2_stagflation_pressure": "R2 滞胀压力",
    "R3_ai_productivity_bull": "R3 AI 生产率牛市",
    "R4_fiscal_credit_shock": "R4 财政信用冲击",
    "R5_recessionary_rate_cut": "R5 衰退式降息",
    "R6_liquidity_cascade": "R6 流动性踩踏",
}

ASSET_LABELS_ZH = {
    "Gold": "黄金",
    "Equity": "美股",
    "Short_Bond": "短债",
    "Intermediate_Bond": "中债",
    "Long_Bond": "长债",
}

NARRATIVE_LABELS_ZH = {
    "inflation_resurgence_score": "通胀再起叙事",
    "fiscal_credit_risk_score": "财政信用风险叙事",
    "gold_safe_haven_score": "黄金避险/去美元化叙事",
    "ai_capex_risk_score": "AI 资本开支风险叙事",
    "ai_revenue_realization_score": "AI 收入兑现叙事",
    "recession_risk_score": "衰退风险叙事",
    "liquidity_pressure_score": "流动性压力叙事",
    "geopolitical_energy_shock_score": "地缘能源冲击叙事",
    "gold_narrative_score": "黄金综合叙事",
    "ai_capex_pressure": "AI Capex 压力",
}


def _utc_now() -> datetime:
    return datetime.now(UTC).replace(tzinfo=None)


@dataclass(frozen=True)
class DailyReportPayload:
    as_of: date
    model_version: str
    data_timestamp: str
    regime_results: list[RegimeResult]
    asset_scores: list[AssetScore]
    portfolio_recommendations: list[PortfolioRecommendation]
    features: dict[str, float]
    narrative_summary: dict[str, float]
    previous_summary: str = "No previous report in database."
    previous_summary_zh: str = "数据库中没有上一份报告。"


@dataclass(frozen=True)
class DailyReportFiles:
    markdown: Path
    html: Path
    markdown_zh: Path
    html_zh: Path


def generate_daily_report(payload: DailyReportPayload, reports_dir: Path) -> DailyReportFiles:
    daily_dir = reports_dir / "daily"
    daily_dir.mkdir(parents=True, exist_ok=True)
    md_path = daily_dir / f"{payload.as_of.isoformat()}_macrolens_report.md"
    html_path = daily_dir / f"{payload.as_of.isoformat()}_macrolens_report.html"
    md_zh_path = daily_dir / f"{payload.as_of.isoformat()}_macrolens_report_zh.md"
    html_zh_path = daily_dir / f"{payload.as_of.isoformat()}_macrolens_report_zh.html"
    markdown = render_daily_markdown(payload)
    markdown_zh = render_daily_markdown_zh(payload)
    md_path.write_text(markdown, encoding="utf-8")
    html_path.write_text(
        _markdown_to_html(markdown, title="MacroLens Daily Report", lang="en"),
        encoding="utf-8",
    )
    md_zh_path.write_text(markdown_zh, encoding="utf-8")
    html_zh_path.write_text(
        _markdown_to_html(markdown_zh, title="宏观复杂系统量化日报", lang="zh-CN"),
        encoding="utf-8",
    )
    return DailyReportFiles(markdown=md_path, html=html_path, markdown_zh=md_zh_path, html_zh=html_zh_path)


def render_daily_markdown(payload: DailyReportPayload) -> str:
    selected = [rec for rec in payload.portfolio_recommendations if rec.selected]
    top_regime = max(payload.regime_results, key=lambda item: item.probability)
    lines = [
        f"# MacroLens Daily Report - {payload.as_of.isoformat()}",
        "",
        "## Today Conclusion",
        f"- Top regime: `{top_regime.regime}` at {top_regime.probability:.1%}.",
        f"- Macro Stress Index: {payload.features.get('macro_stress_index', 0.0):.1f}/100.",
        f"- Phase Transition Risk: {payload.features.get('phase_transition_risk_index', 0.0):.1f}/100.",
        "",
        "## Regime Probabilities",
        "| Regime | Probability | Raw Score | Top Contribution |",
        "|---|---:|---:|---|",
    ]
    for result in sorted(payload.regime_results, key=lambda item: item.probability, reverse=True):
        top_factor = _top_factor(result.contributions)
        lines.append(f"| {result.regime} | {result.probability:.1%} | {result.raw_score:.3f} | {top_factor} |")
    lines.extend(
        [
            "",
            "## Asset Scores",
            "| Asset | Final | Direction | Risk | Valuation | Macro Fit | Interpretation |",
            "|---|---:|---:|---:|---:|---:|---|",
        ]
    )
    for score in sorted(payload.asset_scores, key=lambda item: item.asset):
        lines.append(
            f"| {score.asset} | {score.final_score:.1f} | {score.direction_score:.1f} | "
            f"{score.risk_score:.1f} | {score.valuation_score:.1f} | "
            f"{score.macro_fit_score:.1f} | {_interpret_score(score.final_score)} |"
        )
    lines.extend(
        [
            "",
            "## Narrative Summary",
            "| Narrative Feature | Score |",
            "|---|---:|",
        ]
    )
    for name, value in sorted(payload.narrative_summary.items()):
        if name.endswith("_raw_score"):
            continue
        lines.append(f"| {name} | {value * 100.0:.1f} |")
    lines.extend(
        [
            "",
            "## Selected Portfolio",
            "| Asset | Weight | Reason |",
            "|---|---:|---|",
        ]
    )
    for rec in selected:
        lines.append(f"| {rec.asset} | {rec.weight:.1%} | {rec.reason} |")
    lines.extend(
        [
            "",
            "## Risk Triggers",
            *_risk_triggers(payload.features, payload.asset_scores),
            "",
            "## Change From Previous Run",
            payload.previous_summary,
            "",
            "## Metadata",
            f"- model_version: `{payload.model_version}`",
            f"- data_timestamp: `{payload.data_timestamp}`",
            f"- last_update_time: `{_utc_now().isoformat(timespec='seconds')}Z`",
            "",
            "## Risk Disclaimer",
            RISK_DISCLAIMER,
            "",
        ]
    )
    return "\n".join(lines)


def render_daily_markdown_zh(payload: DailyReportPayload) -> str:
    selected = [rec for rec in payload.portfolio_recommendations if rec.selected]
    top_regime = max(payload.regime_results, key=lambda item: item.probability)
    lines = [
        f"# 宏观透镜日报 - {payload.as_of.isoformat()}",
        "",
        "## 今日结论",
        f"- 当前最高概率状态：`{_regime_label_zh(top_regime.regime)}`，概率 {top_regime.probability:.1%}。",
        f"- 宏观压力指数：{payload.features.get('macro_stress_index', 0.0):.1f}/100。",
        f"- 市场相变风险指数：{payload.features.get('phase_transition_risk_index', 0.0):.1f}/100。",
        "",
        "## Regime 概率",
        "| 宏观状态 | 概率 | 原始分 | 最大贡献因子 |",
        "|---|---:|---:|---|",
    ]
    for result in sorted(payload.regime_results, key=lambda item: item.probability, reverse=True):
        top_factor = _top_factor(result.contributions)
        lines.append(
            f"| {_regime_label_zh(result.regime)} | {result.probability:.1%} | "
            f"{result.raw_score:.3f} | {top_factor} |"
        )
    lines.extend(
        [
            "",
            "## 资产评分",
            "| 资产 | 最终分 | 方向分 | 风险分 | 估值/拥挤度 | 宏观匹配 | 解释 |",
            "|---|---:|---:|---:|---:|---:|---|",
        ]
    )
    for score in sorted(payload.asset_scores, key=lambda item: item.asset):
        lines.append(
            f"| {_asset_label_zh(score.asset)} | {score.final_score:.1f} | {score.direction_score:.1f} | "
            f"{score.risk_score:.1f} | {score.valuation_score:.1f} | "
            f"{score.macro_fit_score:.1f} | {_interpret_score_zh(score.final_score)} |"
        )
    lines.extend(
        [
            "",
            "## 新闻叙事摘要",
            "| 叙事指标 | 分数 |",
            "|---|---:|",
        ]
    )
    for name, value in sorted(payload.narrative_summary.items()):
        if name.endswith("_raw_score"):
            continue
        lines.append(f"| {_narrative_label_zh(name)} | {value * 100.0:.1f} |")
    lines.extend(
        [
            "",
            "## 今日建议组合",
            "| 资产 | 权重 | 理由 |",
            "|---|---:|---|",
        ]
    )
    for rec in selected:
        lines.append(f"| {_asset_label_zh(rec.asset)} | {rec.weight:.1%} | {_portfolio_reason_zh(rec)} |")
    lines.extend(
        [
            "",
            "## 风险触发器",
            *_risk_triggers_zh(payload.features, payload.asset_scores),
            "",
            "## 与上一轮相比",
            payload.previous_summary_zh,
            "",
            "## 元数据",
            f"- model_version: `{payload.model_version}`",
            f"- data_timestamp: `{payload.data_timestamp}`",
            f"- last_update_time: `{_utc_now().isoformat(timespec='seconds')}Z`",
            "",
            "## 风险提示",
            RISK_DISCLAIMER,
            "",
        ]
    )
    return "\n".join(lines)


def _top_factor(contributions: dict[str, float]) -> str:
    if not contributions:
        return "unknown"
    key, value = max(contributions.items(), key=lambda item: abs(item[1]))
    return f"{key} ({value:+.3f})"


def _interpret_score(score: float) -> str:
    if score >= 60:
        return "strong allocation"
    if score >= 20:
        return "moderate allocation"
    if score > -20:
        return "neutral"
    if score > -60:
        return "reduce allocation"
    return "avoid"


def _interpret_score_zh(score: float) -> str:
    if score >= 60:
        return "强配置"
    if score >= 20:
        return "适度配置"
    if score > -20:
        return "中性"
    if score > -60:
        return "降低配置"
    return "明显规避"


def _risk_triggers(features: dict[str, float], scores: list[AssetScore]) -> list[str]:
    score_map = {score.asset: score.final_score for score in scores}
    triggers = []
    if features.get("yield_10y", 0.0) > 4.5 or score_map.get("Long_Bond", 0.0) < -20:
        triggers.append("- 10Y yield or fiscal pressure can keep long-bond scores under pressure.")
    if features.get("gold_crowding_index", 0.0) > 60:
        triggers.append("- Gold crowding is elevated; short-term pullback risk is higher.")
    if features.get("ai_roi_pressure_index", 0.0) > 60:
        triggers.append("- AI capex pressure is elevated; equity multiple compression risk is higher.")
    if features.get("phase_transition_risk_index", 0.0) > 60:
        triggers.append("- Cross-asset fragility is high; small shocks may produce outsized moves.")
    if not triggers:
        triggers.append("- No single risk trigger is above the high-alert threshold today.")
    return triggers


def _risk_triggers_zh(features: dict[str, float], scores: list[AssetScore]) -> list[str]:
    score_map = {score.asset: score.final_score for score in scores}
    triggers = []
    if features.get("yield_10y", 0.0) > 4.5 or score_map.get("Long_Bond", 0.0) < -20:
        triggers.append("- 10Y 收益率或财政压力可能继续压制长债评分。")
    if features.get("gold_crowding_index", 0.0) > 60:
        triggers.append("- 黄金拥挤度偏高，短期回撤风险上升。")
    if features.get("ai_roi_pressure_index", 0.0) > 60:
        triggers.append("- AI Capex 压力偏高，美股估值压缩风险上升。")
    if features.get("phase_transition_risk_index", 0.0) > 60:
        triggers.append("- 跨资产脆弱性偏高，小冲击可能触发较大波动。")
    if not triggers:
        triggers.append("- 今日没有单一风险触发器超过高警戒阈值。")
    return triggers


def _regime_label_zh(regime: str) -> str:
    return REGIME_LABELS_ZH.get(regime, regime)


def _asset_label_zh(asset: str) -> str:
    return ASSET_LABELS_ZH.get(asset, asset)


def _narrative_label_zh(name: str) -> str:
    return NARRATIVE_LABELS_ZH.get(name, name)


def _portfolio_reason_zh(rec: PortfolioRecommendation) -> str:
    if rec.selected:
        return "由 Regime 概率、资产评分和压力指数规则选中。"
    return "用于对照的组合配置。"


def _markdown_to_html(markdown: str, *, title: str, lang: str) -> str:
    escaped = html.escape(markdown)
    body = escaped.replace("\n", "<br>\n")
    return (
        f"<!doctype html><html lang='{lang}'><head><meta charset='utf-8'>"
        f"<title>{html.escape(title)}</title>"
        "<style>body{font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif;"
        "max-width:1100px;margin:40px auto;line-height:1.55;color:#17202a}"
        "code{background:#eef2f5;padding:2px 4px;border-radius:4px}</style>"
        "</head><body>"
        f"{body}</body></html>"
    )
