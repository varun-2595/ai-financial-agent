"""
News Catalyst Engine.

Classifies breaking news headlines into structured actionable catalysts
(Earnings Beat, Analyst Upgrades, FDA Approval, Major Contract, Headline Risk),
scoring sentiment from -1.0 (severe headline risk) to +1.0 (strong bullish catalyst).
Integrates with ModelRouter (M5 Local Node / Gemini Cloud) with robust deterministic fallbacks.
"""
from __future__ import annotations

import re
from typing import Literal, Optional

from pydantic import BaseModel, Field

from src.data.models import NewsItem
from src.gateway.gateway import ModelRequest, TaskComplexity
from src.gateway.router import ModelRouter
from src.utils.logger import logger

CatalystCategory = Literal[
    "EARNINGS_BEAT",
    "UPGRADE_PRICE_TARGET",
    "CONTRACT_DEAL",
    "FDA_APPROVAL",
    "PRODUCT_INNOVATION",
    "MACRO_TAILWIND",
    "LAWSUIT_INVESTIGATION",
    "EARNINGS_MISS",
    "DOWNGRADE",
    "INSIDER_SELLING",
    "NEUTRAL_COMMENTARY",
    "NO_CATALYST",
]

SentimentLabel = Literal["BULLISH_CATALYST", "NEUTRAL", "BEARISH_RISK"]


class CatalystReport(BaseModel):
    """Structured report assessing stock news catalysts and headline risk."""

    ticker: str
    market: Literal["india", "us"] = "us"
    sentiment_score: float = Field(
        ...,
        ge=-1.0,
        le=1.0,
        description="Sentiment score: -1.0 (toxic headline risk) to +1.0 (strong positive catalyst).",
    )
    sentiment_label: SentimentLabel
    catalyst_category: CatalystCategory
    has_headline_risk: bool = Field(
        default=False,
        description="True if major risk: fraud, SEC probe, debt default, regulatory ban, accounting issue.",
    )
    catalyst_summary: str = Field(
        ...,
        description="Concise 1-2 sentence executive summary of the catalyst.",
    )
    key_catalysts: list[str] = Field(
        default_factory=list,
        description="Key bullet points or cited facts from the headlines.",
    )
    confidence: float = Field(
        default=0.7,
        ge=0.0,
        le=1.0,
        description="Model confidence in this classification.",
    )


CATALYST_SYSTEM_PROMPT = """You are a senior institutional quantitative analyst specializing in event-driven equities trading and news sentiment arbitration.
Your objective is to evaluate raw news headlines for a specific security and output a strict JSON CatalystReport.

RULES:
1. Identify high-conviction catalysts:
   - EARNINGS_BEAT: Quarterly beat, raised forward guidance, revenue surprise.
   - UPGRADE_PRICE_TARGET: Wall Street broker upgrade, price target hike.
   - CONTRACT_DEAL: Multi-million/billion dollar contract win, strategic alliance, supply deal.
   - FDA_APPROVAL: Drug phase 3 pass, FDA clearance, patent grant.
   - PRODUCT_INNOVATION: New product launch, AI chip breakthrough, major platform expansion.
   - LAWSUIT_INVESTIGATION: SEC subpoena, fraud allegations, short seller report, CBI/ED raid.
   - EARNINGS_MISS: Revenue miss, lowered guidance, margin compression.
   - DOWNGRADE: Rating cut, price target slash.
   - INSIDER_SELLING: Major promoter/executive dumping shares.

2. HEADLINE RISK FILTER:
   - Set has_headline_risk = True immediately if the news contains fraud, regulatory raids, bankruptcy, accounting restatement, SEC/SEBI ban, or fatal patent loss.

3. SCORING SCALE:
   - Strong Bullish Catalyst: +0.60 to +1.00
   - Mild Bullish Catalyst: +0.20 to +0.59
   - Neutral / Noise: -0.19 to +0.19
   - Mild Bearish Risk: -0.20 to -0.59
   - Severe Headline Risk: -0.60 to -1.00
"""


class NewsCatalystEngine:
    """Evaluates breaking news for catalysts and risks via LLM & heuristics."""

    def __init__(self, router: Optional[ModelRouter] = None):
        self.router = router or ModelRouter()

    def evaluate_catalysts(
        self,
        ticker: str,
        market: Literal["india", "us"],
        news_items: list[NewsItem],
        force_heuristic: bool = False,
    ) -> CatalystReport:
        """Evaluates a list of news items for a ticker and generates a CatalystReport."""
        if not news_items:
            return CatalystReport(
                ticker=ticker,
                market=market,
                sentiment_score=0.0,
                sentiment_label="NEUTRAL",
                catalyst_category="NO_CATALYST",
                has_headline_risk=False,
                catalyst_summary=f"No recent breaking news detected for {ticker}.",
                key_catalysts=[],
                confidence=0.5,
            )

        if force_heuristic:
            return self._heuristic_catalyst_evaluation(ticker, market, news_items)

        news_text = "\n".join(
            f"- [{item.published_at.strftime('%Y-%m-%d') if item.published_at else 'Recent'}] {item.title} ({item.publisher or 'News'})"
            for item in news_items[:6]
        )

        user_prompt = f"""Asset: {ticker} ({market.upper()})

Recent Headlines:
{news_text}

TASK:
Analyze the headlines, identify if there is a primary catalyst, score sentiment from -1.0 to +1.0, flag any headline risk, and return the structured report adhering strictly to the schema.
"""

        req = ModelRequest(
            system_prompt=CATALYST_SYSTEM_PROMPT,
            user_prompt=user_prompt,
            response_schema=CatalystReport,
            task_complexity=TaskComplexity.HIGH_VOLUME,
            temperature=0.1,
            timeout_seconds=25.0,
        )

        resp = self.router.route(req)

        if resp.success:
            if resp.structured_data and isinstance(resp.structured_data, CatalystReport):
                logger.info(
                    f"[CatalystEngine] ✓ {ticker}: {resp.structured_data.catalyst_category} "
                    f"({resp.structured_data.sentiment_score:+.2f}) via {resp.provider_used}"
                )
                return resp.structured_data
            elif resp.content:
                try:
                    report = CatalystReport.model_validate_json(resp.content)
                    return report
                except Exception as parse_err:
                    logger.warning(f"[CatalystEngine] JSON validation error for {ticker}: {parse_err}")

        # Fallback to rule-based parser
        logger.debug(f"[CatalystEngine] Using heuristic evaluation fallback for {ticker}")
        return self._heuristic_catalyst_evaluation(ticker, market, news_items)

    def _heuristic_catalyst_evaluation(
        self,
        ticker: str,
        market: Literal["india", "us"],
        news_items: list[NewsItem],
    ) -> CatalystReport:
        """Fast keyword-based rule evaluation when offline or LLM unavailable."""
        combined_text = " ".join([item.title.lower() for item in news_items])

        # Headline risk keywords
        risk_patterns = [
            r"\bfraud\b", r"\bsec probe\b", r"\bsebi probe\b", r"\blawsuit\b",
            r"\binvestigation\b", r"\braid\b", r"\bdefault\b", r"\bbankruptcy\b",
            r"\baccounting scandal\b", r"\bsubpoena\b", r"\bshort seller\b",
            r"\bclass action\b", r"\brecall\b", r"\bdowngrade\b", r"\bmisses estimates\b",
        ]
        bullish_patterns = [
            (r"\b(record profit|beats estimates|earnings beat|q\d profit jumps|surpasses profit)\b", "EARNINGS_BEAT", 0.75),
            (r"\b(upgraded|price target raised|buy rating|top pick|outperform)\b", "UPGRADE_PRICE_TARGET", 0.65),
            (r"\b(wins contract|bags order|deal with|secures contract|partnership)\b", "CONTRACT_DEAL", 0.70),
            (r"\b(fda approval|fda clears|patent granted|phase \d success)\b", "FDA_APPROVAL", 0.85),
            (r"\b(launches new|breakthrough|ai expansion|next-gen|soars|surges)\b", "PRODUCT_INNOVATION", 0.55),
        ]

        has_risk = any(re.search(pat, combined_text) for pat in risk_patterns)
        
        # Check specific risks
        if has_risk:
            sentiment_score = -0.70
            if any(w in combined_text for w in ("probe", "lawsuit", "investigation", "fraud", "scandal", "subpoena", "raid")):
                category: CatalystCategory = "LAWSUIT_INVESTIGATION"
            elif "miss" in combined_text:
                category: CatalystCategory = "EARNINGS_MISS"
            else:
                category: CatalystCategory = "DOWNGRADE"
            label: SentimentLabel = "BEARISH_RISK"
            summary = f"Identified headline risk/litigation concerns in recent coverage: {news_items[0].title}"
            return CatalystReport(
                ticker=ticker,
                market=market,
                sentiment_score=sentiment_score,
                sentiment_label=label,
                catalyst_category=category,
                has_headline_risk=True,
                catalyst_summary=summary,
                key_catalysts=[news_items[0].title],
                confidence=0.75,
            )

        # Check bullish matches
        for pat, cat, score in bullish_patterns:
            if re.search(pat, combined_text):
                matching_title = next((item.title for item in news_items if re.search(pat, item.title.lower())), news_items[0].title)
                return CatalystReport(
                    ticker=ticker,
                    market=market,
                    sentiment_score=score,
                    sentiment_label="BULLISH_CATALYST",
                    catalyst_category=cat,  # type: ignore
                    has_headline_risk=False,
                    catalyst_summary=f"Positive catalyst identified: {matching_title}",
                    key_catalysts=[matching_title],
                    confidence=0.70,
                )

        # Neutral fallback
        return CatalystReport(
            ticker=ticker,
            market=market,
            sentiment_score=0.10,
            sentiment_label="NEUTRAL",
            catalyst_category="NEUTRAL_COMMENTARY",
            has_headline_risk=False,
            catalyst_summary=f"Routine market news and commentary for {ticker}.",
            key_catalysts=[news_items[0].title] if news_items else [],
            confidence=0.60,
        )
