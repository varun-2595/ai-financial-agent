"""
News & Catalyst Analysis Agent for Aegis Multi-Agent Network.

Evaluates recent headlines, corporate disclosures, earnings catalysts, and sentiment.
"""
from __future__ import annotations

from typing import Optional

from src.agents.base import BaseAgent
from src.agents.models import AgentSignalOutput
from src.agents.prompts import NEWS_AGENT_PROMPT
from src.data.models import StockSnapshot
from src.gateway.gateway import TaskComplexity
from src.gateway.router import ModelRouter


class NewsAgent(BaseAgent):
    """Specialized agent for news sentiment and catalyst analysis."""

    def __init__(self, router: Optional[ModelRouter] = None):
        super().__init__(
            name="NewsAgent",
            system_prompt=NEWS_AGENT_PROMPT,
            task_complexity=TaskComplexity.HIGH_VOLUME,
            router=router,
        )

    def analyze(self, snapshot: StockSnapshot) -> AgentSignalOutput:
        """Evaluate recent news items and corporate catalysts."""
        news_text = ""
        if snapshot.recent_news:
            for idx, item in enumerate(snapshot.recent_news[:5], 1):
                news_text += f"{idx}. [{item.published_at.strftime('%Y-%m-%d')}] {item.title} ({item.publisher})\n"
        else:
            news_text = "No recent major news headlines detected."

        prompt = f"""
Asset: {snapshot.ticker} ({snapshot.market.upper()})
Current Price: {snapshot.currency} {snapshot.current_price:.2f}

Recent Headlines:
{news_text}

Provide your structured NewsAgent sentiment and catalyst evaluation.
"""
        return self._query_gateway(
            user_prompt=prompt,
            fallback_handler=self._heuristic_fallback,
            snapshot=snapshot,
        )

    def _heuristic_fallback(self, snapshot: StockSnapshot) -> AgentSignalOutput:
        """Deterministic rule-based news sentiment evaluation."""
        news_count = len(snapshot.recent_news) if snapshot.recent_news else 0
        positive_keywords = ["growth", "beat", "profit", "expansion", "partnership", "upgrade", "record", "dividend"]
        negative_keywords = ["miss", "investigation", "downgrade", "loss", "lawsuit", "delay", "decline", "warning"]

        pos_score = 0
        neg_score = 0

        if snapshot.recent_news:
            for item in snapshot.recent_news:
                text = (item.title + " " + (item.summary or "")).lower()
                for kw in positive_keywords:
                    if kw in text:
                        pos_score += 1
                for kw in negative_keywords:
                    if kw in text:
                        neg_score += 1

        if pos_score > neg_score:
            signal = "BUY"
            conf = 0.68
            reasons = [f"Positive news sentiment with {pos_score} bullish catalyst triggers detected"]
        elif neg_score > pos_score:
            signal = "HOLD" if neg_score <= 2 else "SELL"
            conf = 0.65
            reasons = [f"Negative catalyst presence with {neg_score} cautionary headline tags"]
        else:
            signal = "HOLD"
            conf = 0.50
            reasons = ["Neutral news environment without high-impact price-moving catalysts"]

        return AgentSignalOutput(
            agent=self.name,
            signal=signal,
            confidence=conf,
            reasons=reasons,
            risks=["Unanticipated news flow or sudden regulatory disclosure"],
            evidence=[
                f"NewsItems={news_count}",
                f"PositiveTags={pos_score}",
                f"NegativeTags={neg_score}",
            ],
        )
