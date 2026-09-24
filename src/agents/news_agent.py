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
from src.rag.citation import CitationVerifier
from src.rag.retriever import FinancialRAGRetriever


class NewsAgent(BaseAgent):
    """Specialized agent for news sentiment and catalyst analysis with RAG grounding."""

    def __init__(
        self,
        router: Optional[ModelRouter] = None,
        retriever: Optional[FinancialRAGRetriever] = None,
    ):
        super().__init__(
            name="NewsAgent",
            system_prompt=NEWS_AGENT_PROMPT,
            task_complexity=TaskComplexity.HIGH_VOLUME,
            router=router,
        )
        self.retriever = retriever or FinancialRAGRetriever()

    def analyze(
        self,
        snapshot: StockSnapshot,
        query_override: Optional[str] = None,
    ) -> AgentSignalOutput:
        """Evaluate recent news items and corporate catalysts with filing grounding."""
        news_text = ""
        if snapshot.recent_news:
            for idx, item in enumerate(snapshot.recent_news[:5], 1):
                date_str = item.published_at.strftime("%Y-%m-%d") if item.published_at else "Recent"
                news_text += f"{idx}. [{date_str}] {item.title} ({item.publisher or 'News'})\n"
        else:
            news_text = "No recent major news headlines detected."

        # Retrieve relevant 8-K disclosures or corporate press releases
        query_text = query_override or f"{snapshot.ticker} 8-K press release earnings announcement guidance"
        retrieval = self.retriever.retrieve_evidence(
            ticker=snapshot.ticker,
            query_text=query_text,
            market=snapshot.market,
            top_k=2,
        )

        prompt = f"""
Asset: {snapshot.ticker} ({snapshot.market.upper()})
Current Price: {snapshot.currency} {snapshot.current_price:.2f}

Recent Headlines:
{news_text}

Authoritative Filing & Disclosure Context:
{retrieval.context_text}

MANDATORY EVIDENCE RULES:
1. Every material catalyst claim MUST cite headline or filing evidence.
2. If evidence is unavailable in both headlines and filings, return "Insufficient evidence."
3. Do NOT fabricate citations or catalysts.
"""
        raw_output = self._query_gateway(
            user_prompt=prompt,
            fallback_handler=lambda snapshot=snapshot, **kw: self._heuristic_fallback(snapshot, retrieval),
            snapshot=snapshot,
        )

        # Check if there is zero evidence in both headlines and filings
        if not snapshot.recent_news and not retrieval.has_sufficient_evidence:
            return AgentSignalOutput(
                agent=self.name,
                signal="HOLD",
                confidence=0.30,
                reasons=["Insufficient evidence from news feeds or corporate filings."],
                risks=["Lack of verified news or filing flow."],
                evidence=["Insufficient evidence."],
            )

        grounded_reasons, grounded_risks, evidence_bullets = CitationVerifier.verify_and_ground_claims(
            reasons=raw_output.reasons,
            retrieval=retrieval,
            agent_name=self.name,
        )

        # Add headline evidence
        if snapshot.recent_news:
            for item in snapshot.recent_news[:3]:
                evidence_bullets.append(f"Headline: \"{item.title}\" ({item.publisher or 'News'})")

        return AgentSignalOutput(
            agent=self.name,
            signal=raw_output.signal,
            confidence=raw_output.confidence,
            reasons=grounded_reasons,
            risks=grounded_risks,
            evidence=evidence_bullets,
        )

    def _heuristic_fallback(
        self,
        snapshot: StockSnapshot,
        retrieval: Optional[Any] = None,
    ) -> AgentSignalOutput:
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

        evidence_list = [
            f"NewsItems={news_count}",
            f"PositiveTags={pos_score}",
            f"NegativeTags={neg_score}",
        ]
        if retrieval and retrieval.citations:
            evidence_list.extend(CitationVerifier.format_citations(retrieval.citations))

        return AgentSignalOutput(
            agent=self.name,
            signal=signal,
            confidence=conf,
            reasons=reasons,
            risks=["Unanticipated news flow or sudden regulatory disclosure"],
            evidence=evidence_list,
        )
