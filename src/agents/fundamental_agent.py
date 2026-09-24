"""
Fundamental Analysis Agent for Aegis Multi-Agent Network.

Evaluates balance sheet health, earnings quality, valuation multiples, ROE, and debt.
"""
from __future__ import annotations

from typing import Optional

from src.agents.base import BaseAgent
from src.agents.models import AgentSignalOutput
from src.agents.prompts import FUNDAMENTAL_AGENT_PROMPT
from src.data.models import StockSnapshot
from src.gateway.gateway import TaskComplexity
from src.gateway.router import ModelRouter
from src.rag.citation import CitationVerifier
from src.rag.retriever import FinancialRAGRetriever


class FundamentalAgent(BaseAgent):
    """Specialized agent for fundamental and financial valuation analysis with RAG grounding."""

    def __init__(
        self,
        router: Optional[ModelRouter] = None,
        retriever: Optional[FinancialRAGRetriever] = None,
    ):
        super().__init__(
            name="FundamentalAgent",
            system_prompt=FUNDAMENTAL_AGENT_PROMPT,
            task_complexity=TaskComplexity.COMPLEX,
            router=router,
        )
        self.retriever = retriever or FinancialRAGRetriever()

    def analyze(
        self,
        snapshot: StockSnapshot,
        query_override: Optional[str] = None,
    ) -> AgentSignalOutput:
        """
        Evaluate fundamental ratios and financial health grounded in corporate filings.
        """
        f = snapshot.fundamentals
        sector = snapshot.sector or "General"
        pe_str = f"{f.pe_ratio:.2f}" if f.pe_ratio else "N/A"
        pb_str = f"{f.pb_ratio:.2f}" if f.pb_ratio else "N/A"
        roe_str = f"{f.return_on_equity * 100:.1f}%" if f.return_on_equity else "N/A"
        margin_str = f"{f.profit_margin * 100:.1f}%" if f.profit_margin else "N/A"
        de_str = f"{f.debt_to_equity:.2f}" if f.debt_to_equity else "N/A"
        mcap_str = f"{f.market_cap:,.0f}" if f.market_cap else "N/A"

        # 1. Retrieve authoritative filing evidence
        query_text = query_override or f"{snapshot.ticker} annual report 10-K quarterly financial results revenue margins debt roe"
        retrieval = self.retriever.retrieve_evidence(
            ticker=snapshot.ticker,
            query_text=query_text,
            market=snapshot.market,
            top_k=3,
        )

        prompt = f"""
Asset: {snapshot.ticker} ({snapshot.market.upper()})
Sector: {sector}
Market Cap: {snapshot.currency} {mcap_str}

Financial Multiples & Metrics:
- P/E Ratio: {pe_str}
- Price-to-Book (P/B): {pb_str}
- Return on Equity (ROE): {roe_str}
- Profit Margin: {margin_str}
- Debt-to-Equity: {de_str}

Authoritative Filing Evidence:
{retrieval.context_text}

MANDATORY EVIDENCE RULES:
1. Every material fundamental claim MUST cite the provided evidence (e.g. [10-K, Item 7, 2024-10-31] or [NSE Quarterly Results, 2024-10-18]).
2. If evidence is unavailable or insufficient in the filings above, explicitly state "Insufficient evidence."
3. Do NOT fabricate or extrapolate unverified financial numbers.
"""
        raw_output = self._query_gateway(
            user_prompt=prompt,
            fallback_handler=lambda snapshot=snapshot, **kw: self._heuristic_fallback(snapshot, retrieval),
            snapshot=snapshot,
        )

        # 2. Strict grounding & citation verification
        grounded_reasons, grounded_risks, evidence_bullets = CitationVerifier.verify_and_ground_claims(
            reasons=raw_output.reasons,
            retrieval=retrieval,
            agent_name=self.name,
        )

        if not retrieval.has_sufficient_evidence:
            return AgentSignalOutput(
                agent=self.name,
                signal="HOLD",
                confidence=0.30,
                reasons=grounded_reasons,
                risks=grounded_risks,
                evidence=["Insufficient evidence."],
            )

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
        """Deterministic rule-based fundamental analysis."""
        f = snapshot.fundamentals
        roe = f.return_on_equity or 0.12
        pe = f.pe_ratio or 22.0
        de = f.debt_to_equity or 0.8
        margin = f.profit_margin or 0.15

        if roe >= 0.15 and pe <= 35.0 and de <= 1.5:
            signal = "BUY"
            conf = 0.72
            reasons = [
                f"Strong return on equity of {roe*100:.1f}% indicates high capital efficiency",
                f"Reasonable valuation at {pe:.1f}x P/E with manageable leverage (D/E: {de:.2f})",
            ]
        elif pe > 65.0 or de > 2.5:
            signal = "HOLD"
            conf = 0.65
            reasons = [
                f"Elevated valuation multiple ({pe:.1f}x P/E) or high leverage (D/E: {de:.2f})",
            ]
        else:
            signal = "HOLD"
            conf = 0.55
            reasons = ["Adequate fundamental profile; balanced risk-reward"]

        evidence_list = [
            f"P/E={pe:.1f}",
            f"ROE={roe*100:.1f}%",
            f"Debt/Equity={de:.2f}",
            f"Margin={margin*100:.1f}%",
        ]
        if retrieval and retrieval.citations:
            evidence_list.extend(CitationVerifier.format_citations(retrieval.citations))

        return AgentSignalOutput(
            agent=self.name,
            signal=signal,
            confidence=conf,
            reasons=reasons,
            risks=[f"Multiple compression risk if earnings growth decelerates"],
            evidence=evidence_list,
        )

