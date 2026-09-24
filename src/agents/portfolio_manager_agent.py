"""
Portfolio Manager Agent for Aegis Multi-Agent Network.

Synthesizes intelligence from Technical, Fundamental, News, Macro, and Risk agents
to form the final coordinated investment recommendation.
NOTE: Does NOT execute trades directly; provides intelligence and conviction.
"""
from __future__ import annotations

from typing import Optional

from src.agents.base import BaseAgent
from src.agents.models import AgentSignalOutput, MultiAgentConsensus
from src.agents.prompts import PORTFOLIO_MANAGER_PROMPT
from src.data.models import StockSnapshot
from src.gateway.gateway import TaskComplexity
from src.gateway.router import ModelRouter


class PortfolioManagerAgent(BaseAgent):
    """
    Lead agent responsible for multi-agent signal synthesis and portfolio conviction.
    """

    def __init__(self, router: Optional[ModelRouter] = None):
        super().__init__(
            name="PortfolioManagerAgent",
            system_prompt=PORTFOLIO_MANAGER_PROMPT,
            task_complexity=TaskComplexity.PORTFOLIO_DECISION,
            router=router,
        )

    def synthesize(
        self,
        snapshot: StockSnapshot,
        consensus: MultiAgentConsensus,
    ) -> AgentSignalOutput:
        """Synthesize signals from all 5 specialized domain agents."""
        tech_str = f"Signal: {consensus.technical.signal if consensus.technical else 'N/A'} (Conf: {consensus.technical.confidence if consensus.technical else 0:.2f})"
        fund_str = f"Signal: {consensus.fundamental.signal if consensus.fundamental else 'N/A'} (Conf: {consensus.fundamental.confidence if consensus.fundamental else 0:.2f})"
        news_str = f"Signal: {consensus.news.signal if consensus.news else 'N/A'} (Conf: {consensus.news.confidence if consensus.news else 0:.2f})"
        macro_str = f"Signal: {consensus.macro.signal if consensus.macro else 'N/A'} (Conf: {consensus.macro.confidence if consensus.macro else 0:.2f})"
        risk_str = f"Signal: {consensus.risk.signal if consensus.risk else 'N/A'} (Conf: {consensus.risk.confidence if consensus.risk else 0:.2f})"

        prompt = f"""
Asset: {snapshot.ticker} ({snapshot.market.upper()})
Current Price: {snapshot.currency} {snapshot.current_price:.2f}

Agent Intelligence Reports:
1. TechnicalAgent:    {tech_str}
2. FundamentalAgent:  {fund_str}
3. NewsAgent:         {news_str}
4. MacroAgent:        {macro_str}
5. RiskAgent (Bear Red-Team): {risk_str}

Detailed Reasoning Highlights:
- Technical: {', '.join(consensus.technical.reasons[:2]) if consensus.technical else 'None'}
- Fundamental: {', '.join(consensus.fundamental.reasons[:2]) if consensus.fundamental else 'None'}
- News: {', '.join(consensus.news.reasons[:1]) if consensus.news else 'None'}
- Macro: {', '.join(consensus.macro.reasons[:1]) if consensus.macro else 'None'}
- Risk / Bear Case: {', '.join(consensus.risk.reasons[:2]) if consensus.risk else 'None'} (Risks: {', '.join(consensus.risk.risks[:2]) if consensus.risk else 'None'})

ADVERSARIAL RED-TEAMING MANDATE:
1. Conduct an adversarial Bull vs Bear debate between upside catalysts and RiskAgent downside objections.
2. If issuing a BUY, you MUST explicitly invalidate or containment-bound the Bear/Risk objections with technical or fundamental data.
3. If Bear/Risk risks are NOT convincingly refuted, downgrade signal to HOLD or SELL.

Synthesize these perspectives and provide your final coordinated PortfolioManagerAgent recommendation.
"""
        return self._query_gateway(
            user_prompt=prompt,
            fallback_handler=self._heuristic_fallback,
            snapshot=snapshot,
            consensus=consensus,
        )

    def _heuristic_fallback(
        self,
        snapshot: StockSnapshot,
        consensus: MultiAgentConsensus,
    ) -> AgentSignalOutput:
        """Deterministic consensus aggregation with adversarial Bear/Risk veto checks."""
        weights = {
            "technical": 0.30,
            "fundamental": 0.25,
            "risk": 0.20,
            "macro": 0.15,
            "news": 0.10,
        }

        agent_map = {
            "technical": consensus.technical,
            "fundamental": consensus.fundamental,
            "risk": consensus.risk,
            "macro": consensus.macro,
            "news": consensus.news,
        }

        buy_weight = 0.0
        sell_weight = 0.0
        hold_weight = 0.0
        all_reasons = []
        all_risks = []
        all_evidence = []

        for key, agent_out in agent_map.items():
            if agent_out is None:
                continue
            w = weights[key]
            conf = agent_out.confidence
            eff_weight = w * conf

            if agent_out.signal == "BUY":
                buy_weight += eff_weight
            elif agent_out.signal == "SELL":
                sell_weight += eff_weight
            else:
                hold_weight += eff_weight

            all_reasons.extend(agent_out.reasons[:1])
            all_risks.extend(agent_out.risks[:1])
            all_evidence.extend(agent_out.evidence[:1])

        # Adversarial Red-Team Check: RiskAgent veto/downside check
        risk_veto = False
        if consensus.risk and consensus.risk.signal == "SELL" and consensus.risk.confidence >= 0.65:
            risk_veto = True

        if buy_weight >= 0.45 and buy_weight > sell_weight * 1.5 and not risk_veto:
            final_signal = "BUY"
            final_conf = min(0.95, round(buy_weight + 0.20, 2))
            all_reasons.append("Adversarial check: Bull thesis invalidates downside objections")
        elif sell_weight >= 0.40 or risk_veto:
            final_signal = "SELL"
            final_conf = min(0.90, round(sell_weight + 0.20, 2)) if not risk_veto else 0.75
            if risk_veto:
                all_risks.append("Adversarial Red-Team veto: RiskAgent identified unacceptable downside tail risk")
        else:
            final_signal = "HOLD"
            final_conf = 0.60
            all_reasons.append("Equilibrium between Bull catalysts and Bear risk considerations")

        return AgentSignalOutput(
            agent=self.name,
            signal=final_signal,
            confidence=final_conf,
            reasons=all_reasons if all_reasons else ["Consensus balance across technical and fundamental signals"],
            risks=all_risks if all_risks else ["Market regime shift or sector rotation"],
            evidence=all_evidence if all_evidence else [f"BuyWeight={buy_weight:.2f}", f"SellWeight={sell_weight:.2f}"],
        )
