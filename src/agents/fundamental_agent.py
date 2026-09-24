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


class FundamentalAgent(BaseAgent):
    """Specialized agent for fundamental and financial valuation analysis."""

    def __init__(self, router: Optional[ModelRouter] = None):
        super().__init__(
            name="FundamentalAgent",
            system_prompt=FUNDAMENTAL_AGENT_PROMPT,
            task_complexity=TaskComplexity.COMPLEX,
            router=router,
        )

    def analyze(self, snapshot: StockSnapshot) -> AgentSignalOutput:
        """Evaluate fundamental ratios and financial health."""
        f = snapshot.fundamentals
        sector = snapshot.sector or "General"
        pe_str = f"{f.pe_ratio:.2f}" if f.pe_ratio else "N/A"
        pb_str = f"{f.pb_ratio:.2f}" if f.pb_ratio else "N/A"
        roe_str = f"{f.return_on_equity * 100:.1f}%" if f.return_on_equity else "N/A"
        margin_str = f"{f.profit_margin * 100:.1f}%" if f.profit_margin else "N/A"
        de_str = f"{f.debt_to_equity:.2f}" if f.debt_to_equity else "N/A"
        mcap_str = f"{f.market_cap:,.0f}" if f.market_cap else "N/A"

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

Provide your structured FundamentalAgent evaluation.
"""
        return self._query_gateway(
            user_prompt=prompt,
            fallback_handler=self._heuristic_fallback,
            snapshot=snapshot,
        )


    def _heuristic_fallback(self, snapshot: StockSnapshot) -> AgentSignalOutput:
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

        return AgentSignalOutput(
            agent=self.name,
            signal=signal,
            confidence=conf,
            reasons=reasons,
            risks=[f"Multiple compression risk if earnings growth decelerates"],
            evidence=[
                f"P/E={pe:.1f}",
                f"ROE={roe*100:.1f}%",
                f"Debt/Equity={de:.2f}",
                f"Margin={margin*100:.1f}%",
            ],
        )

