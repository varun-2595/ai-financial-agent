"""
Macroeconomic & Regime Analysis Agent for Aegis Multi-Agent Network.

Evaluates interest rate trends, inflation regimes, market volatility (VIX), and sector rotation.
"""
from __future__ import annotations

from typing import Optional

from src.agents.base import BaseAgent
from src.agents.models import AgentSignalOutput, MacroContext
from src.agents.prompts import MACRO_AGENT_PROMPT
from src.data.models import StockSnapshot
from src.gateway.gateway import TaskComplexity
from src.gateway.router import ModelRouter


class MacroAgent(BaseAgent):
    """Specialized agent for macroeconomic and market regime analysis."""

    def __init__(self, router: Optional[ModelRouter] = None):
        super().__init__(
            name="MacroAgent",
            system_prompt=MACRO_AGENT_PROMPT,
            task_complexity=TaskComplexity.COMPLEX,
            router=router,
        )

    def analyze(
        self,
        snapshot: StockSnapshot,
        macro_context: Optional[MacroContext] = None,
    ) -> AgentSignalOutput:
        """Evaluate macroeconomic backdrop and regime tailwinds."""
        ctx = macro_context or MacroContext()
        sector = snapshot.sector or "General"
        prompt = f"""
Asset: {snapshot.ticker} ({snapshot.market.upper()})
Sector: {sector}

Macroeconomic Regime:
- Interest Rate Trajectory: {ctx.interest_rate_trend}

- Inflation Environment: {ctx.inflation_regime}
- Market Regime: {ctx.market_regime}
- Implied Volatility (VIX): {ctx.vix_level:.1f}
- FX / Currency Dynamics: {ctx.usd_inr_trend}
- Notes: {ctx.notes or 'None'}

Provide your structured MacroAgent regime evaluation.
"""
        return self._query_gateway(
            user_prompt=prompt,
            fallback_handler=self._heuristic_fallback,
            snapshot=snapshot,
            macro_context=ctx,
        )

    def _heuristic_fallback(
        self,
        snapshot: StockSnapshot,
        macro_context: Optional[MacroContext] = None,
    ) -> AgentSignalOutput:
        """Deterministic rule-based macroeconomic evaluation."""
        ctx = macro_context or MacroContext()
        vix = ctx.vix_level

        if vix <= 20.0 and ctx.market_regime in ("BULLISH_TREND", "SIDEWAYS"):
            signal = "BUY"
            conf = 0.70
            reasons = [
                f"Supportive macro regime with normalized VIX at {vix:.1f}",
                f"Stable monetary and currency backdrop favoring risk assets",
            ]
        elif vix > 25.0 or ctx.market_regime == "BEARISH_TREND":
            signal = "HOLD"
            conf = 0.65
            reasons = [
                f"Elevated macro risk aversion (VIX: {vix:.1f}) and defensive market posture",
            ]
        else:
            signal = "HOLD"
            conf = 0.55
            reasons = ["Neutral macroeconomic backdrop with balanced regime indicators"]

        return AgentSignalOutput(
            agent=self.name,
            signal=signal,
            confidence=conf,
            reasons=reasons,
            risks=["Macro policy surprise, geopolitical shock, or rate spike"],
            evidence=[
                f"VIX={vix:.1f}",
                f"MarketRegime={ctx.market_regime}",
                f"RateTrend={ctx.interest_rate_trend}",
            ],
        )
