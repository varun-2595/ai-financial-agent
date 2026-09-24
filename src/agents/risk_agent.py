"""
Risk & Capital Preservation Agent for Aegis Multi-Agent Network.

Evaluates downside tail risk, invalidation triggers, and volatility exposure.
NOTE: Quantitative position sizing and margin math remain strictly deterministic Python code in RiskEngine.
"""
from __future__ import annotations

from typing import Optional

from src.agents.base import BaseAgent
from src.agents.models import AgentSignalOutput
from src.agents.prompts import RISK_AGENT_PROMPT
from src.data.models import StockSnapshot
from src.gateway.gateway import TaskComplexity
from src.gateway.router import ModelRouter
from src.technicals.indicators import compute_technical_indicators
from src.technicals.levels import compute_support_resistance


class RiskAgent(BaseAgent):
    """Specialized agent for downside risk and scenario invalidation analysis."""

    def __init__(self, router: Optional[ModelRouter] = None):
        super().__init__(
            name="RiskAgent",
            system_prompt=RISK_AGENT_PROMPT,
            task_complexity=TaskComplexity.COMPLEX,
            router=router,
        )

    def analyze(self, snapshot: StockSnapshot) -> AgentSignalOutput:
        """Evaluate structural risk and downside invalidation triggers."""
        tech = compute_technical_indicators(snapshot.history)
        levels = compute_support_resistance(snapshot.history)

        curr = snapshot.current_price
        dist_s1_pct = ((curr - levels.support_1) / curr * 100) if levels.support_1 < curr else 0.0
        atr_val = tech.atr_14 if tech.atr_14 else (curr * 0.02)
        atr_pct = (atr_val / curr * 100)

        prompt = f"""
Asset: {snapshot.ticker} ({snapshot.market.upper()})
Current Price: {snapshot.currency} {curr:.2f}

Volatility & Risk Metrics:
- ATR (14): {snapshot.currency} {atr_val:.2f} ({atr_pct:.2f}% of price)
- Distance to Support 1 ({levels.support_1}): {dist_s1_pct:.2f}%
- Distance to Support 2 ({levels.support_2}): {((curr - levels.support_2)/curr*100):.2f}%
- RSI (14): {tech.rsi_14 if tech.rsi_14 else 50.0:.1f}
- Bollinger Upper: {tech.bb_upper if tech.bb_upper else 0.0:.2f}

Provide your structured RiskAgent evaluation focusing on downside invalidation.
"""

        return self._query_gateway(
            user_prompt=prompt,
            fallback_handler=self._heuristic_fallback,
            snapshot=snapshot,
        )

    def _heuristic_fallback(self, snapshot: StockSnapshot) -> AgentSignalOutput:
        """Deterministic rule-based risk evaluation."""
        tech = compute_technical_indicators(snapshot.history)
        levels = compute_support_resistance(snapshot.history)
        curr = snapshot.current_price
        atr_pct = ((tech.atr_14 or (curr * 0.02)) / curr * 100)

        if atr_pct <= 3.5 and curr > levels.support_1:
            signal = "BUY"
            conf = 0.72
            reasons = [
                f"Controlled structural volatility with ATR at {atr_pct:.2f}% of price",
                f"Solid support buffer above S1 ({levels.support_1}) providing clear invalidation point",
            ]
        elif atr_pct > 6.0:
            signal = "HOLD"
            conf = 0.70
            reasons = [
                f"Excessive price volatility (ATR: {atr_pct:.2f}%); high risk of stop whipsaw",
            ]
        else:
            signal = "HOLD"
            conf = 0.55
            reasons = ["Moderate volatility profile; standard stop loss placement required"]

        return AgentSignalOutput(
            agent=self.name,
            signal=signal,
            confidence=conf,
            reasons=reasons,
            risks=[f"Breakdown below support floor {levels.support_2}"],
            evidence=[
                f"ATR%={atr_pct:.2f}%",
                f"S1={levels.support_1}",
                f"S2={levels.support_2}",
            ],
        )
