"""
Technical Analysis Agent for Aegis Multi-Agent Network.

Evaluates price action, momentum, trend health, volume, and support/resistance levels.
"""
from __future__ import annotations

from typing import Optional

from src.agents.base import BaseAgent
from src.agents.models import AgentSignalOutput
from src.agents.prompts import TECHNICAL_AGENT_PROMPT
from src.data.models import StockSnapshot
from src.gateway.gateway import TaskComplexity
from src.gateway.router import ModelRouter
from src.technicals.indicators import compute_technical_indicators
from src.technicals.levels import compute_support_resistance


class TechnicalAgent(BaseAgent):
    """Specialized agent for quantitative technical analysis."""

    def __init__(self, router: Optional[ModelRouter] = None):
        super().__init__(
            name="TechnicalAgent",
            system_prompt=TECHNICAL_AGENT_PROMPT,
            task_complexity=TaskComplexity.HIGH_VOLUME,
            router=router,
        )

    def analyze(self, snapshot: StockSnapshot) -> AgentSignalOutput:
        """Evaluate technical indicators and price levels."""
        tech = compute_technical_indicators(snapshot.history)
        levels = compute_support_resistance(snapshot.history)

        prompt = f"""
Asset: {snapshot.ticker} ({snapshot.market.upper()})
Current Price: {snapshot.currency} {snapshot.current_price:.2f}

Technical Indicators:
- Short Trend: {tech.trend_short} | Long Trend: {tech.trend_long}
- RSI (14): {tech.rsi_14 if tech.rsi_14 else 50.0:.1f} ({tech.rsi_condition})
- MACD Line: {tech.macd if tech.macd else 0.0:.2f} | Signal: {tech.macd_signal if tech.macd_signal else 0.0:.2f} | Histogram: {tech.macd_hist if tech.macd_hist else 0.0:.2f}
- EMA 20: {tech.ema_20 if tech.ema_20 else 0.0:.2f} | EMA 50: {tech.ema_50 if tech.ema_50 else 0.0:.2f} | EMA 200: {tech.ema_200 if tech.ema_200 else 0.0:.2f}
- ATR (14): {tech.atr_14 if tech.atr_14 else 0.0:.2f} | Bollinger Upper: {tech.bb_upper if tech.bb_upper else 0.0:.2f}

Key Price Levels:
- Pivot: {levels.pivot_point}
- Support 1: {levels.support_1} | Support 2: {levels.support_2}
- Resistance 1: {levels.resistance_1} | Resistance 2: {levels.resistance_2}


Provide your structured TechnicalAgent evaluation.
"""
        return self._query_gateway(
            user_prompt=prompt,
            fallback_handler=self._heuristic_fallback,
            snapshot=snapshot,
        )


    def _heuristic_fallback(self, snapshot: StockSnapshot) -> AgentSignalOutput:
        """Deterministic rule-based technical evaluation."""
        tech = compute_technical_indicators(snapshot.history)
        levels = compute_support_resistance(snapshot.history)
        rsi = tech.rsi_14 or 50.0

        if tech.trend_short == "BULLISH" and rsi < 65.0:
            signal = "BUY"
            conf = 0.75
            reasons = [
                f"Short-term trend is bullish with EMA alignment (EMA20 > EMA50)",
                f"RSI 14 at {rsi:.1f} shows healthy momentum without overbought exhaustion",
            ]
        elif tech.trend_short == "BEARISH" or rsi > 75.0:
            signal = "SELL" if rsi > 80.0 else "HOLD"
            conf = 0.70
            reasons = [
                f"Bearish trend orientation or overbought RSI at {rsi:.1f}",
                f"Near resistance ceiling at {levels.resistance_1}",
            ]
        else:
            signal = "HOLD"
            conf = 0.55
            reasons = ["Sideways consolidation; awaiting clear breakout above pivot"]

        return AgentSignalOutput(
            agent=self.name,
            signal=signal,
            confidence=conf,
            reasons=reasons,
            risks=[f"Breach of primary support at {levels.support_1}"],
            evidence=[
                f"RSI={rsi:.1f}",
                f"Trend={tech.trend_short}",
                f"S1={levels.support_1}",
                f"R1={levels.resistance_1}",
            ],
        )
