"""
LLM Analyst Engine utilizing ModelRouter.
Routes through Local M5 Node or Gemini remote provider with automated structured validation.
"""
from __future__ import annotations

import json
import os
from typing import Optional

from src.analyst.prompts import SYSTEM_PROMPT, build_analysis_prompt
from src.data.models import StockAnalysis, StockSnapshot
from src.gateway.gateway import ModelRequest, TaskComplexity
from src.gateway.router import ModelRouter
from src.technicals.indicators import compute_technical_indicators
from src.technicals.levels import compute_support_resistance
from src.utils.config import get_config
from src.utils.logger import logger


class AnalystEngine:
    """
    Evaluates securities by synthesizing raw data, fundamentals, technical indicators,
    and key price levels into institutional-grade structured research via ModelRouter.
    """

    def __init__(
        self,
        router: Optional[ModelRouter] = None,
        api_key: Optional[str] = None,
        force_heuristic: bool = False,
    ):
        self.config = get_config()
        self.router = router or ModelRouter()
        self.force_heuristic = force_heuristic

    def analyze_stock(self, snapshot: StockSnapshot) -> StockAnalysis:
        """
        Analyze a StockSnapshot and return a validated StockAnalysis object.
        """
        if self.force_heuristic:
            return self._heuristic_fallback(snapshot)

        # 1. Compute technical indicators and key levels
        technicals = compute_technical_indicators(snapshot.history)
        levels = compute_support_resistance(snapshot.history)


        # 2. Build Prompt
        prompt = build_analysis_prompt(snapshot, technicals, levels)

        req = ModelRequest(
            system_prompt=SYSTEM_PROMPT,
            user_prompt=prompt,
            response_schema=StockAnalysis,
            task_complexity=TaskComplexity.COMPLEX,
            temperature=self.config.llm.temperature,
            timeout_seconds=45.0,
        )

        resp = self.router.route(req)

        if resp.success:
            if resp.structured_data and isinstance(resp.structured_data, StockAnalysis):
                logger.success(
                    f"[Analyst] ✓ {snapshot.ticker} ({resp.provider_used}): {resp.structured_data.action} "
                    f"(Confidence: {resp.structured_data.confidence_score:.2f})"
                )
                return resp.structured_data
            elif resp.content:
                try:
                    analysis = StockAnalysis.model_validate_json(resp.content)
                    logger.success(
                        f"[Analyst] ✓ {snapshot.ticker} ({resp.provider_used}): {analysis.action} "
                        f"(Confidence: {analysis.confidence_score:.2f})"
                    )
                    return analysis
                except Exception as parse_err:
                    logger.warning(f"[Analyst] JSON validation fallback for {snapshot.ticker}: {parse_err}")

        logger.warning(f"[Analyst] Model inference failed for {snapshot.ticker} ({resp.error}); applying heuristic fallback.")
        return self._heuristic_fallback(snapshot, error=resp.error)

    def _heuristic_fallback(self, snapshot: StockSnapshot, error: Optional[str] = None) -> StockAnalysis:
        """Deterministic rule-based fallback if LLM is unavailable or unconfigured."""
        technicals = compute_technical_indicators(snapshot.history)
        levels = compute_support_resistance(snapshot.history)

        is_bullish = technicals.trend_short == "BULLISH"
        rsi = technicals.rsi_14 or 50.0

        if is_bullish and rsi < 65:
            action = "Good Buy"
            risk = "Medium"
            conf = 0.70
        elif rsi >= 70:
            action = "Hold-Watch"
            risk = "High"
            conf = 0.60
        elif technicals.trend_short == "BEARISH":
            action = "Hold-Watch"
            risk = "Medium"
            conf = 0.65
        else:
            action = "Hold-Watch"
            risk = "Low"
            conf = 0.50

        return StockAnalysis(
            ticker=snapshot.ticker,
            risk_level=risk,
            action=action,
            reasons_attractive=[
                f"Trend is {technicals.trend_short.lower()}",
                f"RSI 14 at {rsi:.1f} ({technicals.rsi_condition.lower()})",
                f"Trading near support level {levels.support_1}",
            ],
            risks=[
                f"Resistance nearby at {levels.resistance_1}",
                f"Downside invalidation if price breaches support at {levels.support_2}",
            ],
            suggested_approach="Use strict limit orders near key support levels.",
            invalidation_trigger=f"Breakdown below {levels.support_1}",
            entry_price_hint=snapshot.current_price,
            stop_loss_hint=levels.support_1,
            target_price_hint=levels.resistance_1,
            confidence_score=conf,
            strategy_fit=["swing"],
            horizon_fit=["short_term"],
            analyst_notes=f"Generated via technical heuristic rules (LLM note: {error or 'offline'}).",
        )
