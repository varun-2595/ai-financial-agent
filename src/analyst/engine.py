"""
LLM Analyst Engine utilizing the Google GenAI SDK.
Produces validated StockAnalysis structured outputs with fallback handling.
"""
from __future__ import annotations

import os
from typing import Optional

from google import genai
from google.genai import types

from src.analyst.prompts import SYSTEM_PROMPT, build_analysis_prompt
from src.data.models import StockAnalysis, StockSnapshot
from src.technicals.indicators import compute_technical_indicators
from src.technicals.levels import compute_support_resistance
from src.utils.config import get_config
from src.utils.logger import logger


class AnalystEngine:
    """
    Evaluates securities by synthesizing raw data, fundamentals, technical indicators,
    and key price levels into institutional-grade structured research using Gemini.
    """

    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or os.getenv("GEMINI_API_KEY")
        self.config = get_config()
        self.client = genai.Client(api_key=self.api_key) if self.api_key else None

    def analyze_stock(self, snapshot: StockSnapshot) -> StockAnalysis:
        """
        Analyze a StockSnapshot and return a validated StockAnalysis object.
        """
        if not self.client:
            logger.warning("No GEMINI_API_KEY found; generating heuristic-based fallback analysis.")
            return self._heuristic_fallback(snapshot)

        # 1. Compute technical indicators and key levels
        technicals = compute_technical_indicators(snapshot.history)
        levels = compute_support_resistance(snapshot.history)

        # 2. Build Prompt
        prompt = build_analysis_prompt(snapshot, technicals, levels)

        model_name = self.config.llm.model
        logger.info(f"[Analyst] Calling Gemini ({model_name}) for {snapshot.ticker}...")

        try:
            response = self.client.models.generate_content(
                model=model_name,
                contents=prompt,
                config=types.GenerateContentConfig(
                    system_instruction=SYSTEM_PROMPT,
                    temperature=self.config.llm.temperature,
                    response_mime_type="application/json",
                    response_schema=StockAnalysis,
                ),
            )

            # Parse response into StockAnalysis
            analysis = StockAnalysis.model_validate_json(response.text)
            logger.success(f"[Analyst] ✓ {snapshot.ticker}: {analysis.action} (Risk: {analysis.risk_level}, Confidence: {analysis.confidence_score:.2f})")
            return analysis

        except Exception as exc:
            logger.error(f"[Analyst] Error analyzing {snapshot.ticker} with {model_name}: {exc}")
            # Try fallback model if configured
            fallback_model = self.config.llm.fallback_model
            if fallback_model and fallback_model != model_name:
                try:
                    logger.info(f"[Analyst] Retrying with fallback model {fallback_model}...")
                    response = self.client.models.generate_content(
                        model=fallback_model,
                        contents=prompt,
                        config=types.GenerateContentConfig(
                            system_instruction=SYSTEM_PROMPT,
                            temperature=self.config.llm.temperature,
                            response_mime_type="application/json",
                            response_schema=StockAnalysis,
                        ),
                    )
                    analysis = StockAnalysis.model_validate_json(response.text)
                    logger.success(f"[Analyst] ✓ {snapshot.ticker} (via fallback): {analysis.action}")
                    return analysis
                except Exception as fallback_exc:
                    logger.error(f"[Analyst] Fallback model also failed: {fallback_exc}")

            return self._heuristic_fallback(snapshot, error=str(exc))

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
