"""
Pre-Trade 'Red Team' Adversarial Debate Engine.

Executes a structured 3-role adversarial debate before trade approval:
1. Bull Agent: Highlights momentum and catalyst upside.
2. Bear Agent ('Devil's Advocate'): Hunts for bull traps, divergence, and overhead resistance.
3. Chief Risk Judge: Delivers final Verdict (APPROVE / REDUCE / REJECT) with risk-adjusted sizing.

Designed for sub-second execution on Apple Silicon M5 (Qwen 2.5 14B).
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Literal, Optional

from src.analyst.playbook_memory import MemoryWarningReport
from src.data.models import StockSnapshot
from src.gateway.gateway import ModelRequest, TaskComplexity
from src.gateway.router import ModelRouter
from src.technicals.indicators import compute_technical_indicators
from src.technicals.levels import compute_support_resistance
from src.utils.logger import logger


@dataclass
class DebateOutcome:
    ticker: str
    verdict: Literal["APPROVE", "REDUCE", "REJECT"]
    bull_arguments: list[str] = field(default_factory=list)
    bear_counterpoints: list[str] = field(default_factory=list)
    fatal_flaw_detected: bool = False
    verdict_reasoning: str = ""
    confidence_score: float = 0.5


class RedTeamDebateEngine:
    """Orchestrates structured Bull vs. Bear pre-trade debates on local M5 node."""

    def __init__(self, router: Optional[ModelRouter] = None):
        self.router = router or ModelRouter()

    def conduct_debate(
        self,
        snapshot: StockSnapshot,
        direction: Literal["BUY", "SELL"] = "BUY",
        strategy: str = "swing",
        memory_report: Optional[MemoryWarningReport] = None,
        force_heuristic: bool = False,
    ) -> DebateOutcome:
        """
        Runs the pre-trade adversarial debate on local M5.
        """
        ticker = snapshot.ticker
        price = snapshot.current_price
        tech = compute_technical_indicators(snapshot.history)
        levels = compute_support_resistance(snapshot.history)

        # Technical context summary
        rsi_str = f"{tech.rsi_14:.1f}" if (tech and tech.rsi_14 is not None) else "N/A"
        ema20_str = f"{tech.ema_20:.2f}" if (tech and tech.ema_20 is not None) else "N/A"
        ema50_str = f"{tech.ema_50:.2f}" if (tech and tech.ema_50 is not None) else "N/A"
        trend_str = ("BULLISH" if (tech.ema_20 and tech.ema_50 and tech.ema_20 > tech.ema_50) else "BEARISH") if tech else "NEUTRAL"
        s1_str = f"{levels.support_1:.2f}" if (levels and levels.support_1 is not None) else "N/A"
        r1_str = f"{levels.resistance_1:.2f}" if (levels and levels.resistance_1 is not None) else "N/A"

        tech_summary = (
            f"Price: {price} | RSI(14): {rsi_str} | Trend: {trend_str} | "
            f"EMA20: {ema20_str} | EMA50: {ema50_str} | "
            f"Support1: {s1_str} | Resistance1: {r1_str}"
        )

        memory_warning_text = memory_report.formatted_warning if (memory_report and memory_report.has_historical_failures) else "No prior failure patterns."

        if force_heuristic:
            return self._heuristic_debate(snapshot, tech, levels, memory_report)

        prompt = f"""You are the Chief Risk Officer running an institutional Red-Team Pre-Trade Debate.
Asset: {ticker} ({snapshot.market.upper()}) | Direction: {direction} | Strategy: {strategy}
Market Data: {tech_summary}
Historical Failure Memory: {memory_warning_text}

Task:
1. Bull Case: Give 2 strong arguments for this trade.
2. Bear 'Devil's Advocate' Case: Find 2 hidden traps or structural vulnerabilities (e.g. overhead supply, RSI exhaustion, memory warning, low volume).
3. Final Verdict: 'APPROVE', 'REDUCE', or 'REJECT'.

Respond ONLY with valid JSON in this exact structure:
{{
  "bull_arguments": ["point 1", "point 2"],
  "bear_counterpoints": ["risk 1", "risk 2"],
  "fatal_flaw_detected": false,
  "verdict": "APPROVE",
  "confidence_score": 0.85,
  "verdict_reasoning": "Brief justification"
}}
"""

        try:
            req = ModelRequest(
                prompt=prompt,
                task_complexity=TaskComplexity.HIGH_VOLUME,
                temperature=0.2,
                max_tokens=600,
            )
            resp = self.router.route(req)
            if resp.content:
                cleaned = re.sub(r"```json\s*", "", resp.content)
                cleaned = re.sub(r"```\s*", "", cleaned).strip()
                match = re.search(r"\{.*\}", cleaned, re.DOTALL)
                if match:
                    data = json.loads(match.group(0))
                    verdict_str = str(data.get("verdict", "APPROVE")).upper()
                    if verdict_str not in ("APPROVE", "REDUCE", "REJECT"):
                        verdict_str = "APPROVE"

                    outcome = DebateOutcome(
                        ticker=ticker,
                        verdict=verdict_str,  # type: ignore
                        bull_arguments=data.get("bull_arguments", []),
                        bear_counterpoints=data.get("bear_counterpoints", []),
                        fatal_flaw_detected=bool(data.get("fatal_flaw_detected", False)),
                        verdict_reasoning=data.get("verdict_reasoning", "Debate completed on M5"),
                        confidence_score=float(data.get("confidence_score", 0.75)),
                    )
                    logger.info(f"[Red Team Debate] {ticker}: {outcome.verdict} (Conf: {outcome.confidence_score:.2f}) - {outcome.verdict_reasoning}")
                    return outcome
        except Exception as e:
            logger.debug(f"[Red Team Debate] LLM debate routing error for {ticker}: {e}; using heuristic fallback.")

        return self._heuristic_debate(snapshot, tech, levels, memory_report)

    def _heuristic_debate(
        self,
        snapshot: StockSnapshot,
        tech,
        levels,
        memory_report: Optional[MemoryWarningReport],
    ) -> DebateOutcome:
        """Deterministic heuristic fallback for Red Team debate."""
        ticker = snapshot.ticker
        price = snapshot.current_price
        bulls: list[str] = []
        bears: list[str] = []
        fatal_flaw = False

        # Bull points
        if tech and tech.rsi_14 and 45 <= tech.rsi_14 <= 65:
            bulls.append(f"Healthy momentum zone (RSI {tech.rsi_14:.1f})")
        if tech and tech.ema_20 and tech.ema_50 and tech.ema_20 > tech.ema_50:
            bulls.append(f"Price ({price:.2f}) aligned with EMA trend")

        # Bear checks
        if tech and tech.rsi_14 and tech.rsi_14 > 72:
            bears.append(f"Overbought exhaustion risk (RSI {tech.rsi_14:.1f} > 72)")
            fatal_flaw = True
        if levels and levels.resistance_1 and (levels.resistance_1 - price) / price < 0.015:
            bears.append(f"Immediate overhead resistance at {levels.resistance_1:.2f} (<1.5% distance)")

        if memory_report and memory_report.has_historical_failures:
            bears.append(f"Historical failure pattern: {memory_report.key_lessons[0] if memory_report.key_lessons else 'Prior loss on setup'}")
            if memory_report.failure_count >= 2:
                fatal_flaw = True

        if fatal_flaw:
            verdict = "REJECT"
            reason = "Bear Agent uncovered fatal structural trap / historical failure repetition."
        elif bears:
            verdict = "REDUCE"
            reason = f"Approved with reduced size due to minor resistance/pullback risk: {bears[0]}"
        else:
            verdict = "APPROVE"
            reason = "Bull thesis validated by Red Team; no fatal traps detected."

        return DebateOutcome(
            ticker=ticker,
            verdict=verdict,
            bull_arguments=bulls or ["Momentum alignment"],
            bear_counterpoints=bears or ["Standard volatility risk"],
            fatal_flaw_detected=fatal_flaw,
            verdict_reasoning=reason,
            confidence_score=0.80 if verdict == "APPROVE" else (0.60 if verdict == "REDUCE" else 0.30),
        )
