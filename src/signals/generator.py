"""
Trade Signal Generator combining Analyst evaluation, technical indicators,
and key price levels for Intraday, Swing, and Positional trading strategies.
"""
from __future__ import annotations

from typing import Literal, Optional

from src.analyst.engine import AnalystEngine
from src.data.models import StockSnapshot, TradeSignal
from src.risk.position_sizer import RiskEngine
from src.technicals.indicators import compute_technical_indicators
from src.technicals.levels import compute_support_resistance
from src.utils.logger import logger


class SignalGenerator:
    def __init__(self, analyst: Optional[AnalystEngine] = None, risk: Optional[RiskEngine] = None):
        self.analyst = analyst or AnalystEngine()
        self.risk = risk or RiskEngine()

    def generate_signal(
        self,
        snapshot: StockSnapshot,
        strategy: Literal["intraday", "swing", "positional"] = "swing",
        current_cash: float = 100_000.0,
        portfolio_val: float = 1_000_000.0,
        sector_exposure_pct: float = 0.0,
    ) -> Optional[TradeSignal]:
        """
        Synthesizes technical conditions and LLM analysis to produce actionable TradeSignal.
        """
        # 1. Technical calculations
        tech = compute_technical_indicators(snapshot.history)
        levels = compute_support_resistance(snapshot.history)

        # 2. Analyst evaluation
        analysis = self.analyst.analyze_stock(snapshot)

        # Only trigger buy on high conviction
        if analysis.action not in ("Must Buy", "Good Buy"):
            logger.info(f"[Signal] No buy signal for {snapshot.ticker} (action: {analysis.action})")
            return None

        current_p = snapshot.current_price
        direction: Literal["BUY", "SELL", "HOLD"] = "BUY"

        # Calculate entry, SL, and target based on strategy & technical levels
        if strategy == "intraday":
            entry = current_p
            stop_loss = round(levels.support_1 if levels.support_1 < current_p else current_p * 0.99, 2)
            target = round(levels.resistance_1 if levels.resistance_1 > current_p else current_p * 1.015, 2)
        elif strategy == "swing":
            entry = current_p
            # Set stop loss just below key support or ATR
            atr_sl = current_p - (tech.atr_14 * 1.5 if tech.atr_14 else current_p * 0.03)
            stop_loss = round(min(levels.support_1, atr_sl) if levels.support_1 < current_p else atr_sl, 2)
            target = round(levels.resistance_2 if levels.resistance_2 > current_p else current_p * 1.06, 2)
        else: # positional
            entry = current_p
            stop_loss = round(levels.support_2 if levels.support_2 < current_p else current_p * 0.92, 2)
            target = round(current_p * 1.15, 2)

        # 3. Risk Sizing & Approval
        size_res = self.risk.calculate_position_size(
            snapshot=snapshot,
            direction=direction,
            entry_price=entry,
            stop_loss=stop_loss,
            target_price=target,
            current_cash=current_cash,
            total_portfolio_value=portfolio_val,
            sector_exposure_pct=sector_exposure_pct,
        )

        if not size_res.allowed:
            logger.warning(f"[Signal] Signal for {snapshot.ticker} rejected by RiskEngine: {size_res.reason}")
            return None

        sig = TradeSignal(
            ticker=snapshot.ticker,
            market=snapshot.market,
            strategy=strategy,
            direction=direction,
            entry_price=entry,
            stop_loss=stop_loss,
            target_price=target,
            quantity=size_res.quantity,
            confidence=analysis.confidence_score,
            reasoning=f"Action: {analysis.action} | {', '.join(analysis.reasons_attractive[:2])} | {size_res.reason}",
        )
        logger.success(f"[Signal] 🔥 GENERATED {sig.direction} {sig.quantity}x {sig.ticker} @ {sig.entry_price} (SL: {sig.stop_loss}, TGT: {sig.target_price})")
        return sig
