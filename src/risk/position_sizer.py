"""
Position Sizing and Risk Validation Engine.
Strictly enforces ATR-based volatility parity sizing, Half-Kelly scaling,
hard stop losses, single-position value caps, and sector concentration guardrails.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Literal, Optional

from src.data.models import StockSnapshot
from src.technicals.indicators import compute_technical_indicators
from src.utils.config import get_config
from src.utils.logger import logger


@dataclass
class SizingResult:
    allowed: bool
    quantity: int
    entry_price: float
    stop_loss: float
    target_price: float
    capital_allocated: float
    risk_amount: float
    reason: str


class RiskEngine:
    """Volatility-adjusted institutional risk & position sizing engine."""

    def __init__(self):
        self.config = get_config()

    def _get_half_kelly_factor(self) -> float:
        """
        Computes Half-Kelly sizing scale based on trailing 30-trade historical win rate
        and payoff ratio from the decision journal. Clamped between [0.5x, 1.5x].
        """
        try:
            from src.journal.journal_store import DecisionJournalStore
            store = DecisionJournalStore()
            stats = store.get_aggregate_scorecard(days=30)
            if stats["total_trades"] >= 5:
                w = max(min(stats["win_rate_pct"] / 100.0, 0.95), 0.05)
                r = max(stats["win_loss_ratio"], 0.1)
                # Kelly formula: K = W - (1 - W) / R
                kelly = w - ((1.0 - w) / r)
                # Half-Kelly multiplier around 1.0 baseline
                half_kelly = 1.0 + (kelly / 2.0)
                return max(0.50, min(1.50, round(half_kelly, 2)))
        except Exception:
            pass
        return 1.0

    def calculate_position_size(
        self,
        snapshot: StockSnapshot,
        direction: Literal["BUY", "SELL"],
        entry_price: float,
        stop_loss: float,
        target_price: float,
        current_cash: float,
        total_portfolio_value: float,
        sector_exposure_pct: float = 0.0,
        strategy: Literal["scalping", "intraday", "swing", "positional"] = "swing",
    ) -> SizingResult:
        """
        Calculates allowed position size using:
        - ATR-based volatility parity: Shares = floor((Capital * Risk%) / (ATR(14) * 1.5))
        - Half-Kelly scaling factor from recent trade track record
        - Strategy-specific capital caps & margin leverage
        - Sector concentration limits
        - Stop loss validation (minimum Risk-Reward ratio)
        """
        if entry_price <= 0 or stop_loss <= 0 or target_price <= 0:
            return SizingResult(False, 0, entry_price, stop_loss, target_price, 0, 0, "Invalid price values")

        # 1. Direction and Risk-Reward validation
        if direction == "BUY":
            if stop_loss >= entry_price:
                return SizingResult(False, 0, entry_price, stop_loss, target_price, 0, 0, "Stop loss must be below entry for BUY")
            if target_price <= entry_price:
                return SizingResult(False, 0, entry_price, stop_loss, target_price, 0, 0, "Target must be above entry for BUY")
            risk_per_share = entry_price - stop_loss
            reward_per_share = target_price - entry_price
        else:
            if stop_loss <= entry_price:
                return SizingResult(False, 0, entry_price, stop_loss, target_price, 0, 0, "Stop loss must be above entry for SELL")
            if target_price >= entry_price:
                return SizingResult(False, 0, entry_price, stop_loss, target_price, 0, 0, "Target must be below entry for SELL")
            risk_per_share = stop_loss - entry_price
            reward_per_share = entry_price - target_price

        # Check Risk-to-Reward (min 1:1.2 for scalping, 1:1.3 for others)
        min_rr = 1.2 if strategy == "scalping" else 1.3
        rr_ratio = reward_per_share / risk_per_share if risk_per_share > 0 else 0
        if rr_ratio < min_rr:
            return SizingResult(False, 0, entry_price, stop_loss, target_price, 0, 0, f"Unfavorable Risk:Reward ratio ({rr_ratio:.2f} < {min_rr})")

        # 2. Sector Concentration Guardrail
        max_sector_pct = self.config.paper_trading.max_sector_pct
        if sector_exposure_pct >= max_sector_pct:
            return SizingResult(False, 0, entry_price, stop_loss, target_price, 0, 0, f"Sector exposure ({sector_exposure_pct*100:.1f}%) exceeds limit ({max_sector_pct*100:.1f}%)")

        # 3. Strategy Sizing & Margin Leverage
        is_intraday_or_scalp = strategy in ("scalping", "intraday")
        leverage = self.config.paper_trading.intraday_leverage_multiplier if is_intraday_or_scalp else 1.0

        # Maximum loss allowed per trade
        max_risk_pct = 0.035 if strategy == "scalping" else self.config.paper_trading.risk_per_trade_pct
        max_loss_allowed = total_portfolio_value * max_risk_pct

        # 4. ATR(14) Volatility Parity Calculation
        atr_val = None
        if snapshot.history and len(snapshot.history) >= 14:
            try:
                tech = compute_technical_indicators(snapshot.history)
                atr_val = tech.atr_14
            except Exception:
                atr_val = None

        volatility_risk_unit = (atr_val * 1.5) if (atr_val and atr_val > 0) else risk_per_share
        volatility_risk_unit = max(volatility_risk_unit, risk_per_share * 0.5)

        raw_atr_qty = int(max_loss_allowed / volatility_risk_unit) if volatility_risk_unit > 0 else 0

        # Apply Half-Kelly Scaling
        half_kelly_multiplier = self._get_half_kelly_factor()
        qty_by_risk = int(raw_atr_qty * half_kelly_multiplier)

        # Max Position Value Cap
        if strategy == "scalping":
            max_pos_val = total_portfolio_value * 0.50 * leverage
        elif strategy == "intraday":
            max_pos_val = total_portfolio_value * 0.40 * leverage
        elif strategy == "swing":
            max_pos_val = total_portfolio_value * 0.25
        else:  # positional
            max_pos_val = total_portfolio_value * 0.15

        qty_by_val = int(max_pos_val / entry_price)

        # Cash Availability (leveraged for intraday/scalp)
        effective_cash = current_cash * leverage if is_intraday_or_scalp else current_cash
        qty_by_cash = int(effective_cash / entry_price)

        # Conservative minimum across volatility risk, value cap, and cash
        quantity = min(qty_by_risk, qty_by_val, qty_by_cash)

        # Allow 1 share if cash & risk permit on small accounts
        if quantity <= 0 and effective_cash >= entry_price and risk_per_share <= (max_loss_allowed * 1.5):
            quantity = 1

        if quantity <= 0:
            return SizingResult(False, 0, entry_price, stop_loss, target_price, 0, 0, "Insufficient cash or position size calculates to 0")

        total_capital = quantity * entry_price
        total_risk = quantity * risk_per_share

        return SizingResult(
            allowed=True,
            quantity=quantity,
            entry_price=entry_price,
            stop_loss=stop_loss,
            target_price=target_price,
            capital_allocated=round(total_capital, 2),
            risk_amount=round(total_risk, 2),
            reason=f"Approved: {quantity} shares (ATR-Risk: {volatility_risk_unit:.2f}, Half-Kelly: {half_kelly_multiplier:.2f}x, Total Risk: {total_risk:.2f})",
        )
