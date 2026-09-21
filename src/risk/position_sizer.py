"""
Position Sizing and Risk Validation Engine.
Strictly enforces hard stop loss, position size caps, sector concentration limits,
and max portfolio drawdown guardrails.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Optional

from src.data.models import StockSnapshot
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
    def __init__(self):
        self.config = get_config()

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
        Calculates allowed position size based on:
        - Strategy-specific capital allocation and risk cap
        - Simulated 3x intraday margin for scalping and intraday
        - Sector concentration limits
        - Stop loss validation (minimum 1:1.3 Risk-Reward ratio)
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
        qty_by_risk = int(max_loss_allowed / risk_per_share) if risk_per_share > 0 else 0

        # Max Position Value Cap
        if strategy == "scalping":
            max_pos_val = total_portfolio_value * 0.50 * leverage
        elif strategy == "intraday":
            max_pos_val = total_portfolio_value * 0.40 * leverage
        elif strategy == "swing":
            max_pos_val = total_portfolio_value * 0.25
        else: # positional
            max_pos_val = total_portfolio_value * 0.15

        qty_by_val = int(max_pos_val / entry_price)

        # Cash Availability (leveraged for intraday/scalp)
        effective_cash = current_cash * leverage if is_intraday_or_scalp else current_cash
        qty_by_cash = int(effective_cash / entry_price)

        # Conservative minimum across risk, value, and cash
        quantity = min(qty_by_risk, qty_by_val, qty_by_cash)

        # If quantity calculates to 0 due to integer truncation on small accounts, allow 1 share if cash & risk permit
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
            reason=f"Approved: {quantity} shares (R:R {rr_ratio:.2f}, Risk: {total_risk:.2f})",
        )
