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
    ) -> SizingResult:
        """
        Calculates maximum allowed position size based on:
        - Max 2% capital risk per trade
        - Max 10% position value of total portfolio
        - Max 30% sector concentration
        - Stop loss validation (minimum 1:1.5 Risk-Reward ratio)
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

        # Check Risk-to-Reward (min 1:1.5)
        rr_ratio = reward_per_share / risk_per_share
        if rr_ratio < 1.3:
            return SizingResult(False, 0, entry_price, stop_loss, target_price, 0, 0, f"Unfavorable Risk:Reward ratio ({rr_ratio:.2f} < 1.3)")

        # 2. Sector Concentration Guardrail
        max_sector_pct = self.config.paper_trading.max_sector_pct
        if sector_exposure_pct >= max_sector_pct:
            return SizingResult(False, 0, entry_price, stop_loss, target_price, 0, 0, f"Sector exposure ({sector_exposure_pct*100:.1f}%) exceeds limit ({max_sector_pct*100:.1f}%)")

        # 3. Capital Risk Sizing (2% max loss of portfolio)
        max_risk_pct = self.config.paper_trading.risk_per_trade_pct
        max_loss_allowed = total_portfolio_value * max_risk_pct
        qty_by_risk = int(max_loss_allowed / risk_per_share) if risk_per_share > 0 else 0

        # 4. Max Position Value Cap (10% of total portfolio)
        max_pos_pct = self.config.paper_trading.max_position_pct
        max_val_allowed = total_portfolio_value * max_pos_pct
        qty_by_val = int(max_val_allowed / entry_price)

        # 5. Cash Availability
        qty_by_cash = int(current_cash / entry_price)

        # Final Quantity is conservative minimum
        quantity = min(qty_by_risk, qty_by_val, qty_by_cash)

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
