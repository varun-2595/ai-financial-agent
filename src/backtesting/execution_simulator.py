"""
Simulated Execution Engine for Backtesting.

Handles realistic bar-level fills, slippage, brokerage/exchange fees,
stop-loss & take-profit hits, and intraday square-offs.
"""
from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Literal, Optional

from src.data.models import Strategy, TradeSignal
from src.trading.fees import DEFAULT_FEE_SCHEDULE, FeeSchedule


@dataclass
class SimulatedPosition:
    """Represents an active open position in the backtest portfolio."""
    position_id: str
    ticker: str
    market: Literal["india", "us"]
    strategy: Strategy
    direction: Literal["BUY", "SELL"]
    quantity: int
    entry_price: float         # clean price from signal
    avg_cost: float            # filled price including entry slippage
    current_price: float
    stop_loss: float
    target_price: float
    margin_blocked: float      # exact capital collateral deducted from cash
    fees_paid: float           # total fees paid so far on this position
    opened_at: datetime
    entry_order_id: str
    is_paper: bool = True

    @property
    def unrealized_pnl(self) -> float:
        if self.direction == "BUY":
            return (self.current_price - self.avg_cost) * self.quantity
        else:
            return (self.avg_cost - self.current_price) * self.quantity


@dataclass
class SimulatedTradeRecord:
    """Represents a completed (closed) trade with full accounting breakdown."""
    trade_id: str
    position_id: str
    ticker: str
    market: Literal["india", "us"]
    strategy: Strategy
    direction: Literal["BUY", "SELL"]
    quantity: int
    entry_price: float
    entry_fill_price: float
    exit_price_raw: float
    exit_fill_price: float
    margin_blocked: float
    entry_fees: float
    exit_fees: float
    total_fees: float
    gross_pnl: float
    net_pnl: float
    pnl_pct: float
    entry_time: datetime
    exit_time: datetime
    hold_duration_days: float
    exit_reason: str           # "STOP_LOSS", "TARGET_HIT", "INTRADAY_SQUARE_OFF", "SIGNAL_EXIT", "EXPIRY"


class ExecutionSimulator:
    """
    Simulates fills, price executions, and exit triggers without lookahead bias.
    """

    def __init__(self, fee_schedule: Optional[FeeSchedule] = None):
        self.fees = fee_schedule or DEFAULT_FEE_SCHEDULE

    def execute_entry(
        self,
        signal: TradeSignal,
        fill_price_raw: float,
        timestamp: datetime,
        leverage: float = 1.0,
    ) -> tuple[SimulatedPosition, float]:
        """
        Execute a simulated BUY entry.

        Returns:
            (SimulatedPosition, total_cash_deducted)
            where total_cash_deducted = margin_blocked + entry_fees
        """
        # Apply entry adverse slippage (BUY fills higher)
        filled_price = self.fees.apply_slippage(fill_price_raw, "BUY")

        # Margin blocked = (filled_price * qty) / leverage
        margin_blocked = round((filled_price * signal.quantity) / leverage, 4)

        # Entry fees
        entry_fees = self.fees.compute_fees(
            market=signal.market,
            direction="BUY",
            filled_price=filled_price,
            quantity=signal.quantity,
            is_paper=True,
        )

        pos_id = f"POS-{uuid.uuid4().hex[:8].upper()}"
        order_id = f"ORD-{uuid.uuid4().hex[:8].upper()}"

        position = SimulatedPosition(
            position_id=pos_id,
            ticker=signal.ticker,
            market=signal.market,
            strategy=signal.strategy,
            direction=signal.direction if signal.direction in ("BUY", "SELL") else "BUY",
            quantity=signal.quantity,
            entry_price=signal.entry_price,
            avg_cost=filled_price,
            current_price=filled_price,
            stop_loss=signal.stop_loss,
            target_price=signal.target_price,
            margin_blocked=margin_blocked,
            fees_paid=entry_fees,
            opened_at=timestamp,
            entry_order_id=order_id,
        )

        total_cash_deducted = round(margin_blocked + entry_fees, 4)
        return position, total_cash_deducted

    def check_and_execute_exit(
        self,
        position: SimulatedPosition,
        bar_open: float,
        bar_high: float,
        bar_low: float,
        bar_close: float,
        timestamp: datetime,
        is_eod_square_off: bool = False,
    ) -> Optional[tuple[SimulatedTradeRecord, float]]:
        """
        Evaluate if an open position is closed during this bar.

        Checks:
          1. Stop Loss: bar_low <= stop_loss
          2. Target Price: bar_high >= target_price
          3. Conflict (both hit): conservative resolution -> Stop Loss triggered first
          4. Intraday square-off (if applicable): exit at bar_close

        Returns:
            (SimulatedTradeRecord, cash_returned) if closed, else None.
            where cash_returned = margin_blocked + net_pnl
        """
        exit_price_raw: Optional[float] = None
        exit_reason: Optional[str] = None

        sl = position.stop_loss
        tp = position.target_price

        # Check triggers
        sl_hit = sl is not None and bar_low <= sl
        tp_hit = tp is not None and bar_high >= tp

        if sl_hit and tp_hit:
            # Conservative assumption: hit stop-loss first
            # If gap open below SL, exit at bar_open; otherwise at SL
            exit_price_raw = min(bar_open, sl)
            exit_reason = "STOP_LOSS"
        elif sl_hit:
            exit_price_raw = min(bar_open, sl)
            exit_reason = "STOP_LOSS"
        elif tp_hit:
            exit_price_raw = max(bar_open, tp)
            exit_reason = "TARGET_HIT"
        elif is_eod_square_off and position.strategy in ("scalping", "intraday"):
            exit_price_raw = bar_close
            exit_reason = "INTRADAY_SQUARE_OFF"

        if exit_price_raw is None or exit_reason is None:
            # Update current mark-to-market price
            position.current_price = bar_close
            return None

        # Apply exit adverse slippage (SELL fills lower)
        exit_fill_price = self.fees.apply_slippage(exit_price_raw, "SELL")

        # Gross P&L
        if position.direction == "BUY":
            gross_pnl = round((exit_fill_price - position.avg_cost) * position.quantity, 4)
        else:
            gross_pnl = round((position.avg_cost - exit_fill_price) * position.quantity, 4)

        # Exit fees
        exit_fees = self.fees.compute_fees(
            market=position.market,
            direction="SELL",
            filled_price=exit_fill_price,
            quantity=position.quantity,
            is_paper=True,
        )

        entry_fees = position.fees_paid
        total_fees = round(entry_fees + exit_fees, 4)
        net_pnl = round(gross_pnl - exit_fees, 4)
        cash_returned = round(position.margin_blocked + net_pnl, 4)

        duration_days = max(0.0, (timestamp - position.opened_at).total_seconds() / 86400.0)
        pnl_pct = round((gross_pnl / (position.avg_cost * position.quantity)) * 100, 2) if position.avg_cost else 0.0

        trade_record = SimulatedTradeRecord(
            trade_id=f"TRD-{uuid.uuid4().hex[:8].upper()}",
            position_id=position.position_id,
            ticker=position.ticker,
            market=position.market,
            strategy=position.strategy,
            direction=position.direction,
            quantity=position.quantity,
            entry_price=position.entry_price,
            entry_fill_price=position.avg_cost,
            exit_price_raw=exit_price_raw,
            exit_fill_price=exit_fill_price,
            margin_blocked=position.margin_blocked,
            entry_fees=entry_fees,
            exit_fees=exit_fees,
            total_fees=total_fees,
            gross_pnl=gross_pnl,
            net_pnl=net_pnl,
            pnl_pct=pnl_pct,
            entry_time=position.opened_at,
            exit_time=timestamp,
            hold_duration_days=duration_days,
            exit_reason=exit_reason,
        )

        return trade_record, cash_returned
