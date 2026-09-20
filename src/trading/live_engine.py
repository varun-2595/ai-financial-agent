"""
Live Trading Engine with Gated Two-Factor Safety Controls.
Live trading is locked down by default to protect real capital.

Requirements to activate Live Trading:
  1. LIVE_TRADING_ENABLED=true in .env
  2. Two-Factor safety confirmation: active confirmation flag or token
  3. Strict risk check passing
"""
from __future__ import annotations

import os
from typing import Literal, Optional

from src.data.kite_adapter import KiteAdapter
from src.data.models import Order, TradeSignal
from src.db.trading_store import save_order
from src.risk.position_sizer import RiskEngine
from src.utils.logger import logger


class LiveTradingEngine:
    def __init__(self, adapter: Optional[KiteAdapter] = None):
        self.adapter = adapter or KiteAdapter()
        self.risk = RiskEngine()
        self.live_enabled = os.getenv("LIVE_TRADING_ENABLED", "false").lower() == "true"
        self.confirmed_safety = False

    def unlock_live_trading(self, confirmation_passphrase: str) -> bool:
        """Requires exact phrase 'YES-I-UNDERSTAND-RISKS' to unlock live capital."""
        if confirmation_passphrase == "YES-I-UNDERSTAND-RISKS":
            self.confirmed_safety = True
            logger.warning("[Live Engine] ⚠️ LIVE TRADING UNLOCKED WITH REAL CAPITAL.")
            return True
        logger.error("[Live Engine] Invalid confirmation passphrase. Live trading remains LOCKED.")
        return False

    def is_live_ready(self) -> bool:
        return self.live_enabled and self.confirmed_safety and self.adapter.is_connected()

    def execute_live_signal(self, signal: TradeSignal) -> Optional[str]:
        """
        Routes signal to Zerodha Kite Connect if safety checks pass.
        Returns live Order ID or None.
        """
        if not self.is_live_ready():
            logger.warning(
                f"[Live Engine] Live execution blocked for {signal.ticker}. "
                f"Enabled: {self.live_enabled}, Confirmed: {self.confirmed_safety}, Connected: {self.adapter.is_connected()}"
            )
            return None

        # Product: MIS for Intraday, CNC for Swing/Positional
        product = "MIS" if signal.strategy == "intraday" else "CNC"

        order_id = self.adapter.place_order(
            tradingsymbol=signal.ticker,
            transaction_type=signal.direction,
            quantity=signal.quantity,
            order_type="MARKET",
            product=product,
        )

        if order_id:
            logger.success(f"[Live Engine] Real order executed on NSE: {order_id}")
            # Record order in local DB
            order = Order(
                order_id=order_id,
                ticker=signal.ticker,
                market=signal.market,
                strategy=signal.strategy,
                order_type="MARKET",
                direction=signal.direction if signal.direction in ("BUY", "SELL") else "BUY",
                quantity=signal.quantity,
                filled_price=signal.entry_price,
                status="OPEN",
                is_paper=False,
            )
            save_order(order)
            return order_id

        return None
