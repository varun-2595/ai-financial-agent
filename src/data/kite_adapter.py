"""
Zerodha Kite Connect Adapter.
Provides real-time quotes, margins, holdings, and order placement via Kite Connect API.
Safely wraps kiteconnect; falls back gracefully when credentials are not yet configured.
"""
from __future__ import annotations

import os
from typing import Any, Optional

from src.utils.logger import logger


class KiteAdapter:
    def __init__(
        self,
        api_key: Optional[str] = None,
        access_token: Optional[str] = None,
    ):
        self.api_key = api_key or os.getenv("KITE_API_KEY")
        self.access_token = access_token or os.getenv("KITE_ACCESS_TOKEN")
        self.kite: Optional[Any] = None
        self._init_client()

    def _init_client(self) -> None:
        if not self.api_key or not self.access_token:
            logger.info("[Kite] API key or Access token not configured. Running in simulated/mock mode.")
            return

        try:
            from kiteconnect import KiteConnect

            self.kite = KiteConnect(api_key=self.api_key)
            self.kite.set_access_token(self.access_token)
            logger.success("[Kite] KiteConnect client successfully initialized.")
        except ImportError:
            logger.warning("[Kite] 'kiteconnect' package not installed. Live orders disabled.")
        except Exception as exc:
            logger.error(f"[Kite] Failed to initialize KiteConnect: {exc}")

    def is_connected(self) -> bool:
        return self.kite is not None

    def get_margins(self) -> dict:
        """Fetch live equity balance / margins."""
        if not self.kite:
            return {"equity": {"net": 0.0, "available": 0.0}}
        try:
            return self.kite.margins("equity")
        except Exception as exc:
            logger.error(f"[Kite] Failed to fetch margins: {exc}")
            return {}

    def get_positions(self) -> dict:
        """Fetch open net and day positions."""
        if not self.kite:
            return {"net": [], "day": []}
        try:
            return self.kite.positions()
        except Exception as exc:
            logger.error(f"[Kite] Failed to fetch positions: {exc}")
            return {"net": [], "day": []}

    def place_order(
        self,
        tradingsymbol: str,
        transaction_type: str, # "BUY" or "SELL"
        quantity: int,
        order_type: str = "MARKET", # "MARKET", "LIMIT", "SL"
        price: Optional[float] = None,
        trigger_price: Optional[float] = None,
        product: str = "MIS", # "MIS" for Intraday, "CNC" for delivery
    ) -> Optional[str]:
        """
        Places an order on NSE via Kite Connect.
        Returns Kite order_id if successful.
        """
        if not self.kite:
            logger.warning(f"[Kite] Live order placement bypassed for {tradingsymbol} (Kite offline).")
            return None

        # Clean NSE suffix e.g. "RELIANCE.NS" -> "RELIANCE"
        clean_symbol = tradingsymbol.replace(".NS", "").replace(".BO", "")

        try:
            order_id = self.kite.place_order(
                variety=self.kite.VARIETY_REGULAR,
                exchange=self.kite.EXCHANGE_NSE,
                tradingsymbol=clean_symbol,
                transaction_type=transaction_type,
                quantity=quantity,
                product=product,
                order_type=order_type,
                price=price,
                trigger_price=trigger_price,
            )
            logger.success(f"[Kite] Live order placed for {clean_symbol}: Order ID {order_id}")
            return str(order_id)
        except Exception as exc:
            logger.error(f"[Kite] Failed to place order for {clean_symbol}: {exc}")
            return None
