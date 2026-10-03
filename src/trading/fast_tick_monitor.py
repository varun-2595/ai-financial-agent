"""
High-Frequency Fast Tick Position Monitor.
Inspired by BeeBots 1-second Tick Loop architecture.

Runs a continuous lightweight evaluation loop (every 2-3s) on currently open positions.
Executes Stop-Loss exits, Multi-stage +1.5R partial profit bookings (50%), and Breakeven
Stop shifts in real time without waiting for the 15-minute cron scanner.
"""
from __future__ import annotations

import threading
import time
from typing import Callable, Dict, List, Literal, Optional

from src.data.coalescer import market_coalescer
from src.data.models import Quote, StockSnapshot
from src.db.trading_store import get_open_positions
from src.notifier.telegram_bot import TelegramNotifier
from src.trading.paper_engine import PaperTradingEngine
from src.utils.logger import logger
from src.utils.market_hours import is_nse_open, is_nyse_open


class FastTickMonitor:
    """
    Continuous sub-second position risk monitor.
    Decoupled from heavy LLM & screening pipelines to guarantee ultra-fast trade exits.
    """

    def __init__(
        self,
        engine: Optional[PaperTradingEngine] = None,
        tick_interval_seconds: float = 3.0,
        price_fetcher: Optional[Callable[[str, str], Optional[float]]] = None,
        enforce_market_hours: bool = True,
    ):
        self.engine = engine or PaperTradingEngine()
        self.tick_interval = tick_interval_seconds
        self.price_fetcher = price_fetcher or self._default_fetch_price
        self.enforce_market_hours = enforce_market_hours
        self._stop_event = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self.notifier = TelegramNotifier()
        self.ticks_executed = 0

    def start(self) -> None:
        """Starts the Fast Tick Monitor in a background daemon thread."""
        if self._thread and self._thread.is_alive():
            logger.warning("[Fast Tick Monitor] Already running.")
            return

        self._stop_event.clear()
        self._thread = threading.Thread(
            target=self._run_loop,
            name="FastTickPositionMonitor",
            daemon=True,
        )
        self._thread.start()
        logger.info(f"⚡ [Fast Tick Monitor] Started daemon thread (Tick interval: {self.tick_interval:.1f}s)")

    def stop(self) -> None:
        """Signals the background monitor loop to stop and waits for exit."""
        self._stop_event.set()
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=5.0)
            logger.info("⚡ [Fast Tick Monitor] Stopped daemon thread.")

    def is_running(self) -> bool:
        return self._thread is not None and self._thread.is_alive() and not self._stop_event.is_set()

    def _run_loop(self) -> None:
        """Internal continuous polling loop."""
        while not self._stop_event.is_set():
            try:
                self.tick_once()
            except Exception as e:
                logger.error(f"[Fast Tick Monitor] Error in tick loop: {e}", exc_info=True)

            self._stop_event.wait(self.tick_interval)

    def tick_once(self) -> List[str]:
        """
        Executes a single high-frequency risk check across all open positions.
        Returns a list of close/partial report strings.
        """
        self.ticks_executed += 1
        open_positions = get_open_positions()
        if not open_positions:
            return []

        all_reports: List[str] = []

        # Group open positions by market
        markets: Dict[Literal["india", "us"], List[dict]] = {"india": [], "us": []}
        for pos in open_positions:
            m = str(pos.get("market", "india")).lower()
            if m in markets:
                markets[m].append(pos)  # type: ignore

        for m_key, pos_list in markets.items():
            if not pos_list:
                continue

            # Respect market hours unless explicitly disabled (e.g. testing or paper sim)
            if self.enforce_market_hours:
                if m_key == "india" and not is_nse_open():
                    continue
                if m_key == "us" and not is_nyse_open():
                    continue

            # Fetch latest prices for these open positions using request coalescer
            snapshots: Dict[str, StockSnapshot] = {}
            for pos in pos_list:
                ticker = pos["ticker"]
                try:
                    price = market_coalescer.coalesce(
                        key=f"fast_tick_price:{m_key}:{ticker}",
                        fetch_fn=lambda t=ticker, m=m_key: self.price_fetcher(t, m),
                        ttl_seconds=1.5,
                    )
                    if price and price > 0:
                        snapshots[ticker] = StockSnapshot(
                            ticker=ticker,
                            market=m_key,
                            currency="INR" if m_key == "india" else "USD",
                            current_price=price,
                            history=[],
                        )
                except Exception as e:
                    logger.debug(f"[Fast Tick Monitor] Price fetch error for {ticker}: {e}")

            if not snapshots:
                continue

            # Evaluate positions against Stop-Loss, Take-Profit, and Multi-Stage +1.5R breakeven
            reports = self.engine.evaluate_open_positions(
                market=m_key,
                latest_snapshots=snapshots,
            )

            if reports:
                for rep in reports:
                    all_reports.append(rep)
                    logger.success(f"[Fast Tick Monitor] ⚡ ACTION EXECUTED: {rep}")
                    try:
                        self.notifier.send_message(
                            f"⚡ <b>Fast Tick Risk Execution</b>\n<code>{rep}</code>"
                        )
                    except Exception as notify_err:
                        logger.warning(f"[Fast Tick Monitor] Telegram alert failed: {notify_err}")

        return all_reports

    def _default_fetch_price(self, ticker: str, market: str) -> Optional[float]:
        """Fetch current price using existing fetchers or yfinance fast ticker."""
        try:
            if market == "india":
                from src.data.fetcher_india import normalize_india_ticker
                import yfinance as yf
                clean = normalize_india_ticker(ticker)
                t = yf.Ticker(clean)
                fast = t.fast_info
                price = getattr(fast, "last_price", None)
                if price:
                    return float(price)
            else:
                from src.data.fetcher_us import fetch_us_snapshot
                snap = fetch_us_snapshot(ticker)
                if snap:
                    return snap.current_price
        except Exception:
            pass
        return None
