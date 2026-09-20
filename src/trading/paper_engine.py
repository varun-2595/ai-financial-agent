"""
Paper Trading Execution Engine.
Simulates realistic trade fills, tracks active positions, triggers automatic stop loss
and take profit limits, and handles mandatory intraday square-off.
"""
from __future__ import annotations

import uuid
from datetime import datetime, timezone
from typing import Literal, Optional

from src.data.models import Order, Position, StockSnapshot, TradeSignal
from src.db.trading_store import init_trading_db, save_order, save_signal, _conn
from src.utils.config import get_config
from src.utils.logger import logger


class PaperTradingEngine:
    def __init__(self):
        self.config = get_config()
        init_trading_db()
        self._ensure_accounts()

    def _ensure_accounts(self) -> None:
        with _conn() as conn:
            # India Account
            conn.execute("""
                INSERT OR IGNORE INTO accounts (account_id, currency, cash, initial_cash, updated_at)
                VALUES ('paper_inr', 'INR', ?, ?, ?)
            """, (
                self.config.paper_trading.virtual_capital_inr,
                self.config.paper_trading.virtual_capital_inr,
                datetime.now(timezone.utc).isoformat()
            ))
            # US Account
            conn.execute("""
                INSERT OR IGNORE INTO accounts (account_id, currency, cash, initial_cash, updated_at)
                VALUES ('paper_usd', 'USD', ?, ?, ?)
            """, (
                self.config.paper_trading.virtual_capital_usd,
                self.config.paper_trading.virtual_capital_usd,
                datetime.now(timezone.utc).isoformat()
            ))
            conn.commit()

    def get_account_balance(self, market: Literal["india", "us"]) -> float:
        acc_id = "paper_inr" if market == "india" else "paper_usd"
        with _conn() as conn:
            row = conn.execute("SELECT cash FROM accounts WHERE account_id = ?", (acc_id,)).fetchone()
            return float(row["cash"]) if row else 0.0

    def update_cash(self, market: Literal["india", "us"], delta: float) -> None:
        acc_id = "paper_inr" if market == "india" else "paper_usd"
        with _conn() as conn:
            conn.execute("""
                UPDATE accounts SET cash = cash + ?, updated_at = ? WHERE account_id = ?
            """, (delta, datetime.now(timezone.utc).isoformat(), acc_id))
            conn.commit()

    def execute_signal(self, signal: TradeSignal) -> Optional[Order]:
        """
        Executes a paper order from a TradeSignal.
        Deducts cash, records order, and opens a new Position.
        """
        save_signal(signal)

        total_cost = signal.entry_price * signal.quantity
        available_cash = self.get_account_balance(signal.market)

        if available_cash < total_cost:
            logger.warning(f"[Paper Engine] Rejected {signal.ticker}: Insufficient cash ({available_cash:.2f} < {total_cost:.2f})")
            return None

        # Simulate Order Fill (Slight 0.05% slippage simulation)
        slippage = 1.0005 if signal.direction == "BUY" else 0.9995
        filled_price = round(signal.entry_price * slippage, 2)
        order_id = f"ORD-{uuid.uuid4().hex[:8].upper()}"

        order = Order(
            order_id=order_id,
            ticker=signal.ticker,
            market=signal.market,
            strategy=signal.strategy,
            order_type="MARKET",
            direction=signal.direction if signal.direction in ("BUY", "SELL") else "BUY",
            quantity=signal.quantity,
            filled_price=filled_price,
            status="FILLED",
            is_paper=True,
            filled_at=datetime.now(timezone.utc),
        )
        save_order(order)

        # Deduct Cash
        self.update_cash(signal.market, - (filled_price * signal.quantity))

        # Open Position
        with _conn() as conn:
            conn.execute("""
                INSERT INTO positions (ticker, market, strategy, direction, quantity, avg_cost, current_price, stop_loss, target_price, status, is_paper, opened_at)
                VALUES (?, ?, ?, 'LONG', ?, ?, ?, ?, ?, 'OPEN', 1, ?)
            """, (
                signal.ticker, signal.market, signal.strategy, signal.quantity,
                filled_price, filled_price, signal.stop_loss, signal.target_price,
                datetime.now(timezone.utc).isoformat()
            ))
            conn.commit()

        logger.success(f"[Paper Engine] 🚀 FILLED {order.quantity}x {order.ticker} @ {filled_price} (Order ID: {order_id})")
        return order

    def evaluate_open_positions(self, market: Literal["india", "us"], latest_snapshots: dict[str, StockSnapshot]) -> list[str]:
        closed_reports = []
        positions_to_close = []

        with _conn() as conn:
            rows = conn.execute("""
                SELECT * FROM positions WHERE status = 'OPEN' AND market = ?
            """, (market,)).fetchall()

            for row in rows:
                pos_id = row["id"]
                ticker = row["ticker"]
                qty = row["quantity"]
                cost = row["avg_cost"]
                sl = row["stop_loss"]
                tgt = row["target_price"]

                if ticker not in latest_snapshots:
                    continue

                curr_p = latest_snapshots[ticker].current_price
                conn.execute("UPDATE positions SET current_price = ? WHERE id = ?", (curr_p, pos_id))

                if sl and curr_p <= sl:
                    positions_to_close.append((pos_id, ticker, qty, cost, curr_p, f"STOP LOSS HIT @ {curr_p} (SL: {sl})"))
                elif tgt and curr_p >= tgt:
                    positions_to_close.append((pos_id, ticker, qty, cost, curr_p, f"TARGET HIT @ {curr_p} (TGT: {tgt})"))

            conn.commit()

        # Execute closures outside the previous connection context
        for pos_id, ticker, qty, cost, curr_p, reason in positions_to_close:
            realized_pnl = round((curr_p - cost) * qty, 2)
            pnl_pct = round((curr_p - cost) / cost * 100, 2)
            now_str = datetime.now(timezone.utc).isoformat()

            with _conn() as conn:
                conn.execute("""
                    UPDATE positions SET status = 'CLOSED', closed_at = ?, realized_pnl = ?
                    WHERE id = ?
                """, (now_str, realized_pnl, pos_id))
                conn.commit()

            proceeds = curr_p * qty
            self.update_cash(market, proceeds)

            report = f"[EXIT] {ticker} {qty}x closed: {reason} | PnL: {'+' if realized_pnl >= 0 else ''}{realized_pnl} ({pnl_pct:+.2f}%)"
            logger.info(f"[Paper Engine] {report}")
            closed_reports.append(report)

        return closed_reports

    def square_off_intraday(self, market: Literal["india", "us"], latest_snapshots: dict[str, StockSnapshot]) -> list[str]:
        closed_reports = []
        positions_to_close = []

        with _conn() as conn:
            rows = conn.execute("""
                SELECT * FROM positions WHERE status = 'OPEN' AND strategy = 'intraday' AND market = ?
            """, (market,)).fetchall()

            for row in rows:
                pos_id = row["id"]
                ticker = row["ticker"]
                qty = row["quantity"]
                cost = row["avg_cost"]
                curr_p = latest_snapshots[ticker].current_price if ticker in latest_snapshots else cost
                positions_to_close.append((pos_id, ticker, qty, cost, curr_p))

        for pos_id, ticker, qty, cost, curr_p in positions_to_close:
            realized_pnl = round((curr_p - cost) * qty, 2)
            now_str = datetime.now(timezone.utc).isoformat()

            with _conn() as conn:
                conn.execute("""
                    UPDATE positions SET status = 'CLOSED', closed_at = ?, realized_pnl = ? WHERE id = ?
                """, (now_str, realized_pnl, pos_id))
                conn.commit()

            self.update_cash(market, curr_p * qty)
            report = f"[SQUARE OFF] {ticker} {qty}x intraday closed @ {curr_p} | PnL: {realized_pnl}"
            logger.warning(f"[Paper Engine] {report}")
            closed_reports.append(report)

        return closed_reports

    def get_portfolio_summary(self, market: Literal["india", "us"]) -> dict:
        cash = self.get_account_balance(market)
        with _conn() as conn:
            rows = conn.execute("""
                SELECT * FROM positions WHERE status = 'OPEN' AND market = ?
            """, (market,)).fetchall()
            positions = [dict(r) for r in rows]

        invested_val = sum(p["quantity"] * (p["current_price"] or p["avg_cost"]) for p in positions)
        total_val = cash + invested_val
        currency = "₹" if market == "india" else "$"

        return {
            "market": market,
            "currency": currency,
            "cash": round(cash, 2),
            "invested": round(invested_val, 2),
            "total_value": round(total_val, 2),
            "open_positions_count": len(positions),
            "positions": positions,
        }
