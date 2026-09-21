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
            # Check if accounts need syncing with config
            row_inr = conn.execute("SELECT initial_cash FROM accounts WHERE account_id = 'paper_inr'").fetchone()
            if row_inr is None or row_inr["initial_cash"] != self.config.paper_trading.virtual_capital_inr:
                self.reset_account_balances()
                return

    def reset_account_balances(self) -> None:
        """Resets virtual paper accounts to configured settings (₹10,000 INR / $1,000 USD)."""
        now = datetime.now(timezone.utc).isoformat()
        with _conn() as conn:
            conn.execute("""
                INSERT INTO accounts (account_id, currency, cash, initial_cash, updated_at)
                VALUES ('paper_inr', 'INR', ?, ?, ?)
                ON CONFLICT(account_id) DO UPDATE SET
                    cash = excluded.cash,
                    initial_cash = excluded.initial_cash,
                    updated_at = excluded.updated_at
            """, (
                self.config.paper_trading.virtual_capital_inr,
                self.config.paper_trading.virtual_capital_inr,
                now
            ))
            conn.execute("""
                INSERT INTO accounts (account_id, currency, cash, initial_cash, updated_at)
                VALUES ('paper_usd', 'USD', ?, ?, ?)
                ON CONFLICT(account_id) DO UPDATE SET
                    cash = excluded.cash,
                    initial_cash = excluded.initial_cash,
                    updated_at = excluded.updated_at
            """, (
                self.config.paper_trading.virtual_capital_usd,
                self.config.paper_trading.virtual_capital_usd,
                now
            ))
            conn.commit()
        logger.info(f"[Paper Engine] Reset balances: ₹{self.config.paper_trading.virtual_capital_inr:,.2f} INR | ${self.config.paper_trading.virtual_capital_usd:,.2f} USD")

    def full_reset(self) -> dict:
        """
        Nuclear reset: archives open positions as cancelled, resets cash to configured amounts.
        Call this via /reset command to start a clean new trading session.
        """
        now = datetime.now(timezone.utc).isoformat()
        with _conn() as conn:
            # Mark all open positions as CANCELLED
            conn.execute("""
                UPDATE positions SET status = 'CANCELLED', closed_at = ?, realized_pnl = 0.0
                WHERE status = 'OPEN'
            """, (now,))
            cancelled = conn.execute("SELECT changes()").fetchone()[0]
            conn.commit()

        self.reset_account_balances()
        logger.warning(f"[Paper Engine] 🔄 FULL RESET: {cancelled} positions cancelled. Fresh start at ₹10,000 INR / $1,000 USD.")
        return {
            "positions_cancelled": cancelled,
            "new_balance_inr": self.config.paper_trading.virtual_capital_inr,
            "new_balance_usd": self.config.paper_trading.virtual_capital_usd,
        }

    def get_account_balance(self, market: Literal["india", "us"]) -> float:
        acc_id = "paper_inr" if market == "india" else "paper_usd"
        with _conn() as conn:
            row = conn.execute("SELECT cash FROM accounts WHERE account_id = ?", (acc_id,)).fetchone()
            return float(row["cash"]) if row else 0.0

    def get_daily_realized_pnl(self, market: Literal["india", "us"]) -> float:
        """Returns total realized PnL for trades closed today."""
        today_date = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        with _conn() as conn:
            row = conn.execute("""
                SELECT SUM(realized_pnl) as total_pnl
                FROM positions
                WHERE status = 'CLOSED' AND market = ? AND closed_at LIKE ?
            """, (market, f"{today_date}%")).fetchone()
            return float(row["total_pnl"]) if (row and row["total_pnl"] is not None) else 0.0

    def update_cash(self, market: Literal["india", "us"], delta: float) -> None:
        acc_id = "paper_inr" if market == "india" else "paper_usd"
        with _conn() as conn:
            conn.execute("""
                UPDATE accounts SET cash = cash + ?, updated_at = ? WHERE account_id = ?
            """, (delta, datetime.now(timezone.utc).isoformat(), acc_id))
            conn.commit()

    def get_daily_stats(self, market: Literal["india", "us"]) -> dict:
        """Returns today's trade stats: wins, losses, total P&L, unrealized P&L."""
        today_date = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        acc_id = "paper_inr" if market == "india" else "paper_usd"

        with _conn() as conn:
            acc_row = conn.execute(
                "SELECT cash, initial_cash FROM accounts WHERE account_id = ?", (acc_id,)
            ).fetchone()

            # Closed trades today
            closed = conn.execute("""
                SELECT realized_pnl FROM positions
                WHERE status = 'CLOSED' AND market = ? AND closed_at LIKE ?
            """, (market, f"{today_date}%")).fetchall()

            # Open positions
            open_pos = conn.execute("""
                SELECT quantity, avg_cost, current_price FROM positions
                WHERE status = 'OPEN' AND market = ?
            """, (market,)).fetchall()

        realized_pnl = sum(r["realized_pnl"] for r in closed if r["realized_pnl"])
        wins = [r for r in closed if (r["realized_pnl"] or 0) > 0]
        losses = [r for r in closed if (r["realized_pnl"] or 0) < 0]
        unrealized = sum(
            ((r["current_price"] or r["avg_cost"]) - r["avg_cost"]) * r["quantity"]
            for r in open_pos
        )

        initial = float(acc_row["initial_cash"]) if acc_row else 0.0
        target = self.config.paper_trading.daily_profit_target_inr if market == "india" \
            else self.config.paper_trading.daily_profit_target_usd

        return {
            "total_trades": len(closed),
            "wins": len(wins),
            "losses": len(losses),
            "realized_pnl": round(realized_pnl, 2),
            "unrealized_pnl": round(unrealized, 2),
            "daily_target": target,
            "target_met": realized_pnl >= target,
            "initial_capital": initial,
        }

    def execute_signal(self, signal: TradeSignal) -> Optional[Order]:
        """
        Executes a paper order from a TradeSignal.
        Deducts cash, records order, and opens a new Position.
        """
        save_signal(signal)

        total_cost = signal.entry_price * signal.quantity
        available_cash = self.get_account_balance(signal.market)

        # Margin & Leverage check for Intraday & Scalping
        is_margin = signal.strategy in ("scalping", "intraday")
        leverage = self.config.paper_trading.intraday_leverage_multiplier if is_margin else 1.0
        margin_required = total_cost / leverage

        if available_cash < margin_required:
            logger.warning(f"[Paper Engine] Rejected {signal.ticker}: Insufficient margin ({available_cash:.2f} < {margin_required:.2f})")
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

        # Deduct Margin from Cash
        actual_margin_blocked = round((filled_price * signal.quantity) / leverage, 2)
        self.update_cash(signal.market, -actual_margin_blocked)

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

                strategy = row["strategy"]

                if ticker not in latest_snapshots:
                    continue

                curr_p = latest_snapshots[ticker].current_price
                conn.execute("UPDATE positions SET current_price = ? WHERE id = ?", (curr_p, pos_id))

                if sl and curr_p <= sl:
                    positions_to_close.append((pos_id, ticker, qty, cost, curr_p, f"STOP LOSS HIT @ {curr_p} (SL: {sl})", strategy))
                elif tgt and curr_p >= tgt:
                    positions_to_close.append((pos_id, ticker, qty, cost, curr_p, f"TARGET HIT @ {curr_p} (TGT: {tgt})", strategy))

            conn.commit()

        # Execute closures outside the previous connection context
        for pos_id, ticker, qty, cost, curr_p, reason, strategy in positions_to_close:
            realized_pnl = round((curr_p - cost) * qty, 2)
            pnl_pct = round((curr_p - cost) / cost * 100, 2)
            now_str = datetime.now(timezone.utc).isoformat()

            with _conn() as conn:
                conn.execute("""
                    UPDATE positions SET status = 'CLOSED', closed_at = ?, realized_pnl = ?
                    WHERE id = ?
                """, (now_str, realized_pnl, pos_id))
                conn.commit()

            is_margin = strategy in ("scalping", "intraday")
            leverage = self.config.paper_trading.intraday_leverage_multiplier if is_margin else 1.0
            margin_returned = (cost * qty) / leverage
            self.update_cash(market, margin_returned + realized_pnl)

            report = f"[EXIT] {ticker} {qty}x closed: {reason} | PnL: {'+' if realized_pnl >= 0 else ''}{realized_pnl} ({pnl_pct:+.2f}%)"
            logger.info(f"[Paper Engine] {report}")
            closed_reports.append(report)

        return closed_reports

    def square_off_intraday(self, market: Literal["india", "us"], latest_snapshots: dict[str, StockSnapshot]) -> list[str]:
        closed_reports = []
        positions_to_close = []

        with _conn() as conn:
            rows = conn.execute("""
                SELECT * FROM positions WHERE status = 'OPEN' AND strategy IN ('intraday', 'scalping') AND market = ?
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

            margin_returned = (cost * qty) / self.config.paper_trading.intraday_leverage_multiplier
            self.update_cash(market, margin_returned + realized_pnl)
            report = f"[SQUARE OFF] {ticker} {qty}x intraday closed @ {curr_p} | PnL: {realized_pnl}"
            logger.warning(f"[Paper Engine] {report}")
            closed_reports.append(report)

        return closed_reports

    def close_all_positions(self, market: Literal["india", "us"]) -> list[str]:
        """Emergency method to close ALL open positions for a market regardless of strategy."""
        closed_reports = []
        with _conn() as conn:
            rows = conn.execute("""
                SELECT * FROM positions WHERE status = 'OPEN' AND market = ?
            """, (market,)).fetchall()
            positions = [dict(r) for r in rows]

        now_str = datetime.now(timezone.utc).isoformat()
        for pos in positions:
            pos_id = pos["id"]
            ticker = pos["ticker"]
            qty = pos["quantity"]
            cost = pos["avg_cost"]
            curr_p = pos["current_price"] or cost
            strategy = pos.get("strategy", "swing")
            realized_pnl = round((curr_p - cost) * qty, 2)

            with _conn() as conn:
                conn.execute("""
                    UPDATE positions SET status = 'CLOSED', closed_at = ?, realized_pnl = ? WHERE id = ?
                """, (now_str, realized_pnl, pos_id))
                conn.commit()

            is_margin = strategy in ("scalping", "intraday")
            leverage = self.config.paper_trading.intraday_leverage_multiplier if is_margin else 1.0
            margin_returned = (cost * qty) / leverage
            self.update_cash(market, margin_returned + realized_pnl)
            report = f"[EMERGENCY CLOSE] {ticker} {qty}x closed @ {curr_p} | PnL: {realized_pnl}"
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
