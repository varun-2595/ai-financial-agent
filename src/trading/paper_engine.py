"""
Paper Trading Execution Engine — Phase 1: Correct Portfolio Accounting.

Accounting model
================
On BUY (entry)
--------------
  filled_price  = signal.entry_price * (1 + slippage_pct)   # slippage on entry
  margin_blocked = filled_price * qty / leverage             # cash actually reserved
  entry_fees     = fee_schedule.compute_fees(BUY, ...)
  cash_deducted  = margin_blocked + entry_fees

  accounts.cash            -= cash_deducted
  accounts.reserved_margin += margin_blocked
  positions.margin_blocked  = margin_blocked
  positions.fees_paid       = entry_fees
  ledger: MARGIN_BLOCK (-margin_blocked), FEE (-entry_fees)

On SELL / close (exit)
-----------------------
  exit_price  = current_price * (1 - slippage_pct)          # slippage on exit
  gross_pnl   = (exit_price - avg_cost) * qty               # for LONG positions
  exit_fees   = fee_schedule.compute_fees(SELL, ...)
  net_pnl     = gross_pnl - exit_fees

  cash_returned = margin_blocked + net_pnl

  accounts.cash            += cash_returned
  accounts.reserved_margin -= margin_blocked
  positions.realized_pnl    = net_pnl  (fees already subtracted)
  positions.fees_paid      += exit_fees
  ledger: MARGIN_RELEASE (+margin_blocked), FEE (-exit_fees), REALIZED_PNL (+/- net_pnl)

Portfolio NAV
-------------
  NAV = cash + reserved_margin + unrealized_pnl_of_open_positions

  This equals the true equity the account controls:
    - cash: liquid, free to use
    - reserved_margin: locked in open trades but still yours
    - unrealized_pnl: paper gain/loss on open trades
  The leveraged "notional" value of open trades is NOT added to NAV.

Buying Power
------------
  buying_power        = cash - 0                  (cash is already net of reserved_margin)
  effective_bp_intraday = cash * leverage
"""
from __future__ import annotations

import uuid
from datetime import datetime, timezone
from typing import Literal, Optional

from src.data.models import Order, Position, StockSnapshot, TradeSignal
from src.db.trading_store import (
    _conn,
    append_ledger,
    get_ledger,
    get_open_positions,
    init_trading_db,
    save_order,
    save_signal,
)
from src.risk.models import PortfolioRiskState, RiskDecision
from src.risk.portfolio_risk_manager import PortfolioRiskManager
from src.trading.fees import DEFAULT_FEE_SCHEDULE, FeeSchedule
from src.utils.config import get_config
from src.utils.logger import logger


class PaperTradingEngine:
    def __init__(
        self,
        fee_schedule: Optional[FeeSchedule] = None,
        risk_manager: Optional[PortfolioRiskManager] = None,
    ):
        self.config = get_config()
        self.fees = fee_schedule or DEFAULT_FEE_SCHEDULE
        self.risk_manager = risk_manager or PortfolioRiskManager()
        init_trading_db()
        self._ensure_accounts()

    # ── Account bootstrap ──────────────────────────────────────────────────────

    def _ensure_accounts(self) -> None:
        with _conn() as conn:
            row_inr = conn.execute(
                "SELECT initial_cash FROM accounts WHERE account_id = 'paper_inr'"
            ).fetchone()
            if row_inr is None or row_inr["initial_cash"] != self.config.paper_trading.virtual_capital_inr:
                self.reset_account_balances()

    def reset_account_balances(self) -> None:
        """Reset virtual paper accounts to configured capital. Preserves open positions."""
        now = datetime.now(timezone.utc).isoformat()
        cfg = self.config.paper_trading
        with _conn() as conn:
            for acc_id, currency, capital in [
                ("paper_inr", "INR", cfg.virtual_capital_inr),
                ("paper_usd", "USD", cfg.virtual_capital_usd),
            ]:
                conn.execute("""
                    INSERT INTO accounts (account_id, currency, cash, initial_cash, reserved_margin, updated_at)
                    VALUES (?, ?, ?, ?, 0.0, ?)
                    ON CONFLICT(account_id) DO UPDATE SET
                        cash            = excluded.cash,
                        initial_cash    = excluded.initial_cash,
                        reserved_margin = 0.0,
                        updated_at      = excluded.updated_at
                """, (acc_id, currency, capital, capital, now))
                append_ledger(conn, acc_id, "RESET", capital, capital,
                              description=f"Account reset to {capital} {currency}")
            conn.commit()
        logger.info(
            f"[Paper Engine] Reset balances: ₹{cfg.virtual_capital_inr:,.2f} INR"
            f" | ${cfg.virtual_capital_usd:,.2f} USD"
        )

    def full_reset(self) -> dict:
        """
        Nuclear reset: archives all open positions as CANCELLED, resets cash.
        Use via /reset Telegram command to start a clean session.
        """
        now = datetime.now(timezone.utc).isoformat()
        with _conn() as conn:
            conn.execute("""
                UPDATE positions SET status = 'CANCELLED', closed_at = ?, realized_pnl = 0.0
                WHERE status = 'OPEN'
            """, (now,))
            cancelled = conn.execute("SELECT changes()").fetchone()[0]
            conn.commit()

        self.reset_account_balances()
        logger.warning(
            f"[Paper Engine] 🔄 FULL RESET: {cancelled} positions cancelled."
            " Fresh start."
        )
        return {
            "positions_cancelled": cancelled,
            "new_balance_inr": self.config.paper_trading.virtual_capital_inr,
            "new_balance_usd": self.config.paper_trading.virtual_capital_usd,
        }

    # ── Account queries ────────────────────────────────────────────────────────

    def _acc_id(self, market: Literal["india", "us"]) -> str:
        return "paper_inr" if market == "india" else "paper_usd"

    def get_account_balance(self, market: Literal["india", "us"]) -> float:
        """Returns free cash (not including reserved margin)."""
        with _conn() as conn:
            row = conn.execute(
                "SELECT cash FROM accounts WHERE account_id = ?", (self._acc_id(market),)
            ).fetchone()
            return float(row["cash"]) if row else 0.0

    def get_reserved_margin(self, market: Literal["india", "us"]) -> float:
        """Returns total margin currently blocked by open positions."""
        with _conn() as conn:
            row = conn.execute(
                "SELECT reserved_margin FROM accounts WHERE account_id = ?",
                (self._acc_id(market),)
            ).fetchone()
            return float(row["reserved_margin"]) if row else 0.0

    def get_buying_power(self, market: Literal["india", "us"]) -> dict:
        """
        Returns buying power breakdown:
          - cash: free liquid cash
          - buying_power_equity: cash (1:1 for swing/positional)
          - buying_power_intraday: cash × leverage (for scalping/intraday)
        """
        cash = self.get_account_balance(market)
        leverage = self.config.paper_trading.intraday_leverage_multiplier
        return {
            "cash": round(cash, 2),
            "buying_power_equity": round(cash, 2),
            "buying_power_intraday": round(cash * leverage, 2),
            "leverage": leverage,
        }

    def get_daily_realized_pnl(self, market: Literal["india", "us"]) -> float:
        today_date = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        with _conn() as conn:
            row = conn.execute("""
                SELECT SUM(realized_pnl) as total_pnl
                FROM positions
                WHERE status = 'CLOSED' AND market = ? AND closed_at LIKE ?
            """, (market, f"{today_date}%")).fetchone()
            return float(row["total_pnl"]) if (row and row["total_pnl"] is not None) else 0.0

    def get_daily_stats(self, market: Literal["india", "us"]) -> dict:
        """Returns today's trade stats: wins, losses, P&L, unrealized P&L."""
        today_date = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        acc_id = self._acc_id(market)

        with _conn() as conn:
            acc_row = conn.execute(
                "SELECT cash, initial_cash, reserved_margin FROM accounts WHERE account_id = ?",
                (acc_id,)
            ).fetchone()
            closed = conn.execute("""
                SELECT realized_pnl FROM positions
                WHERE status = 'CLOSED' AND market = ? AND closed_at LIKE ?
            """, (market, f"{today_date}%")).fetchall()
            open_pos = conn.execute("""
                SELECT quantity, avg_cost, current_price, direction FROM positions
                WHERE status = 'OPEN' AND market = ?
            """, (market,)).fetchall()

        realized_pnl = sum(r["realized_pnl"] for r in closed if r["realized_pnl"])
        wins = [r for r in closed if (r["realized_pnl"] or 0) > 0]
        losses = [r for r in closed if (r["realized_pnl"] or 0) < 0]

        # Unrealized P&L: direction-aware
        unrealized = 0.0
        for r in open_pos:
            curr = r["current_price"] or r["avg_cost"]
            if r["direction"] == "LONG":
                unrealized += (curr - r["avg_cost"]) * r["quantity"]
            else:
                unrealized += (r["avg_cost"] - curr) * r["quantity"]

        initial = float(acc_row["initial_cash"]) if acc_row else 0.0
        target = (self.config.paper_trading.daily_profit_target_inr
                  if market == "india"
                  else self.config.paper_trading.daily_profit_target_usd)

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

    def get_portfolio_risk_state(self, market: Literal["india", "us"]) -> PortfolioRiskState:
        """Construct live portfolio risk state snapshot for PortfolioRiskManager."""
        nav_dict = self.get_portfolio_nav(market)
        nav_val = float(nav_dict.get("nav", 1000.0))
        cash = self.get_account_balance(market)
        reserved_margin = self.get_reserved_margin(market)
        open_pos = get_open_positions(market)
        daily_stats = self.get_daily_stats(market)
        initial_cash = float(daily_stats.get("initial_capital", nav_val))

        gross_exposure = 0.0
        sector_exposures: dict[str, float] = {}
        sector_counts: dict[str, int] = {}

        for p in open_pos:
            qty = p.get("quantity", 0)
            price = p.get("current_price") or p.get("avg_cost", 0.0)
            val = qty * price
            gross_exposure += val
            sec = p.get("sector") or "General"
            sector_exposures[sec] = sector_exposures.get(sec, 0.0) + val
            sector_counts[sec] = sector_counts.get(sec, 0) + 1

        peak_nav = max(initial_cash, nav_val)
        drawdown_pct = max(0.0, (peak_nav - nav_val) / peak_nav) if peak_nav > 0 else 0.0
        leverage = gross_exposure / nav_val if nav_val > 0 else 0.0

        currency = "INR" if market == "india" else "USD"

        return PortfolioRiskState(
            market=market,
            currency=currency,
            nav=nav_val,
            peak_nav=peak_nav,
            cash=cash,
            reserved_margin=reserved_margin,
            gross_exposure=gross_exposure,
            daily_realized_pnl=daily_stats.get("realized_pnl", 0.0),
            daily_unrealized_pnl=daily_stats.get("unrealized_pnl", 0.0),
            current_drawdown_pct=drawdown_pct,
            current_leverage=leverage,
            portfolio_beta=1.0,
            sector_exposures=sector_exposures,
            sector_position_counts=sector_counts,
            open_positions=open_pos,
        )

    # ── Internal cash helpers ──────────────────────────────────────────────────

    def _update_cash(
        self,
        conn: sqlite3.Connection,   # type: ignore[name-defined]  # imported lazily below
        acc_id: str,
        delta: float,
        delta_margin: float = 0.0,
    ) -> float:
        """
        Apply delta to cash and delta_margin to reserved_margin atomically.
        Returns the new cash balance.
        """
        import sqlite3 as _sqlite3
        conn.execute("""
            UPDATE accounts
            SET cash            = cash + ?,
                reserved_margin = MAX(0, reserved_margin + ?),
                updated_at      = ?
            WHERE account_id = ?
        """, (delta, delta_margin, datetime.now(timezone.utc).isoformat(), acc_id))
        row = conn.execute("SELECT cash FROM accounts WHERE account_id = ?", (acc_id,)).fetchone()
        return float(row["cash"])

    # ── Trade execution ────────────────────────────────────────────────────────

    def execute_signal(self, signal: TradeSignal) -> Optional[Order]:
        """
        Execute a paper BUY order from a TradeSignal after deterministic Risk Engine gatekeeper.

        Order execution flow:
        AI Proposal → Portfolio Manager → PortfolioRiskManager (APPROVE/REDUCE/REJECT) → Execution.
        """
        save_signal(signal)

        if signal.direction not in ("BUY", "SELL"):
            logger.warning(f"[Paper Engine] Skipping signal with direction={signal.direction}")
            return None

        # ── Deterministic Portfolio-Level Risk Gate ─────────────────────────
        risk_state = self.get_portfolio_risk_state(signal.market)
        risk_eval = self.risk_manager.evaluate_trade(signal, risk_state)

        if risk_eval.decision == RiskDecision.REJECT:
            logger.warning(f"[Paper Engine] ❌ Trade REJECTED by Risk Engine: {risk_eval.reason}")
            return None

        if risk_eval.decision == RiskDecision.REDUCE:
            logger.info(f"[Paper Engine] ⚠️ Trade REDUCED by Risk Engine: {risk_eval.reason}")
            signal.quantity = risk_eval.approved_quantity

        # ── Slippage on fill price ─────────────────────────────────────────
        filled_price = self.fees.apply_slippage(signal.entry_price, signal.direction)

        # ── Leverage & margin calculation ──────────────────────────────────
        is_margin = signal.strategy in ("scalping", "intraday")
        leverage = self.config.paper_trading.intraday_leverage_multiplier if is_margin else 1.0
        margin_blocked = round((filled_price * signal.quantity) / leverage, 4)

        # ── Fees ───────────────────────────────────────────────────────────
        entry_fees = self.fees.compute_fees(
            market=signal.market,
            direction="BUY",
            filled_price=filled_price,
            quantity=signal.quantity,
            is_paper=True,
        )
        total_deduction = round(margin_blocked + entry_fees, 4)

        # ── Cash availability check ────────────────────────────────────────
        available_cash = self.get_account_balance(signal.market)
        if available_cash < total_deduction:
            logger.warning(
                f"[Paper Engine] Rejected {signal.ticker}: Insufficient cash "
                f"({available_cash:.2f} < {total_deduction:.2f} needed)"
            )
            return None

        # ── Build & persist order ──────────────────────────────────────────
        order_id = f"ORD-{uuid.uuid4().hex[:8].upper()}"
        order = Order(
            order_id=order_id,
            ticker=signal.ticker,
            market=signal.market,
            strategy=signal.strategy,
            order_type="MARKET",
            direction="BUY",
            quantity=signal.quantity,
            filled_price=filled_price,
            status="FILLED",
            is_paper=True,
            filled_at=datetime.now(timezone.utc),
        )
        save_order(order, fees=entry_fees, slippage_pct=self.fees.slippage_pct)

        # ── Atomic DB update: cash, reserved_margin, position, ledger ─────
        acc_id = self._acc_id(signal.market)
        now_str = datetime.now(timezone.utc).isoformat()
        with _conn() as conn:
            # 1. Deduct cash & block margin
            new_cash = self._update_cash(conn, acc_id,
                                          delta=-total_deduction,
                                          delta_margin=+margin_blocked)

            # 2. Open position record
            cursor = conn.execute("""
                INSERT INTO positions
                    (ticker, market, strategy, direction, quantity, avg_cost, current_price,
                     stop_loss, target_price, margin_blocked, fees_paid, status, is_paper, opened_at)
                VALUES (?, ?, ?, 'LONG', ?, ?, ?, ?, ?, ?, ?, 'OPEN', 1, ?)
            """, (
                signal.ticker, signal.market, signal.strategy,
                signal.quantity, filled_price, filled_price,
                signal.stop_loss, signal.target_price,
                margin_blocked, entry_fees, now_str
            ))
            pos_id = cursor.lastrowid

            # 3. Ledger entries
            append_ledger(conn, acc_id, "MARGIN_BLOCK", -margin_blocked, new_cash + entry_fees,
                          ref_order_id=order_id, ref_position_id=pos_id,
                          description=f"BUY {signal.quantity}×{signal.ticker} @ {filled_price}")
            append_ledger(conn, acc_id, "FEE", -entry_fees, new_cash,
                          ref_order_id=order_id, ref_position_id=pos_id,
                          description=f"Entry fees {signal.ticker}")
            conn.commit()

        logger.success(
            f"[Paper Engine] 🚀 FILLED {order.quantity}×{order.ticker} @ {filled_price}"
            f" | Margin: {margin_blocked:.2f} | Fee: {entry_fees:.4f}"
            f" | Cash left: {available_cash - total_deduction:.2f}"
            f" (Order: {order_id})"
        )
        return order

    # ── Position closing helpers ───────────────────────────────────────────────

    def _close_position_record(
        self,
        pos_id: int,
        ticker: str,
        qty: int,
        avg_cost: float,
        exit_price_raw: float,
        margin_blocked: float,
        fees_paid_so_far: float,
        market: Literal["india", "us"],
        strategy: str,
        reason: str,
    ) -> Optional[str]:
        """
        Core close logic shared by all exit paths.

        Returns a human-readable report string, or None on error.
        """
        # Slippage on exit (sell fills slightly below)
        exit_price = self.fees.apply_slippage(exit_price_raw, "SELL")

        # Gross P&L (direction: always LONG in current engine)
        gross_pnl = round((exit_price - avg_cost) * qty, 4)

        # Exit fees
        exit_fees = self.fees.compute_fees(
            market=market, direction="SELL",
            filled_price=exit_price, quantity=qty, is_paper=True,
        )

        # Net P&L after all fees
        net_pnl = round(gross_pnl - exit_fees, 4)

        # Cash returned = original margin + net P&L
        cash_returned = round(margin_blocked + net_pnl, 4)

        acc_id = self._acc_id(market)
        now_str = datetime.now(timezone.utc).isoformat()
        pnl_pct = round((exit_price - avg_cost) / avg_cost * 100, 2) if avg_cost else 0.0

        with _conn() as conn:
            # 1. Close position
            conn.execute("""
                UPDATE positions
                SET status = 'CLOSED', closed_at = ?, realized_pnl = ?,
                    fees_paid = fees_paid + ?, current_price = ?
                WHERE id = ?
            """, (now_str, net_pnl, exit_fees, exit_price, pos_id))

            # 2. Return margin + net P&L to cash; release reserved margin
            new_cash = self._update_cash(conn, acc_id,
                                          delta=+cash_returned,
                                          delta_margin=-margin_blocked)

            # 3. Ledger entries
            append_ledger(conn, acc_id, "MARGIN_RELEASE", +margin_blocked, new_cash - net_pnl,
                          ref_position_id=pos_id,
                          description=f"Close {qty}×{ticker}: {reason}")
            append_ledger(conn, acc_id, "FEE", -exit_fees, new_cash - net_pnl + exit_fees,
                          ref_position_id=pos_id,
                          description=f"Exit fees {ticker}")
            append_ledger(conn, acc_id, "REALIZED_PNL", net_pnl, new_cash,
                          ref_position_id=pos_id,
                          description=f"Net PnL {ticker} {pnl_pct:+.2f}%")
            conn.commit()

        pnl_sign = "+" if net_pnl >= 0 else ""
        report = (
            f"[EXIT] {qty}×{ticker} @ {exit_price:.4f} | "
            f"Gross: {pnl_sign}{gross_pnl:.2f} | Fees: -{exit_fees:.4f} | "
            f"Net: {pnl_sign}{net_pnl:.2f} ({pnl_pct:+.2f}%) | {reason}"
        )
        logger.info(f"[Paper Engine] {report}")
        return report

    # ── Public exit methods ────────────────────────────────────────────────────

    def evaluate_open_positions(
        self,
        market: Literal["india", "us"],
        latest_snapshots: dict[str, StockSnapshot],
    ) -> list[str]:
        """
        Check all open positions against SL/TP. Close triggered positions.
        Returns list of close report strings.
        """
        positions_to_close: list[tuple] = []

        with _conn() as conn:
            rows = conn.execute("""
                SELECT * FROM positions WHERE status = 'OPEN' AND market = ?
            """, (market,)).fetchall()

            for row in rows:
                ticker = row["ticker"]
                if ticker not in latest_snapshots:
                    continue
                curr_p = latest_snapshots[ticker].current_price
                conn.execute(
                    "UPDATE positions SET current_price = ? WHERE id = ?",
                    (curr_p, row["id"])
                )
                sl, tgt = row["stop_loss"], row["target_price"]
                if sl and curr_p <= sl:
                    positions_to_close.append((
                        row["id"], ticker, row["quantity"], row["avg_cost"],
                        curr_p, row["margin_blocked"], row["fees_paid"],
                        f"STOP LOSS HIT @ {curr_p:.4f} (SL: {sl})", row["strategy"],
                    ))
                elif tgt and curr_p >= tgt:
                    positions_to_close.append((
                        row["id"], ticker, row["quantity"], row["avg_cost"],
                        curr_p, row["margin_blocked"], row["fees_paid"],
                        f"TARGET HIT @ {curr_p:.4f} (TGT: {tgt})", row["strategy"],
                    ))
            conn.commit()

        closed_reports = []
        for pos_id, ticker, qty, avg_cost, curr_p, margin_blocked, fees_paid, reason, strategy in positions_to_close:
            report = self._close_position_record(
                pos_id=pos_id, ticker=ticker, qty=qty, avg_cost=avg_cost,
                exit_price_raw=curr_p, margin_blocked=margin_blocked,
                fees_paid_so_far=fees_paid, market=market,
                strategy=strategy, reason=reason,
            )
            if report:
                closed_reports.append(report)
        return closed_reports

    def sell_partial(
        self,
        ticker: str,
        market: Literal["india", "us"],
        exit_price_raw: float,
        sell_quantity: int,
    ) -> Optional[str]:
        """
        Partially close an open position.
        Uses proportional margin release and FIFO average cost.

        Returns a report string, or None if position not found or invalid qty.
        """
        with _conn() as conn:
            row = conn.execute("""
                SELECT * FROM positions WHERE ticker = ? AND market = ? AND status = 'OPEN'
                ORDER BY opened_at ASC LIMIT 1
            """, (ticker, market)).fetchone()

        if not row:
            logger.warning(f"[Paper Engine] sell_partial: no open position for {ticker}")
            return None

        actual_qty = int(row["quantity"])
        if sell_quantity >= actual_qty:
            # Delegate to full close via _close_position_record
            return self._close_position_record(
                pos_id=row["id"], ticker=ticker, qty=actual_qty,
                avg_cost=row["avg_cost"], exit_price_raw=exit_price_raw,
                margin_blocked=row["margin_blocked"], fees_paid_so_far=row["fees_paid"],
                market=market, strategy=row["strategy"], reason="FULL EXIT (via sell_partial)",
            )

        # ── Partial close: do accounting inline (don't fully close the row) ──
        exit_price = self.fees.apply_slippage(exit_price_raw, "SELL")
        proportion = sell_quantity / actual_qty
        partial_margin = round(float(row["margin_blocked"]) * proportion, 4)

        # Gross P&L on the lot being sold
        gross_pnl = round((exit_price - float(row["avg_cost"])) * sell_quantity, 4)
        exit_fees = self.fees.compute_fees(
            market=market, direction="SELL",
            filled_price=exit_price, quantity=sell_quantity, is_paper=True,
        )
        net_pnl = round(gross_pnl - exit_fees, 4)
        cash_returned = round(partial_margin + net_pnl, 4)

        acc_id = self._acc_id(market)
        now_str = datetime.now(timezone.utc).isoformat()
        pnl_pct = round((exit_price - float(row["avg_cost"])) / float(row["avg_cost"]) * 100, 2)
        pnl_sign = "+" if net_pnl >= 0 else ""

        # New values for the remaining open position
        new_qty = actual_qty - sell_quantity
        new_margin = round(float(row["margin_blocked"]) - partial_margin, 4)
        new_fees_paid = round(float(row["fees_paid"]) - round(float(row["fees_paid"]) * proportion, 4), 4)

        with _conn() as conn:
            # 1. Update position: reduce quantity and margin (keep OPEN)
            conn.execute("""
                UPDATE positions
                SET quantity = ?, margin_blocked = ?, fees_paid = ?
                WHERE id = ? AND status = 'OPEN'
            """, (new_qty, new_margin, new_fees_paid, row["id"]))

            # 2. Credit cash and release partial margin
            new_cash = self._update_cash(conn, acc_id,
                                          delta=+cash_returned,
                                          delta_margin=-partial_margin)

            # 3. Ledger entries
            append_ledger(conn, acc_id, "MARGIN_RELEASE", +partial_margin, new_cash - net_pnl,
                          ref_position_id=row["id"],
                          description=f"Partial close {sell_quantity}/{actual_qty}×{ticker}")
            if exit_fees > 0:
                append_ledger(conn, acc_id, "FEE", -exit_fees, new_cash - net_pnl + exit_fees,
                              ref_position_id=row["id"],
                              description=f"Partial exit fees {ticker}")
            append_ledger(conn, acc_id, "REALIZED_PNL", net_pnl, new_cash,
                          ref_position_id=row["id"],
                          description=f"Partial PnL {ticker} {pnl_pct:+.2f}%")
            conn.commit()

        report = (
            f"[PARTIAL EXIT] {sell_quantity}/{actual_qty}×{ticker} @ {exit_price:.4f} | "
            f"Gross: {pnl_sign}{gross_pnl:.2f} | Fees: -{exit_fees:.4f} | "
            f"Net: {pnl_sign}{net_pnl:.2f} ({pnl_pct:+.2f}%) | "
            f"Remaining: {new_qty} shares"
        )
        logger.info(f"[Paper Engine] {report}")
        return report


    def square_off_intraday(
        self,
        market: Literal["india", "us"],
        latest_snapshots: dict[str, StockSnapshot],
    ) -> list[str]:
        """Force-close all OPEN intraday/scalping positions at current price."""
        with _conn() as conn:
            rows = conn.execute("""
                SELECT * FROM positions
                WHERE status = 'OPEN'
                  AND strategy IN ('intraday', 'scalping')
                  AND market = ?
            """, (market,)).fetchall()
            positions = [dict(r) for r in rows]

        closed_reports = []
        for pos in positions:
            curr_p = (latest_snapshots[pos["ticker"]].current_price
                      if pos["ticker"] in latest_snapshots
                      else pos["current_price"] or pos["avg_cost"])
            report = self._close_position_record(
                pos_id=pos["id"], ticker=pos["ticker"],
                qty=pos["quantity"], avg_cost=pos["avg_cost"],
                exit_price_raw=curr_p,
                margin_blocked=pos["margin_blocked"],
                fees_paid_so_far=pos["fees_paid"],
                market=market, strategy=pos["strategy"],
                reason="INTRADAY SQUARE-OFF",
            )
            if report:
                closed_reports.append(report)
        return closed_reports

    def close_all_positions(
        self,
        market: Literal["india", "us"],
        latest_snapshots: Optional[dict[str, StockSnapshot]] = None,
    ) -> list[str]:
        """Emergency close ALL open positions for a market."""
        with _conn() as conn:
            rows = conn.execute("""
                SELECT * FROM positions WHERE status = 'OPEN' AND market = ?
            """, (market,)).fetchall()
            positions = [dict(r) for r in rows]

        if not positions:
            return []

        if latest_snapshots is None:
            logger.warning("[Paper Engine] close_all_positions: no live prices provided — using stale DB prices")

        closed_reports = []
        for pos in positions:
            ticker = pos["ticker"]
            if latest_snapshots and ticker in latest_snapshots:
                curr_p = latest_snapshots[ticker].current_price
            else:
                curr_p = pos["current_price"] or pos["avg_cost"]

            report = self._close_position_record(
                pos_id=pos["id"], ticker=ticker,
                qty=pos["quantity"], avg_cost=pos["avg_cost"],
                exit_price_raw=curr_p,
                margin_blocked=pos["margin_blocked"],
                fees_paid_so_far=pos["fees_paid"],
                market=market, strategy=pos.get("strategy", "swing"),
                reason="EMERGENCY CLOSE",
            )
            if report:
                closed_reports.append(report)
        return closed_reports

    # ── Portfolio summary / NAV ────────────────────────────────────────────────

    def get_portfolio_nav(self, market: Literal["india", "us"]) -> dict:
        """
        Returns the true portfolio equity (NAV).

        NAV = cash + reserved_margin + unrealized_pnl

        cash:             free liquid cash
        reserved_margin:  margin locked in open positions (still your equity)
        unrealized_pnl:   paper gain/loss on open positions

        The leveraged notional of open positions is NOT added to NAV.
        """
        acc_id = self._acc_id(market)
        currency = "₹" if market == "india" else "$"

        with _conn() as conn:
            acc = conn.execute(
                "SELECT cash, initial_cash, reserved_margin FROM accounts WHERE account_id = ?",
                (acc_id,)
            ).fetchone()
            open_rows = conn.execute("""
                SELECT ticker, quantity, avg_cost, current_price, direction, margin_blocked
                FROM positions WHERE status = 'OPEN' AND market = ?
            """, (market,)).fetchall()

        if not acc:
            return {}

        cash = float(acc["cash"])
        reserved_margin = float(acc["reserved_margin"])
        initial_cash = float(acc["initial_cash"])

        positions_data = []
        total_unrealized = 0.0
        total_notional = 0.0

        for row in open_rows:
            curr = row["current_price"] or row["avg_cost"]
            direction = row["direction"] or "LONG"
            if direction == "LONG":
                upnl = (curr - row["avg_cost"]) * row["quantity"]
            else:
                upnl = (row["avg_cost"] - curr) * row["quantity"]

            notional = curr * row["quantity"]
            total_unrealized += upnl
            total_notional += notional
            positions_data.append({
                "ticker": row["ticker"],
                "quantity": row["quantity"],
                "avg_cost": row["avg_cost"],
                "current_price": curr,
                "unrealized_pnl": round(upnl, 2),
                "margin_blocked": row["margin_blocked"],
            })

        nav = round(cash + reserved_margin + total_unrealized, 2)
        total_return_pct = round((nav - initial_cash) / initial_cash * 100, 2) if initial_cash else 0.0

        return {
            "market": market,
            "currency": currency,
            "cash": round(cash, 2),
            "reserved_margin": round(reserved_margin, 2),
            "unrealized_pnl": round(total_unrealized, 2),
            "nav": nav,
            "initial_capital": round(initial_cash, 2),
            "total_return_pct": total_return_pct,
            "open_positions_count": len(positions_data),
            "positions": positions_data,
        }

    def get_portfolio_summary(self, market: Literal["india", "us"]) -> dict:
        """
        Backward-compatible summary used by Telegram /status.
        Delegates to get_portfolio_nav() but adds legacy keys.
        """
        nav = self.get_portfolio_nav(market)
        if not nav:
            return {
                "market": market,
                "currency": "₹" if market == "india" else "$",
                "cash": 0.0, "invested": 0.0, "total_value": 0.0,
                "open_positions_count": 0, "positions": [],
            }
        return {
            **nav,
            # Legacy keys kept for Telegram bot compatibility
            "invested": nav["reserved_margin"],
            "total_value": nav["nav"],
        }

    # ── Ledger access ──────────────────────────────────────────────────────────

    def get_transaction_ledger(
        self, market: Literal["india", "us"], limit: int = 50
    ) -> list[dict]:
        """Returns the most recent ledger entries for this market's account."""
        return get_ledger(self._acc_id(market), limit=limit)
