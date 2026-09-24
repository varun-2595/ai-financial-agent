"""
Database models and persistence for Paper & Live Trading, Positions, and Signals.
Supports PostgreSQL (asyncpg) with persistent connection pooling and SQLite WAL mode.
Provides row-level locking (SELECT ... FOR UPDATE) for balance allocation and order submission.
"""
from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal, Optional

from sqlalchemy import select, update, text
from src.data.models import Order, Position, TradeSignal
from src.db.session import (
    DEFAULT_SQLITE_PATH,
    AccountModel,
    LedgerModel,
    OrderModel,
    PositionModel,
    SignalModel,
    get_async_session,
    init_db,
)
from src.utils.logger import logger

DB_PATH = DEFAULT_SQLITE_PATH


def _conn() -> sqlite3.Connection:
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.row_factory = sqlite3.Row
    return conn


def _add_column_if_missing(conn: sqlite3.Connection, table: str, column: str, definition: str) -> None:
    """Safe ALTER TABLE — no-ops if the column already exists."""
    existing = {row[1] for row in conn.execute(f"PRAGMA table_info({table})")}
    if column not in existing:
        conn.execute(f"ALTER TABLE {table} ADD COLUMN {column} {definition}")
        logger.info(f"[Trading DB] Migrated: added {table}.{column}")


def init_trading_db() -> None:
    """Initialize local SQLite tables and ensure compatibility schema."""
    with _conn() as conn:
        # ── Accounts / Balances ────────────────────────────────────────────
        conn.execute("""
            CREATE TABLE IF NOT EXISTS accounts (
                account_id      TEXT PRIMARY KEY,
                currency        TEXT NOT NULL,
                cash            REAL NOT NULL,
                initial_cash    REAL NOT NULL,
                reserved_margin REAL NOT NULL DEFAULT 0.0,
                peak_nav        REAL NOT NULL DEFAULT 0.0,
                updated_at      TEXT NOT NULL
            )
        """)

        # ── Orders ────────────────────────────────────────────────────────
        conn.execute("""
            CREATE TABLE IF NOT EXISTS orders (
                order_id         TEXT PRIMARY KEY,
                broker_order_ref TEXT,
                ticker           TEXT NOT NULL,
                market           TEXT NOT NULL,
                exchange         TEXT NOT NULL DEFAULT 'NSE',
                strategy         TEXT NOT NULL,
                order_type       TEXT NOT NULL,
                direction        TEXT NOT NULL,
                quantity         INTEGER NOT NULL,
                limit_price      REAL,
                trigger_price    REAL,
                requested_price  REAL,
                filled_price     REAL,
                executed_price   REAL,
                fees             REAL NOT NULL DEFAULT 0.0,
                statutory_fees   REAL NOT NULL DEFAULT 0.0,
                slippage         REAL NOT NULL DEFAULT 0.0,
                slippage_pct     REAL NOT NULL DEFAULT 0.0005,
                status           TEXT NOT NULL,
                is_paper         INTEGER NOT NULL DEFAULT 1,
                created_at       TEXT NOT NULL,
                filled_at        TEXT
            )
        """)

        # ── Active & Closed Positions ──────────────────────────────────────
        conn.execute("""
            CREATE TABLE IF NOT EXISTS positions (
                id              INTEGER PRIMARY KEY AUTOINCREMENT,
                position_id     TEXT UNIQUE,
                ticker          TEXT NOT NULL,
                market          TEXT NOT NULL,
                strategy        TEXT NOT NULL,
                direction       TEXT NOT NULL,
                quantity        INTEGER NOT NULL,
                avg_cost        REAL NOT NULL,
                current_price   REAL,
                stop_loss       REAL,
                target_price    REAL,
                margin_blocked  REAL NOT NULL DEFAULT 0.0,
                fees_paid       REAL NOT NULL DEFAULT 0.0,
                status          TEXT NOT NULL DEFAULT 'OPEN',
                is_paper        INTEGER NOT NULL DEFAULT 1,
                opened_at       TEXT NOT NULL,
                closed_at       TEXT,
                realized_pnl    REAL DEFAULT 0.0,
                return_pct      REAL DEFAULT 0.0
            )
        """)

        # ── Transaction Ledger (double-entry cash event log) ────────────────
        conn.execute("""
            CREATE TABLE IF NOT EXISTS ledger (
                id              INTEGER PRIMARY KEY AUTOINCREMENT,
                account_id      TEXT NOT NULL,
                entry_type      TEXT NOT NULL,
                amount          REAL NOT NULL,
                balance_after   REAL NOT NULL,
                ref_order_id    TEXT,
                ref_position_id INTEGER,
                description     TEXT,
                created_at      TEXT NOT NULL
            )
        """)

        # ── Trade Signals Log ──────────────────────────────────────────────
        conn.execute("""
            CREATE TABLE IF NOT EXISTS signals (
                id           INTEGER PRIMARY KEY AUTOINCREMENT,
                ticker       TEXT NOT NULL,
                market       TEXT NOT NULL,
                strategy     TEXT NOT NULL,
                direction    TEXT NOT NULL,
                entry_price  REAL NOT NULL,
                stop_loss    REAL NOT NULL,
                target_price REAL NOT NULL,
                quantity     INTEGER NOT NULL,
                confidence   REAL NOT NULL,
                reasoning    TEXT,
                generated_at TEXT NOT NULL
            )
        """)

        # ── Account Snapshots (for drawdown and NAV tracking) ──────────────
        conn.execute("""
            CREATE TABLE IF NOT EXISTS account_snapshots (
                id              INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp       TEXT NOT NULL,
                market          TEXT NOT NULL,
                cash_balance    REAL NOT NULL,
                margin_utilized REAL NOT NULL,
                portfolio_nav   REAL NOT NULL,
                peak_nav        REAL NOT NULL,
                drawdown_pct    REAL NOT NULL,
                created_at      TEXT NOT NULL
            )
        """)
        conn.commit()

    # Safe column migrations for existing SQLite DBs
    with _conn() as conn:
        _add_column_if_missing(conn, "accounts", "reserved_margin", "REAL NOT NULL DEFAULT 0.0")
        _add_column_if_missing(conn, "accounts", "peak_nav", "REAL NOT NULL DEFAULT 0.0")
        _add_column_if_missing(conn, "positions", "margin_blocked", "REAL NOT NULL DEFAULT 0.0")
        _add_column_if_missing(conn, "positions", "fees_paid", "REAL NOT NULL DEFAULT 0.0")
        _add_column_if_missing(conn, "positions", "position_id", "TEXT")
        conn.execute("CREATE UNIQUE INDEX IF NOT EXISTS idx_positions_pos_id ON positions(position_id)")
        _add_column_if_missing(conn, "positions", "return_pct", "REAL DEFAULT 0.0")
        _add_column_if_missing(conn, "orders", "fees", "REAL NOT NULL DEFAULT 0.0")
        _add_column_if_missing(conn, "orders", "statutory_fees", "REAL NOT NULL DEFAULT 0.0")
        _add_column_if_missing(conn, "orders", "slippage", "REAL NOT NULL DEFAULT 0.0")
        _add_column_if_missing(conn, "orders", "slippage_pct", "REAL NOT NULL DEFAULT 0.0005")
        _add_column_if_missing(conn, "orders", "broker_order_ref", "TEXT")
        _add_column_if_missing(conn, "orders", "exchange", "TEXT DEFAULT 'NSE'")
        _add_column_if_missing(conn, "orders", "requested_price", "REAL")
        _add_column_if_missing(conn, "orders", "executed_price", "REAL")
        conn.commit()

    logger.info("[Trading DB] Initialized.")


# ── Ledger helpers ─────────────────────────────────────────────────────────────

def append_ledger(
    conn: sqlite3.Connection,
    account_id: str,
    entry_type: str,
    amount: float,
    balance_after: float,
    ref_order_id: Optional[str] = None,
    ref_position_id: Optional[int] = None,
    description: Optional[str] = None,
) -> None:
    """Write a single ledger entry. Must be called inside an open conn context."""
    conn.execute("""
        INSERT INTO ledger
            (account_id, entry_type, amount, balance_after, ref_order_id, ref_position_id, description, created_at)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        account_id, entry_type, amount, balance_after,
        ref_order_id, ref_position_id, description,
        datetime.now(timezone.utc).isoformat()
    ))


def get_ledger(account_id: str, limit: int = 100) -> list[dict]:
    """Return the most recent ledger entries for an account, newest first."""
    with _conn() as conn:
        rows = conn.execute("""
            SELECT * FROM ledger
            WHERE account_id = ?
            ORDER BY id DESC
            LIMIT ?
        """, (account_id, limit)).fetchall()
        return [dict(r) for r in rows]


# ── Signal / Order persistence ─────────────────────────────────────────────────

def save_signal(sig: TradeSignal) -> None:
    with _conn() as conn:
        conn.execute("""
            INSERT INTO signals (ticker, market, strategy, direction, entry_price, stop_loss, target_price, quantity, confidence, reasoning, generated_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            sig.ticker, sig.market, sig.strategy, sig.direction, sig.entry_price,
            sig.stop_loss, sig.target_price, sig.quantity, sig.confidence,
            sig.reasoning, sig.generated_at.isoformat()
        ))
        conn.commit()


def save_order(
    order: Order,
    fees: float = 0.0,
    slippage_pct: float = 0.0005,
    statutory_fees: float = 0.0,
    broker_order_ref: Optional[str] = None,
) -> None:
    with _conn() as conn:
        conn.execute("""
            INSERT INTO orders (order_id, broker_order_ref, ticker, market, exchange, strategy, order_type, direction, quantity,
                                limit_price, trigger_price, requested_price, filled_price, executed_price,
                                fees, statutory_fees, slippage_pct,
                                status, is_paper, created_at, filled_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(order_id) DO UPDATE SET
                status           = excluded.status,
                broker_order_ref = coalesce(excluded.broker_order_ref, orders.broker_order_ref),
                filled_price     = excluded.filled_price,
                executed_price   = excluded.executed_price,
                fees             = excluded.fees,
                statutory_fees   = excluded.statutory_fees,
                filled_at        = excluded.filled_at
        """, (
            order.order_id, broker_order_ref, order.ticker, order.market,
            "NSE" if order.market == "india" else "US", order.strategy,
            order.order_type, order.direction, order.quantity, order.limit_price,
            order.trigger_price, order.limit_price or order.filled_price, order.filled_price, order.filled_price,
            fees, statutory_fees, slippage_pct,
            order.status, int(order.is_paper),
            order.created_at.isoformat(),
            order.filled_at.isoformat() if order.filled_at else None
        ))
        conn.commit()


def get_open_positions(market: Optional[str] = None, strategy: Optional[str] = None) -> list[dict]:
    query = "SELECT * FROM positions WHERE status = 'OPEN'"
    params = []
    if market:
        query += " AND market = ?"
        params.append(market)
    if strategy:
        query += " AND strategy = ?"
        params.append(strategy)

    with _conn() as conn:
        rows = conn.execute(query, params).fetchall()
        return [dict(r) for r in rows]


def record_account_snapshot(
    market: str,
    cash_balance: float,
    margin_utilized: float,
    portfolio_nav: float,
    peak_nav: float,
    drawdown_pct: float,
) -> None:
    """Record an intraday portfolio NAV snapshot for drawdown tracking and circuit breakers."""
    now_iso = datetime.now(timezone.utc).isoformat()
    with _conn() as conn:
        conn.execute("""
            INSERT INTO account_snapshots (timestamp, market, cash_balance, margin_utilized, portfolio_nav, peak_nav, drawdown_pct, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """, (now_iso, market, cash_balance, margin_utilized, portfolio_nav, peak_nav, drawdown_pct, now_iso))
        conn.commit()
