"""
Database models and persistence for Paper & Live Trading, Positions, and Signals.
Backed by SQLite in WAL mode.

Phase 1 additions:
  - `accounts.reserved_margin`   — margin currently blocked by open positions
  - `positions.margin_blocked`   — exact margin deducted when position was opened
  - `positions.fees_paid`        — cumulative brokerage + exchange fees for this position
  - `ledger` table               — double-entry transaction log for every cash event
"""
from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal, Optional

from src.data.models import Order, Position, TradeSignal
from src.utils.logger import logger

DB_PATH = Path(__file__).parent.parent.parent / "data" / "trading.db"


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
    with _conn() as conn:
        # ── Accounts / Balances ────────────────────────────────────────────
        conn.execute("""
            CREATE TABLE IF NOT EXISTS accounts (
                account_id      TEXT PRIMARY KEY,
                currency        TEXT NOT NULL,
                cash            REAL NOT NULL,
                initial_cash    REAL NOT NULL,
                reserved_margin REAL NOT NULL DEFAULT 0.0,
                updated_at      TEXT NOT NULL
            )
        """)

        # ── Orders ────────────────────────────────────────────────────────
        conn.execute("""
            CREATE TABLE IF NOT EXISTS orders (
                order_id      TEXT PRIMARY KEY,
                ticker        TEXT NOT NULL,
                market        TEXT NOT NULL,
                strategy      TEXT NOT NULL,
                order_type    TEXT NOT NULL,
                direction     TEXT NOT NULL,
                quantity      INTEGER NOT NULL,
                limit_price   REAL,
                trigger_price REAL,
                filled_price  REAL,
                fees          REAL NOT NULL DEFAULT 0.0,
                slippage_pct  REAL NOT NULL DEFAULT 0.0005,
                status        TEXT NOT NULL,
                is_paper      INTEGER NOT NULL DEFAULT 1,
                created_at    TEXT NOT NULL,
                filled_at     TEXT
            )
        """)

        # ── Active & Closed Positions ──────────────────────────────────────
        conn.execute("""
            CREATE TABLE IF NOT EXISTS positions (
                id              INTEGER PRIMARY KEY AUTOINCREMENT,
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
                realized_pnl    REAL DEFAULT 0.0
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
        conn.commit()

    # ── Safe column migrations for existing DBs ────────────────────────────
    with _conn() as conn:
        _add_column_if_missing(conn, "accounts", "reserved_margin", "REAL NOT NULL DEFAULT 0.0")
        _add_column_if_missing(conn, "positions", "margin_blocked", "REAL NOT NULL DEFAULT 0.0")
        _add_column_if_missing(conn, "positions", "fees_paid", "REAL NOT NULL DEFAULT 0.0")
        _add_column_if_missing(conn, "orders", "fees", "REAL NOT NULL DEFAULT 0.0")
        _add_column_if_missing(conn, "orders", "slippage_pct", "REAL NOT NULL DEFAULT 0.0005")
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


def save_order(order: Order, fees: float = 0.0, slippage_pct: float = 0.0005) -> None:
    with _conn() as conn:
        conn.execute("""
            INSERT INTO orders (order_id, ticker, market, strategy, order_type, direction, quantity,
                                limit_price, trigger_price, filled_price, fees, slippage_pct,
                                status, is_paper, created_at, filled_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(order_id) DO UPDATE SET
                status       = excluded.status,
                filled_price = excluded.filled_price,
                fees         = excluded.fees,
                filled_at    = excluded.filled_at
        """, (
            order.order_id, order.ticker, order.market, order.strategy,
            order.order_type, order.direction, order.quantity, order.limit_price,
            order.trigger_price, order.filled_price, fees, slippage_pct,
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
