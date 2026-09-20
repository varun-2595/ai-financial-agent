"""
Database models and persistence for Financial Advisory Portfolio & Recommendations.
Kept strictly separate from the Trading Mode portfolio.
Backed by SQLite in WAL mode.
"""
from __future__ import annotations

import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal, Optional

from src.data.models import AdvisoryHolding
from src.utils.logger import logger

DB_PATH = Path(__file__).parent.parent.parent / "data" / "advisory.db"


def _conn() -> sqlite3.Connection:
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.row_factory = sqlite3.Row
    return conn


def init_advisory_db() -> None:
    with _conn() as conn:
        # Advisory Recommended Holdings
        conn.execute("""
            CREATE TABLE IF NOT EXISTS advisory_holdings (
                id                   INTEGER PRIMARY KEY AUTOINCREMENT,
                ticker               TEXT NOT NULL,
                name                 TEXT,
                market               TEXT NOT NULL,
                horizon              TEXT NOT NULL, -- 'short_term' (1-6mo) or 'long_term' (1-5yr)
                entry_price          REAL NOT NULL,
                current_price        REAL,
                suggested_allocation_pct REAL,
                quantity_suggested   INTEGER,
                stop_loss            REAL,
                target_price         REAL,
                thesis               TEXT,
                invalidation_trigger TEXT,
                status               TEXT NOT NULL DEFAULT 'ACTIVE', -- 'ACTIVE', 'EXITED', 'TRIMMED'
                recommended_at       TEXT NOT NULL,
                exited_at            TEXT,
                exit_price           REAL,
                realized_return_pct  REAL
            )
        """)

        # Advisory Action Log (Sell / Trim alerts)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS advisory_alerts (
                id             INTEGER PRIMARY KEY AUTOINCREMENT,
                holding_id     INTEGER NOT NULL,
                ticker         TEXT NOT NULL,
                action_type    TEXT NOT NULL, -- 'SELL', 'TRIM', 'STOP_LOSS', 'TARGET_HIT', 'TIME_STOP'
                reason         TEXT NOT NULL,
                price_at_alert REAL NOT NULL,
                created_at     TEXT NOT NULL,
                FOREIGN KEY (holding_id) REFERENCES advisory_holdings (id)
            )
        """)
        conn.commit()
    logger.info("[Advisory DB] Initialized.")


def save_advisory_recommendation(
    ticker: str,
    name: str,
    market: Literal["india", "us"],
    horizon: Literal["short_term", "long_term"],
    entry_price: float,
    suggested_alloc_pct: float,
    quantity_suggested: int,
    stop_loss: float,
    target_price: float,
    thesis: str,
    invalidation_trigger: str,
) -> int:
    now_str = datetime.now(timezone.utc).isoformat()
    with _conn() as conn:
        cur = conn.execute("""
            INSERT INTO advisory_holdings (
                ticker, name, market, horizon, entry_price, current_price,
                suggested_allocation_pct, quantity_suggested, stop_loss,
                target_price, thesis, invalidation_trigger, status, recommended_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 'ACTIVE', ?)
        """, (
            ticker, name, market, horizon, entry_price, entry_price,
            suggested_alloc_pct, quantity_suggested, stop_loss,
            target_price, thesis, invalidation_trigger, now_str
        ))
        conn.commit()
        return cur.lastrowid


def get_active_advisory_holdings(market: Optional[str] = None) -> list[dict]:
    query = "SELECT * FROM advisory_holdings WHERE status = 'ACTIVE'"
    params = []
    if market:
        query += " AND market = ?"
        params.append(market)

    with _conn() as conn:
        rows = conn.execute(query, params).fetchall()
        return [dict(r) for r in rows]


def log_advisory_alert(holding_id: int, ticker: str, action_type: str, reason: str, price: float) -> None:
    now_str = datetime.now(timezone.utc).isoformat()
    with _conn() as conn:
        conn.execute("""
            INSERT INTO advisory_alerts (holding_id, ticker, action_type, reason, price_at_alert, created_at)
            VALUES (?, ?, ?, ?, ?, ?)
        """, (holding_id, ticker, action_type, reason, price, now_str))
        conn.commit()
