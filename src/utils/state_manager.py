"""
System State Manager for Aegis.
Tracks runtime state flags (e.g. trading pause/resume) persisted to SQLite.
"""
from __future__ import annotations

import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal

from src.utils.logger import logger

DB_PATH = Path(__file__).parent.parent.parent / "data" / "trading.db"


def _conn() -> sqlite3.Connection:
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.row_factory = sqlite3.Row
    return conn


def init_state_table() -> None:
    with _conn() as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS system_state (
                key        TEXT PRIMARY KEY,
                value      TEXT NOT NULL,
                updated_at TEXT NOT NULL
            )
        """)
        conn.commit()


def get_state(key: str, default: str = "") -> str:
    init_state_table()
    with _conn() as conn:
        row = conn.execute("SELECT value FROM system_state WHERE key = ?", (key,)).fetchone()
        return row["value"] if row else default


def set_state(key: str, value: str) -> None:
    init_state_table()
    now_iso = datetime.now(timezone.utc).isoformat()
    with _conn() as conn:
        conn.execute("""
            INSERT INTO system_state (key, value, updated_at)
            VALUES (?, ?, ?)
            ON CONFLICT(key) DO UPDATE SET
                value = excluded.value,
                updated_at = excluded.updated_at
        """, (key, value, now_iso))
        conn.commit()


def is_trading_paused(market: str = "all") -> bool:
    """
    Returns True if trading is paused for the given market ('india', 'us', or 'all').
    If global 'all' is paused, both 'india' and 'us' are considered paused.
    """
    global_paused = get_state("pause_all", "0") == "1"
    if global_paused:
        return True

    m = market.lower()
    if m in ("all", "both"):
        return global_paused or (get_state("pause_india", "0") == "1" and get_state("pause_us", "0") == "1")
    elif m in ("india", "nse", "bse"):
        return get_state("pause_india", "0") == "1"
    elif m in ("us", "nyse", "nasdaq"):
        return get_state("pause_us", "0") == "1"
    return False


def set_trading_paused(market: str, paused: bool) -> None:
    """
    Set trading pause state for 'india', 'us', or 'all'.
    """
    val = "1" if paused else "0"
    m = market.lower()
    if m in ("all", "both"):
        set_state("pause_all", val)
        set_state("pause_india", val)
        set_state("pause_us", val)
        logger.info(f"[State] Global trading pause set to: {paused}")
    elif m in ("india", "nse", "bse"):
        set_state("pause_india", val)
        if not paused:
            set_state("pause_all", "0")
        logger.info(f"[State] India trading pause set to: {paused}")
    elif m in ("us", "nyse", "nasdaq"):
        set_state("pause_us", val)
        if not paused:
            set_state("pause_all", "0")
        logger.info(f"[State] US trading pause set to: {paused}")
