"""
Dynamic Watchlist Manager — the agent's live, self-updating ticker registry.
Backed by SQLite with WAL mode for safe concurrency.
"""
from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal

import yaml

from src.utils.logger import logger

DB_PATH = Path(__file__).parent.parent.parent / "data" / "watchlist.db"
MANUAL_YAML = Path(__file__).parent.parent.parent / "config" / "watchlist.yaml"

Source = Literal["auto", "manual"]
Strategy = Literal["intraday", "swing", "positional"]


def _utcnow_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _conn() -> sqlite3.Connection:
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.row_factory = sqlite3.Row
    return conn


def _init_db() -> None:
    with _conn() as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS watchlist (
                ticker      TEXT NOT NULL,
                market      TEXT NOT NULL CHECK(market IN ('india', 'us')),
                source      TEXT NOT NULL DEFAULT 'auto',
                strategies  TEXT NOT NULL DEFAULT '["swing"]',
                pinned      INTEGER NOT NULL DEFAULT 0,
                banned      INTEGER NOT NULL DEFAULT 0,
                added_at    TEXT NOT NULL,
                updated_at  TEXT NOT NULL,
                reason      TEXT,
                PRIMARY KEY (ticker)
            )
        """)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS watchlist_log (
                id          INTEGER PRIMARY KEY AUTOINCREMENT,
                ticker      TEXT NOT NULL,
                action      TEXT NOT NULL,
                source      TEXT,
                reason      TEXT,
                timestamp   TEXT NOT NULL
            )
        """)
        conn.commit()


def _log_action(ticker: str, action: str, source: str | None, reason: str | None) -> None:
    with _conn() as conn:
        conn.execute(
            "INSERT INTO watchlist_log (ticker, action, source, reason, timestamp) "
            "VALUES (?, ?, ?, ?, ?)",
            (ticker, action, source, reason, _utcnow_iso()),
        )
        conn.commit()


def init_watchlist() -> None:
    _init_db()
    _seed_from_yaml()
    logger.info("[Watchlist] Initialised.")


def _seed_from_yaml() -> None:
    if not MANUAL_YAML.exists():
        return
    with open(MANUAL_YAML) as f:
        config = yaml.safe_load(f) or {}

    wl = config.get("watchlist", {})
    for market, entries in wl.items():
        if not isinstance(entries, list):
            continue
        for entry in entries:
            ticker = entry.get("ticker", "").upper()
            strategies = entry.get("strategies") or entry.get("strategy", ["swing"])
            pinned = bool(entry.get("pinned", False))
            if ticker:
                add_ticker(
                    ticker=ticker,
                    market=market,
                    strategies=strategies,
                    source="manual",
                    reason="Seeded from watchlist.yaml",
                    pinned=pinned,
                )
    logger.info("[Watchlist] Seeded from watchlist.yaml")


def add_ticker(
    ticker: str,
    market: Literal["india", "us"],
    strategies: list[Strategy],
    source: Source = "auto",
    reason: str | None = None,
    pinned: bool = False,
) -> bool:
    ticker = ticker.upper()
    now = _utcnow_iso()

    with _conn() as conn:
        row = conn.execute(
            "SELECT banned FROM watchlist WHERE ticker = ?", (ticker,)
        ).fetchone()
        if row and row["banned"]:
            logger.debug(f"[Watchlist] {ticker} is banned — skipping add.")
            return False

        conn.execute("""
            INSERT INTO watchlist (ticker, market, source, strategies, pinned, banned, added_at, updated_at, reason)
            VALUES (?, ?, ?, ?, ?, 0, ?, ?, ?)
            ON CONFLICT(ticker) DO UPDATE SET
                source     = CASE WHEN excluded.source = 'manual' THEN 'manual' ELSE watchlist.source END,
                strategies = excluded.strategies,
                pinned     = MAX(watchlist.pinned, excluded.pinned),
                updated_at = excluded.updated_at,
                reason     = excluded.reason
        """, (
            ticker, market, source, json.dumps(strategies), int(pinned), now, now, reason,
        ))
        conn.commit()

    _log_action(ticker, "added", source, reason)
    logger.info(f"[Watchlist] ➕ Added {ticker} ({market}, {strategies}, source={source})")
    return True


def remove_ticker(
    ticker: str,
    reason: str | None = None,
    force: bool = False,
) -> bool:
    ticker = ticker.upper()
    with _conn() as conn:
        row = conn.execute(
            "SELECT source, pinned FROM watchlist WHERE ticker = ?", (ticker,)
        ).fetchone()

        if row is None:
            return False

        if not force and (row["source"] == "manual" or row["pinned"]):
            logger.debug(
                f"[Watchlist] {ticker} is {'pinned' if row['pinned'] else 'manual'} "
                "— skipping auto-remove. Use force=True to override."
            )
            return False

        conn.execute("DELETE FROM watchlist WHERE ticker = ?", (ticker,))
        conn.commit()

    _log_action(ticker, "removed", None, reason)
    logger.info(f"[Watchlist] ➖ Removed {ticker}: {reason}")
    return True


def pin_ticker(ticker: str, reason: str = "User pinned") -> bool:
    ticker = ticker.upper()
    with _conn() as conn:
        cur = conn.execute(
            "UPDATE watchlist SET pinned = 1, updated_at = ? WHERE ticker = ?",
            (_utcnow_iso(), ticker),
        )
        conn.commit()
        if cur.rowcount == 0:
            logger.warning(f"[Watchlist] Cannot pin {ticker}: ticker not found.")
            return False
    _log_action(ticker, "pinned", "manual", reason)
    logger.info(f"[Watchlist] 📌 Pinned {ticker}")
    return True


def ban_ticker(ticker: str, market: Literal["india", "us"] | None = None, reason: str = "User banned") -> None:
    ticker = ticker.upper()
    now = _utcnow_iso()

    # Determine market if not explicitly provided
    if not market:
        market = "india" if ticker.endswith(".NS") or ticker.endswith(".BO") else "us"

    with _conn() as conn:
        conn.execute("""
            INSERT INTO watchlist (ticker, market, source, strategies, pinned, banned, added_at, updated_at, reason)
            VALUES (?, ?, 'manual', '[]', 0, 1, ?, ?, ?)
            ON CONFLICT(ticker) DO UPDATE SET
                banned = 1,
                market = excluded.market,
                updated_at = excluded.updated_at,
                reason = excluded.reason
        """, (ticker, market, now, now, reason))
        conn.commit()
    _log_action(ticker, "banned", "manual", reason)
    logger.warning(f"[Watchlist] 🚫 Banned {ticker} ({market}): {reason}")


def unban_ticker(ticker: str) -> bool:
    ticker = ticker.upper()
    now = _utcnow_iso()
    with _conn() as conn:
        cur = conn.execute(
            "UPDATE watchlist SET banned = 0, updated_at = ? WHERE ticker = ?",
            (now, ticker),
        )
        conn.commit()
        if cur.rowcount == 0:
            logger.warning(f"[Watchlist] Cannot unban {ticker}: ticker not found.")
            return False
    _log_action(ticker, "unbanned", "manual", "User unbanned")
    logger.info(f"[Watchlist] 🟢 Unbanned {ticker}")
    return True


def get_banned_tickers() -> list[str]:
    _init_db()
    with _conn() as conn:
        rows = conn.execute("SELECT ticker FROM watchlist WHERE banned = 1").fetchall()
        return [r["ticker"] for r in rows]


def get_active_tickers(
    market: Literal["india", "us"] | None = None,
    strategy: Strategy | None = None,
) -> list[dict]:
    _init_db()
    query = "SELECT ticker, market, source, strategies, pinned FROM watchlist WHERE banned = 0"
    params: list = []

    if market:
        query += " AND market = ?"
        params.append(market)

    rows = []
    with _conn() as conn:
        for row in conn.execute(query, params).fetchall():
            strats = json.loads(row["strategies"])
            if strategy and strategy not in strats:
                continue
            rows.append({
                "ticker":     row["ticker"],
                "market":     row["market"],
                "source":     row["source"],
                "strategies": strats,
                "pinned":     bool(row["pinned"]),
            })
    return rows


def get_watchlist_summary() -> dict:
    all_tickers = get_active_tickers()
    banned = get_banned_tickers()
    india = [t for t in all_tickers if t["market"] == "india"]
    us    = [t for t in all_tickers if t["market"] == "us"]
    auto  = [t for t in all_tickers if t["source"] == "auto"]
    manual = [t for t in all_tickers if t["source"] == "manual"]

    return {
        "total":   len(all_tickers),
        "india":   len(india),
        "us":      len(us),
        "auto":    len(auto),
        "manual":  len(manual),
        "banned":  len(banned),
        "tickers": [t["ticker"] for t in all_tickers],
    }
