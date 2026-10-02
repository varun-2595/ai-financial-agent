"""
Pre-Trade Autopsy Memory & Mistake Retrieval Engine.

Queries historical loss autopsies and prescriptive lessons from trade_playbook
to prevent the agent from repeating identical failure modes on the same ticker or setup.
"""
from __future__ import annotations

import sqlite3
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal, Optional

from src.utils.logger import logger

DB_PATH = Path(__file__).parent.parent.parent / "data" / "trading.db"


def _conn() -> sqlite3.Connection:
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.row_factory = sqlite3.Row
    return conn


@dataclass
class MemoryWarningReport:
    ticker: str
    market: Literal["india", "us"]
    strategy: str
    has_historical_failures: bool
    failure_count: int = 0
    common_failure_categories: list[str] = field(default_factory=list)
    key_lessons: list[str] = field(default_factory=list)
    formatted_warning: str = ""


class PlaybookMemoryRetriever:
    """Retrieves and formats historical trade autopsy lessons prior to signal generation."""

    def __init__(self, db_path: Optional[Path] = None):
        self.db_path = db_path or DB_PATH

    def get_memory_warning(
        self,
        ticker: str,
        market: Literal["india", "us"],
        strategy: str,
        limit: int = 5,
    ) -> MemoryWarningReport:
        """
        Queries the trade_playbook for past autopsies on this specific ticker or setup.
        """
        ticker_clean = ticker.upper().strip()
        with _conn() as conn:
            # Query past autopsies for this specific ticker
            rows = conn.execute("""
                SELECT failure_category, root_cause, prescriptive_lesson, realized_pnl, created_at
                FROM trade_playbook
                WHERE ticker = ? AND market = ?
                ORDER BY created_at DESC LIMIT ?
            """, (ticker_clean, market, limit)).fetchall()

            # If no ticker-specific autopsies, query general strategy autopsies
            if not rows:
                rows = conn.execute("""
                    SELECT failure_category, root_cause, prescriptive_lesson, realized_pnl, created_at
                    FROM trade_playbook
                    WHERE market = ? AND strategy = ?
                    ORDER BY created_at DESC LIMIT 3
                """, (market, strategy)).fetchall()

        if not rows:
            return MemoryWarningReport(
                ticker=ticker_clean,
                market=market,
                strategy=strategy,
                has_historical_failures=False,
                failure_count=0,
                common_failure_categories=[],
                key_lessons=[],
                formatted_warning="No prior failure patterns recorded for this setup.",
            )

        failure_count = len(rows)
        categories = [r["failure_category"] for r in rows if r["failure_category"]]
        lessons = [r["prescriptive_lesson"] for r in rows if r["prescriptive_lesson"]]

        lines = [f"⚠️ <b>Historical Memory Brief for {ticker_clean}:</b>"]
        for i, r in enumerate(rows[:3], 1):
            lines.append(f"  {i}. [{r['failure_category']}] {r['prescriptive_lesson']}")

        formatted = "\n".join(lines)
        logger.info(f"[Playbook Memory] Ingested {failure_count} historical lessons for {ticker_clean}")

        return MemoryWarningReport(
            ticker=ticker_clean,
            market=market,
            strategy=strategy,
            has_historical_failures=True,
            failure_count=failure_count,
            common_failure_categories=list(set(categories)),
            key_lessons=lessons[:3],
            formatted_warning=formatted,
        )
