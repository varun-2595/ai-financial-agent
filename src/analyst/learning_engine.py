"""
EOD Mistake-Learning & Trade Autopsy Engine.
Analyzes losing and winning trades to curate an evolving playbook of market lessons.
"""
from __future__ import annotations

import json
import os
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal, Optional

from google import genai
from google.genai import types

from src.data.models import Strategy, TradePlaybookEntry
from src.utils.config import get_config
from src.utils.logger import logger

DB_PATH = Path(__file__).parent.parent.parent / "data" / "trading.db"


def _conn() -> sqlite3.Connection:
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.row_factory = sqlite3.Row
    return conn


def init_playbook_table() -> None:
    with _conn() as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS trade_playbook (
                id                  INTEGER PRIMARY KEY AUTOINCREMENT,
                ticker              TEXT NOT NULL,
                market              TEXT NOT NULL,
                strategy            TEXT NOT NULL,
                direction           TEXT NOT NULL,
                entry_price         REAL NOT NULL,
                exit_price          REAL NOT NULL,
                realized_pnl        REAL NOT NULL,
                realized_pnl_pct    REAL NOT NULL,
                failure_category    TEXT NOT NULL,
                root_cause          TEXT NOT NULL,
                prescriptive_lesson TEXT NOT NULL,
                created_at          TEXT NOT NULL
            )
        """)
        conn.commit()


class LearningEngine:
    """
    Evaluates completed trades at EOD to extract concrete rules and mistake autopsies.
    """
    def __init__(self, api_key: Optional[str] = None):
        self.config = get_config()
        self.api_key = api_key or os.getenv("GEMINI_API_KEY")
        self.client = genai.Client(api_key=self.api_key) if self.api_key else None
        init_playbook_table()

    def autopsy_losing_trade(self, pos: dict) -> TradePlaybookEntry:
        """
        Runs an institutional loss autopsy on a closed trade.
        """
        ticker = pos["ticker"]
        market = pos["market"]
        strategy = pos.get("strategy", "intraday")
        direction = pos.get("direction", "BUY")
        avg_cost = pos["avg_cost"]
        current_p = pos.get("current_price") or avg_cost
        realized_pnl = pos.get("realized_pnl", 0.0)
        pnl_pct = ((current_p - avg_cost) / avg_cost * 100) if avg_cost else 0.0

        failure_cat = "FALSE_BREAKOUT"
        root_cause = f"Position stopped out at {current_p} from entry {avg_cost}."
        lesson = "Verify 15-minute volume surge (>1.5x) and market trend confirmation before entry."

        if self.client:
            prompt = (
                f"You are a hedge fund risk analyst performing an objective post-mortem on a closed losing trade.\n\n"
                f"Trade Details:\n"
                f"• Ticker: {ticker} ({market.upper()})\n"
                f"• Strategy: {strategy}\n"
                f"• Direction: {direction}\n"
                f"• Entry Price: {avg_cost}\n"
                f"• Exit Price: {current_p}\n"
                f"• Realized P&L: {realized_pnl} ({pnl_pct:+.2f}%)\n\n"
                f"Identify why this trade failed (e.g. false breakout, overextended RSI, stop too tight, market breadth headwind) "
                f"and provide 1 concrete rule to avoid repeating this loss.\n"
                f"Return JSON strictly matching this schema:\n"
                f'{{"failure_category": "string", "root_cause": "string", "prescriptive_lesson": "string"}}'
            )
            try:
                resp = self.client.models.generate_content(
                    model=self.config.llm.model,
                    contents=prompt,
                    config=types.GenerateContentConfig(
                        temperature=0.2,
                        response_mime_type="application/json",
                    ),
                )
                if resp.text:
                    parsed = json.loads(resp.text)
                    failure_cat = parsed.get("failure_category", failure_cat)
                    root_cause = parsed.get("root_cause", root_cause)
                    lesson = parsed.get("prescriptive_lesson", lesson)
            except Exception as exc:
                logger.warning(f"[Learning Engine] Gemini autopsy failed ({exc}); using rule-based lesson.")

        now_iso = datetime.now(timezone.utc).isoformat()
        with _conn() as conn:
            cur = conn.execute("""
                INSERT INTO trade_playbook (
                    ticker, market, strategy, direction, entry_price, exit_price,
                    realized_pnl, realized_pnl_pct, failure_category, root_cause,
                    prescriptive_lesson, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                ticker, market, strategy, direction, avg_cost, current_p,
                realized_pnl, round(pnl_pct, 2), failure_cat, root_cause, lesson, now_iso
            ))
            conn.commit()
            entry_id = cur.lastrowid

        entry = TradePlaybookEntry(
            id=entry_id,
            ticker=ticker,
            market=market,
            strategy=strategy,
            direction=direction,
            entry_price=avg_cost,
            exit_price=current_p,
            realized_pnl=realized_pnl,
            realized_pnl_pct=round(pnl_pct, 2),
            failure_category=failure_cat,
            root_cause=root_cause,
            prescriptive_lesson=lesson,
            created_at=datetime.now(timezone.utc),
        )
        logger.warning(f"[Playbook] 📝 Logged Lesson for {ticker}: {lesson}")
        return entry

    def get_recent_lessons(self, market: Optional[str] = None, limit: int = 5) -> list[TradePlaybookEntry]:
        """Fetches recent lessons from the playbook database."""
        query = "SELECT * FROM trade_playbook"
        params = []
        if market:
            query += " WHERE market = ?"
            params.append(market)
        query += " ORDER BY id DESC LIMIT ?"
        params.append(limit)

        with _conn() as conn:
            rows = conn.execute(query, params).fetchall()
            return [
                TradePlaybookEntry(
                    id=r["id"],
                    ticker=r["ticker"],
                    market=r["market"],
                    strategy=r["strategy"],
                    direction=r["direction"],
                    entry_price=r["entry_price"],
                    exit_price=r["exit_price"],
                    realized_pnl=r["realized_pnl"],
                    realized_pnl_pct=r["realized_pnl_pct"],
                    failure_category=r["failure_category"],
                    root_cause=r["root_cause"],
                    prescriptive_lesson=r["prescriptive_lesson"],
                )
                for r in rows
            ]

    def run_daily_eod_learning(self, market: Literal["india", "us"]) -> dict:
        """
        Reviews today's closed trades, runs autopsies on any losses,
        and returns a performance & learning summary.
        """
        today_date = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        with _conn() as conn:
            rows = conn.execute("""
                SELECT * FROM positions
                WHERE status = 'CLOSED' AND market = ? AND closed_at LIKE ?
            """, (market, f"{today_date}%")).fetchall()
            closed_trades = [dict(r) for r in rows]

        total_trades = len(closed_trades)
        wins = [t for t in closed_trades if (t.get("realized_pnl") or 0) > 0]
        losses = [t for t in closed_trades if (t.get("realized_pnl") or 0) < 0]
        total_pnl = sum(t.get("realized_pnl", 0.0) for t in closed_trades)

        target = self.config.paper_trading.daily_profit_target_inr if market == "india" else self.config.paper_trading.daily_profit_target_usd
        target_met = total_pnl >= target

        autopsies: list[TradePlaybookEntry] = []
        for loss in losses:
            entry = self.autopsy_losing_trade(loss)
            autopsies.append(entry)

        summary = {
            "market": market,
            "total_trades": total_trades,
            "wins": len(wins),
            "losses": len(losses),
            "win_rate_pct": round((len(wins) / total_trades * 100), 1) if total_trades else 0.0,
            "total_pnl": round(total_pnl, 2),
            "daily_target": target,
            "target_met": target_met,
            "autopsies": autopsies,
        }
        logger.info(
            f"[Learning Engine] EOD {market.upper()}: {total_trades} trades, "
            f"PnL: {total_pnl:+.2f} (Target: {target:,.2f} | Met: {target_met})"
        )
        return summary
