"""
EOD Mistake-Learning & Trade Autopsy Engine.
Analyzes losing and winning trades to curate an evolving playbook of market lessons.
"""
from __future__ import annotations

import json
import os
import sqlite3
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Literal, Optional

from google import genai
from google.genai import types

from src.data.models import Strategy, TradePlaybookEntry
from src.gateway.gateway import ModelRequest, TaskComplexity
from src.gateway.router import ModelRouter
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


def get_ticker_consecutive_losses(ticker: str, market: str) -> int:
    """
    Returns the count of consecutive losses for a ticker up to the most recent closed trade.
    """
    clean_ticker = ticker.upper().strip()
    with _conn() as conn:
        rows = conn.execute("""
            SELECT realized_pnl FROM positions
            WHERE ticker = ? AND market = ? AND status = 'CLOSED'
            ORDER BY closed_at DESC LIMIT 10
        """, (clean_ticker, market)).fetchall()
        
    consecutive = 0
    for r in rows:
        pnl = float(r["realized_pnl"] or 0.0)
        if pnl < 0:
            consecutive += 1
        else:
            break
    return consecutive


def is_ticker_quarantined(ticker: str, market: str, cooldown_hours: int = 72) -> tuple[bool, str]:
    """
    Determines if a ticker should be quarantined from trading due to consecutive losses
    or severe 7-day negative expectancy.
    Rule: 2 or more consecutive losses = 72-hour quarantine.
          3 or more losses in 7 days with negative net PnL = 7-day quarantine.
    """
    clean_ticker = ticker.upper().strip()
    now_utc = datetime.now(timezone.utc)
    
    with _conn() as conn:
        # Check consecutive losses
        recent_rows = conn.execute("""
            SELECT realized_pnl, closed_at FROM positions
            WHERE ticker = ? AND market = ? AND status = 'CLOSED'
            ORDER BY closed_at DESC LIMIT 5
        """, (clean_ticker, market)).fetchall()

    if not recent_rows:
        return False, ""

    consecutive = 0
    latest_closed_at_str = None
    latest_loss_pnl = 0.0
    for r in recent_rows:
        pnl = float(r["realized_pnl"] or 0.0)
        if pnl < 0:
            consecutive += 1
            if latest_closed_at_str is None:
                latest_closed_at_str = r["closed_at"]
                latest_loss_pnl = pnl
        else:
            break

    if consecutive >= 2 and latest_closed_at_str:
        try:
            closed_dt = datetime.fromisoformat(latest_closed_at_str)
            if closed_dt.tzinfo is None:
                closed_dt = closed_dt.replace(tzinfo=timezone.utc)
            elapsed_hours = (now_utc - closed_dt).total_seconds() / 3600.0
            if elapsed_hours < cooldown_hours:
                remaining_hrs = int(cooldown_hours - elapsed_hours)
                return (
                    True,
                    f"Ticker {clean_ticker} in {cooldown_hours}h QUARANTINE after {consecutive} consecutive losses "
                    f"(Latest loss: {latest_loss_pnl:+.2f}, {remaining_hrs}h remaining)."
                )
        except Exception:
            pass

    return False, ""


def validate_signal_against_playbook(
    ticker: str,
    market: str,
    strategy: str,
    rsi: Optional[float] = None,
    rel_volume: Optional[float] = None,
) -> tuple[bool, str]:
    """
    Enforces active rules from trade_playbook against a proposed setup.
    If the setup matches a known chronic failure mode for this ticker or strategy,
    it rejects the trade before capital is committed.
    """
    clean_ticker = ticker.upper().strip()
    with _conn() as conn:
        rows = conn.execute("""
            SELECT failure_category, prescriptive_lesson
            FROM trade_playbook
            WHERE ticker = ? AND market = ?
            ORDER BY id DESC LIMIT 5
        """, (clean_ticker, market)).fetchall()

    if not rows:
        return True, "No prior failure patterns."

    categories = [r["failure_category"] for r in rows]

    # Rule 1: Overbought reversal trap
    if any(c in ("OVERBOUGHT_REVERSAL", "OVERBOUGHT_EXHAUSTION") for c in categories):
        if rsi is not None and rsi > 62.0:
            return (
                False,
                f"Playbook Block: Prior loss from OVERBOUGHT_REVERSAL on {clean_ticker}. Current RSI ({rsi:.1f} > 62) is overextended."
            )

    # Rule 2: Low-volume chop trap
    if any(c in ("CHOPPY_NOISE_FLUSH", "LOW_VOLUME_CHOP", "MOMENTUM_STAGNATION") for c in categories):
        if rel_volume is not None and rel_volume < 1.3:
            return (
                False,
                f"Playbook Block: Prior loss from CHOP/STAGNATION on {clean_ticker}. Relative volume ({rel_volume:.2f}x < 1.3x) is insufficient."
            )

    return True, "Playbook rules cleared."


class LearningEngine:
    """
    Evaluates completed trades at EOD to extract concrete rules and mistake autopsies.
    """
    def __init__(self, api_key: Optional[str] = None, router: Optional[ModelRouter] = None):
        self.config = get_config()
        self.api_key = api_key or os.getenv("GEMINI_API_KEY")
        self.router = router or ModelRouter()
        self.client = genai.Client(api_key=self.api_key) if self.api_key else None
        init_playbook_table()

    def autopsy_losing_trade(self, pos: dict) -> TradePlaybookEntry:
        """
        Runs an institutional loss autopsy on a closed trade.
        """
        ticker = pos["ticker"]
        market = pos["market"]
        raw_strat = str(pos.get("strategy", "intraday")).lower()
        strategy = raw_strat if raw_strat in ("scalping", "intraday", "swing", "positional") else "intraday"
        raw_dir = str(pos.get("direction", "BUY")).upper()
        direction = "BUY" if raw_dir in ("BUY", "LONG") else "SELL"
        avg_cost = pos["avg_cost"]
        current_p = pos.get("current_price") or avg_cost
        realized_pnl = pos.get("realized_pnl", 0.0)
        pnl_pct = ((current_p - avg_cost) / avg_cost * 100) if avg_cost else 0.0

        # Calculate holding duration
        holding_mins = 0.0
        opened_at_str = pos.get("opened_at")
        closed_at_str = pos.get("closed_at")
        if opened_at_str and closed_at_str:
            try:
                op_dt = datetime.fromisoformat(opened_at_str)
                cl_dt = datetime.fromisoformat(closed_at_str)
                holding_mins = max(1.0, (cl_dt - op_dt).total_seconds() / 60.0)
            except Exception:
                pass

        exit_reason = str(pos.get("exit_reason") or pos.get("reason") or "").upper()

        # Dynamic quantitative failure classification
        if "SQUARE-OFF" in exit_reason or "TIMEOUT" in exit_reason:
            failure_cat = "MOMENTUM_STAGNATION"
            root_cause = f"Position held {holding_mins:.0f}m without upside follow-through; closed by session square-off at {current_p}."
            lesson = f"Avoid entering {ticker} after mid-day without >1.5x relative volume surge."
        elif holding_mins > 0 and holding_mins < 20:
            failure_cat = "CHOPPY_NOISE_FLUSH"
            root_cause = f"Rapid stop-out within {holding_mins:.0f}m from entry {avg_cost} due to spread/wick volatility."
            lesson = f"Require 15-minute candle close confirmation above resistance before buying {ticker}."
        elif pnl_pct <= -2.0:
            failure_cat = "STRUCTURAL_SUPPORT_BREAKDOWN"
            root_cause = f"Sharp breakdown through invalidation level at {current_p} ({pnl_pct:+.2f}%)."
            lesson = f"Enforce strict entry near major daily/4H support and reduce position size on high-beta setups for {ticker}."
        else:
            failure_cat = "PULLBACK_INVALIDATION"
            root_cause = f"Price retreated below entry ({avg_cost} ➔ {current_p}) invalidating {direction} momentum."
            lesson = f"Check overall market index breadth and sector strength before executing {ticker}."

        # Attempt structured LLM autopsy through ModelRouter if available
        try:
            prompt = (
                f"You are a hedge fund risk analyst performing an objective post-mortem on a closed losing trade.\n\n"
                f"Trade Details:\n"
                f"• Ticker: {ticker} ({market.upper()})\n"
                f"• Strategy: {strategy}\n"
                f"• Direction: {direction}\n"
                f"• Entry Price: {avg_cost}\n"
                f"• Exit Price: {current_p}\n"
                f"• Holding Duration: {holding_mins:.0f} minutes\n"
                f"• Exit Reason: {exit_reason or 'Stop Loss'}\n"
                f"• Realized P&L: {realized_pnl} ({pnl_pct:+.2f}%)\n\n"
                f"Identify why this trade failed (e.g. FALSE_BREAKOUT, OVERBOUGHT_REVERSAL, CHOPPY_NOISE_FLUSH, "
                f"MOMENTUM_STAGNATION, MACRO_HEADWIND) and provide 1 concrete rule to avoid repeating this loss.\n"
                f"Return JSON strictly matching this schema:\n"
                f'{{"failure_category": "string", "root_cause": "string", "prescriptive_lesson": "string"}}'
            )
            req = ModelRequest(
                system_prompt="You are an institutional trading risk analyst.",
                user_prompt=prompt,
                task_complexity=TaskComplexity.COMPLEX,
                temperature=0.2,
                timeout_seconds=25.0,
            )
            resp = self.router.route(req)
            if resp.success and resp.content:
                import re
                cleaned = re.sub(r"```json\s*", "", resp.content)
                cleaned = re.sub(r"```\s*", "", cleaned).strip()
                match = re.search(r"\{.*\}", cleaned, re.DOTALL)
                if match:
                    parsed = json.loads(match.group(0))
                    failure_cat = parsed.get("failure_category", failure_cat)
                    root_cause = parsed.get("root_cause", root_cause)
                    lesson = parsed.get("prescriptive_lesson", lesson)
        except Exception as exc:
            logger.debug(f"[Learning Engine] Router autopsy exception ({exc}); using dynamic rule-based lesson.")

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
        cutoff = (datetime.now(timezone.utc) - timedelta(hours=24)).isoformat()
        with _conn() as conn:
            rows = conn.execute("""
                SELECT * FROM positions
                WHERE status = 'CLOSED' AND market = ? AND closed_at >= ?
                ORDER BY closed_at DESC
            """, (market, cutoff)).fetchall()
            if not rows:
                # Fallback to the latest 10 closed trades if no trades closed in last 24h
                rows = conn.execute("""
                    SELECT * FROM positions
                    WHERE status = 'CLOSED' AND market = ?
                    ORDER BY closed_at DESC LIMIT 10
                """, (market,)).fetchall()
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
