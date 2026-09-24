"""
Persistent Storage and Retrieval for the Immutable Decision Journal & Post-Trade Evaluations.
Phase 7: Decision observability, attribution tracking, and agent scorecards.
Backed by SQLite in WAL mode.
"""
from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

from src.journal.models import (
    AgentAttributionScore,
    AgentScorecard,
    DecisionJournalEntry,
    TradeEvaluation,
)
from src.utils.logger import logger

DB_PATH = Path(__file__).parent.parent.parent / "data" / "trading.db"


class DecisionJournalStore:
    """SQLite-backed store for trade decisions, post-mortem evaluations, and agent scorecards."""

    def __init__(self, db_path: Optional[Path] = None):
        self.db_path = db_path or DB_PATH
        self._init_db()

    def _conn(self) -> sqlite3.Connection:
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(self.db_path)
        conn.execute("PRAGMA journal_mode=WAL")
        conn.row_factory = sqlite3.Row
        return conn

    def _init_db(self) -> None:
        """Create decision_journal and trade_evaluations tables if not present."""
        with self._conn() as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS decision_journal (
                    journal_id             TEXT PRIMARY KEY,
                    timestamp              TEXT NOT NULL,
                    symbol                 TEXT NOT NULL,
                    market                 TEXT NOT NULL,
                    strategy               TEXT NOT NULL,
                    direction              TEXT NOT NULL,
                    market_snapshot        TEXT NOT NULL,
                    agent_outputs          TEXT NOT NULL,
                    confidence             REAL NOT NULL,
                    evidence               TEXT NOT NULL,
                    final_thesis           TEXT NOT NULL,
                    risk_decision          TEXT NOT NULL,
                    risk_reasons           TEXT NOT NULL,
                    requested_quantity     INTEGER NOT NULL,
                    approved_quantity      INTEGER NOT NULL,
                    entry_price            REAL NOT NULL,
                    stop_loss              REAL NOT NULL,
                    target_price           REAL NOT NULL,
                    order_id               TEXT,
                    position_id            INTEGER,
                    status                 TEXT NOT NULL DEFAULT 'PROPOSED',
                    exit_price             REAL,
                    exit_timestamp         TEXT,
                    exit_reason            TEXT,
                    realized_pnl           REAL,
                    return_pct             REAL,
                    holding_period_seconds REAL,
                    created_at             TEXT NOT NULL,
                    updated_at             TEXT NOT NULL
                )
            """)

            conn.execute("""
                CREATE TABLE IF NOT EXISTS trade_evaluations (
                    evaluation_id            TEXT PRIMARY KEY,
                    journal_id               TEXT NOT NULL,
                    symbol                   TEXT NOT NULL,
                    market                   TEXT NOT NULL,
                    direction                TEXT NOT NULL,
                    realized_pnl             REAL NOT NULL,
                    return_pct               REAL NOT NULL,
                    is_winner                INTEGER NOT NULL,
                    holding_period_seconds   REAL NOT NULL,
                    directional_accuracy     REAL NOT NULL,
                    thesis_accuracy          REAL NOT NULL,
                    thesis_notes             TEXT NOT NULL,
                    agent_accuracy           TEXT NOT NULL,
                    risk_decision_evaluation TEXT NOT NULL,
                    major_failure_reason     TEXT NOT NULL,
                    failure_details          TEXT NOT NULL,
                    evaluated_at             TEXT NOT NULL,
                    FOREIGN KEY(journal_id) REFERENCES decision_journal(journal_id)
                )
            """)

            # Indices for rapid querying
            conn.execute("CREATE INDEX IF NOT EXISTS idx_journal_symbol ON decision_journal(symbol)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_journal_pos_id ON decision_journal(position_id)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_journal_status ON decision_journal(status)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_eval_journal_id ON trade_evaluations(journal_id)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_eval_symbol ON trade_evaluations(symbol)")
            conn.commit()

    # ── Decision Journal CRUD ──────────────────────────────────────────────────

    def record_decision(self, entry: DecisionJournalEntry) -> str:
        """Store immutable pre-trade decision snapshot."""
        now_str = datetime.now(timezone.utc).isoformat()
        ts_str = entry.timestamp.isoformat() if isinstance(entry.timestamp, datetime) else str(entry.timestamp)

        with self._conn() as conn:
            conn.execute("""
                INSERT INTO decision_journal (
                    journal_id, timestamp, symbol, market, strategy, direction,
                    market_snapshot, agent_outputs, confidence, evidence,
                    final_thesis, risk_decision, risk_reasons,
                    requested_quantity, approved_quantity, entry_price,
                    stop_loss, target_price, order_id, position_id,
                    status, exit_price, exit_timestamp, exit_reason,
                    realized_pnl, return_pct, holding_period_seconds,
                    created_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                entry.journal_id,
                ts_str,
                entry.symbol.upper(),
                entry.market.lower(),
                entry.strategy,
                entry.direction.upper(),
                json.dumps(entry.market_snapshot),
                json.dumps(self._serialize_agent_outputs(entry.agent_outputs)),
                entry.confidence,
                json.dumps(entry.evidence),
                entry.final_thesis,
                entry.risk_decision,
                json.dumps(entry.risk_reasons),
                entry.requested_quantity,
                entry.approved_quantity,
                entry.entry_price,
                entry.stop_loss,
                entry.target_price,
                entry.order_id,
                entry.position_id,
                entry.status,
                entry.exit_price,
                entry.exit_timestamp.isoformat() if entry.exit_timestamp else None,
                entry.exit_reason,
                entry.realized_pnl,
                entry.return_pct,
                entry.holding_period_seconds,
                now_str,
                now_str,
            ))
            conn.commit()

        logger.info(f"[DecisionJournal] Saved decision {entry.journal_id} for {entry.symbol} (Status={entry.status})")
        return entry.journal_id

    def get_decision(self, journal_id: str) -> Optional[DecisionJournalEntry]:
        """Fetch decision record by ID."""
        with self._conn() as conn:
            row = conn.execute(
                "SELECT * FROM decision_journal WHERE journal_id = ?", (journal_id,)
            ).fetchone()
            return self._row_to_entry(row) if row else None

    def get_decision_by_position_id(self, position_id: int) -> Optional[DecisionJournalEntry]:
        """Lookup decision associated with an active or closed position."""
        with self._conn() as conn:
            row = conn.execute(
                "SELECT * FROM decision_journal WHERE position_id = ? ORDER BY timestamp DESC LIMIT 1",
                (position_id,)
            ).fetchone()
            return self._row_to_entry(row) if row else None

    def update_decision_execution(
        self,
        journal_id: str,
        order_id: str,
        position_id: Optional[int] = None,
        status: str = "EXECUTED",
    ) -> None:
        """Attach executed order and position IDs to decision record."""
        now_str = datetime.now(timezone.utc).isoformat()
        with self._conn() as conn:
            conn.execute("""
                UPDATE decision_journal
                SET order_id = ?, position_id = ?, status = ?, updated_at = ?
                WHERE journal_id = ?
            """, (order_id, position_id, status, now_str, journal_id))
            conn.commit()

    def update_decision_exit(
        self,
        journal_id: str,
        exit_price: float,
        exit_timestamp: datetime,
        exit_reason: str,
        realized_pnl: float,
        return_pct: float,
        holding_period_seconds: float,
    ) -> None:
        """Update decision record with final closed trade metrics."""
        now_str = datetime.now(timezone.utc).isoformat()
        ts_str = exit_timestamp.isoformat() if isinstance(exit_timestamp, datetime) else str(exit_timestamp)
        with self._conn() as conn:
            conn.execute("""
                UPDATE decision_journal
                SET exit_price = ?, exit_timestamp = ?, exit_reason = ?,
                    realized_pnl = ?, return_pct = ?, holding_period_seconds = ?,
                    status = 'CLOSED', updated_at = ?
                WHERE journal_id = ?
            """, (
                exit_price, ts_str, exit_reason,
                realized_pnl, return_pct, holding_period_seconds,
                now_str, journal_id
            ))
            conn.commit()

    # ── Post-Trade Evaluation CRUD ─────────────────────────────────────────────

    def record_evaluation(self, evaluation: TradeEvaluation) -> str:
        """Persist post-trade evaluation."""
        eval_dict = evaluation.model_dump()
        with self._conn() as conn:
            conn.execute("""
                INSERT INTO trade_evaluations (
                    evaluation_id, journal_id, symbol, market, direction,
                    realized_pnl, return_pct, is_winner, holding_period_seconds,
                    directional_accuracy, thesis_accuracy, thesis_notes,
                    agent_accuracy, risk_decision_evaluation,
                    major_failure_reason, failure_details, evaluated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(evaluation_id) DO UPDATE SET
                    realized_pnl = excluded.realized_pnl,
                    return_pct = excluded.return_pct,
                    is_winner = excluded.is_winner,
                    directional_accuracy = excluded.directional_accuracy,
                    thesis_accuracy = excluded.thesis_accuracy,
                    thesis_notes = excluded.thesis_notes,
                    agent_accuracy = excluded.agent_accuracy,
                    risk_decision_evaluation = excluded.risk_decision_evaluation,
                    major_failure_reason = excluded.major_failure_reason,
                    failure_details = excluded.failure_details,
                    evaluated_at = excluded.evaluated_at
            """, (
                evaluation.evaluation_id,
                evaluation.journal_id,
                evaluation.symbol,
                evaluation.market,
                evaluation.direction,
                evaluation.realized_pnl,
                evaluation.return_pct,
                1 if evaluation.is_winner else 0,
                evaluation.holding_period_seconds,
                evaluation.directional_accuracy,
                evaluation.thesis_accuracy,
                evaluation.thesis_notes,
                json.dumps({k: v.model_dump() for k, v in evaluation.agent_accuracy.items()}),
                evaluation.risk_decision_evaluation,
                evaluation.major_failure_reason,
                evaluation.failure_details,
                evaluation.evaluated_at.isoformat(),
            ))
            conn.commit()
        return evaluation.evaluation_id

    def get_evaluation(self, journal_id: str) -> Optional[TradeEvaluation]:
        """Fetch evaluation by associated journal_id."""
        with self._conn() as conn:
            row = conn.execute(
                "SELECT * FROM trade_evaluations WHERE journal_id = ?", (journal_id,)
            ).fetchone()
            return self._row_to_evaluation(row) if row else None

    def get_evaluations(self, symbol: Optional[str] = None, limit: int = 100) -> list[TradeEvaluation]:
        """Retrieve recent trade evaluations."""
        query = "SELECT * FROM trade_evaluations"
        params = []
        if symbol:
            query += " WHERE symbol = ?"
            params.append(symbol.upper())
        query += " ORDER BY evaluated_at DESC LIMIT ?"
        params.append(limit)

        with self._conn() as conn:
            rows = conn.execute(query, params).fetchall()
            return [self._row_to_evaluation(r) for r in rows]

    def get_journal_entries(
        self,
        symbol: Optional[str] = None,
        status: Optional[str] = None,
        limit: int = 100,
    ) -> list[DecisionJournalEntry]:
        """Retrieve journal history."""
        query = "SELECT * FROM decision_journal WHERE 1=1"
        params = []
        if symbol:
            query += " AND symbol = ?"
            params.append(symbol.upper())
        if status:
            query += " AND status = ?"
            params.append(status.upper())
        query += " ORDER BY timestamp DESC LIMIT ?"
        params.append(limit)

        with self._conn() as conn:
            rows = conn.execute(query, params).fetchall()
            return [self._row_to_entry(r) for r in rows]

    def get_recent_decisions(self, limit: int = 5) -> list[DecisionJournalEntry]:
        """Fetch the most recent decision entries."""
        return self.get_journal_entries(limit=limit)

    def get_recent_closed_trades(self, limit: int = 5) -> list[dict[str, Any]]:
        """
        Fetch recent closed trades joining decision journal and trade evaluations
        for Telegram /journal reporting.
        """
        with self._conn() as conn:
            rows = conn.execute("""
                SELECT
                    j.journal_id, j.symbol, j.market, j.strategy, j.direction,
                    j.entry_price, j.exit_price, j.realized_pnl, j.return_pct,
                    j.holding_period_seconds, j.final_thesis, j.agent_outputs,
                    j.exit_reason, j.exit_timestamp,
                    e.thesis_notes, e.major_failure_reason, e.failure_details
                FROM decision_journal j
                LEFT JOIN trade_evaluations e ON j.journal_id = e.journal_id
                WHERE j.status = 'CLOSED' AND j.realized_pnl IS NOT NULL
                ORDER BY j.exit_timestamp DESC, j.updated_at DESC
                LIMIT ?
            """, (limit,)).fetchall()

            results: list[dict[str, Any]] = []
            for r in rows:
                agent_outs = json.loads(r["agent_outputs"]) if r["agent_outputs"] else {}
                pm_decision = agent_outs.get("PortfolioManagerAgent", {})
                pm_driver = ""
                if isinstance(pm_decision, dict):
                    reasons = pm_decision.get("reasons", [])
                    if reasons:
                        pm_driver = reasons[0]
                if not pm_driver:
                    pm_driver = r["final_thesis"][:100]

                learning_note = r["thesis_notes"] or r["failure_details"] or r["exit_reason"] or "Standard rule exit"

                results.append({
                    "symbol": r["symbol"],
                    "market": r["market"],
                    "strategy": r["strategy"],
                    "direction": r["direction"],
                    "entry_price": r["entry_price"],
                    "exit_price": r["exit_price"],
                    "realized_pnl": r["realized_pnl"],
                    "return_pct": r["return_pct"],
                    "holding_period_seconds": r["holding_period_seconds"] or 0.0,
                    "pm_driver": pm_driver,
                    "learning_note": learning_note,
                    "exit_reason": r["exit_reason"],
                })
            return results

    def get_aggregate_scorecard(self, days: Optional[int] = None) -> dict[str, Any]:
        """
        Aggregate lifetime and trailing performance directly via SQL aggregations
        across all closed journal entries.
        """
        query = """
            SELECT
                symbol, market, direction, realized_pnl, return_pct, holding_period_seconds
            FROM decision_journal
            WHERE status = 'CLOSED' AND realized_pnl IS NOT NULL
        """
        params: list[Any] = []
        if days:
            query += " AND datetime(exit_timestamp) >= datetime('now', ?)"
            params.append(f"-{days} days")

        query += " ORDER BY exit_timestamp ASC"

        with self._conn() as conn:
            rows = conn.execute(query, params).fetchall()

        if not rows:
            return {
                "total_trades": 0,
                "winning_trades": 0,
                "losing_trades": 0,
                "win_rate_pct": 0.0,
                "profit_factor": 0.0,
                "cumulative_pnl_inr": 0.0,
                "cumulative_pnl_usd": 0.0,
                "avg_win": 0.0,
                "avg_loss": 0.0,
                "win_loss_ratio": 0.0,
                "max_consecutive_losses": 0,
            }

        total_trades = len(rows)
        wins = [r for r in rows if (r["realized_pnl"] or 0) > 0]
        losses = [r for r in rows if (r["realized_pnl"] or 0) < 0]

        winning_trades = len(wins)
        losing_trades = len(losses)
        win_rate_pct = round((winning_trades / total_trades) * 100, 1) if total_trades > 0 else 0.0

        gross_profit = sum((r["realized_pnl"] or 0) for r in wins)
        gross_loss = sum(abs(r["realized_pnl"] or 0) for r in losses)
        profit_factor = round(gross_profit / gross_loss, 2) if gross_loss > 0 else (round(gross_profit, 2) if gross_profit > 0 else 1.0)

        pnl_inr = sum((r["realized_pnl"] or 0) for r in rows if str(r["market"]).lower() == "india")
        pnl_usd = sum((r["realized_pnl"] or 0) for r in rows if str(r["market"]).lower() == "us")

        avg_win = round(gross_profit / winning_trades, 2) if winning_trades > 0 else 0.0
        avg_loss = round(gross_loss / losing_trades, 2) if losing_trades > 0 else 0.0
        win_loss_ratio = round(avg_win / avg_loss, 2) if avg_loss > 0 else (round(avg_win, 2) if avg_win > 0 else 1.0)

        # Max consecutive losses
        max_consec_losses = 0
        current_streak = 0
        for r in rows:
            if (r["realized_pnl"] or 0) < 0:
                current_streak += 1
                if current_streak > max_consec_losses:
                    max_consec_losses = current_streak
            else:
                current_streak = 0

        return {
            "total_trades": total_trades,
            "winning_trades": winning_trades,
            "losing_trades": losing_trades,
            "win_rate_pct": win_rate_pct,
            "profit_factor": profit_factor,
            "cumulative_pnl_inr": round(pnl_inr, 2),
            "cumulative_pnl_usd": round(pnl_usd, 2),
            "avg_win": avg_win,
            "avg_loss": avg_loss,
            "win_loss_ratio": win_loss_ratio,
            "max_consecutive_losses": max_consec_losses,
        }

    # ── Helpers ────────────────────────────────────────────────────────────────

    def _serialize_agent_outputs(self, outputs: dict[str, Any]) -> dict[str, Any]:
        serialized = {}
        for k, v in outputs.items():
            if hasattr(v, "model_dump"):
                serialized[k] = v.model_dump()
            elif isinstance(v, dict):
                serialized[k] = v
            else:
                serialized[k] = str(v)
        return serialized

    def _row_to_entry(self, row: sqlite3.Row) -> DecisionJournalEntry:
        return DecisionJournalEntry(
            journal_id=row["journal_id"],
            timestamp=datetime.fromisoformat(row["timestamp"]),
            symbol=row["symbol"],
            market=row["market"],
            strategy=row["strategy"],
            direction=row["direction"],
            market_snapshot=json.loads(row["market_snapshot"]),
            agent_outputs=json.loads(row["agent_outputs"]),
            confidence=float(row["confidence"]),
            evidence=json.loads(row["evidence"]),
            final_thesis=row["final_thesis"],
            risk_decision=row["risk_decision"],
            risk_reasons=json.loads(row["risk_reasons"]),
            requested_quantity=int(row["requested_quantity"]),
            approved_quantity=int(row["approved_quantity"]),
            entry_price=float(row["entry_price"]),
            stop_loss=float(row["stop_loss"]),
            target_price=float(row["target_price"]),
            order_id=row["order_id"],
            position_id=row["position_id"],
            status=row["status"],
            exit_price=float(row["exit_price"]) if row["exit_price"] is not None else None,
            exit_timestamp=datetime.fromisoformat(row["exit_timestamp"]) if row["exit_timestamp"] else None,
            exit_reason=row["exit_reason"],
            realized_pnl=float(row["realized_pnl"]) if row["realized_pnl"] is not None else None,
            return_pct=float(row["return_pct"]) if row["return_pct"] is not None else None,
            holding_period_seconds=float(row["holding_period_seconds"]) if row["holding_period_seconds"] is not None else None,
        )

    def _row_to_evaluation(self, row: sqlite3.Row) -> TradeEvaluation:
        agent_acc_dict = json.loads(row["agent_accuracy"])
        agent_accuracy = {
            k: AgentAttributionScore(**v) for k, v in agent_acc_dict.items()
        }
        return TradeEvaluation(
            evaluation_id=row["evaluation_id"],
            journal_id=row["journal_id"],
            symbol=row["symbol"],
            market=row["market"],
            direction=row["direction"],
            realized_pnl=float(row["realized_pnl"]),
            return_pct=float(row["return_pct"]),
            is_winner=bool(row["is_winner"]),
            holding_period_seconds=float(row["holding_period_seconds"]),
            directional_accuracy=float(row["directional_accuracy"]),
            thesis_accuracy=float(row["thesis_accuracy"]),
            thesis_notes=row["thesis_notes"],
            agent_accuracy=agent_accuracy,
            risk_decision_evaluation=row["risk_decision_evaluation"],
            major_failure_reason=row["major_failure_reason"],
            failure_details=row["failure_details"],
            evaluated_at=datetime.fromisoformat(row["evaluated_at"]),
        )
