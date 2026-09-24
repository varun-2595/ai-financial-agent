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

    # ── Agent Performance Scorecards (Observability Only) ──────────────────────

    def get_agent_scorecards(self) -> dict[str, AgentScorecard]:
        """
        Aggregate post-trade attribution metrics across all historical evaluations.
        Strictly for observability and performance measurement.
        """
        scorecards: dict[str, AgentScorecard] = {}
        evals = self.get_evaluations(limit=10_000)

        for ev in evals:
            pnl = ev.realized_pnl
            ret = ev.return_pct

            for agent_name, att in ev.agent_accuracy.items():
                if agent_name not in scorecards:
                    scorecards[agent_name] = AgentScorecard(agent_name=agent_name)

                sc = scorecards[agent_name]
                sc.total_evaluations += 1
                sig = att.signal.upper()

                if sig == "BUY":
                    sc.bullish_calls += 1
                    sc.avg_return_when_bullish += ret
                elif sig == "SELL":
                    sc.bearish_calls += 1
                    sc.avg_return_when_bearish += ret
                else:
                    sc.neutral_calls += 1

                if att.directional_accuracy == 1.0:
                    sc.correct_calls += 1
                elif att.directional_accuracy == 0.0:
                    sc.incorrect_calls += 1

                sc.avg_confidence += att.confidence
                sc.brier_calibration_score += att.brier_score_loss

                # PnL attribution
                if (sig == "BUY" and ev.direction == "BUY") or (sig == "SELL" and ev.direction == "SELL"):
                    sc.total_pnl_attributed += pnl

        # Normalize averages
        for agent_name, sc in scorecards.items():
            n = sc.total_evaluations
            if n > 0:
                sc.accuracy_rate = round(sc.correct_calls / n, 4)
                sc.avg_confidence = round(sc.avg_confidence / n, 4)
                sc.brier_calibration_score = round(sc.brier_calibration_score / n, 4)
                sc.total_pnl_attributed = round(sc.total_pnl_attributed, 2)
            
            decisive = sc.correct_calls + sc.incorrect_calls
            if decisive > 0:
                sc.win_rate = round(sc.correct_calls / decisive, 4)

            if sc.bullish_calls > 0:
                sc.avg_return_when_bullish = round(sc.avg_return_when_bullish / sc.bullish_calls, 2)
            if sc.bearish_calls > 0:
                sc.avg_return_when_bearish = round(sc.avg_return_when_bearish / sc.bearish_calls, 2)

        return scorecards

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
