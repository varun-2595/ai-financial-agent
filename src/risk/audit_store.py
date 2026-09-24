"""
Risk Decision Audit Store.

Persists full quantitative risk evaluation snapshots to SQLite,
providing an immutable audit trail for every trade approval, reduction, and rejection.
"""
from __future__ import annotations

import json
import sqlite3
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

from src.risk.models import PortfolioRiskState, RiskAuditRecord, RiskEvaluationResult
from src.utils.logger import logger

DB_PATH = Path(__file__).parent.parent.parent / "data" / "trading.db"


class RiskAuditStore:
    """Manages persistence of risk evaluations and rejection logs."""

    def __init__(self, db_path: Optional[Path] = None):
        self.db_path = db_path or DB_PATH
        self._init_table()

    def _get_conn(self) -> sqlite3.Connection:
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def _init_table(self) -> None:
        """Create risk_audit_log table if not exists."""
        with self._get_conn() as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS risk_audit_log (
                    audit_id            TEXT PRIMARY KEY,
                    timestamp           TEXT NOT NULL,
                    ticker              TEXT NOT NULL,
                    market              TEXT NOT NULL,
                    strategy            TEXT NOT NULL,
                    direction           TEXT NOT NULL,
                    requested_quantity  INTEGER NOT NULL,
                    approved_quantity   INTEGER NOT NULL,
                    decision            TEXT NOT NULL,
                    entry_price         REAL NOT NULL,
                    approved_margin     REAL NOT NULL,
                    nav_at_decision     REAL NOT NULL,
                    cash_at_decision    REAL NOT NULL,
                    drawdown_at_decision REAL NOT NULL,
                    leverage_at_decision REAL NOT NULL,
                    violations_json     TEXT NOT NULL,
                    checks_json         TEXT NOT NULL,
                    reason              TEXT NOT NULL
                )
            """)
            conn.execute("CREATE INDEX IF NOT EXISTS idx_ral_ticker ON risk_audit_log(ticker)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_ral_decision ON risk_audit_log(decision)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_ral_timestamp ON risk_audit_log(timestamp)")
            conn.commit()

    def record_evaluation(
        self,
        eval_result: RiskEvaluationResult,
        risk_state: PortfolioRiskState,
    ) -> str:
        """Record a risk decision into the audit log."""
        audit_id = f"AUD-{uuid.uuid4().hex[:10].upper()}"
        now_str = datetime.now(timezone.utc).isoformat()

        checks_data = [
            {
                "check_name": c.check_name.value,
                "passed": c.passed,
                "limit_value": c.limit_value,
                "current_value": c.current_value,
                "projected_value": c.projected_value,
                "message": c.message,
            }
            for c in eval_result.checks
        ]

        with self._get_conn() as conn:
            conn.execute("""
                INSERT INTO risk_audit_log (
                    audit_id, timestamp, ticker, market, strategy, direction,
                    requested_quantity, approved_quantity, decision, entry_price,
                    approved_margin, nav_at_decision, cash_at_decision,
                    drawdown_at_decision, leverage_at_decision, violations_json,
                    checks_json, reason
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                audit_id,
                now_str,
                eval_result.ticker,
                eval_result.market,
                eval_result.strategy,
                eval_result.direction,
                eval_result.requested_quantity,
                eval_result.approved_quantity,
                eval_result.decision.value,
                eval_result.entry_price,
                eval_result.approved_margin,
                risk_state.nav,
                risk_state.cash,
                risk_state.current_drawdown_pct,
                risk_state.current_leverage,
                json.dumps(eval_result.violations),
                json.dumps(checks_data),
                eval_result.reason,
            ))
            conn.commit()

        logger.debug(
            f"[RiskAudit] Logged decision {eval_result.decision.value} for {eval_result.ticker} "
            f"(Req: {eval_result.requested_quantity}, Appr: {eval_result.approved_quantity}, ID: {audit_id})"
        )
        return audit_id

    def get_audit_history(
        self,
        ticker: Optional[str] = None,
        market: Optional[str] = None,
        decision: Optional[str] = None,
        limit: int = 50,
    ) -> list[dict[str, Any]]:
        """Query recent risk audit records with optional filters."""
        sql = "SELECT * FROM risk_audit_log WHERE 1=1"
        params: list[Any] = []

        if ticker:
            sql += " AND ticker = ?"
            params.append(ticker.upper())
        if market:
            sql += " AND market = ?"
            params.append(market.lower())
        if decision:
            sql += " AND decision = ?"
            params.append(decision.upper())

        sql += " ORDER BY timestamp DESC LIMIT ?"
        params.append(limit)

        with self._get_conn() as conn:
            rows = conn.execute(sql, params).fetchall()
            results = []
            for row in rows:
                item = dict(row)
                item["violations"] = json.loads(item["violations_json"])
                item["checks"] = json.loads(item["checks_json"])
                results.append(item)
            return results
