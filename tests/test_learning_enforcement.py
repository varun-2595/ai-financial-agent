"""
Unit tests for Closed-Loop Learning, Ticker Quarantine, Playbook Enforcement, and Diagnostic Cards.
"""
import pytest
from datetime import datetime, timezone, timedelta
from unittest.mock import patch

from src.analyst.learning_engine import (
    get_ticker_consecutive_losses,
    is_ticker_quarantined,
    validate_signal_against_playbook,
    LearningEngine,
    init_playbook_table,
)
from src.data.models import StockSnapshot, TradeSignal
from src.trading.paper_engine import PaperTradingEngine


def test_consecutive_losses_and_quarantine(tmp_path):
    test_db = tmp_path / "test_trading.db"
    with patch("src.analyst.learning_engine.DB_PATH", test_db), \
         patch("src.db.trading_store.DB_PATH", test_db), \
         patch("src.journal.journal_store.DB_PATH", test_db), \
         patch("src.risk.audit_store.DB_PATH", test_db):

        from src.db.trading_store import init_trading_db, _conn
        init_trading_db()
        init_playbook_table()

        # Initially zero consecutive losses
        assert get_ticker_consecutive_losses("ACMESOLAR.NS", "india") == 0
        quarantined, _ = is_ticker_quarantined("ACMESOLAR.NS", "india")
        assert not quarantined

        now = datetime.now(timezone.utc).isoformat()
        with _conn() as conn:
            # Insert 1 losing trade
            conn.execute("""
                INSERT INTO positions (
                    ticker, market, strategy, direction, quantity, avg_cost, current_price,
                    status, is_paper, opened_at, closed_at, realized_pnl
                ) VALUES ('ACMESOLAR.NS', 'india', 'intraday', 'LONG', 10, 100.0, 98.0, 'CLOSED', 1, ?, ?, -20.0)
            """, (now, now))
            conn.commit()

        # 1 loss: consecutive is 1, not yet quarantined (rule requires >= 2)
        assert get_ticker_consecutive_losses("ACMESOLAR.NS", "india") == 1
        quarantined, _ = is_ticker_quarantined("ACMESOLAR.NS", "india")
        assert not quarantined

        # Insert 2nd consecutive losing trade
        with _conn() as conn:
            conn.execute("""
                INSERT INTO positions (
                    ticker, market, strategy, direction, quantity, avg_cost, current_price,
                    status, is_paper, opened_at, closed_at, realized_pnl
                ) VALUES ('ACMESOLAR.NS', 'india', 'intraday', 'LONG', 10, 100.0, 97.0, 'CLOSED', 1, ?, ?, -30.0)
            """, (now, now))
            conn.commit()

        # 2 consecutive losses: quarantine MUST trigger
        assert get_ticker_consecutive_losses("ACMESOLAR.NS", "india") == 2
        quarantined, reason = is_ticker_quarantined("ACMESOLAR.NS", "india")
        assert quarantined
        assert "QUARANTINE" in reason
        assert "2 consecutive losses" in reason


def test_playbook_signal_validation(tmp_path):
    test_db = tmp_path / "test_trading.db"
    with patch("src.analyst.learning_engine.DB_PATH", test_db), \
         patch("src.db.trading_store.DB_PATH", test_db):

        init_playbook_table()
        learning = LearningEngine()

        # Log an autopsy with OVERBOUGHT_REVERSAL
        pos = {
            "ticker": "COIN",
            "market": "us",
            "strategy": "intraday",
            "direction": "BUY",
            "avg_cost": 200.0,
            "current_price": 195.0,
            "realized_pnl": -15.0,
            "exit_reason": "STOP LOSS HIT",
        }
        # Force rule-based autopsy
        learning.router = None
        learning.client = None
        learning.autopsy_losing_trade(pos)

        # In trade_playbook, insert explicitly an OVERBOUGHT_REVERSAL rule for COIN
        from src.analyst.learning_engine import _conn
        with _conn() as conn:
            conn.execute("""
                INSERT INTO trade_playbook (
                    ticker, market, strategy, direction, entry_price, exit_price,
                    realized_pnl, realized_pnl_pct, failure_category, root_cause,
                    prescriptive_lesson, created_at
                ) VALUES ('COIN', 'us', 'intraday', 'BUY', 200.0, 195.0, -15.0, -2.5,
                          'OVERBOUGHT_REVERSAL', 'RSI 75 at entry', 'Never buy COIN when RSI > 62', ?)
            """, (datetime.now(timezone.utc).isoformat(),))
            conn.commit()

        # Testing with overextended RSI (> 62): should be REJECTED
        allowed, reason = validate_signal_against_playbook("COIN", "us", "intraday", rsi=70.0, rel_volume=1.5)
        assert not allowed
        assert "OVERBOUGHT_REVERSAL" in reason

        # Testing with safe RSI (< 62): should pass
        allowed, reason = validate_signal_against_playbook("COIN", "us", "intraday", rsi=52.0, rel_volume=1.5)
        assert allowed


def test_diagnostic_exit_card_formatting(tmp_path):
    test_db = tmp_path / "test_trading.db"
    with patch("src.analyst.learning_engine.DB_PATH", test_db), \
         patch("src.db.trading_store.DB_PATH", test_db), \
         patch("src.journal.journal_store.DB_PATH", test_db), \
         patch("src.risk.audit_store.DB_PATH", test_db):

        from src.db.trading_store import init_trading_db
        init_trading_db()

        engine = PaperTradingEngine()
        engine.reset_account_balances()

        signal = TradeSignal(
            ticker="AAPL",
            market="us",
            direction="BUY",
            strategy="intraday",
            entry_price=220.0,
            stop_loss=216.0,
            target_price=226.0,
            quantity=5,
            confidence=0.85,
            reasoning="Momentum breakout",
        )
        engine.execute_signal(signal)

        # Trigger Stop Loss
        snapshots = {
            "AAPL": StockSnapshot(
                ticker="AAPL",
                market="us",
                currency="USD",
                current_price=215.50,
                history=[],
            )
        }
        reports = engine.evaluate_open_positions(market="us", latest_snapshots=snapshots)
        assert len(reports) == 1
        rep = reports[0]

        # Verify rich diagnostic HTML elements
        assert "🛑 <b>STOP-LOSS EXECUTED: AAPL (US)</b>" in rep
        assert "Net P&L:" in rep
        assert "R-Multiple:" in rep
        assert "Diagnostics:" in rep
        assert "Action:" in rep
        assert "$" in rep  # US currency symbol
