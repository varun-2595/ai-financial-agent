"""
Comprehensive test suite for 24/7 Crypto Market Integration.
Tests crypto data normalization, live snapshot structure, paper engine accounting,
risk management, fast tick monitor 24/7 bypass, and notification rendering.
"""
from datetime import datetime, timezone
from unittest.mock import MagicMock, patch

import pytest

from src.data.fetcher_crypto import (
    normalize_crypto_ticker,
    to_exchange_symbol,
)
from src.data.models import Order, Quote, TradeSignal, StockSnapshot, Fundamentals
from src.trading.paper_engine import PaperTradingEngine
from src.trading.fast_tick_monitor import FastTickMonitor


def test_crypto_ticker_normalization():
    assert normalize_crypto_ticker("BTC") == "BTC/USDT"
    assert normalize_crypto_ticker("btc/usdt") == "BTC/USDT"
    assert normalize_crypto_ticker("ETHUSDT") == "ETH/USDT"
    assert normalize_crypto_ticker("SOL-USDT") == "SOL/USDT"
    assert normalize_crypto_ticker("DOGE") == "DOGE/USDT"

    assert to_exchange_symbol("BTC/USDT") == "BTCUSDT"
    assert to_exchange_symbol("ETH-USDT") == "ETHUSDT"
    assert to_exchange_symbol("SOL") == "SOLUSDT"


def test_crypto_paper_engine_accounting(tmp_path):
    test_db = tmp_path / "test_trading.db"

    with patch("src.db.trading_store.DB_PATH", test_db), \
         patch("src.journal.journal_store.DB_PATH", test_db), \
         patch("src.risk.audit_store.DB_PATH", test_db):

        from src.db.trading_store import init_trading_db
        init_trading_db()

        engine = PaperTradingEngine()
        engine.reset_account_balances()

        # Check initial crypto balance is 1,000 USDT
        summary = engine.get_portfolio_summary("crypto")
        assert summary["cash"] == 1000.0
        assert summary["total_value"] == 1000.0
        assert summary["open_positions_count"] == 0

        # Execute a crypto BUY signal
        signal = TradeSignal(
            ticker="BTC/USDT",
            market="crypto",
            direction="BUY",
            strategy="scalping",
            entry_price=60000.0,
            stop_loss=58000.0,
            target_price=64000.0,
            quantity=0.01,
            confidence=0.85,
            reasoning="Bullish momentum breakout",
        )

        order = engine.execute_signal(signal)
        assert order is not None
        assert order.status == "FILLED"
        assert order.market == "crypto"
        assert order.ticker == "BTC/USDT"

        # Check updated portfolio summary
        updated_summary = engine.get_portfolio_summary("crypto")
        assert updated_summary["open_positions_count"] == 1
        assert updated_summary["cash"] < 1000.0  # Cash deducted for margin + fee

        # Check fee schedule
        fees = engine.fees.compute_fees(
            market="crypto",
            direction="BUY",
            filled_price=order.filled_price,
            quantity=order.quantity,
            is_paper=True,
        )
        assert fees > 0.0  # Flat 0.05% fee applied

        # Evaluate positions at target price (should close trade with profit)
        mock_snapshot = StockSnapshot(
            ticker="BTC/USDT",
            name="BTC / Tether USD",
            market="crypto",
            sector="Crypto Assets",
            currency="USDT",
            is_etf=False,
            current_price=64500.0,
            price_change_pct_1d=7.5,
            history=[],
            fundamentals=Fundamentals(avg_volume_30d=10000000),
            data_source="binance_public",
        )

        reports = engine.evaluate_open_positions("crypto", {"BTC/USDT": mock_snapshot})
        assert len(reports) == 1
        assert "TARGET HIT" in reports[0]

        # Verify position closed
        final_summary = engine.get_portfolio_summary("crypto")
        assert final_summary["open_positions_count"] == 0
        assert final_summary["cash"] > 1000.0  # Net profit realized


def test_fast_tick_monitor_247_crypto(tmp_path):
    test_db = tmp_path / "test_trading.db"

    with patch("src.db.trading_store.DB_PATH", test_db), \
         patch("src.journal.journal_store.DB_PATH", test_db), \
         patch("src.risk.audit_store.DB_PATH", test_db):

        from src.db.trading_store import init_trading_db
        init_trading_db()

        engine = PaperTradingEngine()
        engine.reset_account_balances()

        # Open a crypto position
        signal = TradeSignal(
            ticker="ETH/USDT",
            market="crypto",
            direction="BUY",
            strategy="scalping",
            entry_price=3000.0,
            stop_loss=2900.0,
            target_price=3300.0,
            quantity=0.1,
            confidence=0.9,
            reasoning="Ethereum layer-2 volume surge",
        )
        engine.execute_signal(signal)

        # FastTickMonitor should check crypto regardless of stock market hours
        monitor = FastTickMonitor(
            tick_interval_seconds=1.0,
            price_fetcher=lambda ticker, market: 2850.0,
        )

        exits = monitor.tick_once()
        assert len(exits) == 1
        assert "ETH/USDT" in exits[0]
        assert "STOP LOSS HIT" in exits[0]

        # Open positions should now be 0
        assert engine.get_portfolio_summary("crypto")["open_positions_count"] == 0


@pytest.mark.asyncio
async def test_telegram_status_crypto(tmp_path):
    from unittest.mock import AsyncMock
    from src.notifier.telegram_bot import cmd_status

    test_db = tmp_path / "test_trading.db"
    with patch("src.db.trading_store.DB_PATH", test_db), \
         patch("src.journal.journal_store.DB_PATH", test_db), \
         patch("src.risk.audit_store.DB_PATH", test_db), \
         patch("src.utils.state_manager.DB_PATH", test_db), \
         patch("src.notifier.telegram_bot._is_authorized", return_value=True):

        from src.db.trading_store import init_trading_db
        init_trading_db()

        mock_update = MagicMock()
        mock_update.message = MagicMock()
        mock_update.message.reply_text = AsyncMock()
        mock_context = MagicMock()

        await cmd_status(mock_update, mock_context)

        mock_update.message.reply_text.assert_called_once()
        text_arg = mock_update.message.reply_text.call_args[0][0]

        assert "Crypto (24/7/365):" in text_arg
        assert "24/7 ACTIVE" in text_arg
        assert "₮1,000.00" in text_arg


def test_scheduler_runner_crypto_job():
    import sys
    from unittest.mock import MagicMock
    if "platformdirs" not in sys.modules:
        sys.modules["platformdirs"] = MagicMock()
    from src.scheduler.runner import build_scheduler
    sched = build_scheduler(background=True)
    job_ids = [j.id for j in sched.get_jobs()]
    assert "crypto_247_intraday_scan" in job_ids
