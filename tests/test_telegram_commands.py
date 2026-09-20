"""
Unit tests for Telegram Bot State Manager and Commands.
"""
import os
import pytest
from unittest.mock import MagicMock

from src.utils.state_manager import is_trading_paused, set_trading_paused, init_state_table
from src.trading.paper_engine import PaperTradingEngine
from src.screener.watchlist_manager import init_watchlist, add_ticker, remove_ticker, get_active_tickers
from src.notifier.telegram_bot import _is_authorized


def test_state_manager_pause_resume():
    init_state_table()
    
    # Resume all initially
    set_trading_paused("all", False)
    assert not is_trading_paused("india")
    assert not is_trading_paused("us")
    assert not is_trading_paused("all")

    # Pause India only
    set_trading_paused("india", True)
    assert is_trading_paused("india")
    assert not is_trading_paused("us")

    # Resume India
    set_trading_paused("india", False)
    assert not is_trading_paused("india")

    # Pause Global All
    set_trading_paused("all", True)
    assert is_trading_paused("india")
    assert is_trading_paused("us")
    assert is_trading_paused("all")

    # Clean up
    set_trading_paused("all", False)


def test_watchlist_include_and_exclude():
    init_watchlist()

    ticker = "TEST_TELEGRAM_TICKER.NS"
    # Include with pin
    success = add_ticker(
        ticker=ticker,
        market="india",
        strategies=["swing"],
        source="manual",
        pinned=True,
        reason="Telegram test"
    )
    assert success is True

    # Verify present and pinned
    active = get_active_tickers("india")
    found = [t for t in active if t["ticker"] == ticker]
    assert len(found) == 1
    assert found[0]["pinned"] == 1

    # Exclude with force=True
    removed = remove_ticker(ticker, reason="Telegram exclude test", force=True)
    assert removed is True

    # Verify removed
    active_after = get_active_tickers("india")
    assert not any(t["ticker"] == ticker for t in active_after)


def test_authorization_check(monkeypatch):
    monkeypatch.setenv("TELEGRAM_CHAT_ID", "123456789")

    # Authorized mock update
    mock_update_good = MagicMock()
    mock_update_good.effective_chat.id = 123456789
    assert _is_authorized(mock_update_good) is True

    # Unauthorized mock update
    mock_update_bad = MagicMock()
    mock_update_bad.effective_chat.id = 987654321
    assert _is_authorized(mock_update_bad) is False


def test_close_all_positions():
    engine = PaperTradingEngine()
    # Ensure method runs without exception even with empty/existing DB
    closed = engine.close_all_positions("india")
    assert isinstance(closed, list)
