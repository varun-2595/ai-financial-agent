"""
Tests for Intraday Hardware Kill-Switch and Peak-to-Trough NAV Circuit Breaker.
"""
from __future__ import annotations

import pytest
from src.db.trading_store import init_trading_db, record_account_snapshot, get_open_positions
from src.risk.portfolio_risk_manager import PortfolioRiskManager
from src.trading.paper_engine import PaperTradingEngine
from src.utils.state_manager import is_trading_paused, set_trading_paused, init_state_table
from src.data.models import TradeSignal


@pytest.fixture(autouse=True)
def setup_db():
    init_trading_db()
    init_state_table()
    set_trading_paused("all", False)
    engine = PaperTradingEngine()
    engine.reset_account_balances()


def test_kill_switch_triggers_on_peak_to_trough_drawdown():
    risk_mgr = PortfolioRiskManager()
    assert is_trading_paused("india") is False

    # 1. Normal state: Peak NAV 100k, current NAV 99.5k (0.5% drawdown)
    triggered = risk_mgr.check_and_enforce_intraday_circuit_breaker(
        market="india",
        current_nav=99500.0,
        peak_nav=100000.0,
        max_drawdown_limit_pct=0.02,  # 2.0%
    )
    assert triggered is False
    assert is_trading_paused("india") is False

    # 2. Crash scenario: Peak NAV 100k, current NAV 97.5k (2.5% drawdown > 2.0% limit)
    triggered = risk_mgr.check_and_enforce_intraday_circuit_breaker(
        market="india",
        current_nav=97500.0,
        peak_nav=100000.0,
        max_drawdown_limit_pct=0.02,
    )
    assert triggered is True
    # Trading must now be paused
    assert is_trading_paused("india") is True


def test_kill_switch_squares_off_intraday_positions():
    engine = PaperTradingEngine()
    risk_mgr = PortfolioRiskManager()

    # Place an intraday trade
    sig = TradeSignal(
        ticker="TATAMOTORS.NS",
        market="india",
        strategy="scalping",
        direction="BUY",
        entry_price=600.0,
        stop_loss=595.0,
        target_price=615.0,
        quantity=5,
        confidence=0.85,
        reasoning="Scalp signal for kill switch test",
    )
    engine.execute_signal(sig)
    positions = get_open_positions(market="india")
    assert len(positions) == 1

    # Execute emergency kill switch
    risk_mgr.execute_emergency_kill_switch(
        market="india",
        current_nav=97000.0,
        peak_nav=100000.0,
        drawdown_pct=0.03,
    )

    # All intraday positions should now be squared off
    open_positions = get_open_positions(market="india")
    assert len(open_positions) == 0
    assert is_trading_paused("india") is True
