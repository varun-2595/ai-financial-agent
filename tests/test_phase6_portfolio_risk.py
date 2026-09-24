"""
Test Suite for Phase 6: Portfolio-Level Risk Management.

Comprehensive verification of all 10 deterministic risk rules,
rejection conditions, size reduction behavior, and audit logging.
"""
from __future__ import annotations

import tempfile
from datetime import datetime, timezone
from pathlib import Path
import pytest

from src.data.models import Fundamentals, Quote, StockSnapshot, TradeSignal
from src.risk.audit_store import RiskAuditStore
from src.risk.models import (
    PortfolioRiskState,
    RiskCheckName,
    RiskDecision,
)
from src.risk.portfolio_risk_manager import PortfolioRiskManager
from src.trading.paper_engine import PaperTradingEngine


def _make_signal(
    signal_id: str,
    ticker: str,
    strategy: str = "swing",
    direction: str = "BUY",
    entry_price: float = 100.0,
    stop_loss: float = 90.0,
    target_price: float = 120.0,
    quantity: int = 2,
    market: str = "us",
) -> TradeSignal:
    return TradeSignal(
        signal_id=signal_id,
        ticker=ticker,
        market=market,
        strategy=strategy,
        direction=direction,
        entry_price=entry_price,
        stop_loss=stop_loss,
        target_price=target_price,
        quantity=quantity,
        confidence=0.85,
        reasoning="Multi-agent high conviction setup",
    )


@pytest.fixture
def temp_audit_store():
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = Path(tmpdir) / "test_trading.db"
        yield RiskAuditStore(db_path=db_path)


@pytest.fixture
def risk_mgr(temp_audit_store):
    return PortfolioRiskManager(
        audit_store=temp_audit_store,
        max_portfolio_beta=1.50,
        max_positions_per_sector=3,
        max_adv_pct=0.02,
        min_adv_volume=10_000,
    )


@pytest.fixture
def base_risk_state():
    return PortfolioRiskState(
        market="us",
        currency="USD",
        nav=1000.0,
        peak_nav=1000.0,
        cash=1000.0,
        reserved_margin=0.0,
        gross_exposure=0.0,
        daily_realized_pnl=0.0,
        daily_unrealized_pnl=0.0,
        current_drawdown_pct=0.0,
        current_leverage=0.0,
        portfolio_beta=1.0,
        sector_exposures={"Technology": 0.0},
        sector_position_counts={"Technology": 0},
        open_positions=[],
    )


def test_approve_valid_trade(risk_mgr, base_risk_state):
    """Test standard valid trade passes all checks and is approved with full size."""
    signal = _make_signal("SIG-001", "AAPL", quantity=2)
    snapshot = StockSnapshot(
        ticker="AAPL",
        market="us",
        currency="USD",
        sector="Technology",
        current_price=100.0,
        fundamentals=Fundamentals(avg_volume_30d=5_000_000, beta=1.1),
    )

    result = risk_mgr.evaluate_trade(signal, base_risk_state, snapshot)
    assert result.decision == RiskDecision.APPROVE
    assert result.approved_quantity == 2
    assert result.approved_margin == 200.0
    assert len(result.violations) == 0


def test_rejection_daily_loss_circuit_breaker(risk_mgr, base_risk_state):
    """Test daily loss circuit breaker rejects new trade when daily losses reach limit."""
    base_risk_state.daily_realized_pnl = -55.0  # Tripped circuit breaker ($50 limit)
    signal = _make_signal("SIG-002", "AAPL")

    result = risk_mgr.evaluate_trade(signal, base_risk_state)
    assert result.decision == RiskDecision.REJECT
    assert result.approved_quantity == 0
    assert any("Daily loss circuit breaker tripped" in v for v in result.violations)


def test_rejection_max_portfolio_drawdown(risk_mgr, base_risk_state):
    """Test max portfolio drawdown circuit breaker rejects trade when drawdown >= 15%."""
    base_risk_state.current_drawdown_pct = 0.16  # 16% drawdown > 15% limit
    signal = _make_signal("SIG-003", "AAPL")

    result = risk_mgr.evaluate_trade(signal, base_risk_state)
    assert result.decision == RiskDecision.REJECT
    assert result.approved_quantity == 0
    assert any("Portfolio drawdown circuit breaker active" in v for v in result.violations)


def test_rejection_available_cash_insufficient(risk_mgr, base_risk_state):
    """Test rejection when unallocated cash is zero or insufficient for even 1 share."""
    base_risk_state.cash = 10.0  # Price is 100
    signal = _make_signal("SIG-004", "AAPL", quantity=5)

    result = risk_mgr.evaluate_trade(signal, base_risk_state)
    assert result.decision == RiskDecision.REJECT
    assert result.approved_quantity == 0
    assert "Insufficient" in result.reason


def test_rejection_max_positions_per_sector(risk_mgr, base_risk_state):
    """Test concentration rule rejects 4th position in the same sector (max 3)."""
    base_risk_state.sector_position_counts["Technology"] = 3
    base_risk_state.open_positions = [
        {"ticker": "MSFT", "sector": "Technology"},
        {"ticker": "NVDA", "sector": "Technology"},
        {"ticker": "AMD", "sector": "Technology"},
    ]

    signal = _make_signal("SIG-005", "AAPL", quantity=2)
    snapshot = StockSnapshot(
        ticker="AAPL",
        market="us",
        currency="USD",
        sector="Technology",
        current_price=100.0,
    )

    result = risk_mgr.evaluate_trade(signal, base_risk_state, snapshot)
    assert result.decision == RiskDecision.REJECT
    assert result.approved_quantity == 0
    assert any("Max positions in sector 'Technology' reached" in v for v in result.violations)


def test_rejection_illiquid_adv(risk_mgr, base_risk_state):
    """Test illiquid stock with 30-day ADV < 10,000 shares is rejected."""
    snapshot = StockSnapshot(
        ticker="PENNY_STOCK",
        market="us",
        currency="USD",
        current_price=5.0,
        fundamentals=Fundamentals(avg_volume_30d=2_500),
    )
    signal = _make_signal("SIG-006", "PENNY_STOCK", entry_price=5.0, stop_loss=4.0, target_price=7.0, quantity=50)

    result = risk_mgr.evaluate_trade(signal, base_risk_state, snapshot)
    assert result.decision == RiskDecision.REJECT
    assert result.approved_quantity == 0
    assert any("Stock is illiquid" in v for v in result.violations)


def test_rejection_unfavorable_risk_reward(risk_mgr, base_risk_state):
    """Test rejection when Risk-to-Reward ratio is below threshold (e.g. < 1:1.3)."""
    # Risk = 100 - 90 = 10, Reward = 105 - 100 = 5 -> R:R = 0.50 (< 1.3)
    signal = _make_signal("SIG-007", "AAPL", entry_price=100.0, stop_loss=90.0, target_price=105.0)

    result = risk_mgr.evaluate_trade(signal, base_risk_state)
    assert result.decision == RiskDecision.REJECT
    assert result.approved_quantity == 0
    assert any("Unfavorable Risk:Reward ratio" in v for v in result.violations)


def test_rejection_invalid_pricing_or_bounds(risk_mgr, base_risk_state):
    """Test inverted stop-loss boundary is rejected immediately."""
    signal = _make_signal("SIG-008", "AAPL", entry_price=100.0, stop_loss=110.0, target_price=120.0)

    result = risk_mgr.evaluate_trade(signal, base_risk_state)
    assert result.decision == RiskDecision.REJECT
    assert result.approved_quantity == 0
    assert any("Invalid BUY stop loss" in v for v in result.violations)


def test_reduce_quantity_by_risk_budget(risk_mgr, base_risk_state):
    """Test trade quantity is scaled down (REDUCE) when risk budget limits total loss."""
    base_risk_state.cash = 1000.0  # Cash is ample ($1,000)
    # Risk budget = 5% of $1,000 = $50. Risk per share = $10. Max allowed = 5 shares.
    signal = _make_signal("SIG-009", "AAPL", strategy="intraday", entry_price=100.0, stop_loss=90.0, target_price=120.0, quantity=8)

    result = risk_mgr.evaluate_trade(signal, base_risk_state)
    assert result.decision == RiskDecision.REDUCE
    assert result.approved_quantity == 5
    assert "risk budget" in result.reason


def test_reduce_quantity_by_position_cap(risk_mgr, base_risk_state):
    """Test trade quantity is scaled down (REDUCE) to stay within 50% max position cap."""
    base_risk_state.cash = 1000.0
    signal = _make_signal("SIG-010", "AAPL", quantity=8)  # Requests $800 position (80% of NAV)

    result = risk_mgr.evaluate_trade(signal, base_risk_state)
    assert result.decision == RiskDecision.REDUCE
    assert result.approved_quantity == 5
    assert result.approved_quantity * 100.0 <= 500.0


def test_reduce_quantity_by_sector_cap(risk_mgr, base_risk_state):
    """Test trade quantity is scaled down (REDUCE) when sector room is limited."""
    base_risk_state.sector_exposures["Technology"] = 400.0
    base_risk_state.cash = 600.0

    signal = _make_signal("SIG-011", "AAPL", quantity=5)
    snapshot = StockSnapshot(
        ticker="AAPL",
        market="us",
        currency="USD",
        sector="Technology",
        current_price=100.0,
    )

    result = risk_mgr.evaluate_trade(signal, base_risk_state, snapshot)
    assert result.decision == RiskDecision.REDUCE
    assert result.approved_quantity == 1
    assert "sector room" in result.reason


def test_reduce_quantity_by_adv_liquidity(risk_mgr, base_risk_state):
    """Test trade quantity is scaled down (REDUCE) to stay within 2% of 30-day ADV."""
    snapshot = StockSnapshot(
        ticker="MID_CAP",
        market="us",
        currency="USD",
        current_price=0.10,
        fundamentals=Fundamentals(avg_volume_30d=100_000),
    )
    signal = _make_signal("SIG-012", "MID_CAP", entry_price=0.10, stop_loss=0.09, target_price=0.15, quantity=10_000)

    result = risk_mgr.evaluate_trade(signal, base_risk_state, snapshot)
    assert result.decision == RiskDecision.REDUCE
    assert result.approved_quantity == 2_000
    assert "2% ADV" in result.reason


def test_risk_audit_log_persistence_and_retrieval(temp_audit_store, risk_mgr, base_risk_state):
    """Test that all risk evaluation decisions are recorded in SQLite audit log and queryable."""
    signal = _make_signal("SIG-AUDIT-1", "AAPL", quantity=2)

    result = risk_mgr.evaluate_trade(signal, base_risk_state)
    assert result.decision == RiskDecision.APPROVE

    history = temp_audit_store.get_audit_history(ticker="AAPL")
    assert len(history) >= 1
    latest = history[0]

    assert latest["ticker"] == "AAPL"
    assert latest["decision"] == "APPROVE"
    assert latest["requested_quantity"] == 2
    assert latest["approved_quantity"] == 2
    assert latest["nav_at_decision"] == 1000.0
    assert len(latest["checks"]) >= 5


def test_paper_engine_executes_only_approved_or_reduced_trades(risk_mgr):
    """Test that PaperTradingEngine routes trades through risk gate, reducing or rejecting orders."""
    engine = PaperTradingEngine(risk_manager=risk_mgr)
    engine.full_reset()

    # 1. Valid order within capital ($1,000 USD virtual capital)
    valid_signal = _make_signal("SIG-PE-1", "AAPL", entry_price=150.0, stop_loss=140.0, target_price=175.0, quantity=1)
    order1 = engine.execute_signal(valid_signal)
    assert order1 is not None
    assert order1.quantity == 1
    assert order1.status == "FILLED"

    # 2. Over-sized order ($600 position > $250 position cap, but within $850 cash) -> Sized down by risk engine to 1 share
    large_signal = _make_signal("SIG-PE-2", "MSFT", entry_price=200.0, stop_loss=185.0, target_price=235.0, quantity=3)
    order2 = engine.execute_signal(large_signal)
    assert order2 is not None
    assert order2.quantity < 3  # Reduced to safe allocation (1 share)
    assert order2.quantity == 1

    # 3. Bad trade with negative risk-reward (150 -> stop 140, target 145) -> REJECTED
    bad_signal = _make_signal("SIG-PE-3", "TSLA", entry_price=150.0, stop_loss=140.0, target_price=135.0, quantity=1)
    order3 = engine.execute_signal(bad_signal)
    assert order3 is None  # Hard rejected
