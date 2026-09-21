"""
Tests for Scalping Strategy, Capital Sizing (₹10,000 INR / $1,000 USD),
Intraday/Scalping Square-off, and EOD Mistake-Learning Engine.
"""
from __future__ import annotations

from datetime import datetime, timezone
import pytest

from src.data.models import Quote, StockAnalysis, StockSnapshot, TradeSignal
from src.risk.position_sizer import RiskEngine
from src.signals.generator import SignalGenerator
from src.trading.paper_engine import PaperTradingEngine, init_trading_db
from src.analyst.learning_engine import LearningEngine, init_playbook_table


@pytest.fixture(autouse=True)
def setup_db():
    init_trading_db()
    init_playbook_table()


def test_account_reset_and_capital():
    """Verify virtual capital resets to exactly ₹10,000 INR and $1,000 USD."""
    engine = PaperTradingEngine()
    engine.reset_account_balances()

    inr_balance = engine.get_account_balance("india")
    usd_balance = engine.get_account_balance("us")

    assert inr_balance == 10000.0, f"Expected INR 10,000 but got {inr_balance}"
    assert usd_balance == 1000.0, f"Expected USD 1,000 but got {usd_balance}"


def test_scalping_position_sizing():
    """Verify scalping position sizing allows sufficient leverage and allocation for small accounts."""
    risk = RiskEngine()
    snap = StockSnapshot(
        ticker="SUZLON.NS", market="india", currency="INR", current_price=60.0
    )

    # Scalping trade with tight SL (0.8%) and good R:R (1:2.4)
    # Entry 60.0, SL 59.5, Target 61.2
    res = risk.calculate_position_size(
        snapshot=snap,
        direction="BUY",
        entry_price=60.0,
        stop_loss=59.5,
        target_price=61.2,
        current_cash=10000.0,
        total_portfolio_value=10000.0,
        strategy="scalping",
    )

    assert res.allowed, f"Scalp sizing rejected: {res.reason}"
    assert res.quantity > 0, "Scalp quantity should be positive"
    assert res.quantity >= 1, "Should allow at least 1 share"


def test_scalping_signal_generation():
    """Verify SignalGenerator creates scalping signals with tight stops and appropriate targets."""
    from unittest.mock import MagicMock

    mock_analyst = MagicMock()
    mock_analyst.analyze_stock.return_value = StockAnalysis(
        ticker="RVNL.NS",
        risk_level="Medium",
        action="Good Buy",
        reasons_attractive=["Strong momentum", "Breakout above resistance"],
        risks=["High volatility"],
        suggested_approach="Aggressive scalp",
        invalidation_trigger="Below 113",
        confidence_score=0.85,
        strategy_fit=["scalping"],
        horizon_fit=["short_term"],
    )

    generator = SignalGenerator(analyst=mock_analyst)
    quotes = []
    base_price = 100.0
    for i in range(40):
        p = base_price + (i * 0.4)
        quotes.append(
            Quote(
                timestamp=datetime(2026, 1, 1, 10, i, tzinfo=timezone.utc),
                open=p - 0.1,
                high=p + 0.5,
                low=p - 0.2,
                close=p,
                volume=50000,
            )
        )

    snap = StockSnapshot(
        ticker="RVNL.NS", market="india", currency="INR", current_price=115.0, history=quotes
    )

    sig = generator.generate_signal(snap, strategy="scalping", current_cash=10000.0, portfolio_val=10000.0)
    assert sig is not None, "Should generate a scalping signal"
    assert sig.strategy == "scalping"
    assert sig.entry_price > 0
    # Stop loss should be within ~1% of entry for scalping
    sl_pct = (sig.entry_price - sig.stop_loss) / sig.entry_price
    assert 0.005 <= sl_pct <= 0.02, f"Scalp stop loss pct out of expected tight range: {sl_pct:.3%}"


def test_square_off_intraday_and_scalping():
    """Verify square_off_intraday closes both intraday and scalping positions."""
    engine = PaperTradingEngine()
    engine.reset_account_balances()

    # Enter an intraday position
    sig_intra = TradeSignal(
        ticker="INTRADAY_TEST",
        market="india",
        strategy="intraday",
        direction="BUY",
        entry_price=100.0,
        stop_loss=98.0,
        target_price=105.0,
        quantity=5,
        confidence=0.8,
        reasoning="Intraday test",
    )
    # Enter a scalping position
    sig_scalp = TradeSignal(
        ticker="SCALP_TEST",
        market="india",
        strategy="scalping",
        direction="BUY",
        entry_price=50.0,
        stop_loss=49.5,
        target_price=51.0,
        quantity=10,
        confidence=0.85,
        reasoning="Scalp test",
    )

    engine.execute_signal(sig_intra)
    engine.execute_signal(sig_scalp)

    summary = engine.get_portfolio_summary("india")
    open_tickers = [p["ticker"] for p in summary["positions"]]
    assert "INTRADAY_TEST" in open_tickers
    assert "SCALP_TEST" in open_tickers

    # Square off at 15:15 IST equivalent
    snapshots = {
        "INTRADAY_TEST": StockSnapshot(ticker="INTRADAY_TEST", market="india", currency="INR", current_price=102.0),
        "SCALP_TEST": StockSnapshot(ticker="SCALP_TEST", market="india", currency="INR", current_price=50.8),
    }
    closed_reports = engine.square_off_intraday("india", snapshots)
    assert len(closed_reports) >= 2
    assert any("INTRADAY_TEST" in r for r in closed_reports)
    assert any("SCALP_TEST" in r for r in closed_reports)

    # Verify positions are no longer open
    summary_after = engine.get_portfolio_summary("india")
    remaining_tickers = [p["ticker"] for p in summary_after["positions"]]
    assert "INTRADAY_TEST" not in remaining_tickers
    assert "SCALP_TEST" not in remaining_tickers


def test_learning_engine_autopsy_and_retrieval():
    """Verify the Learning Engine records autopsies and retrieves lessons from the playbook."""
    # Test fallback rule-based autopsy when no live API key or client is provided
    engine = LearningEngine(api_key=None)

    losing_pos = {
        "ticker": "COCHINSHIP.NS",
        "market": "india",
        "strategy": "scalping",
        "direction": "BUY",
        "avg_cost": 1500.0,
        "current_price": 1475.0,
        "realized_pnl": -250.0,
    }

    entry = engine.autopsy_losing_trade(losing_pos)
    assert entry.id is not None
    assert entry.ticker == "COCHINSHIP.NS"
    assert entry.failure_category != ""
    assert entry.prescriptive_lesson != ""
    assert entry.realized_pnl == -250.0

    # Retrieve lessons
    lessons = engine.get_recent_lessons("india", limit=5)
    assert len(lessons) > 0
    assert any(l.ticker == "COCHINSHIP.NS" for l in lessons)
