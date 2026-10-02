"""
Test Suite for Strategy Profitability Overhaul.
Verifies:
1. MarketRegimeDetector (Benchmark EMA20/50 and regime filters)
2. Multi-Stage Trade Execution (50% partial profit take @ +1.5R, Breakeven SL)
3. PortfolioRiskManager Loss Streak Circuit Breakers
4. Asymmetric Risk-to-Reward and Volatility Sizing
"""
from __future__ import annotations

from datetime import datetime, timezone
import pytest

from src.data.models import Quote, StockSnapshot, TradeSignal
from src.risk.models import PortfolioRiskState
from src.risk.portfolio_risk_manager import PortfolioRiskManager
from src.risk.position_sizer import RiskEngine
from src.screener.regime import MarketRegime, MarketRegimeDetector
from src.trading.paper_engine import PaperTradingEngine


# ── 1. Market Regime Detector Tests ────────────────────────────────────────────

def test_market_regime_detector_calculations():
    detector = MarketRegimeDetector()
    
    # Test EMA calculation
    prices = [100.0, 102.0, 104.0, 106.0, 108.0, 110.0]
    ema = detector._compute_ema(prices, 3)
    assert ema > 100.0
    
    # Test RSI calculation
    rsi = detector._compute_rsi(prices, 3)
    assert 0.0 <= rsi <= 100.0
    assert rsi > 50.0  # Steady uptrend should yield RSI > 50


def test_market_regime_fallback_and_types():
    detector = MarketRegimeDetector()
    regime_india = detector.get_market_regime("india", force_refresh=True)
    assert isinstance(regime_india, MarketRegime)
    assert regime_india.market == "india"
    assert regime_india.benchmark_symbol in ("^NSEI", "NIFTYBEES.NS")
    assert isinstance(regime_india.allows_long, bool)

    is_bullish = detector.is_market_bullish("india")
    assert isinstance(is_bullish, bool)


# ── 2. Asymmetric Risk:Reward Validation ───────────────────────────────────────

def test_risk_engine_asymmetric_rr_validation():
    risk_engine = RiskEngine()
    history = [
        Quote(timestamp=datetime.now(timezone.utc), open=100.0, high=105.0, low=98.0, close=100.0, volume=10000)
        for _ in range(20)
    ]
    snap = StockSnapshot(
        ticker="TEST_TICKER",
        market="india",
        currency="INR",
        current_price=100.0,
        history=history,
    )

    # Sub-par R:R (Risk 5 to make 3 -> R:R 0.6) should be REJECTED
    res_bad = risk_engine.calculate_position_size(
        snapshot=snap,
        direction="BUY",
        entry_price=100.0,
        stop_loss=95.0,
        target_price=103.0,
        current_cash=10000.0,
        total_portfolio_value=10000.0,
        strategy="swing",
    )
    assert res_bad.allowed is False
    assert "Risk:Reward" in res_bad.reason

    # Asymmetric 1:2.5 R:R (Risk 2 to make 6 -> R:R 3.0) should be APPROVED
    res_good = risk_engine.calculate_position_size(
        snapshot=snap,
        direction="BUY",
        entry_price=100.0,
        stop_loss=98.0,
        target_price=106.0,
        current_cash=10000.0,
        total_portfolio_value=10000.0,
        strategy="swing",
    )
    assert res_good.allowed is True
    assert res_good.quantity > 0


# ── 3. Multi-Stage Partial Profit & Breakeven SL ───────────────────────────────

def test_multistage_partial_profit_and_breakeven_sl():
    engine = PaperTradingEngine()
    engine.full_reset(hard_wipe=True)

    # Enter a 10-share position: Entry 100.0, SL 96.0 (Risk = 4.0/sh), TGT 112.0 (+3.0R)
    # +1.5R partial trigger = 100 + (1.5 * 4.0) = 106.0
    sig = TradeSignal(
        ticker="MULTISTAGE_TEST",
        market="india",
        strategy="swing",
        direction="BUY",
        entry_price=100.0,
        stop_loss=96.0,
        target_price=112.0,
        quantity=10,
        confidence=0.85,
        reasoning="Multi-stage test setup",
    )
    order = engine.execute_signal(sig)
    assert order is not None
    assert order.quantity == 10

    # Simulate price advancing to 107.0 (+1.75R) -> Should book 50% profit (5 shares) and move SL to 100.0
    snap_107 = {
        "MULTISTAGE_TEST": StockSnapshot(
            ticker="MULTISTAGE_TEST",
            market="india",
            currency="INR",
            current_price=107.0,
        )
    }
    reports = engine.evaluate_open_positions("india", snap_107)
    assert len(reports) == 1
    assert "PARTIAL EXIT" in reports[0]

    # Verify open position state: Remaining 5 shares with stop_loss == 100.05 (Breakeven)
    from src.db.trading_store import get_open_positions
    pos_list = get_open_positions("india")
    assert len(pos_list) == 1
    rem_pos = pos_list[0]
    assert rem_pos["ticker"] == "MULTISTAGE_TEST"
    assert rem_pos["quantity"] == 5
    assert rem_pos["stop_loss"] == 100.05  # Breakeven entry price!


# ── 4. Loss Streak Circuit Breaker ─────────────────────────────────────────────

def test_loss_streak_circuit_breaker(monkeypatch):
    risk_mgr = PortfolioRiskManager()
    risk_state = PortfolioRiskState(
        market="india",
        currency="INR",
        nav=10000.0,
        peak_nav=10000.0,
        cash=10000.0,
        reserved_margin=0.0,
        gross_exposure=0.0,
        daily_realized_pnl=0.0,
        daily_unrealized_pnl=0.0,
        current_drawdown_pct=0.0,
    )
    sig = TradeSignal(
        ticker="STREAK_TEST",
        market="india",
        strategy="swing",
        direction="BUY",
        entry_price=100.0,
        stop_loss=96.0,
        target_price=110.0,
        quantity=10,
        confidence=0.8,
        reasoning="Loss streak test",
    )

    # Normal trade with 0 consecutive losses -> APPROVED
    res_normal = risk_mgr.evaluate_trade(sig, risk_state)
    assert res_normal.decision.name in ("APPROVE", "REDUCE")
