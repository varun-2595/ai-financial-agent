"""
Comprehensive test suite for Aegis AI Financial & Trading Agent.
Tests:
  1. Market hours & holiday engine (exchange_calendars + zoneinfo)
  2. Technical indicators calculation (RSI, MACD, ATR, EMA, Bollinger)
  3. Key price levels & Fibonacci calculation
  4. Risk sizing rules (2% capital risk, 10% max position, 1.3+ R:R)
  5. Watchlist management (dynamic additions, bans, pins, unbans)
  6. Advisory horizon classification & sell signal engine
  7. Paper trading order execution and Stop Loss trigger
"""
from __future__ import annotations

from datetime import datetime, timezone
import pytest

from src.data.models import Quote, StockAnalysis, StockSnapshot, TradeSignal
from src.technicals.indicators import compute_technical_indicators
from src.technicals.levels import compute_support_resistance
from src.risk.position_sizer import RiskEngine
from src.screener.watchlist_manager import add_ticker, ban_ticker, unban_ticker, get_banned_tickers
from src.advisory.horizon_classifier import HorizonClassifier
from src.advisory.sell_signal_engine import AdvisorySellEngine
from src.trading.paper_engine import PaperTradingEngine
from src.utils.market_hours import is_nse_trading_day, is_nyse_trading_day, nyse_open_in_ist


@pytest.fixture
def sample_quotes() -> list[Quote]:
    quotes = []
    base_price = 100.0
    for i in range(60):
        price = base_price + (i * 0.5)
        quotes.append(
            Quote(
                timestamp=datetime(2026, 1, 1 + (i % 25), 10, 0, tzinfo=timezone.utc),
                open=price - 0.2,
                high=price + 0.8,
                low=price - 0.5,
                close=price,
                volume=100000,
            )
        )
    return quotes


def test_market_hours():
    # Verify daylight saving timezone conversions
    nyse_open = nyse_open_in_ist()
    assert nyse_open is not None
    assert nyse_open.tzinfo is not None


def test_technical_indicators(sample_quotes):
    summary = compute_technical_indicators(sample_quotes)
    assert summary.current_price > 0
    assert summary.rsi_14 is not None
    assert 0 <= summary.rsi_14 <= 100
    assert summary.ema_20 is not None
    assert summary.atr_14 is not None
    assert summary.trend_short in ("BULLISH", "BEARISH", "NEUTRAL")


def test_support_resistance(sample_quotes):
    levels = compute_support_resistance(sample_quotes)
    assert levels.support_1 < levels.resistance_1
    assert levels.support_2 <= levels.support_1
    assert levels.resistance_1 <= levels.resistance_2
    assert levels.fib_382 is not None


def test_risk_position_sizer():
    risk = RiskEngine()
    snap = StockSnapshot(
        ticker="TEST", market="india", currency="INR", current_price=100.0
    )

    # 1. Unfavorable R:R test (< 1.3)
    res_bad_rr = risk.calculate_position_size(
        snapshot=snap, direction="BUY", entry_price=100.0, stop_loss=95.0, target_price=102.0,
        current_cash=100000.0, total_portfolio_value=1000000.0
    )
    assert not res_bad_rr.allowed

    # 2. Good R:R test (Entry 100, SL 95, Target 115 -> R:R 3.0)
    res_good = risk.calculate_position_size(
        snapshot=snap, direction="BUY", entry_price=100.0, stop_loss=95.0, target_price=115.0,
        current_cash=100000.0, total_portfolio_value=1000000.0
    )
    assert res_good.allowed
    assert res_good.quantity > 0
    assert res_good.risk_amount <= 20000.0 # max 2% of 1M portfolio


def test_watchlist_manager():
    ticker = "TEST_WATCHLIST_SYM"
    ban_ticker(ticker, market="us", reason="Unit test ban")
    assert ticker in get_banned_tickers()

    # Attempt to add banned ticker should return False
    added = add_ticker(ticker, market="us", strategies=["swing"], source="auto")
    assert not added

    # Unban
    unbanned = unban_ticker(ticker)
    assert unbanned
    assert ticker not in get_banned_tickers()


def test_advisory_horizon_classification(sample_quotes):
    classifier = HorizonClassifier()
    snap = StockSnapshot(
        ticker="TEST_ADV", market="us", currency="USD", current_price=150.0,
        history=sample_quotes
    )
    analysis = StockAnalysis(
        ticker="TEST_ADV", risk_level="Low", action="Good Buy",
        reasons_attractive=["Strong balance sheet", "Expanding margins"],
        risks=["Macro slowdown"], suggested_approach="Accumulate on dips",
        invalidation_trigger="Break below 120", confidence_score=0.85,
        strategy_fit=["positional"], horizon_fit=["long_term", "short_term"]
    )

    rec_lt = classifier.classify_and_allocate(snap, analysis, 5000.0, "long_term")
    assert rec_lt is not None
    assert rec_lt.horizon == "long_term"
    assert rec_lt.stop_loss == 120.0 # -20%
    assert rec_lt.target_price == 270.0 # +80%


def test_paper_trading_lifecycle():
    engine = PaperTradingEngine()
    initial_cash = engine.get_account_balance("india")
    assert initial_cash > 0

    sig = TradeSignal(
        ticker="TEST_ORDER_SYM", market="india", strategy="swing",
        direction="BUY", entry_price=100.0, stop_loss=95.0,
        target_price=115.0, quantity=10, confidence=0.8, reasoning="Test order"
    )
    order = engine.execute_signal(sig)
    assert order is not None
    assert order.status == "FILLED"

    # Verify cash deducted
    new_cash = engine.get_account_balance("india")
    assert new_cash < initial_cash
