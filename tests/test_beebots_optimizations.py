"""
Tests for BeeBots-inspired High-Throughput & Low-Latency Architecture upgrades:
1. In-flight request coalescer & micro-TTL cache
2. Continuous Fast Tick Position Monitor (SL, +1.5R partial TP, breakeven SL)
3. Constrained Action Menus for Apple Silicon M5 debate
4. Pre-Order Two-Phase Commit Journaling
"""
from __future__ import annotations

import concurrent.futures
import time
from datetime import datetime, timezone
from unittest.mock import MagicMock, patch

import pytest
from src.analyst.debate_engine import RedTeamDebateEngine
from src.data.coalescer import RequestCoalescer
from src.data.models import Order, Quote, StockAnalysis, StockSnapshot, TradeSignal
from src.db.trading_store import _conn, get_open_positions, get_pending_orders, init_trading_db
from src.trading.fast_tick_monitor import FastTickMonitor
from src.trading.paper_engine import PaperTradingEngine


@pytest.fixture(autouse=True)
def setup_db():
    init_trading_db()
    with _conn() as conn:
        conn.execute("DELETE FROM orders")
        conn.execute("DELETE FROM positions")
        conn.execute("DELETE FROM ledger")
        conn.execute("DELETE FROM accounts")
        conn.commit()
    engine = PaperTradingEngine()
    engine.full_reset(hard_wipe=True)


# ── 1. Request Coalescer Tests ────────────────────────────────────────────────

def test_request_coalescer_deduplication():
    coalescer = RequestCoalescer(default_ttl_seconds=2.0)
    call_count = 0

    def mock_network_fetch():
        nonlocal call_count
        call_count += 1
        time.sleep(0.05)  # Simulate network latency
        return {"ticker": "RELIANCE.NS", "price": 2500.0}

    # Launch 10 concurrent requests for the same key
    with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
        futures = [
            executor.submit(
                coalescer.coalesce,
                "quote:RELIANCE.NS",
                mock_network_fetch,
                2.0,
            )
            for _ in range(10)
        ]
        results = [f.result() for f in futures]

    # Verify all 10 got the exact same data
    for res in results:
        assert res["ticker"] == "RELIANCE.NS"
        assert res["price"] == 2500.0

    # Only 1 network call should have occurred
    assert call_count == 1
    assert coalescer.stats["network_fetches"] == 1
    assert coalescer.stats["coalesced_waits"] == 9


def test_request_coalescer_ttl_expiry():
    coalescer = RequestCoalescer(default_ttl_seconds=0.1)
    call_count = 0

    def fetch_price():
        nonlocal call_count
        call_count += 1
        return 100.0 + call_count

    # First call -> fetches
    val1 = coalescer.coalesce("quote:TEST", fetch_price, ttl_seconds=0.1)
    assert val1 == 101.0
    assert call_count == 1

    # Immediate second call -> cached (hits)
    val2 = coalescer.coalesce("quote:TEST", fetch_price, ttl_seconds=0.1)
    assert val2 == 101.0
    assert call_count == 1
    assert coalescer.stats["cache_hits"] == 1

    # Sleep past TTL
    time.sleep(0.15)

    # Third call -> cache expired, refetches
    val3 = coalescer.coalesce("quote:TEST", fetch_price, ttl_seconds=0.1)
    assert val3 == 102.0
    assert call_count == 2


# ── 2. Fast Tick Position Monitor Tests ───────────────────────────────────────

def test_fast_tick_monitor_triggers_stop_loss():
    engine = PaperTradingEngine()
    # Execute a buy order: 10 shs of INFY @ 100, SL at 95, TGT at 110
    sig = TradeSignal(
        ticker="INFY.NS",
        market="india",
        strategy="swing",
        direction="BUY",
        entry_price=100.0,
        stop_loss=95.0,
        target_price=110.0,
        quantity=10,
        confidence=0.85,
        reasoning="Test breakout",
    )
    order = engine.execute_signal(sig)
    assert order is not None

    positions = get_open_positions()
    assert len(positions) == 1

    # Create FastTickMonitor with custom price fetcher dropping price to 94.0 (breaching SL)
    def mock_price(ticker, market):
        if ticker == "INFY.NS":
            return 94.0
        return None

    monitor = FastTickMonitor(
        engine=engine,
        tick_interval_seconds=0.1,
        price_fetcher=mock_price,
        enforce_market_hours=False,
    )

    reports = monitor.tick_once()
    assert len(reports) == 1
    assert "STOP LOSS HIT" in reports[0]

    # Verify position is now closed
    open_pos = get_open_positions()
    assert len(open_pos) == 0


def test_fast_tick_monitor_triggers_partial_tp_and_breakeven():
    engine = PaperTradingEngine()
    # Buy 10 shs @ 100, SL at 90 (R = 10, +1.5R = 115)
    sig = TradeSignal(
        ticker="TCS.NS",
        market="india",
        strategy="swing",
        direction="BUY",
        entry_price=100.0,
        stop_loss=90.0,
        target_price=130.0,
        quantity=10,
        confidence=0.90,
        reasoning="Test partial TP",
    )
    order = engine.execute_signal(sig)
    assert order is not None

    # Price spikes to 116 (+1.6R)
    def mock_price(ticker, market):
        if ticker == "TCS.NS":
            return 116.0
        return None

    monitor = FastTickMonitor(
        engine=engine,
        tick_interval_seconds=0.1,
        price_fetcher=mock_price,
        enforce_market_hours=False,
    )

    reports = monitor.tick_once()
    assert len(reports) == 1
    assert "PARTIAL" in reports[0]

    # Verify remaining position has 5 shares and stop_loss moved to breakeven (100.0)
    open_pos = get_open_positions()
    assert len(open_pos) == 1
    assert open_pos[0]["quantity"] == 5
    assert open_pos[0]["stop_loss"] == pytest.approx(100.0, abs=1.0)


# ── 3. Constrained M5 Debate Action Menu Tests ───────────────────────────────

def test_constrained_m5_debate_menu_approval():
    engine = RedTeamDebateEngine()
    snap = StockSnapshot(
        ticker="HDFCBANK.NS",
        market="india",
        currency="INR",
        current_price=1600.0,
        history=[
            Quote(
                timestamp=datetime.now(timezone.utc),
                open=1590,
                high=1610,
                low=1585,
                close=1600,
                volume=100000,
            )
            for _ in range(30)
        ],
    )

    # Mock router to return fast constrained menu JSON: choice 1 (APPROVE)
    mock_resp = MagicMock()
    mock_resp.content = '{"choice": 1, "key_flaw": "none", "confidence": 0.88}'
    engine.router.route = MagicMock(return_value=mock_resp)

    outcome = engine.conduct_debate(
        snapshot=snap,
        direction="BUY",
        strategy="intraday",
    )

    assert outcome.verdict == "APPROVE"
    assert outcome.confidence_score == 0.88
    assert not outcome.fatal_flaw_detected


def test_constrained_m5_debate_menu_rejection():
    engine = RedTeamDebateEngine()
    snap = StockSnapshot(
        ticker="ZOMATO.NS",
        market="india",
        currency="INR",
        current_price=250.0,
        history=[
            Quote(
                timestamp=datetime.now(timezone.utc),
                open=245,
                high=252,
                low=240,
                close=250,
                volume=200000,
            )
            for _ in range(30)
        ],
    )

    # Mock router to return fast constrained menu JSON: choice 3 (REJECT)
    mock_resp = MagicMock()
    mock_resp.content = '{"choice": 3, "key_flaw": "Overhead supply zone and RSI exhaustion", "confidence": 0.92}'
    engine.router.route = MagicMock(return_value=mock_resp)

    outcome = engine.conduct_debate(
        snapshot=snap,
        direction="BUY",
        strategy="scalping",
    )

    assert outcome.verdict == "REJECT"
    assert outcome.fatal_flaw_detected
    assert "Red-Team Veto" in outcome.verdict_reasoning


# ── 4. Pre-Order Two-Phase Commit Tests ───────────────────────────────────────

def test_pre_order_two_phase_commit():
    engine = PaperTradingEngine()
    sig = TradeSignal(
        ticker="WIPRO.NS",
        market="india",
        strategy="swing",
        direction="BUY",
        entry_price=450.0,
        stop_loss=435.0,
        target_price=480.0,
        quantity=5,
        confidence=0.80,
        reasoning="Test two phase commit",
    )

    order = engine.execute_signal(sig)
    assert order is not None

    with _conn() as conn:
        row = conn.execute("SELECT * FROM orders WHERE order_id = ?", (order.order_id,)).fetchone()
        assert row is not None
        assert row["status"] == "FILLED"
        assert row["filled_price"] is not None
        assert row["filled_at"] is not None

    # Check pending orders is empty after fill
    pending = get_pending_orders()
    assert len(pending) == 0
