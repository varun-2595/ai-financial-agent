"""
Test Suite for Apple Silicon M5 Intelligence Suite.
Verifies:
1. PlaybookMemoryRetriever (Historical loss autopsy lookup)
2. RedTeamDebateEngine (Adversarial Bull vs. Bear pre-trade debate)
3. NewsStreamSieve (Zero-cost local M5 news classification)
4. PositionStressTester (Synthetic market shock simulation)
5. SignalGenerator with integrated M5 intelligence suite
"""
from __future__ import annotations

from datetime import datetime, timezone
import pytest

from src.analyst.debate_engine import DebateOutcome, RedTeamDebateEngine
from src.analyst.playbook_memory import MemoryWarningReport, PlaybookMemoryRetriever
from src.data.models import NewsItem, Quote, StockSnapshot, TradeSignal
from src.news.stream_sieve import NewsStreamSieve
from src.risk.stress_tester import PositionStressTester
from src.signals.generator import SignalGenerator


def _sample_snapshot(ticker: str = "TCS.NS", price: float = 3500.0) -> StockSnapshot:
    history = [
        Quote(timestamp=datetime.now(timezone.utc), open=price - 10, high=price + 20, low=price - 15, close=price, volume=50000)
        for _ in range(30)
    ]
    return StockSnapshot(
        ticker=ticker,
        market="india",
        currency="INR",
        current_price=price,
        history=history,
    )


# ── 1. Playbook Memory Retriever ───────────────────────────────────────────────

def test_playbook_memory_retriever():
    retriever = PlaybookMemoryRetriever()
    report = retriever.get_memory_warning(ticker="INFY.NS", market="india", strategy="swing")
    assert isinstance(report, MemoryWarningReport)
    assert report.ticker == "INFY.NS"
    assert report.market == "india"
    assert isinstance(report.has_historical_failures, bool)


# ── 2. Red Team Debate Engine ──────────────────────────────────────────────────

def test_red_team_debate_engine_heuristic():
    engine = RedTeamDebateEngine()
    snap = _sample_snapshot("RELIANCE.NS", 2500.0)
    
    outcome = engine.conduct_debate(snap, direction="BUY", strategy="swing", force_heuristic=True)
    assert isinstance(outcome, DebateOutcome)
    assert outcome.ticker == "RELIANCE.NS"
    assert outcome.verdict in ("APPROVE", "REDUCE", "REJECT")
    assert len(outcome.bull_arguments) > 0
    assert len(outcome.bear_counterpoints) > 0
    assert 0.0 <= outcome.confidence_score <= 1.0


# ── 3. High-Velocity News Sieve ───────────────────────────────────────────────

def test_news_stream_sieve():
    sieve = NewsStreamSieve()
    items = [
        NewsItem(title="Tech Mahindra wins $500M mega defense cloud contract"),
        NewsItem(title="SEBI initiates probe against promoter group for alleged fund diversion"),
        NewsItem(title="Markets trade flat in morning trade amidst global cues"),
    ]
    results = sieve.filter_batch(items, market="india", force_heuristic=True)
    assert len(results) == 3
    
    # 1st item should be BREAKING_ALPHA
    assert results[0].category == "BREAKING_ALPHA"
    assert results[0].impact_score > 0
    
    # 2nd item should be HEADLINE_RISK
    assert results[1].category == "HEADLINE_RISK"
    assert results[1].impact_score < 0
    
    # 3rd item should be NOISE
    assert results[2].category == "NOISE"


# ── 4. Position Stress Tester ──────────────────────────────────────────────────

def test_position_stress_tester():
    tester = PositionStressTester()
    snap = _sample_snapshot("HDFCBANK.NS", 1600.0)
    
    outcome = tester.test_position_resilience(
        snapshot=snap,
        quantity=5,
        entry_price=1600.0,
        stop_loss=1560.0,
        total_capital=10000.0,
        max_allowed_loss_pct=0.05,
    )
    assert outcome.ticker == "HDFCBANK.NS"
    assert len(outcome.scenarios) == 5
    assert outcome.worst_case_loss > 0
    assert isinstance(outcome.passed, bool)


# ── 5. End-to-End Signal Generation with M5 Suite ──────────────────────────────

def test_signal_generator_with_m5_suite():
    generator = SignalGenerator()
    snap = _sample_snapshot("TRENT.NS", 7000.0)
    
    sig = generator.generate_signal(
        snapshot=snap,
        strategy="swing",
        current_cash=20000.0,
        portfolio_val=20000.0,
        force_heuristic=True,
    )
    # If approved by analyst, verify signal metadata contains Red-Team and Stress verification
    if sig:
        assert isinstance(sig, TradeSignal)
        assert sig.ticker == "TRENT.NS"
        assert "Red-Team" in sig.reasoning
        assert sig.quantity > 0
