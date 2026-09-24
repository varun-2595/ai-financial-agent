"""
Phase 7 Tests: Immutable Decision Journal & Agent Attribution Observability.

Verifies:
1. Complete pre-trade decision recording (market snapshot, agent outputs, evidence, risk decision, sizing).
2. Post-trade evaluation calculation (directional accuracy, thesis accuracy, agent attribution, Brier calibration).
3. Root cause failure taxonomy classification.
4. Risk decision efficacy analysis.
5. Cumulative agent performance scorecards.
6. End-to-end integration with PaperTradingEngine.
7. Strict preservation of immutability and non-mutation of agent weights.
"""
from __future__ import annotations

import os
import sqlite3
import tempfile
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from src.agents.models import AgentSignalOutput
from src.data.models import StockSnapshot, TradeSignal
from src.journal.evaluator import PostTradeEvaluator
from src.journal.journal_store import DecisionJournalStore
from src.journal.models import (
    AgentAttributionScore,
    AgentScorecard,
    DecisionJournalEntry,
    TradeEvaluation,
)
from src.trading.fees import FeeSchedule
from src.trading.paper_engine import PaperTradingEngine


from src.db.trading_store import _conn, init_trading_db


@pytest.fixture(autouse=True)
def clean_test_tables():
    """Ensure trading tables and journal tables start empty for each test."""
    init_trading_db()
    with _conn() as conn:
        conn.execute("DELETE FROM positions")
        conn.execute("DELETE FROM orders")
        conn.execute("DELETE FROM signals")
        conn.execute("DELETE FROM ledger")
        conn.commit()


@pytest.fixture
def temp_journal_store(tmp_path):
    """Provide an isolated SQLite database for journal testing."""
    db_file = tmp_path / "test_journal.db"
    store = DecisionJournalStore(db_path=db_file)
    return store


@pytest.fixture
def sample_decision_entry():
    """Create a realistic sample multi-agent decision journal entry."""
    now = datetime.now(timezone.utc)
    return DecisionJournalEntry(
        journal_id="DEC-TEST-001",
        timestamp=now,
        symbol="RELIANCE.NS",
        market="india",
        strategy="multi_agent_consensus",
        direction="BUY",
        market_snapshot={
            "price": 2500.0,
            "rsi": 42.5,
            "macd": "BULLISH_CROSS",
            "atr_14": 35.0,
            "sma_50": 2480.0,
            "sma_200": 2400.0,
            "sector": "Energy & Conglomerates",
            "vix": 14.5,
        },
        agent_outputs={
            "TechnicalAgent": AgentSignalOutput(
                agent="TechnicalAgent",
                signal="BUY",
                confidence=0.85,
                reasons=["Golden cross on 4H", "RSI divergence above 40 support"],
                risks=["Immediate resistance at 2580"],
                evidence=["RSI=42.5", "MACD histogram > 0"],
            ),
            "FundamentalAgent": AgentSignalOutput(
                agent="FundamentalAgent",
                signal="BUY",
                confidence=0.75,
                reasons=["Strong refining margins in Q3", "Oil-to-Chemicals revenue +12%"],
                risks=["High Capex in 5G rollouts"],
                evidence=["Filing: Q3 EBITDA grew 10.5% YoY"],
            ),
            "NewsAgent": AgentSignalOutput(
                agent="NewsAgent",
                signal="BUY",
                confidence=0.70,
                reasons=["New green hydrogen contract signed"],
                risks=["Crude oil volatility"],
                evidence=["Press release: 1GW green hydrogen award"],
            ),
            "MacroAgent": AgentSignalOutput(
                agent="MacroAgent",
                signal="HOLD",
                confidence=0.60,
                reasons=["RBI kept repo rates steady"],
                risks=["Sticky global inflation"],
                evidence=["RBI policy rate at 6.50%"],
            ),
            "RiskAgent": AgentSignalOutput(
                agent="RiskAgent",
                signal="BUY",
                confidence=0.90,
                reasons=["2.4x Risk-Reward ratio", "Max risk within 1% capital"],
                risks=["Gap down risk on international crude shock"],
                evidence=["Stop loss at 2420, Target at 2700"],
            ),
        },
        confidence=0.78,
        evidence=[
            "Filing: Q3 EBITDA grew 10.5% YoY",
            "Press release: 1GW green hydrogen award",
            "Technical: 50-SMA crossing 200-SMA",
        ],
        final_thesis="High-conviction long on refining margins and green energy expansion with favorable 2.4x R:R.",
        risk_decision="APPROVE",
        risk_reasons=["Passed all 10 portfolio-level risk checks"],
        requested_quantity=100,
        approved_quantity=100,
        entry_price=2500.0,
        stop_loss=2420.0,
        target_price=2700.0,
        status="PROPOSED",
    )


# ── 1. Immutable Decision Journal Recording ───────────────────────────────────

def test_record_and_retrieve_decision_journal(temp_journal_store, sample_decision_entry):
    """Ensure pre-trade snapshot is stored with full fidelity and immutability."""
    jid = temp_journal_store.record_decision(sample_decision_entry)
    assert jid == "DEC-TEST-001"

    fetched = temp_journal_store.get_decision("DEC-TEST-001")
    assert fetched is not None
    assert fetched.symbol == "RELIANCE.NS"
    assert fetched.market == "india"
    assert fetched.confidence == 0.78
    assert fetched.entry_price == 2500.0
    assert fetched.stop_loss == 2420.0
    assert fetched.target_price == 2700.0
    assert fetched.requested_quantity == 100
    assert fetched.approved_quantity == 100
    assert "TechnicalAgent" in fetched.agent_outputs
    assert fetched.agent_outputs["TechnicalAgent"]["signal"] == "BUY"
    assert len(fetched.evidence) == 3
    assert fetched.final_thesis.startswith("High-conviction")


def test_update_decision_execution_and_exit(temp_journal_store, sample_decision_entry):
    """Test lifecycle status updates from PROPOSED -> EXECUTED -> CLOSED."""
    temp_journal_store.record_decision(sample_decision_entry)

    # Execute
    temp_journal_store.update_decision_execution(
        journal_id=sample_decision_entry.journal_id,
        order_id="ORD-12345",
        position_id=42,
        status="EXECUTED",
    )
    exec_entry = temp_journal_store.get_decision(sample_decision_entry.journal_id)
    assert exec_entry.status == "EXECUTED"
    assert exec_entry.order_id == "ORD-12345"
    assert exec_entry.position_id == 42

    # Lookup by position ID
    by_pos = temp_journal_store.get_decision_by_position_id(42)
    assert by_pos is not None
    assert by_pos.journal_id == sample_decision_entry.journal_id

    # Exit
    exit_dt = datetime.now(timezone.utc) + timedelta(hours=6)
    temp_journal_store.update_decision_exit(
        journal_id=sample_decision_entry.journal_id,
        exit_price=2710.0,
        exit_timestamp=exit_dt,
        exit_reason="TAKE_PROFIT_TRIGGERED",
        realized_pnl=21000.0,
        return_pct=8.4,
        holding_period_seconds=21600.0,
    )

    closed_entry = temp_journal_store.get_decision(sample_decision_entry.journal_id)
    assert closed_entry.status == "CLOSED"
    assert closed_entry.exit_price == 2710.0
    assert closed_entry.realized_pnl == 21000.0
    assert closed_entry.return_pct == 8.4
    assert closed_entry.exit_reason == "TAKE_PROFIT_TRIGGERED"


# ── 2. Post-Trade Evaluation: Winning Trade ────────────────────────────────────

def test_post_trade_evaluator_winning_trade(sample_decision_entry):
    """Test evaluation logic for a target-hitting winning trade."""
    sample_decision_entry.approved_quantity = 100
    exit_time = sample_decision_entry.timestamp + timedelta(hours=4)

    evaluation = PostTradeEvaluator.evaluate(
        entry=sample_decision_entry,
        exit_price=2705.0,  # Above target 2700.0
        exit_timestamp=exit_time,
        exit_reason="TAKE_PROFIT",
        realized_pnl=20500.0,
        return_pct=8.2,
    )

    assert evaluation.is_winner is True
    assert evaluation.directional_accuracy == 1.0
    assert evaluation.thesis_accuracy == 1.0
    assert "Target price" in evaluation.thesis_notes
    assert evaluation.major_failure_reason == "NONE_WINNING_TRADE"
    assert "OPTIMAL_SIZING" in evaluation.risk_decision_evaluation

    # Per-Agent Attribution
    agents = evaluation.agent_accuracy
    assert "TechnicalAgent" in agents
    assert agents["TechnicalAgent"].directional_accuracy == 1.0
    assert agents["TechnicalAgent"].signal == "BUY"
    # Brier loss = (0.85 - 1.0)^2 = 0.0225
    assert abs(agents["TechnicalAgent"].brier_score_loss - 0.0225) < 1e-4

    assert "FundamentalAgent" in agents
    assert agents["FundamentalAgent"].directional_accuracy == 1.0

    # MacroAgent called HOLD -> 0.5 neutral
    assert "MacroAgent" in agents
    assert agents["MacroAgent"].directional_accuracy == 0.5


# ── 3. Post-Trade Evaluation: Losing Trade (Stop Loss Hit) ─────────────────────

def test_post_trade_evaluator_losing_trade_stop_loss(sample_decision_entry):
    """Test evaluation for a losing trade hitting stop loss."""
    exit_time = sample_decision_entry.timestamp + timedelta(hours=2)

    evaluation = PostTradeEvaluator.evaluate(
        entry=sample_decision_entry,
        exit_price=2415.0,  # Below stop loss 2420.0
        exit_timestamp=exit_time,
        exit_reason="STOP_LOSS",
        realized_pnl=-8500.0,
        return_pct=-3.4,
    )

    assert evaluation.is_winner is False
    assert evaluation.directional_accuracy == 0.0
    assert evaluation.thesis_accuracy == 0.0
    assert "Stop loss" in evaluation.thesis_notes
    assert evaluation.major_failure_reason == "STOP_LOSS_HIT"
    assert "CONTROLLED_LOSS" in evaluation.risk_decision_evaluation

    # Per-Agent Attribution
    agents = evaluation.agent_accuracy
    assert agents["TechnicalAgent"].directional_accuracy == 0.0
    # Brier loss on incorrect BUY with 0.85 confidence = (0.85 - 0.0)^2 = 0.7225
    assert abs(agents["TechnicalAgent"].brier_score_loss - 0.7225) < 1e-4
    assert agents["FundamentalAgent"].directional_accuracy == 0.0


# ── 4. Failure Taxonomy Classification ────────────────────────────────────────

def test_failure_taxonomy_macro_reversal(sample_decision_entry):
    """Verify classification of MACRO_REVERSAL when MacroAgent signaled warning."""
    sample_decision_entry.agent_outputs["MacroAgent"] = AgentSignalOutput(
        agent="MacroAgent",
        signal="SELL",
        confidence=0.80,
        reasons=["Surprise CPI inflation jump"],
        risks=["Macro interest rate hike likely"],
        evidence=["US CPI 3.8% vs 3.2% expected"],
    )

    evaluation = PostTradeEvaluator.evaluate(
        entry=sample_decision_entry,
        exit_price=2410.0,
        exit_reason="MANUAL_CLOSE",
        realized_pnl=-9000.0,
    )

    assert evaluation.major_failure_reason == "MACRO_REVERSAL"
    assert "Macro" in evaluation.failure_details


def test_failure_taxonomy_news_headline_risk(sample_decision_entry):
    """Verify classification of NEWS_HEADLINE_RISK when NewsAgent cited regulatory probe."""
    sample_decision_entry.agent_outputs["MacroAgent"] = AgentSignalOutput(
        agent="MacroAgent", signal="HOLD", confidence=0.5, reasons=[], risks=[], evidence=[]
    )
    sample_decision_entry.agent_outputs["NewsAgent"] = AgentSignalOutput(
        agent="NewsAgent",
        signal="BUY",
        confidence=0.6,
        reasons=["Good earnings"],
        risks=["SEBI regulatory probe into subsidiary headline"],
        evidence=[],
    )

    evaluation = PostTradeEvaluator.evaluate(
        entry=sample_decision_entry,
        exit_price=2430.0,
        exit_reason="MANUAL_CLOSE",
        realized_pnl=-7000.0,
    )

    assert evaluation.major_failure_reason == "NEWS_HEADLINE_RISK"
    assert "headline" in evaluation.failure_details.lower() or "regulatory" in evaluation.failure_details.lower()


# ── 5. Risk Decision Sizing Evaluation ─────────────────────────────────────────

def test_risk_decision_protective_reduction(sample_decision_entry):
    """Verify assessment when Risk Engine reduced sizing and averted deeper loss."""
    sample_decision_entry.risk_decision = "REDUCE"
    sample_decision_entry.requested_quantity = 200
    sample_decision_entry.approved_quantity = 50  # Risk Engine cut 150 shares

    evaluation = PostTradeEvaluator.evaluate(
        entry=sample_decision_entry,
        exit_price=2400.0,
        exit_reason="STOP_LOSS",
        realized_pnl=-5000.0,  # (2400-2500)*50
    )

    assert "PROTECTIVE_REDUCTION" in evaluation.risk_decision_evaluation
    assert "200 to 50" in evaluation.risk_decision_evaluation


def test_risk_decision_conservative_cap(sample_decision_entry):
    """Verify assessment when Risk Engine reduced sizing on a winning trade."""
    sample_decision_entry.risk_decision = "REDUCE"
    sample_decision_entry.requested_quantity = 200
    sample_decision_entry.approved_quantity = 100

    evaluation = PostTradeEvaluator.evaluate(
        entry=sample_decision_entry,
        exit_price=2600.0,
        exit_reason="TAKE_PROFIT",
        realized_pnl=10000.0,
    )

    assert "CONSERVATIVE_CAP" in evaluation.risk_decision_evaluation


# ── 6. Cumulative Agent Scorecards & Observability ─────────────────────────────

def test_cumulative_agent_scorecards(temp_journal_store, sample_decision_entry):
    """
    Test cumulative scorecard generation across multiple trades without retraining.
    """
    # Trade 1: Win
    entry1 = sample_decision_entry.model_copy(deep=True)
    entry1.journal_id = "DEC-001"
    entry1.symbol = "RELIANCE.NS"
    temp_journal_store.record_decision(entry1)
    eval1 = PostTradeEvaluator.evaluate(entry1, exit_price=2700.0, realized_pnl=20000.0, return_pct=8.0)
    temp_journal_store.record_evaluation(eval1)

    # Trade 2: Loss
    entry2 = sample_decision_entry.model_copy(deep=True)
    entry2.journal_id = "DEC-002"
    entry2.symbol = "TCS.NS"
    entry2.agent_outputs["TechnicalAgent"] = AgentSignalOutput(
        agent="TechnicalAgent", signal="BUY", confidence=0.90, reasons=[], risks=[], evidence=[]
    )
    entry2.agent_outputs["FundamentalAgent"] = AgentSignalOutput(
        agent="FundamentalAgent", signal="SELL", confidence=0.80, reasons=[], risks=[], evidence=[]
    )
    temp_journal_store.record_decision(entry2)
    eval2 = PostTradeEvaluator.evaluate(entry2, exit_price=2400.0, realized_pnl=-10000.0, return_pct=-4.0)
    temp_journal_store.record_evaluation(eval2)

    # Retrieve cumulative scorecards
    scorecards = temp_journal_store.get_agent_scorecards()
    assert "TechnicalAgent" in scorecards
    assert "FundamentalAgent" in scorecards

    tech_sc = scorecards["TechnicalAgent"]
    assert tech_sc.total_evaluations == 2
    assert tech_sc.bullish_calls == 2
    assert tech_sc.correct_calls == 1  # 1 win, 1 loss
    assert tech_sc.incorrect_calls == 1
    assert tech_sc.win_rate == 0.5
    assert tech_sc.accuracy_rate == 0.5
    assert tech_sc.total_pnl_attributed == 10000.0  # 20000 + (-10000)

    fund_sc = scorecards["FundamentalAgent"]
    assert fund_sc.total_evaluations == 2
    assert fund_sc.bullish_calls == 1
    assert fund_sc.bearish_calls == 1
    # Trade 1 (BUY, price up) -> correct. Trade 2 (SELL, price down) -> correct!
    assert fund_sc.correct_calls == 2
    assert fund_sc.win_rate == 1.0


# ── 7. End-to-End PaperTradingEngine Integration ───────────────────────────────

def test_paper_engine_records_decision_and_evaluates_on_close(tmp_path):
    """
    Verify complete flow:
    Signal -> Risk Gate -> Paper Order -> Decision Journal -> Position Open -> Close -> PostTradeEvaluator -> Scorecard.
    """
    db_file = tmp_path / "paper_test.db"
    store = DecisionJournalStore(db_path=db_file)

    engine = PaperTradingEngine(
        fee_schedule=FeeSchedule(paper_per_side_pct=0.0005, slippage_pct=0.0005),
        journal_store=store,
    )
    engine.reset_account_balances()

    # Generate a BUY signal
    signal = TradeSignal(
        ticker="INFY.NS",
        market="india",
        strategy="swing",
        direction="BUY",
        entry_price=1500.0,
        stop_loss=1440.0,
        target_price=1620.0,
        quantity=5,
        confidence=0.82,
        reasoning="Multi-agent consensus breakout with strong IT spending guidance",
    )

    order = engine.execute_signal(signal)
    assert order is not None
    assert order.status == "FILLED"

    # Verify Decision Journal has recorded the executed decision
    entries = engine.get_journal_entries(symbol="INFY.NS")
    assert len(entries) == 1
    rec = entries[0]
    assert rec.status == "EXECUTED"
    assert rec.requested_quantity == 5
    assert rec.approved_quantity == order.quantity
    assert rec.order_id == order.order_id
    assert rec.position_id is not None
    pos_id = rec.position_id

    # Now close position at target price
    report = engine._close_position_record(
        pos_id=pos_id,
        ticker="INFY.NS",
        qty=order.quantity,
        avg_cost=order.filled_price,
        exit_price_raw=1620.0,
        margin_blocked=order.filled_price * order.quantity,
        fees_paid_so_far=engine.fees.compute_fees("india", "BUY", order.filled_price, order.quantity, True),
        market="india",
        strategy="swing",
        reason="TARGET_REACHED",
    )
    assert report is not None
    assert "[EXIT]" in report

    # Verify Decision Journal is now CLOSED
    updated_entries = engine.get_journal_entries(symbol="INFY.NS")
    assert len(updated_entries) == 1
    closed_rec = updated_entries[0]
    assert closed_rec.status == "CLOSED"
    assert closed_rec.realized_pnl is not None
    assert closed_rec.realized_pnl > 0

    # Verify Post-Trade Evaluation record was created
    evals = engine.get_trade_evaluations(symbol="INFY.NS")
    assert len(evals) == 1
    trade_eval = evals[0]
    assert trade_eval.symbol == "INFY.NS"
    assert trade_eval.is_winner is True
    assert trade_eval.directional_accuracy == 1.0
    assert trade_eval.major_failure_reason == "NONE_WINNING_TRADE"

    # Verify Scorecards reflect the trade
    scorecards = engine.get_agent_scorecards()
    assert len(scorecards) > 0


def test_short_position_evaluation(sample_decision_entry):
    """Verify evaluation for short positions."""
    sample_decision_entry.direction = "SELL"
    sample_decision_entry.entry_price = 100.0
    sample_decision_entry.stop_loss = 105.0
    sample_decision_entry.target_price = 90.0
    sample_decision_entry.agent_outputs["TechnicalAgent"] = AgentSignalOutput(
        agent="TechnicalAgent", signal="SELL", confidence=0.85, reasons=[], risks=[], evidence=[]
    )

    evaluation = PostTradeEvaluator.evaluate(
        entry=sample_decision_entry,
        exit_price=88.0,  # Short target achieved
        exit_reason="TAKE_PROFIT",
        realized_pnl=1200.0,
        return_pct=12.0,
    )

    assert evaluation.is_winner is True
    assert evaluation.directional_accuracy == 1.0
    assert evaluation.thesis_accuracy == 1.0
    assert evaluation.agent_accuracy["TechnicalAgent"].directional_accuracy == 1.0


def test_immutability_and_no_weight_retraining(temp_journal_store, sample_decision_entry):
    """
    Strict requirement: Phase 7 is measurement and observability only.
    No weights or agent models are mutated.
    """
    initial_confidence = sample_decision_entry.confidence
    initial_agent_outputs = sample_decision_entry.agent_outputs.copy()

    temp_journal_store.record_decision(sample_decision_entry)
    eval_res = PostTradeEvaluator.evaluate(sample_decision_entry, exit_price=2400.0, realized_pnl=-5000.0)
    temp_journal_store.record_evaluation(eval_res)
    scorecards = temp_journal_store.get_agent_scorecards()

    # Verify original decision entry remains identical
    stored = temp_journal_store.get_decision(sample_decision_entry.journal_id)
    assert stored.confidence == initial_confidence
    assert stored.agent_outputs["TechnicalAgent"]["confidence"] == initial_agent_outputs["TechnicalAgent"].confidence
    assert len(scorecards) > 0

