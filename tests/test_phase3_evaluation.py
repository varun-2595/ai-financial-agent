"""
Comprehensive Unit & Integration Test Suite for Phase 3: Performance Evaluation & Benchmark Analytics.

Tests:
  1. BenchmarkProvider for India (NIFTY 50) and US (S&P 500)
  2. Gross vs Net return breakdown and cost drag computation
  3. Risk-adjusted metrics (Sharpe, Sortino, Calmar, Beta, Jensen's Alpha, Information Ratio, Treynor)
  4. Drawdown mechanics, duration, and recovery factor
  5. Portfolio turnover rate and market exposure calculation
  6. Trade distribution statistics (Win Rate, Profit Factor, Expectancy, Payoff Ratio, Streaks)
  7. Tearsheet generation and Markdown report export
"""
from __future__ import annotations

import math
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd
import pytest

from src.backtesting.execution_simulator import SimulatedTradeRecord
from src.evaluation.benchmark import BenchmarkProvider, BenchmarkSummary
from src.evaluation.evaluator import ComprehensiveMetrics, PerformanceEvaluator
from src.evaluation.tearsheet import generate_strategy_vs_benchmark_report, print_tearsheet


# ── Fixtures ───────────────────────────────────────────────────────────────────

@pytest.fixture
def sample_dates() -> list[date]:
    start = date(2025, 1, 1)
    return [start + timedelta(days=i) for i in range(100) if (start + timedelta(days=i)).weekday() < 5]


@pytest.fixture
def sample_benchmark_summary(sample_dates) -> BenchmarkSummary:
    days = len(sample_dates)
    # 8% total return with 12% vol
    prices = [100.0 * (1.0 + 0.08 * (i / days) + 0.01 * math.sin(i / 3.0)) for i in range(days)]
    returns = np.diff(prices) / prices[:-1]
    return BenchmarkSummary(
        symbol="^NSEI",
        name="NIFTY 50",
        market="india",
        start_date=sample_dates[0].strftime("%Y-%m-%d"),
        end_date=sample_dates[-1].strftime("%Y-%m-%d"),
        trading_days=days,
        initial_price=prices[0],
        ending_price=prices[-1],
        total_return_pct=8.0,
        cagr_pct=21.5,
        annualized_volatility_pct=12.0,
        sharpe_ratio=1.35,
        sortino_ratio=1.95,
        max_drawdown_pct=3.5,
        daily_returns=returns.tolist(),
        price_series=prices,
    )


@pytest.fixture
def sample_trades() -> list[SimulatedTradeRecord]:
    t1 = SimulatedTradeRecord(
        trade_id="TRD-1", position_id="POS-1", ticker="RELIANCE.NS", market="india",
        strategy="swing", direction="BUY", quantity=10,
        entry_price=100.0, entry_fill_price=100.05,
        exit_price_raw=115.0, exit_fill_price=114.95,
        margin_blocked=1000.50, entry_fees=1.0, exit_fees=1.15, total_fees=2.15,
        gross_pnl=149.0, net_pnl=146.85, pnl_pct=14.68,
        entry_time=datetime(2025, 1, 5, tzinfo=timezone.utc),
        exit_time=datetime(2025, 1, 15, tzinfo=timezone.utc),
        hold_duration_days=10.0, exit_reason="TARGET_HIT",
    )
    t2 = SimulatedTradeRecord(
        trade_id="TRD-2", position_id="POS-2", ticker="TCS.NS", market="india",
        strategy="swing", direction="BUY", quantity=5,
        entry_price=200.0, entry_fill_price=200.10,
        exit_price_raw=190.0, exit_fill_price=189.90,
        margin_blocked=1000.50, entry_fees=1.0, exit_fees=0.95, total_fees=1.95,
        gross_pnl=-51.0, net_pnl=-52.95, pnl_pct=-5.29,
        entry_time=datetime(2025, 1, 16, tzinfo=timezone.utc),
        exit_time=datetime(2025, 1, 20, tzinfo=timezone.utc),
        hold_duration_days=4.0, exit_reason="STOP_LOSS",
    )
    t3 = SimulatedTradeRecord(
        trade_id="TRD-3", position_id="POS-3", ticker="INFY.NS", market="india",
        strategy="swing", direction="BUY", quantity=8,
        entry_price=150.0, entry_fill_price=150.08,
        exit_price_raw=165.0, exit_fill_price=164.92,
        margin_blocked=1200.64, entry_fees=1.2, exit_fees=1.32, total_fees=2.52,
        gross_pnl=118.72, net_pnl=116.20, pnl_pct=9.68,
        entry_time=datetime(2025, 1, 22, tzinfo=timezone.utc),
        exit_time=datetime(2025, 2, 2, tzinfo=timezone.utc),
        hold_duration_days=11.0, exit_reason="TARGET_HIT",
    )
    return [t1, t2, t3]


@pytest.fixture
def sample_nav_history(sample_dates) -> list[dict]:
    days = len(sample_dates)
    navs = []
    base_nav = 10_000.0
    for i, d in enumerate(sample_dates):
        growth = 1.0 + 0.10 * (i / days) + 0.008 * math.sin(i / 2.5)
        nav_val = base_nav * growth
        navs.append({
            "date": d.strftime("%Y-%m-%d"),
            "cash": nav_val * 0.7,
            "reserved_margin": nav_val * 0.3,
            "unrealized_pnl": 0.0,
            "nav": round(nav_val, 2),
            "open_positions_count": 1 if i % 2 == 0 else 2,
        })
    return navs


# ── 1. Benchmark Provider Tests ────────────────────────────────────────────────

def test_benchmark_provider_india(sample_dates, tmp_path):
    """Verify BenchmarkProvider returns valid NIFTY 50 benchmark profile."""
    provider = BenchmarkProvider(cache_dir=tmp_path, use_cache=False)
    summary = provider.compute_summary("india", sample_dates, risk_free_rate=0.05)

    assert summary.symbol == "^NSEI"
    assert summary.name == "NIFTY 50"
    assert summary.market == "india"
    assert summary.trading_days == len(sample_dates)
    assert len(summary.daily_returns) == len(sample_dates) - 1
    assert summary.initial_price > 0
    assert summary.annualized_volatility_pct > 0


def test_benchmark_provider_us(sample_dates, tmp_path):
    """Verify BenchmarkProvider returns valid S&P 500 benchmark profile."""
    provider = BenchmarkProvider(cache_dir=tmp_path, use_cache=False)
    summary = provider.compute_summary("us", sample_dates, risk_free_rate=0.05)

    assert summary.symbol == "^GSPC"
    assert summary.name == "S&P 500"
    assert summary.market == "us"
    assert summary.trading_days == len(sample_dates)


# ── 2. Gross vs Net Returns Breakdown ──────────────────────────────────────────

def test_gross_vs_net_return_breakdown(sample_nav_history, sample_trades, sample_benchmark_summary):
    """
    Verify gross return vs net return distinction:
    Gross Profit = Net Profit + Total Brokerage Fees + Total Adverse Slippage Cost.
    """
    evaluator = PerformanceEvaluator(risk_free_rate=0.05)
    m = evaluator.evaluate(
        daily_nav_history=sample_nav_history,
        trade_records=sample_trades,
        initial_capital=10_000.0,
        benchmark=sample_benchmark_summary,
        market="india",
    )

    # Verify fees & slippage math
    total_fees = sum(t.total_fees for t in sample_trades)
    total_slippage = sum(
        abs(t.entry_fill_price - t.entry_price) * t.quantity +
        abs(t.exit_fill_price - t.exit_price_raw) * t.quantity
        for t in sample_trades
    )

    assert abs(m.total_brokerage_fees - total_fees) < 0.01
    assert abs(m.total_slippage_cost - total_slippage) < 0.01
    assert abs(m.total_transaction_friction - (total_fees + total_slippage)) < 0.01

    # Gross Profit = Net Profit + Total Friction
    assert abs(m.gross_profit - (m.net_profit + m.total_transaction_friction)) < 0.01
    assert m.gross_return_pct > m.net_return_pct
    assert abs(m.cost_drag_pct - (m.gross_return_pct - m.net_return_pct)) < 0.01


# ── 3. Risk-Adjusted Return Metrics ────────────────────────────────────────────

def test_risk_adjusted_metrics(sample_nav_history, sample_trades, sample_benchmark_summary):
    """
    Verify Sharpe, Sortino, Calmar, Beta, Jensen's Alpha, and Information Ratio.
    """
    evaluator = PerformanceEvaluator(risk_free_rate=0.05)
    m = evaluator.evaluate(
        daily_nav_history=sample_nav_history,
        trade_records=sample_trades,
        initial_capital=10_000.0,
        benchmark=sample_benchmark_summary,
        market="india",
    )

    assert m.annualized_volatility_pct > 0
    assert m.sharpe_ratio != 0.0
    assert m.sortino_ratio != 0.0
    assert m.calmar_ratio > 0
    assert m.beta > 0
    assert isinstance(m.alpha_pct, float)
    assert isinstance(m.information_ratio, float)
    assert isinstance(m.treynor_ratio, float)


# ── 4. Drawdown & Tail Risk ────────────────────────────────────────────────────

def test_drawdown_calculation():
    """Verify exact Peak-to-Trough Drawdown %, amount, and duration."""
    nav_series = [
        {"date": "2025-01-01", "nav": 10000.0, "open_positions_count": 0},
        {"date": "2025-01-02", "nav": 11000.0, "open_positions_count": 1},  # Peak
        {"date": "2025-01-03", "nav": 9900.0, "open_positions_count": 1},   # -10% DD (11000 -> 9900)
        {"date": "2025-01-04", "nav": 10450.0, "open_positions_count": 1},  # -5% DD
        {"date": "2025-01-05", "nav": 11550.0, "open_positions_count": 0},  # New Peak
    ]

    dates = [date(2025, 1, i + 1) for i in range(5)]
    bench = BenchmarkSummary(
        symbol="^NSEI", name="NIFTY", market="india", start_date="2025-01-01",
        end_date="2025-01-05", trading_days=5, initial_price=100.0, ending_price=105.0,
        total_return_pct=5.0, cagr_pct=5.0, annualized_volatility_pct=10.0,
        sharpe_ratio=1.0, sortino_ratio=1.0, max_drawdown_pct=2.0,
        daily_returns=[0.01, 0.01, 0.01, 0.02], price_series=[100, 101, 102, 103, 105],
    )

    evaluator = PerformanceEvaluator(risk_free_rate=0.05)
    m = evaluator.evaluate(nav_series, [], 10000.0, bench, "india")

    # Max DD from 11000 to 9900 = 1100 amount = 10.0%
    assert abs(m.max_drawdown_amount - 1100.0) < 0.01
    assert abs(m.max_drawdown_pct - 10.0) < 0.01
    assert m.peak_nav == 11550.0
    assert m.current_drawdown_pct == 0.0  # Ending at new peak


# ── 5. Portfolio Turnover & Exposure ───────────────────────────────────────────

def test_portfolio_turnover_and_exposure(sample_nav_history, sample_trades, sample_benchmark_summary):
    """Verify annualized portfolio turnover and market exposure calculations."""
    evaluator = PerformanceEvaluator(risk_free_rate=0.05)
    m = evaluator.evaluate(sample_nav_history, sample_trades, 10_000.0, sample_benchmark_summary, "india")

    assert m.total_traded_volume > 0
    assert m.annualized_turnover_pct > 0
    assert 0 <= m.market_exposure_pct <= 100.0
    assert m.avg_hold_duration_days > 0


# ── 6. Trade Distribution Analytics ────────────────────────────────────────────

def test_trade_distribution_statistics(sample_trades, sample_benchmark_summary, sample_nav_history):
    """Verify Win Rate, Profit Factor, Expectancy, Payoff Ratio, and streaks."""
    evaluator = PerformanceEvaluator(risk_free_rate=0.05)
    m = evaluator.evaluate(sample_nav_history, sample_trades, 10_000.0, sample_benchmark_summary, "india")

    assert m.total_trades == 3
    assert m.winning_trades == 2
    assert m.losing_trades == 1
    assert abs(m.win_rate_pct - 66.67) < 0.1

    # Gains: 146.85 + 116.20 = 263.05. Losses: 52.95
    # Profit factor: 263.05 / 52.95 ≈ 4.97
    assert abs(m.profit_factor - 4.97) < 0.1
    # Expectancy > 0
    assert m.expectancy > 0
    assert m.max_consecutive_wins == 1
    assert m.max_consecutive_losses == 1


# ── 7. Strategy vs Benchmark Tearsheet Reporting ───────────────────────────────

def test_tearsheet_and_markdown_report_generation(tmp_path, sample_nav_history, sample_trades, sample_benchmark_summary):
    """Verify that Markdown tearsheet contains all required sections."""
    evaluator = PerformanceEvaluator(risk_free_rate=0.05)
    m = evaluator.evaluate(sample_nav_history, sample_trades, 10_000.0, sample_benchmark_summary, "india")

    out_file = tmp_path / "tearsheet_report.md"
    md = generate_strategy_vs_benchmark_report(m, output_path=out_file)

    assert out_file.exists()
    assert "Executive Comparison: Strategy vs. Benchmark" in md
    assert "Risk-Adjusted & Factor Analysis" in md
    assert "Execution Costs & Slippage Drag" in md
    assert "Drawdown & Tail Risk Profile" in md
    assert "Portfolio Turnover & Capital Efficiency" in md
    assert "Trade Analytics & Statistical Distribution" in md

    # Ensure stdout tearsheet executes cleanly
    print_tearsheet(m)
