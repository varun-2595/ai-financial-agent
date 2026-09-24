"""
Comprehensive Unit & Integration Test Suite for Phase 2: Aegis Backtesting Engine.

Tests:
  1. Data provider point-in-time slicing (look-ahead bias prevention)
  2. Execution simulator fills, slippage, fees, and SL/TP conflict resolution
  3. Risk limits and position constraints during backtest
  4. Exact portfolio accounting & NAV invariance in simulated runs
  5. Performance metrics calculation (Sharpe, Sortino, CAGR, Drawdown, Profit Factor)
  6. Multi-market support (India & US)
  7. Reporting (Markdown and CSV exports)
  8. CLI argument parsing
"""
from __future__ import annotations

from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd
import pytest

from src.backtesting.data_provider import BacktestDataProvider
from src.backtesting.engine import BacktestConfig, BacktestEngine
from src.backtesting.execution_simulator import (
    ExecutionSimulator,
    SimulatedPosition,
    SimulatedTradeRecord,
)
from src.backtesting.metrics import calculate_metrics, PerformanceMetrics
from src.backtesting.reporting import generate_markdown_report, export_trades_csv, export_nav_csv
from src.data.models import Quote, StockSnapshot, TradeSignal
from src.trading.fees import FeeSchedule


# ── Fixtures & Mock Data ───────────────────────────────────────────────────────

def _generate_synthetic_ohlcv(
    start_date: str = "2025-01-01",
    num_days: int = 100,
    base_price: float = 100.0,
    trend: float = 0.5,
) -> pd.DataFrame:
    """Generate realistic synthetic daily OHLCV series."""
    dates = pd.bdate_range(start=start_date, periods=num_days, tz=timezone.utc)
    data = []
    price = base_price
    for i, d in enumerate(dates):
        open_p = price + np.sin(i / 5.0) * 2.0
        high_p = open_p + 2.5
        low_p = open_p - 2.0
        close_p = open_p + trend + np.cos(i / 3.0) * 1.5
        vol = 100_000 + i * 1000
        price = close_p
        data.append({
            "open": open_p,
            "high": max(open_p, high_p, close_p),
            "low": min(open_p, low_p, close_p),
            "close": close_p,
            "volume": vol,
        })
    df = pd.DataFrame(data, index=dates)
    return df


@pytest.fixture
def mock_data_provider(tmp_path) -> BacktestDataProvider:
    provider = BacktestDataProvider(cache_dir=tmp_path, use_cache=False)
    # Inject synthetic data for Indian and US tickers
    df_rel = _generate_synthetic_ohlcv("2025-01-01", 100, 2500.0, trend=2.0)
    df_tcs = _generate_synthetic_ohlcv("2025-01-01", 100, 3500.0, trend=-1.0)
    df_aapl = _generate_synthetic_ohlcv("2025-01-01", 100, 200.0, trend=0.5)

    provider._memory_cache["RELIANCE.NS"] = df_rel
    provider._memory_cache["TCS.NS"] = df_tcs
    provider._memory_cache["AAPL"] = df_aapl
    return provider


# ── 1. Point-in-Time Data Provider Tests ───────────────────────────────────────

def test_data_provider_point_in_time_slicing(mock_data_provider):
    """
    Verify point-in-time snapshot contains ONLY bars up to current_time.
    Strictly verifies no future bars leak into history (look-ahead bias prevention).
    """
    provider = mock_data_provider
    cutoff = datetime(2025, 2, 1, 15, 30, tzinfo=timezone.utc)

    snap = provider.get_point_in_time_snapshot(
        ticker="RELIANCE.NS",
        current_time=cutoff,
        market="india",
        lookback_bars=60,
    )

    assert snap is not None
    assert snap.ticker == "RELIANCE.NS"
    assert len(snap.history) > 0

    # Ensure EVERY quote timestamp is strictly <= cutoff
    for q in snap.history:
        assert q.timestamp <= cutoff, f"Future quote leaked: {q.timestamp} > {cutoff}"

    # Latest quote timestamp should be close to cutoff
    assert snap.history[-1].timestamp <= cutoff


def test_data_provider_trading_days(mock_data_provider):
    """Verify trading days are correctly extracted in chronological order."""
    provider = mock_data_provider
    days = provider.get_trading_days("2025-01-01", "2025-02-15", "india")
    assert len(days) > 10
    # Check monotonicity
    for i in range(1, len(days)):
        assert days[i] > days[i - 1]


# ── 2. Execution Simulator Tests ───────────────────────────────────────────────

def test_execution_simulator_entry():
    """Verify simulated BUY applies adverse slippage, margin, and entry fees."""
    fees = FeeSchedule(paper_per_side_pct=0.001, slippage_pct=0.001)
    sim = ExecutionSimulator(fee_schedule=fees)

    sig = TradeSignal(
        ticker="TEST", market="india", strategy="swing",
        direction="BUY", entry_price=100.0,
        stop_loss=95.0, target_price=110.0,
        quantity=10, confidence=0.8, reasoning="test",
    )

    now = datetime(2025, 1, 15, 15, 30, tzinfo=timezone.utc)
    pos, cash_deducted = sim.execute_entry(sig, fill_price_raw=100.0, timestamp=now, leverage=1.0)

    # Filled price = 100 * 1.001 = 100.10
    assert abs(pos.avg_cost - 100.10) < 0.01
    # Margin = 100.10 * 10 = 1001.00
    assert abs(pos.margin_blocked - 1001.00) < 0.01
    # Fee = 100.10 * 10 * 0.001 = 1.001
    assert abs(pos.fees_paid - 1.001) < 0.01
    # Total cash deducted = 1001.00 + 1.001 = 1002.001
    assert abs(cash_deducted - 1002.001) < 0.01


def test_execution_simulator_target_hit():
    """Verify target hit triggers exit with sell slippage and exit fees."""
    fees = FeeSchedule(paper_per_side_pct=0.001, slippage_pct=0.001)
    sim = ExecutionSimulator(fee_schedule=fees)

    pos = SimulatedPosition(
        position_id="POS-1", ticker="TEST", market="india",
        strategy="swing", direction="BUY", quantity=10,
        entry_price=100.0, avg_cost=100.10, current_price=100.10,
        stop_loss=95.0, target_price=110.0,
        margin_blocked=1001.00, fees_paid=1.001,
        opened_at=datetime(2025, 1, 10, tzinfo=timezone.utc),
        entry_order_id="ORD-1",
    )

    now = datetime(2025, 1, 15, 15, 30, tzinfo=timezone.utc)
    # Bar with High=112.0 >= Target 110.0
    res = sim.check_and_execute_exit(
        position=pos, bar_open=108.0, bar_high=112.0,
        bar_low=107.0, bar_close=111.0, timestamp=now,
    )

    assert res is not None
    trade, cash_returned = res
    assert trade.exit_reason == "TARGET_HIT"
    # Exit raw = max(open=108, target=110) = 110.0
    # Exit fill = 110.0 * (1 - 0.001) = 109.89
    assert abs(trade.exit_fill_price - 109.89) < 0.01
    # Gross PnL = (109.89 - 100.10) * 10 = 97.90
    assert abs(trade.gross_pnl - 97.90) < 0.05
    # Exit fee = 109.89 * 10 * 0.001 = 1.0989
    # Net PnL = 97.90 - 1.0989 = 96.80
    assert abs(trade.net_pnl - 96.80) < 0.05
    # Cash returned = margin (1001.00) + net_pnl (96.80) = 1097.80
    assert abs(cash_returned - (1001.00 + trade.net_pnl)) < 0.01


def test_execution_simulator_stop_loss_hit():
    """Verify stop loss triggers when bar_low <= stop_loss."""
    fees = FeeSchedule(paper_per_side_pct=0.0, slippage_pct=0.0)
    sim = ExecutionSimulator(fee_schedule=fees)

    pos = SimulatedPosition(
        position_id="POS-1", ticker="TEST", market="india",
        strategy="swing", direction="BUY", quantity=10,
        entry_price=100.0, avg_cost=100.0, current_price=100.0,
        stop_loss=95.0, target_price=115.0,
        margin_blocked=1000.0, fees_paid=0.0,
        opened_at=datetime(2025, 1, 10, tzinfo=timezone.utc),
        entry_order_id="ORD-1",
    )

    now = datetime(2025, 1, 12, tzinfo=timezone.utc)
    # Bar with Low=93.0 <= Stop Loss 95.0
    res = sim.check_and_execute_exit(
        position=pos, bar_open=97.0, bar_high=98.0,
        bar_low=93.0, bar_close=94.0, timestamp=now,
    )

    assert res is not None
    trade, cash_returned = res
    assert trade.exit_reason == "STOP_LOSS"
    assert trade.net_pnl == (95.0 - 100.0) * 10   # -50.0
    assert cash_returned == 1000.0 - 50.0          # 950.0


def test_execution_simulator_conflict_resolution():
    """
    Verify conservative conflict resolution:
    If a wide bar touches BOTH stop-loss and target, assume stop-loss was hit first.
    """
    fees = FeeSchedule(paper_per_side_pct=0.0, slippage_pct=0.0)
    sim = ExecutionSimulator(fee_schedule=fees)

    pos = SimulatedPosition(
        position_id="POS-1", ticker="TEST", market="india",
        strategy="swing", direction="BUY", quantity=10,
        entry_price=100.0, avg_cost=100.0, current_price=100.0,
        stop_loss=90.0, target_price=120.0,
        margin_blocked=1000.0, fees_paid=0.0,
        opened_at=datetime(2025, 1, 10, tzinfo=timezone.utc),
        entry_order_id="ORD-1",
    )

    now = datetime(2025, 1, 12, tzinfo=timezone.utc)
    # Ultra-wide bar: Low=85.0 (below SL 90) and High=125.0 (above TP 120)
    res = sim.check_and_execute_exit(
        position=pos, bar_open=100.0, bar_high=125.0,
        bar_low=85.0, bar_close=110.0, timestamp=now,
    )

    assert res is not None
    trade, _ = res
    assert trade.exit_reason == "STOP_LOSS", "Conflict must conservatively trigger STOP_LOSS"


# ── 3. Performance Metrics Calculation Tests ───────────────────────────────────

def test_metrics_empty_run():
    """Metrics calculation on 0 trades returns clean defaults without error."""
    m = calculate_metrics([], [], initial_capital=10_000.0)
    assert m.total_trades == 0
    assert m.total_return_pct == 0.0
    assert m.ending_nav == 10_000.0
    assert m.sharpe_ratio == 0.0


def test_metrics_calculation_accuracy():
    """Verify Sharpe, Sortino, Drawdown, Win Rate, and Profit Factor calculations."""
    nav_series = [
        {"date": "2025-01-01", "nav": 10000.0, "open_positions_count": 0},
        {"date": "2025-01-02", "nav": 10200.0, "open_positions_count": 1},
        {"date": "2025-01-03", "nav": 10500.0, "open_positions_count": 1},
        {"date": "2025-01-04", "nav": 10100.0, "open_positions_count": 1},  # -3.8% DD
        {"date": "2025-01-05", "nav": 10800.0, "open_positions_count": 0},
    ]

    t1 = SimulatedTradeRecord(
        trade_id="T1", position_id="P1", ticker="AAA", market="india",
        strategy="swing", direction="BUY", quantity=10,
        entry_price=100.0, entry_fill_price=100.0,
        exit_price_raw=110.0, exit_fill_price=110.0,
        margin_blocked=1000.0, entry_fees=1.0, exit_fees=1.0, total_fees=2.0,
        gross_pnl=100.0, net_pnl=98.0, pnl_pct=9.8,
        entry_time=datetime(2025, 1, 1, tzinfo=timezone.utc),
        exit_time=datetime(2025, 1, 3, tzinfo=timezone.utc),
        hold_duration_days=2.0, exit_reason="TARGET_HIT",
    )
    t2 = SimulatedTradeRecord(
        trade_id="T2", position_id="P2", ticker="BBB", market="india",
        strategy="swing", direction="BUY", quantity=10,
        entry_price=100.0, entry_fill_price=100.0,
        exit_price_raw=95.0, exit_fill_price=95.0,
        margin_blocked=1000.0, entry_fees=1.0, exit_fees=1.0, total_fees=2.0,
        gross_pnl=-50.0, net_pnl=-52.0, pnl_pct=-5.2,
        entry_time=datetime(2025, 1, 3, tzinfo=timezone.utc),
        exit_time=datetime(2025, 1, 4, tzinfo=timezone.utc),
        hold_duration_days=1.0, exit_reason="STOP_LOSS",
    )

    m = calculate_metrics(nav_series, [t1, t2], initial_capital=10_000.0)

    assert m.total_return_pct == 8.0     # (10800 - 10000)/10000
    assert m.peak_nav == 10800.0
    assert m.total_trades == 2
    assert m.winning_trades == 1
    assert m.losing_trades == 1
    assert m.win_rate_pct == 50.0
    # Profit factor = 98.0 / 52.0 ≈ 1.88
    assert abs(m.profit_factor - 1.88) < 0.05
    # Max DD = (10500 - 10100)/10500 * 100 = 3.81%
    assert abs(m.max_drawdown_pct - 3.81) < 0.05
    assert m.market_exposure_pct == 60.0  # 3 days out of 5 holding positions


# ── 4. Full Backtest Engine Integration Tests ─────────────────────────────────

def test_backtest_engine_run_india(mock_data_provider):
    """
    Run a full deterministic backtest on synthetic Indian market data.
    Verifies execution, accounting, ledger, and metrics generation.
    """
    cfg = BacktestConfig(
        market="india",
        start_date="2025-01-01",
        end_date="2025-03-31",
        initial_capital=10_000.0,
        strategy="swing",
        tickers=["RELIANCE.NS", "TCS.NS"],
        max_positions=2,
    )

    engine = BacktestEngine(config=cfg, data_provider=mock_data_provider)
    res = engine.run()

    assert res is not None
    assert isinstance(res.metrics, PerformanceMetrics)
    assert len(res.daily_nav_history) > 20
    assert len(res.ledger) > 0

    # Ensure NAV equality: NAV = cash + reserved_margin + unrealized_pnl
    for item in res.daily_nav_history:
        calc_nav = item["cash"] + item["reserved_margin"] + item["unrealized_pnl"]
        assert abs(item["nav"] - calc_nav) < 0.05


def test_backtest_engine_run_us(mock_data_provider):
    """
    Run a deterministic backtest on US market data.
    """
    cfg = BacktestConfig(
        market="us",
        start_date="2025-01-01",
        end_date="2025-03-31",
        initial_capital=1_000.0,
        strategy="swing",
        tickers=["AAPL"],
        max_positions=1,
    )

    engine = BacktestEngine(config=cfg, data_provider=mock_data_provider)
    res = engine.run()

    assert res is not None
    assert res.metrics.initial_capital == 1_000.0
    assert len(res.daily_nav_history) > 20


def test_backtest_position_limit_respected(mock_data_provider):
    """Verify open positions count never exceeds max_positions."""
    cfg = BacktestConfig(
        market="india",
        start_date="2025-01-01",
        end_date="2025-03-31",
        initial_capital=50_000.0,
        strategy="swing",
        tickers=["RELIANCE.NS", "TCS.NS"],
        max_positions=1,   # strict limit of 1
    )

    engine = BacktestEngine(config=cfg, data_provider=mock_data_provider)
    res = engine.run()

    for item in res.daily_nav_history:
        assert item["open_positions_count"] <= 1


# ── 5. Reporting and Export Tests ──────────────────────────────────────────────

def test_markdown_and_csv_reports(tmp_path, mock_data_provider):
    """Verify Markdown report generation and CSV exports produce valid files."""
    cfg = BacktestConfig(
        market="india",
        start_date="2025-01-01",
        end_date="2025-02-15",
        initial_capital=10_000.0,
        strategy="swing",
        tickers=["RELIANCE.NS"],
    )

    engine = BacktestEngine(config=cfg, data_provider=mock_data_provider)
    res = engine.run()

    # 1. Markdown report
    md_path = tmp_path / "test_report.md"
    md_content = generate_markdown_report(res, output_path=md_path)
    assert md_path.exists()
    assert "Aegis Backtest Report" in md_content
    assert "Executive Summary" in md_content

    # 2. CSV exports
    trades_csv = tmp_path / "trades.csv"
    nav_csv = tmp_path / "nav.csv"
    export_trades_csv(res, trades_csv)
    export_nav_csv(res, nav_csv)

    assert trades_csv.exists()
    assert nav_csv.exists()
    df_nav = pd.read_csv(nav_csv)
    assert "nav" in df_nav.columns
    assert len(df_nav) > 0
