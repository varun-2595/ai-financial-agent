"""
Aegis Historical Backtesting Engine Package.

Provides deterministic historical simulation using the exact same strategy,
risk sizing, fee schedules, slippage, and portfolio accounting as paper/live trading.
"""
from __future__ import annotations

from src.backtesting.data_provider import BacktestDataProvider
from src.backtesting.engine import BacktestConfig, BacktestEngine, BacktestResult
from src.backtesting.execution_simulator import ExecutionSimulator
from src.backtesting.metrics import calculate_metrics, PerformanceMetrics
from src.backtesting.reporting import generate_markdown_report, print_backtest_summary

__all__ = [
    "BacktestConfig",
    "BacktestDataProvider",
    "BacktestEngine",
    "BacktestResult",
    "ExecutionSimulator",
    "PerformanceMetrics",
    "calculate_metrics",
    "generate_markdown_report",
    "print_backtest_summary",
]
