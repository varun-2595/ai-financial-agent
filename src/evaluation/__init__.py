"""
Aegis Quantitative Performance Evaluation Package.

Provides institutional-grade metrics calculation, benchmark analytics (NIFTY 50 & S&P 500),
Strategy vs Benchmark comparative tearsheet generation, and Walk-Forward Validation.
"""
from __future__ import annotations

from src.evaluation.benchmark import BenchmarkProvider, BenchmarkSummary
from src.evaluation.evaluator import ComprehensiveMetrics, PerformanceEvaluator
from src.evaluation.tearsheet import generate_strategy_vs_benchmark_report, print_tearsheet
from src.evaluation.walk_forward import (
    PeriodMetrics,
    RobustnessDiagnostics,
    WalkForwardConfig,
    WalkForwardPeriod,
    WalkForwardReport,
    WalkForwardValidator,
)

__all__ = [
    "BenchmarkProvider",
    "BenchmarkSummary",
    "ComprehensiveMetrics",
    "PerformanceEvaluator",
    "PeriodMetrics",
    "RobustnessDiagnostics",
    "WalkForwardConfig",
    "WalkForwardPeriod",
    "WalkForwardReport",
    "WalkForwardValidator",
    "generate_strategy_vs_benchmark_report",
    "print_tearsheet",
]
