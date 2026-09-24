"""
Aegis Quantitative Performance Evaluation Package.

Provides institutional-grade metrics calculation, benchmark analytics (NIFTY 50 & S&P 500),
and Strategy vs Benchmark comparative tearsheet generation.
"""
from __future__ import annotations

from src.evaluation.benchmark import BenchmarkProvider, BenchmarkSummary
from src.evaluation.evaluator import PerformanceEvaluator, ComprehensiveMetrics
from src.evaluation.tearsheet import generate_strategy_vs_benchmark_report, print_tearsheet

__all__ = [
    "BenchmarkProvider",
    "BenchmarkSummary",
    "ComprehensiveMetrics",
    "PerformanceEvaluator",
    "generate_strategy_vs_benchmark_report",
    "print_tearsheet",
]
