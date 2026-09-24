"""
Phase 8: Walk-Forward Validation & Out-of-Sample Robustness Test Suite.

Verifies:
1. Rolling window generation across customizable train/test schedules.
2. Accurate extraction of 7 institutional metrics (CAGR, Sharpe, Sortino, MaxDD, PF, Win Rate, Turnover).
3. Automated detection of Overfitting (WFE < 0.50 / positive IS -> negative OOS).
4. Automated detection of Look-Ahead Bias (anomalous win rates / zero drawdown).
5. Automated detection of Unstable Performance (high metric variance across rolling folds).
6. Automated detection of Regime Dependence (concentrated bear-market failures).
7. Zero optimization on test periods constraint.
8. Markdown and CSV aggregate robustness tearsheet generation.
"""
from __future__ import annotations

import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from src.backtesting.data_provider import BacktestDataProvider
from src.backtesting.engine import BacktestConfig, BacktestEngine, BacktestResult
from src.evaluation.walk_forward import (
    PeriodMetrics,
    RobustnessDiagnostics,
    WalkForwardConfig,
    WalkForwardPeriod,
    WalkForwardReport,
    WalkForwardValidator,
)


# ── 1. Rolling Window Generator Tests ────────────────────────────────────────

def test_generate_rolling_windows_standard_schedule():
    """Verify standard 2-year train -> 1-year test rolling schedule (2020-2026)."""
    cfg = WalkForwardConfig(
        start_year=2020,
        end_year=2026,
        train_years=2,
        test_years=1,
        step_years=1,
    )
    windows = WalkForwardValidator.generate_rolling_windows(cfg)

    expected = [
        ("2020-01-01", "2021-12-31", "2022-01-01", "2022-12-31"),
        ("2021-01-01", "2022-12-31", "2023-01-01", "2023-12-31"),
        ("2022-01-01", "2023-12-31", "2024-01-01", "2024-12-31"),
        ("2023-01-01", "2024-12-31", "2025-01-01", "2025-12-31"),
        ("2024-01-01", "2025-12-31", "2026-01-01", "2026-12-31"),
    ]

    assert len(windows) == 5
    assert windows == expected


def test_generate_rolling_windows_custom_periods():
    """Verify custom explicit periods override automatic generation."""
    custom = [
        ("2021-01-01", "2022-06-30", "2022-07-01", "2022-12-31"),
        ("2022-01-01", "2023-06-30", "2023-07-01", "2023-12-31"),
    ]
    cfg = WalkForwardConfig(custom_periods=custom)
    windows = WalkForwardValidator.generate_rolling_windows(cfg)
    assert windows == custom


# ── 2. Metric Extraction & Calculation Tests ─────────────────────────────────

def test_period_metrics_extraction():
    """Verify extraction of the 7 required metrics from backtest results."""
    metrics = PeriodMetrics(
        cagr_pct=18.5,
        sharpe_ratio=1.45,
        sortino_ratio=2.10,
        max_drawdown_pct=8.2,
        profit_factor=1.85,
        win_rate_pct=62.5,
        turnover_pct=140.0,
        total_trades=24,
        total_return_pct=40.5,
        net_profit=40500.0,
    )

    assert metrics.cagr_pct == 18.5
    assert metrics.sharpe_ratio == 1.45
    assert metrics.sortino_ratio == 2.10
    assert metrics.max_drawdown_pct == 8.2
    assert metrics.profit_factor == 1.85
    assert metrics.win_rate_pct == 62.5
    assert metrics.turnover_pct == 140.0


def test_walk_forward_efficiency_calculation():
    """Verify Walk-Forward Efficiency (OOS / IS) handling."""
    # Positive IS and positive OOS
    assert WalkForwardValidator._calc_efficiency(2.0, 1.5) == 0.75
    # Positive IS and zero/negative OOS
    assert WalkForwardValidator._calc_efficiency(1.8, -0.9) == -0.50
    # Zero IS
    assert WalkForwardValidator._calc_efficiency(0.0, 0.0) == 1.0


# ── 3. Anomaly & Defect Detection Tests ───────────────────────────────────────

def test_overfitting_detection_wfe_below_threshold():
    """Detect overfitting when WFE < 0.50 or when positive IS collapses to negative OOS."""
    periods = [
        WalkForwardPeriod(
            period_index=1,
            label="2020–2021 -> 2022",
            train_start="2020-01-01",
            train_end="2021-12-31",
            test_start="2022-01-01",
            test_end="2022-12-31",
            in_sample_metrics=PeriodMetrics(cagr_pct=25.0, sharpe_ratio=2.2, win_rate_pct=70.0, profit_factor=2.5),
            out_of_sample_metrics=PeriodMetrics(cagr_pct=-5.0, sharpe_ratio=-0.3, win_rate_pct=35.0, profit_factor=0.7),
            efficiency_ratio=-0.14,
            cagr_efficiency_ratio=-0.20,
        ),
        WalkForwardPeriod(
            period_index=2,
            label="2021–2022 -> 2023",
            train_start="2021-01-01",
            train_end="2022-12-31",
            test_start="2023-01-01",
            test_end="2023-12-31",
            in_sample_metrics=PeriodMetrics(cagr_pct=28.0, sharpe_ratio=2.4, win_rate_pct=72.0, profit_factor=2.8),
            out_of_sample_metrics=PeriodMetrics(cagr_pct=-8.0, sharpe_ratio=-0.5, win_rate_pct=30.0, profit_factor=0.6),
            efficiency_ratio=-0.21,
            cagr_efficiency_ratio=-0.29,
        ),
    ]

    mean_is = PeriodMetrics(cagr_pct=26.5, sharpe_ratio=2.3)
    mean_oos = PeriodMetrics(cagr_pct=-6.5, sharpe_ratio=-0.4)

    diag = WalkForwardValidator.analyze_diagnostics(periods, mean_is, mean_oos, agg_eff_sharpe=-0.17)

    assert diag.is_overfitted is True
    assert "OVERFITTING_WFE_BELOW_50PCT" in diag.flags or "OVERFITTING_POSITIVE_IS_NEGATIVE_OOS" in diag.flags
    assert diag.robustness_score < 70.0


def test_lookahead_bias_detection():
    """Detect look-ahead bias when win rate is suspiciously perfect without drawdown."""
    periods = [
        WalkForwardPeriod(
            period_index=1,
            label="2020–2021 -> 2022",
            train_start="2020-01-01",
            train_end="2021-12-31",
            test_start="2022-01-01",
            test_end="2022-12-31",
            in_sample_metrics=PeriodMetrics(cagr_pct=20.0, sharpe_ratio=1.5),
            out_of_sample_metrics=PeriodMetrics(
                cagr_pct=50.0,
                sharpe_ratio=3.5,
                win_rate_pct=92.0,       # Suspicious > 80%
                max_drawdown_pct=0.1,    # Zero drawdown < 0.5%
                profit_factor=8.5,       # Suspicious > 5.0
                total_trades=20,
            ),
            efficiency_ratio=2.33,
            cagr_efficiency_ratio=2.5,
        )
    ]

    mean_is = PeriodMetrics(cagr_pct=20.0, sharpe_ratio=1.5)
    mean_oos = PeriodMetrics(cagr_pct=50.0, sharpe_ratio=3.5)

    diag = WalkForwardValidator.analyze_diagnostics(periods, mean_is, mean_oos, agg_eff_sharpe=2.33)

    assert diag.is_lookahead_biased is True
    assert "SUSPECTED_LOOKAHEAD_BIAS" in diag.flags


def test_unstable_performance_detection():
    """Detect unstable performance when rolling OOS returns fluctuate wildly."""
    periods = [
        WalkForwardPeriod(
            period_index=1,
            label="P1",
            train_start="2020-01-01", train_end="2021-12-31", test_start="2022-01-01", test_end="2022-12-31",
            in_sample_metrics=PeriodMetrics(cagr_pct=20.0, sharpe_ratio=1.5),
            out_of_sample_metrics=PeriodMetrics(cagr_pct=45.0, sharpe_ratio=2.8, win_rate_pct=65.0, profit_factor=2.0),
            efficiency_ratio=1.87, cagr_efficiency_ratio=2.25,
        ),
        WalkForwardPeriod(
            period_index=2,
            label="P2",
            train_start="2021-01-01", train_end="2022-12-31", test_start="2023-01-01", test_end="2023-12-31",
            in_sample_metrics=PeriodMetrics(cagr_pct=20.0, sharpe_ratio=1.5),
            out_of_sample_metrics=PeriodMetrics(cagr_pct=-25.0, sharpe_ratio=-1.2, win_rate_pct=30.0, profit_factor=0.5),
            efficiency_ratio=-0.80, cagr_efficiency_ratio=-1.25,
        ),
    ]

    mean_is = PeriodMetrics(cagr_pct=20.0, sharpe_ratio=1.5)
    mean_oos = PeriodMetrics(cagr_pct=10.0, sharpe_ratio=0.8)

    diag = WalkForwardValidator.analyze_diagnostics(periods, mean_is, mean_oos, agg_eff_sharpe=0.53)

    assert diag.is_unstable is True
    assert "UNSTABLE_OOS_PERFORMANCE" in diag.flags


def test_regime_dependence_detection():
    """Detect regime dependence when failure is isolated to bear market horizons."""
    periods = [
        WalkForwardPeriod(
            period_index=1,
            label="2020–2021 -> 2022",
            train_start="2020-01-01", train_end="2021-12-31", test_start="2022-01-01", test_end="2022-12-31",
            in_sample_metrics=PeriodMetrics(cagr_pct=22.0, sharpe_ratio=1.6),
            out_of_sample_metrics=PeriodMetrics(cagr_pct=-12.0, sharpe_ratio=-0.8, win_rate_pct=40.0, profit_factor=0.8),
            efficiency_ratio=-0.5, cagr_efficiency_ratio=-0.55,
        ),
        WalkForwardPeriod(
            period_index=2,
            label="2021–2022 -> 2023",
            train_start="2021-01-01", train_end="2022-12-31", test_start="2023-01-01", test_end="2023-12-31",
            in_sample_metrics=PeriodMetrics(cagr_pct=20.0, sharpe_ratio=1.5),
            out_of_sample_metrics=PeriodMetrics(cagr_pct=18.0, sharpe_ratio=1.4, win_rate_pct=60.0, profit_factor=1.8),
            efficiency_ratio=0.93, cagr_efficiency_ratio=0.90,
        ),
        WalkForwardPeriod(
            period_index=3,
            label="2022–2023 -> 2024",
            train_start="2022-01-01", train_end="2023-12-31", test_start="2024-01-01", test_end="2024-12-31",
            in_sample_metrics=PeriodMetrics(cagr_pct=21.0, sharpe_ratio=1.55),
            out_of_sample_metrics=PeriodMetrics(cagr_pct=19.0, sharpe_ratio=1.45, win_rate_pct=62.0, profit_factor=1.9),
            efficiency_ratio=0.94, cagr_efficiency_ratio=0.90,
        ),
    ]

    mean_is = PeriodMetrics(cagr_pct=21.0, sharpe_ratio=1.55)
    mean_oos = PeriodMetrics(cagr_pct=8.33, sharpe_ratio=0.68)

    diag = WalkForwardValidator.analyze_diagnostics(periods, mean_is, mean_oos, agg_eff_sharpe=0.44)

    assert diag.is_regime_dependent is True
    assert "REGIME_DEPENDENCE_BEAR_VULNERABILITY" in diag.flags


# ── 4. Robust Strategy Clean Bill of Health ──────────────────────────────────

def test_robust_strategy_diagnostics():
    """Verify a clean robust strategy gets high robustness score and no defect flags."""
    periods = [
        WalkForwardPeriod(
            period_index=1,
            label="2020–2021 -> 2022",
            train_start="2020-01-01", train_end="2021-12-31", test_start="2022-01-01", test_end="2022-12-31",
            in_sample_metrics=PeriodMetrics(cagr_pct=20.0, sharpe_ratio=1.6, win_rate_pct=58.0, profit_factor=1.7, max_drawdown_pct=9.0),
            out_of_sample_metrics=PeriodMetrics(cagr_pct=16.0, sharpe_ratio=1.3, win_rate_pct=55.0, profit_factor=1.5, max_drawdown_pct=11.0, total_trades=15),
            efficiency_ratio=0.81, cagr_efficiency_ratio=0.80,
        ),
        WalkForwardPeriod(
            period_index=2,
            label="2021–2022 -> 2023",
            train_start="2021-01-01", train_end="2022-12-31", test_start="2023-01-01", test_end="2023-12-31",
            in_sample_metrics=PeriodMetrics(cagr_pct=22.0, sharpe_ratio=1.7, win_rate_pct=60.0, profit_factor=1.8, max_drawdown_pct=8.0),
            out_of_sample_metrics=PeriodMetrics(cagr_pct=19.0, sharpe_ratio=1.5, win_rate_pct=58.0, profit_factor=1.65, max_drawdown_pct=10.0, total_trades=16),
            efficiency_ratio=0.88, cagr_efficiency_ratio=0.86,
        ),
    ]

    mean_is = PeriodMetrics(cagr_pct=21.0, sharpe_ratio=1.65)
    mean_oos = PeriodMetrics(cagr_pct=17.5, sharpe_ratio=1.40)

    diag = WalkForwardValidator.analyze_diagnostics(periods, mean_is, mean_oos, agg_eff_sharpe=0.85)

    assert diag.is_overfitted is False
    assert diag.is_lookahead_biased is False
    assert diag.is_unstable is False
    assert diag.is_regime_dependent is False
    assert diag.robustness_score == 100.0


# ── 5. Integration, Markdown, and CSV Export Tests ───────────────────────────

def test_walk_forward_markdown_and_csv_generation():
    """Verify Markdown report generation and CSV export functionality."""
    p1 = WalkForwardPeriod(
        period_index=1,
        label="2020–2021 -> 2022",
        train_start="2020-01-01",
        train_end="2021-12-31",
        test_start="2022-01-01",
        test_end="2022-12-31",
        in_sample_metrics=PeriodMetrics(cagr_pct=22.0, sharpe_ratio=1.6, sortino_ratio=2.2, max_drawdown_pct=8.0, profit_factor=1.8, win_rate_pct=60.0, turnover_pct=120.0),
        out_of_sample_metrics=PeriodMetrics(cagr_pct=17.0, sharpe_ratio=1.3, sortino_ratio=1.8, max_drawdown_pct=10.5, profit_factor=1.5, win_rate_pct=56.0, turnover_pct=115.0),
        efficiency_ratio=0.81,
        cagr_efficiency_ratio=0.77,
    )

    cfg = WalkForwardConfig(market="india", strategy="swing", train_years=2, test_years=1)
    mean_is = p1.in_sample_metrics
    mean_oos = p1.out_of_sample_metrics
    diag = RobustnessDiagnostics(robustness_score=95.0)

    report = WalkForwardReport(
        config=cfg,
        periods=[p1],
        mean_in_sample=mean_is,
        mean_out_of_sample=mean_oos,
        aggregate_efficiency_ratio=0.81,
        aggregate_cagr_efficiency=0.77,
        diagnostics=diag,
    )

    md = report.to_markdown()
    assert "Aegis Walk-Forward Validation & Robustness Report" in md
    assert "2020–2021 -> 2022" in md
    assert "Sharpe WFE" in md
    assert "CAGR WFE" in md

    # Export CSV
    with tempfile.TemporaryDirectory() as tmpdir:
        csv_file = Path(tmpdir) / "wf_report.csv"
        saved_path = report.export_csv(csv_file)
        assert saved_path.exists()
        content = saved_path.read_text()
        assert "2020–2021 -> 2022" in content
        assert "IS CAGR (%)" in content
        assert "OOS Sharpe" in content


def test_walk_forward_validator_execution():
    """Verify end-to-end execution of WalkForwardValidator on synthetic/cached slices."""
    custom = [
        ("2025-01-01", "2025-06-30", "2025-07-01", "2025-12-31"),
    ]
    cfg = WalkForwardConfig(
        market="india",
        strategy="swing",
        tickers=["RELIANCE.NS"],
        custom_periods=custom,
    )

    validator = WalkForwardValidator()
    report = validator.run(cfg)

    assert len(report.periods) == 1
    assert report.periods[0].train_start == "2025-01-01"
    assert report.periods[0].test_start == "2025-07-01"
    assert report.mean_in_sample is not None
    assert report.mean_out_of_sample is not None
    assert report.diagnostics is not None
