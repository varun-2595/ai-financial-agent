"""
Walk-Forward Validation Engine for Aegis AI Trading Agent.

Implements rigorous rolling out-of-sample (OOS) validation without look-ahead
bias or parameter optimization on test periods.

Generates rolling walk-forward schedules (e.g., 2020-2021 -> 2022, 2021-2022 -> 2023),
evaluates key quantitative metrics (CAGR, Sharpe, Sortino, Max Drawdown, Profit Factor,
Win Rate, Turnover), and runs automated defect detection for overfitting, look-ahead bias,
unstable performance, and regime dependence.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
import numpy as np
import pandas as pd
from typing import TYPE_CHECKING, Any, Literal, Optional

if TYPE_CHECKING:
    from src.backtesting.data_provider import BacktestDataProvider
    from src.backtesting.engine import BacktestConfig, BacktestResult

from src.data.models import Strategy
from src.utils.logger import logger




@dataclass
class WalkForwardConfig:
    """Configuration for rolling walk-forward validation."""

    market: Literal["india", "us"] = "india"
    start_year: int = 2020
    end_year: int = 2026
    train_years: int = 2
    test_years: int = 1
    step_years: int = 1
    strategy: Strategy = "swing"
    tickers: Optional[list[str]] = None
    initial_capital: Optional[float] = None
    leverage: Optional[float] = None
    max_positions: int = 5
    risk_per_trade_pct: float = 0.03
    use_cache: bool = True
    benchmark_ticker: Optional[str] = None
    custom_periods: Optional[list[tuple[str, str, str, str]]] = None


@dataclass
class PeriodMetrics:
    """Core metrics required for In-Sample and Out-of-Sample periods."""

    cagr_pct: float = 0.0
    sharpe_ratio: float = 0.0
    sortino_ratio: float = 0.0
    max_drawdown_pct: float = 0.0
    profit_factor: float = 0.0
    win_rate_pct: float = 0.0
    turnover_pct: float = 0.0
    total_trades: int = 0
    total_return_pct: float = 0.0
    net_profit: float = 0.0

    @classmethod
    def from_backtest_result(cls, result: BacktestResult) -> PeriodMetrics:
        """Extract standardized metrics from a BacktestResult or ComprehensiveMetrics."""
        if result.evaluation is not None:
            ev = result.evaluation
            return cls(
                cagr_pct=round(ev.net_cagr_pct, 2),
                sharpe_ratio=round(ev.sharpe_ratio, 2),
                sortino_ratio=round(ev.sortino_ratio, 2),
                max_drawdown_pct=round(ev.max_drawdown_pct, 2),
                profit_factor=round(ev.profit_factor, 2),
                win_rate_pct=round(ev.win_rate_pct, 2),
                turnover_pct=round(ev.annualized_turnover_pct, 2),
                total_trades=ev.total_trades,
                total_return_pct=round(ev.net_return_pct, 2),
                net_profit=round(ev.net_profit, 2),
            )

        m = result.metrics
        # Fallback to standard BacktestMetrics
        turnover = 0.0
        if result.trades and m.initial_capital > 0:
            traded_vol = sum(t.quantity * t.entry_fill_price + t.quantity * t.exit_fill_price for t in result.trades)
            years = max(m.trading_days_count / 252.0, m.total_calendar_days / 365.25, 0.01)
            turnover = round(((traded_vol / 2.0) / m.initial_capital) / years * 100.0, 2)

        return cls(
            cagr_pct=round(m.cagr_pct, 2),
            sharpe_ratio=round(m.sharpe_ratio, 2),
            sortino_ratio=round(m.sortino_ratio, 2),
            max_drawdown_pct=round(m.max_drawdown_pct, 2),
            profit_factor=round(m.profit_factor, 2),
            win_rate_pct=round(m.win_rate_pct, 2),
            turnover_pct=turnover,
            total_trades=m.total_trades,
            total_return_pct=round(m.total_return_pct, 2),
            net_profit=round(m.total_net_profit, 2),
        )


@dataclass
class WalkForwardPeriod:
    """Results for a single In-Sample / Out-of-Sample rolling window."""

    period_index: int
    label: str
    train_start: str
    train_end: str
    test_start: str
    test_end: str
    in_sample_metrics: PeriodMetrics
    out_of_sample_metrics: PeriodMetrics
    efficiency_ratio: float  # OOS Sharpe / IS Sharpe
    cagr_efficiency_ratio: float  # OOS CAGR / IS CAGR
    is_result: Optional[BacktestResult] = None
    oos_result: Optional[BacktestResult] = None


@dataclass
class RobustnessDiagnostics:
    """Automated anomaly and defect detection flags."""

    is_overfitted: bool = False
    is_lookahead_biased: bool = False
    is_unstable: bool = False
    is_regime_dependent: bool = False
    robustness_score: float = 100.0
    flags: list[str] = field(default_factory=list)
    details: list[str] = field(default_factory=list)
    recommendations: list[str] = field(default_factory=list)


@dataclass
class WalkForwardReport:
    """Comprehensive aggregate report across all walk-forward validation periods."""

    config: WalkForwardConfig
    periods: list[WalkForwardPeriod]
    mean_in_sample: PeriodMetrics
    mean_out_of_sample: PeriodMetrics
    aggregate_efficiency_ratio: float
    aggregate_cagr_efficiency: float
    diagnostics: RobustnessDiagnostics
    generated_at: str = field(default_factory=lambda: datetime.now().strftime("%Y-%m-%d %H:%M:%S"))

    def to_dataframe(self) -> pd.DataFrame:
        """Convert periods summary into a pandas DataFrame."""
        rows = []
        for p in self.periods:
            rows.append({
                "Period": p.label,
                "Train Dates": f"{p.train_start} to {p.train_end}",
                "Test Dates": f"{p.test_start} to {p.test_end}",
                "IS CAGR (%)": p.in_sample_metrics.cagr_pct,
                "OOS CAGR (%)": p.out_of_sample_metrics.cagr_pct,
                "IS Sharpe": p.in_sample_metrics.sharpe_ratio,
                "OOS Sharpe": p.out_of_sample_metrics.sharpe_ratio,
                "IS Sortino": p.in_sample_metrics.sortino_ratio,
                "OOS Sortino": p.out_of_sample_metrics.sortino_ratio,
                "IS MaxDD (%)": p.in_sample_metrics.max_drawdown_pct,
                "OOS MaxDD (%)": p.out_of_sample_metrics.max_drawdown_pct,
                "IS Profit Factor": p.in_sample_metrics.profit_factor,
                "OOS Profit Factor": p.out_of_sample_metrics.profit_factor,
                "IS Win Rate (%)": p.in_sample_metrics.win_rate_pct,
                "OOS Win Rate (%)": p.out_of_sample_metrics.win_rate_pct,
                "IS Turnover (%)": p.in_sample_metrics.turnover_pct,
                "OOS Turnover (%)": p.out_of_sample_metrics.turnover_pct,
                "Sharpe WFE": p.efficiency_ratio,
                "CAGR WFE": p.cagr_efficiency_ratio,
            })
        return pd.DataFrame(rows)

    def to_markdown(self) -> str:
        """Generate structured Markdown tearsheet."""
        diag = self.diagnostics
        status_icon = "🟢 ROBUST" if diag.robustness_score >= 70 else ("🟡 MODERATE" if diag.robustness_score >= 45 else "🔴 FRAGILE")

        md = [
            f"# Aegis Walk-Forward Validation & Robustness Report",
            f"",
            f"**Status**: {status_icon} | **Robustness Score**: `{diag.robustness_score:.1f}/100` | **Generated**: `{self.generated_at}`",
            f"- **Market**: `{self.config.market.upper()}` | **Strategy**: `{self.config.strategy.capitalize()}`",
            f"- **Rolling Configuration**: Train `{self.config.train_years}y` $\\rightarrow$ Test `{self.config.test_years}y` (Step `{self.config.step_years}y`)",
            f"- **Rolling Windows Evaluated**: `{len(self.periods)}`",
            f"",
            f"---",
            f"",
            f"## 1. Executive Summary & Aggregate Robustness",
            f"",
            f"| Metric | Mean In-Sample (IS) | Mean Out-of-Sample (OOS) | Degradation / Efficiency |",
            f"|:---|---:|---:|---:|",
            f"| **CAGR** | `{self.mean_in_sample.cagr_pct:+.2f}%` | `{self.mean_out_of_sample.cagr_pct:+.2f}%` | **CAGR WFE**: `{self.aggregate_cagr_efficiency:.2f}` |",
            f"| **Sharpe Ratio** | `{self.mean_in_sample.sharpe_ratio:.2f}` | `{self.mean_out_of_sample.sharpe_ratio:.2f}` | **Sharpe WFE**: `{self.aggregate_efficiency_ratio:.2f}` |",
            f"| **Sortino Ratio** | `{self.mean_in_sample.sortino_ratio:.2f}` | `{self.mean_out_of_sample.sortino_ratio:.2f}` | `{((self.mean_out_of_sample.sortino_ratio / max(self.mean_in_sample.sortino_ratio, 0.01)) * 100):.1f}% retention` |",
            f"| **Max Drawdown** | `{self.mean_in_sample.max_drawdown_pct:.2f}%` | `{self.mean_out_of_sample.max_drawdown_pct:.2f}%` | `{self.mean_out_of_sample.max_drawdown_pct - self.mean_in_sample.max_drawdown_pct:+.2f}% $\\Delta$` |",
            f"| **Profit Factor** | `{self.mean_in_sample.profit_factor:.2f}` | `{self.mean_out_of_sample.profit_factor:.2f}` | `{self.mean_out_of_sample.profit_factor - self.mean_in_sample.profit_factor:+.2f} $\\Delta$` |",
            f"| **Win Rate** | `{self.mean_in_sample.win_rate_pct:.1f}%` | `{self.mean_out_of_sample.win_rate_pct:.1f}%` | `{self.mean_out_of_sample.win_rate_pct - self.mean_in_sample.win_rate_pct:+.1f}% $\\Delta$` |",
            f"| **Turnover** | `{self.mean_in_sample.turnover_pct:.1f}%` | `{self.mean_out_of_sample.turnover_pct:.1f}%` | `{self.mean_out_of_sample.turnover_pct - self.mean_in_sample.turnover_pct:+.1f}% $\\Delta$` |",
            f"",
            f"---",
            f"",
            f"## 2. Automated Defect & Anomaly Diagnostics",
            f"",
            f"- **Overfitting Detected**: `{'⚠️ YES' if diag.is_overfitted else '✅ NO'}`",
            f"- **Look-Ahead Bias Detected**: `{'⚠️ YES' if diag.is_lookahead_biased else '✅ NO'}`",
            f"- **Unstable Performance**: `{'⚠️ YES' if diag.is_unstable else '✅ NO'}`",
            f"- **Regime Dependence**: `{'⚠️ YES' if diag.is_regime_dependent else '✅ NO'}`",
            f"",
        ]

        if diag.details:
            md.append("### Diagnostic Findings:")
            for detail in diag.details:
                md.append(f"- {detail}")
            md.append("")

        if diag.recommendations:
            md.append("### Recommendations:")
            for rec in diag.recommendations:
                md.append(f"- 💡 {rec}")
            md.append("")

        md.extend([
            f"---",
            f"",
            f"## 3. Rolling Period Breakdown",
            f"",
            f"| Period | Train Range | Test Range | IS Sharpe | OOS Sharpe | WFE (Sharpe) | IS CAGR | OOS CAGR | OOS MaxDD | OOS Win Rate | OOS PF |",
            f"|:---|:---|:---|---:|---:|---:|---:|---:|---:|---:|---:|",
        ])

        for p in self.periods:
            md.append(
                f"| **{p.label}** | {p.train_start} to {p.train_end} | {p.test_start} to {p.test_end} | "
                f"`{p.in_sample_metrics.sharpe_ratio:.2f}` | `{p.out_of_sample_metrics.sharpe_ratio:.2f}` | "
                f"`{p.efficiency_ratio:.2f}` | `{p.in_sample_metrics.cagr_pct:+.2f}%` | "
                f"`{p.out_of_sample_metrics.cagr_pct:+.2f}%` | `{p.out_of_sample_metrics.max_drawdown_pct:.2f}%` | "
                f"`{p.out_of_sample_metrics.win_rate_pct:.1f}%` | `{p.out_of_sample_metrics.profit_factor:.2f}` |"
            )

        md.extend([
            f"",
            f"---",
            f"",
            f"## 4. Methodology & Walk-Forward Integrity",
            f"",
            f"- **Zero Parameter Optimization on OOS**: Strategy parameters, risk sizing rules, and agent thresholds are frozen prior to test period evaluation.",
            f"- **Point-in-Time Realism**: Data is strictly sliced chronologically up to each decision bar with no forward leakage.",
            f"- **Execution Friction**: Full brokerage fees, taxes (STT/stamp duty), and slippage models applied across all in-sample and out-of-sample periods.",
            f"",
        ])

        return "\n".join(md)

    def export_csv(self, filepath: str | Path) -> Path:
        """Export rolling periods summary to CSV."""
        path = Path(filepath)
        path.parent.mkdir(parents=True, exist_ok=True)
        df = self.to_dataframe()
        df.to_csv(path, index=False)
        return path


class WalkForwardValidator:
    """
    Orchestrates Rolling Walk-Forward Validation and Robustness Testing.
    """

    def __init__(self, data_provider: Optional[Any] = None):
        if data_provider is None:
            from src.backtesting.data_provider import BacktestDataProvider
            self.data_provider = BacktestDataProvider(use_cache=True)
        else:
            self.data_provider = data_provider

    @staticmethod
    def generate_rolling_windows(config: WalkForwardConfig) -> list[tuple[str, str, str, str]]:
        """
        Generate chronological rolling train and test date windows.
        Example:
          start_year=2020, end_year=2026, train_years=2, test_years=1, step_years=1:
          - (2020-01-01, 2021-12-31, 2022-01-01, 2022-12-31)
          - (2021-01-01, 2022-12-31, 2023-01-01, 2023-12-31)
          - (2022-01-01, 2023-12-31, 2024-01-01, 2024-12-31)
          - (2023-01-01, 2024-12-31, 2025-01-01, 2025-12-31)
          - (2024-01-01, 2025-12-31, 2026-01-01, 2026-12-31)
        """
        if config.custom_periods:
            return config.custom_periods

        windows = []
        curr_train_start = config.start_year

        while True:
            train_end_year = curr_train_start + config.train_years - 1
            test_start_year = train_end_year + 1
            test_end_year = test_start_year + config.test_years - 1

            if test_end_year > config.end_year:
                break

            train_start_str = f"{curr_train_start:04d}-01-01"
            train_end_str = f"{train_end_year:04d}-12-31"
            test_start_str = f"{test_start_year:04d}-01-01"
            test_end_str = f"{test_end_year:04d}-12-31"

            windows.append((train_start_str, train_end_str, test_start_str, test_end_str))
            curr_train_start += config.step_years

        return windows

    def run(self, config: WalkForwardConfig) -> WalkForwardReport:
        """
        Execute rolling walk-forward backtests and evaluate out-of-sample robustness.
        """
        from src.backtesting.engine import BacktestConfig, BacktestEngine

        windows = self.generate_rolling_windows(config)

        if not windows:
            raise ValueError(
                f"No rolling windows generated for range {config.start_year}-{config.end_year} "
                f"with train_years={config.train_years} and test_years={config.test_years}."
            )

        logger.info(f"[WalkForward] Starting validation across {len(windows)} rolling windows for {config.market.upper()}...")
        periods: list[WalkForwardPeriod] = []

        for idx, (train_start, train_end, test_start, test_end) in enumerate(windows, 1):
            train_label = f"{train_start[:4]}–{train_end[:4]}" if train_start[:4] != train_end[:4] else train_start[:4]
            test_label = f"{test_start[:4]}–{test_end[:4]}" if test_start[:4] != test_end[:4] else test_start[:4]
            label = f"{train_label} \u2192 {test_label}"

            logger.info(f"[WalkForward] [{idx}/{len(windows)}] Running Fold {label} (IS: {train_start}..{train_end}, OOS: {test_start}..{test_end})")

            # 1. In-Sample Backtest
            is_cfg = BacktestConfig(
                market=config.market,
                start_date=train_start,
                end_date=train_end,
                strategy=config.strategy,
                tickers=config.tickers,
                initial_capital=config.initial_capital,
                leverage=config.leverage,
                max_positions=config.max_positions,
                risk_per_trade_pct=config.risk_per_trade_pct,
                use_cache=config.use_cache,
                benchmark_ticker=config.benchmark_ticker,
            )
            is_engine = BacktestEngine(config=is_cfg, data_provider=self.data_provider)
            is_result = is_engine.run()
            is_metrics = PeriodMetrics.from_backtest_result(is_result)

            # 2. Out-of-Sample Backtest (Strictly NO re-optimization on test data)
            oos_cfg = BacktestConfig(
                market=config.market,
                start_date=test_start,
                end_date=test_end,
                strategy=config.strategy,
                tickers=config.tickers,
                initial_capital=config.initial_capital,
                leverage=config.leverage,
                max_positions=config.max_positions,
                risk_per_trade_pct=config.risk_per_trade_pct,
                use_cache=config.use_cache,
                benchmark_ticker=config.benchmark_ticker,
            )
            oos_engine = BacktestEngine(config=oos_cfg, data_provider=self.data_provider)
            oos_result = oos_engine.run()
            oos_metrics = PeriodMetrics.from_backtest_result(oos_result)

            # 3. Efficiency calculation
            eff_sharpe = self._calc_efficiency(is_metrics.sharpe_ratio, oos_metrics.sharpe_ratio)
            eff_cagr = self._calc_efficiency(is_metrics.cagr_pct, oos_metrics.cagr_pct)

            period = WalkForwardPeriod(
                period_index=idx,
                label=label,
                train_start=train_start,
                train_end=train_end,
                test_start=test_start,
                test_end=test_end,
                in_sample_metrics=is_metrics,
                out_of_sample_metrics=oos_metrics,
                efficiency_ratio=eff_sharpe,
                cagr_efficiency_ratio=eff_cagr,
                is_result=is_result,
                oos_result=oos_result,
            )
            periods.append(period)

        # 4. Compute Aggregate Metrics across periods
        mean_is = self._aggregate_period_metrics([p.in_sample_metrics for p in periods])
        mean_oos = self._aggregate_period_metrics([p.out_of_sample_metrics for p in periods])
        agg_eff_sharpe = self._calc_efficiency(mean_is.sharpe_ratio, mean_oos.sharpe_ratio)
        agg_eff_cagr = self._calc_efficiency(mean_is.cagr_pct, mean_oos.cagr_pct)

        # 5. Defect & Anomaly Diagnostics
        diagnostics = self.analyze_diagnostics(periods, mean_is, mean_oos, agg_eff_sharpe)

        return WalkForwardReport(
            config=config,
            periods=periods,
            mean_in_sample=mean_is,
            mean_out_of_sample=mean_oos,
            aggregate_efficiency_ratio=agg_eff_sharpe,
            aggregate_cagr_efficiency=agg_eff_cagr,
            diagnostics=diagnostics,
        )

    @staticmethod
    def _calc_efficiency(is_val: float, oos_val: float) -> float:
        """Safely calculate Walk-Forward Efficiency (OOS / IS)."""
        if abs(is_val) < 1e-4:
            return round(1.0 if abs(oos_val) < 1e-4 else (0.0 if oos_val <= 0 else 2.0), 2)
        if is_val < 0 and oos_val < 0:
            # Both negative: if OOS is less negative, efficiency is positive
            return round(is_val / oos_val, 2)
        if is_val > 0 and oos_val <= 0:
            return round(oos_val / is_val, 2)
        return round(oos_val / is_val, 2)

    @staticmethod
    def _aggregate_period_metrics(metric_list: list[PeriodMetrics]) -> PeriodMetrics:
        """Calculate mean metrics across multiple periods."""
        if not metric_list:
            return PeriodMetrics()

        n = len(metric_list)
        return PeriodMetrics(
            cagr_pct=round(sum(m.cagr_pct for m in metric_list) / n, 2),
            sharpe_ratio=round(sum(m.sharpe_ratio for m in metric_list) / n, 2),
            sortino_ratio=round(sum(m.sortino_ratio for m in metric_list) / n, 2),
            max_drawdown_pct=round(sum(m.max_drawdown_pct for m in metric_list) / n, 2),
            profit_factor=round(sum(m.profit_factor for m in metric_list) / n, 2),
            win_rate_pct=round(sum(m.win_rate_pct for m in metric_list) / n, 2),
            turnover_pct=round(sum(m.turnover_pct for m in metric_list) / n, 2),
            total_trades=int(round(sum(m.total_trades for m in metric_list) / n)),
            total_return_pct=round(sum(m.total_return_pct for m in metric_list) / n, 2),
            net_profit=round(sum(m.net_profit for m in metric_list) / n, 2),
        )

    @classmethod
    def analyze_diagnostics(
        cls,
        periods: list[WalkForwardPeriod],
        mean_is: PeriodMetrics,
        mean_oos: PeriodMetrics,
        agg_eff_sharpe: float,
    ) -> RobustnessDiagnostics:
        """
        Run automated heuristic detection for:
        1. Overfitting (WFE < 0.50 or positive IS becoming negative OOS)
        2. Look-ahead bias (abnormally high win rate > 80% without drawdown or instant perfection)
        3. Unstable performance (high variance in OOS Sharpe/CAGR across periods)
        4. Regime dependence (concentrated failure in specific down/bear regimes)
        """
        flags: list[str] = []
        details: list[str] = []
        recommendations: list[str] = []
        score = 100.0

        oos_sharpes = [p.out_of_sample_metrics.sharpe_ratio for p in periods]
        oos_cagrs = [p.out_of_sample_metrics.cagr_pct for p in periods]
        oos_drawdowns = [p.out_of_sample_metrics.max_drawdown_pct for p in periods]
        oos_win_rates = [p.out_of_sample_metrics.win_rate_pct for p in periods]
        oos_profit_factors = [p.out_of_sample_metrics.profit_factor for p in periods]

        # ── 1. Overfitting Detection ─────────────────────────────────────────
        # Walk-Forward Efficiency < 50% or Sharpe drop > 50%
        is_overfitted = False
        if mean_is.sharpe_ratio > 0.5 and agg_eff_sharpe < 0.50:
            is_overfitted = True
            flags.append("OVERFITTING_WFE_BELOW_50PCT")
            details.append(
                f"Walk-Forward Efficiency ({agg_eff_sharpe:.2f}) is below 0.50. "
                f"Out-of-sample Sharpe ({mean_oos.sharpe_ratio:.2f}) degraded severely from In-Sample ({mean_is.sharpe_ratio:.2f})."
            )
            recommendations.append("Reduce strategy parameter complexity, tighten entry filters, or increase training sample size.")
            score -= 30.0
        elif mean_is.cagr_pct > 10.0 and mean_oos.cagr_pct < 0.0:
            is_overfitted = True
            flags.append("OVERFITTING_POSITIVE_IS_NEGATIVE_OOS")
            details.append(
                f"Strategy was profitable in-sample (CAGR {mean_is.cagr_pct:+.2f}%) but lost money out-of-sample (CAGR {mean_oos.cagr_pct:+.2f}%)."
            )
            recommendations.append("Review signal features for curve-fitting to specific historical price patterns.")
            score -= 35.0

        # Count periods where OOS Sharpe is negative while IS Sharpe is strongly positive
        severe_drops = sum(1 for p in periods if p.in_sample_metrics.sharpe_ratio >= 1.0 and p.out_of_sample_metrics.sharpe_ratio < 0.0)
        if severe_drops >= max(1, len(periods) // 2):
            is_overfitted = True
            if "OVERFITTING_PERSISTENT_DROPS" not in flags:
                flags.append("OVERFITTING_PERSISTENT_DROPS")
                details.append(f"{severe_drops}/{len(periods)} periods suffered total Sharpe collapse (IS >= 1.0 -> OOS < 0.0).")
                score -= 20.0

        # ── 2. Look-Ahead Bias Detection ─────────────────────────────────────
        # Unrealistic perfection: Win rate > 80% with low drawdown (< 3%), or profit factor > 5.0 with > 10 trades
        is_lookahead_biased = False
        suspicious_high_win_rate = any(w > 80.0 and t > 5 for w, t in zip(oos_win_rates, [p.out_of_sample_metrics.total_trades for p in periods]))
        suspicious_pf = any(pf > 5.0 and t >= 10 for pf, t in zip(oos_profit_factors, [p.out_of_sample_metrics.total_trades for p in periods]))
        zero_drawdown = any(dd < 0.5 and t >= 10 for dd, t in zip(oos_drawdowns, [p.out_of_sample_metrics.total_trades for p in periods]))

        if suspicious_high_win_rate or suspicious_pf or zero_drawdown:
            is_lookahead_biased = True
            flags.append("SUSPECTED_LOOKAHEAD_BIAS")
            if suspicious_high_win_rate:
                details.append(f"Uncharacteristically high OOS win rate (>80%) detected in one or more test periods.")
            if suspicious_pf:
                details.append(f"Abnormally high profit factor (>5.0) detected with substantial trade count.")
            if zero_drawdown:
                details.append(f"Virtually zero maximum drawdown (<0.5%) detected, which often indicates future price leakage.")
            recommendations.append("Inspect indicator lag and shift bars. Verify that signals only use data available strictly at bar open/close.")
            score -= 40.0

        # ── 3. Unstable Performance Detection ────────────────────────────────
        # High variance in OOS metrics across folds
        is_unstable = False
        sharpe_std = float(np.std(oos_sharpes)) if len(oos_sharpes) > 1 else 0.0
        cagr_std = float(np.std(oos_cagrs)) if len(oos_cagrs) > 1 else 0.0

        # Flip-flops between large gains and large losses
        has_extreme_swings = False
        if len(oos_cagrs) >= 2:
            max_cagr = max(oos_cagrs)
            min_cagr = min(oos_cagrs)
            if (max_cagr - min_cagr) > 40.0:
                has_extreme_swings = True

        if sharpe_std > 1.2 or (cagr_std > 20.0 and has_extreme_swings):
            is_unstable = True
            flags.append("UNSTABLE_OOS_PERFORMANCE")
            details.append(
                f"High cross-period variance detected (Sharpe \u03c3={sharpe_std:.2f}, CAGR \u03c3={cagr_std:.2f}%). "
                f"Performance fluctuates significantly across rolling horizons."
            )
            recommendations.append("Add dynamic volatility scaling or regime-adaptive position sizing to stabilize rolling returns.")
            score -= 20.0

        # ── 4. Regime Dependence Detection ───────────────────────────────────
        # Check if negative performance is concentrated in specific periods (e.g. 2022 bear market)
        is_regime_dependent = False
        negative_periods = [p for p in periods if p.out_of_sample_metrics.cagr_pct < 0]
        if 0 < len(negative_periods) < len(periods):
            # Check if negative periods are associated with specific market downturns
            bear_years = {"2022", "2020"}  # Classical high-volatility / drawdown years
            bear_failures = [p for p in negative_periods if any(y in p.test_start for y in bear_years)]
            if len(bear_failures) == len(negative_periods) and len(bear_failures) > 0:
                is_regime_dependent = True
                flags.append("REGIME_DEPENDENCE_BEAR_VULNERABILITY")
                details.append(
                    f"Performance degradation is exclusively concentrated in bear/high-volatility test windows ({', '.join(p.label for p in bear_failures)})."
                )
                recommendations.append("Implement trend filters (e.g. 200 EMA / ADX macro filter) to disarm aggressive buying in bear regimes.")
                score -= 15.0

        # Ensure robustness score is bounded [0, 100]
        score = max(0.0, min(100.0, score))

        if not flags:
            details.append("No significant overfitting, look-ahead bias, or severe regime fragility detected across rolling out-of-sample periods.")
            recommendations.append("Strategy exhibits robust out-of-sample stability and walk-forward efficiency.")

        return RobustnessDiagnostics(
            is_overfitted=is_overfitted,
            is_lookahead_biased=is_lookahead_biased,
            is_unstable=is_unstable,
            is_regime_dependent=is_regime_dependent,
            robustness_score=round(score, 1),
            flags=flags,
            details=details,
            recommendations=recommendations,
        )
