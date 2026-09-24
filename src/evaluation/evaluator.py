"""
Comprehensive Performance Evaluator for Aegis AI Trading Agent.

Calculates institutional quantitative metrics, gross vs net return breakdowns,
risk-adjusted benchmarks, drawdowns, turnover, and execution friction.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from datetime import date, datetime
from typing import Any, Literal, Optional

import numpy as np
import pandas as pd

from src.backtesting.execution_simulator import SimulatedTradeRecord
from src.evaluation.benchmark import BenchmarkSummary


@dataclass
class ComprehensiveMetrics:
    """Rigorous quantitative performance report containing all Phase 3 metrics."""

    # ── Horizon & Capital ──────────────────────────────────────────────────
    market: Literal["india", "us"]
    start_date: str
    end_date: str
    calendar_days: int
    trading_days: int
    initial_capital: float
    ending_nav: float
    peak_nav: float

    # ── 1. Returns Breakdown (Gross vs Net) ─────────────────────────────────
    gross_profit: float
    gross_return_pct: float
    gross_cagr_pct: float
    net_profit: float
    net_return_pct: float
    net_cagr_pct: float

    # ── 2. Risk-Adjusted Returns ───────────────────────────────────────────
    annualized_volatility_pct: float
    downside_volatility_pct: float
    sharpe_ratio: float
    sortino_ratio: float
    calmar_ratio: float
    beta: float
    alpha_pct: float
    information_ratio: float
    treynor_ratio: float

    # ── 3. Drawdowns ───────────────────────────────────────────────────────
    max_drawdown_amount: float
    max_drawdown_pct: float
    max_drawdown_duration_days: int
    current_drawdown_pct: float
    recovery_factor: float

    # ── 4. Execution Costs & Slippage Impact ────────────────────────────────
    total_brokerage_fees: float
    total_slippage_cost: float
    total_transaction_friction: float
    cost_drag_pct: float             # gross_return_pct - net_return_pct
    friction_bps_on_volume: float    # total friction / total traded volume * 10000

    # ── 5. Turnover & Exposure ─────────────────────────────────────────────
    total_traded_volume: float
    annualized_turnover_pct: float
    market_exposure_pct: float
    avg_hold_duration_days: float

    # ── 6. Trade Analytics ─────────────────────────────────────────────────
    total_trades: int
    winning_trades: int
    losing_trades: int
    breakeven_trades: int
    win_rate_pct: float
    profit_factor: float
    expectancy: float
    avg_trade_pnl: float
    avg_win_amount: float
    avg_loss_amount: float
    win_loss_ratio: float
    largest_win: float
    largest_loss: float
    max_consecutive_wins: int
    max_consecutive_losses: int

    # ── 7. Benchmark Comparison ────────────────────────────────────────────
    benchmark_symbol: str
    benchmark_name: str
    benchmark_return_pct: float
    benchmark_cagr_pct: float
    benchmark_volatility_pct: float
    benchmark_sharpe_ratio: float
    benchmark_max_drawdown_pct: float
    excess_return_pct: float

    # ── Daily series (for inspection / plots) ──────────────────────────────
    daily_returns: list[float] = field(default_factory=list)
    daily_nav: list[dict[str, Any]] = field(default_factory=list)


class PerformanceEvaluator:
    """
    Computes rigorous evaluation metrics from NAV histories, trade logs, and benchmarks.
    """

    def __init__(self, risk_free_rate: float = 0.05):
        self.risk_free_rate = risk_free_rate

    def evaluate(
        self,
        daily_nav_history: list[dict[str, Any]],
        trade_records: list[SimulatedTradeRecord],
        initial_capital: float,
        benchmark: BenchmarkSummary,
        market: Literal["india", "us"] = "india",
    ) -> ComprehensiveMetrics:
        """
        Evaluate full quantitative performance metrics.
        """
        if not daily_nav_history:
            return self._empty_metrics(initial_capital, benchmark, market)

        df_nav = pd.DataFrame(daily_nav_history)
        df_nav["nav"] = df_nav["nav"].astype(float)
        df_nav["date"] = pd.to_datetime(df_nav["date"])
        df_nav = df_nav.sort_values("date").reset_index(drop=True)

        start_date_str = df_nav["date"].iloc[0].strftime("%Y-%m-%d")
        end_date_str = df_nav["date"].iloc[-1].strftime("%Y-%m-%d")
        calendar_days = max(1, (df_nav["date"].iloc[-1] - df_nav["date"].iloc[0]).days)
        trading_days = len(df_nav)
        years = max(trading_days / 252.0, calendar_days / 365.25)

        ending_nav = float(df_nav["nav"].iloc[-1])
        peak_nav = float(df_nav["nav"].max())
        net_profit = round(ending_nav - initial_capital, 2)
        net_return_pct = round((net_profit / initial_capital) * 100.0, 2)

        net_cagr = ((ending_nav / initial_capital) ** (1.0 / years) - 1.0) * 100.0 if (years > 0 and initial_capital > 0 and ending_nav > 0) else 0.0
        net_cagr_pct = round(net_cagr, 2)

        # ── 1. Costs & Gross Return Breakdown ──────────────────────────────
        total_brokerage = sum(t.total_fees for t in trade_records)
        total_slippage = sum(
            abs(t.entry_fill_price - t.entry_price) * t.quantity +
            abs(t.exit_fill_price - t.exit_price_raw) * t.quantity
            for t in trade_records
        )
        total_friction = total_brokerage + total_slippage
        gross_profit = net_profit + total_friction
        gross_return_pct = round((gross_profit / initial_capital) * 100.0, 2)

        gross_ending = initial_capital + gross_profit
        gross_cagr = ((gross_ending / initial_capital) ** (1.0 / years) - 1.0) * 100.0 if (years > 0 and initial_capital > 0 and gross_ending > 0) else 0.0
        gross_cagr_pct = round(gross_cagr, 2)
        cost_drag_pct = round(gross_return_pct - net_return_pct, 2)

        # Total traded volume
        total_traded_volume = sum(
            (t.entry_fill_price * t.quantity) + (t.exit_fill_price * t.quantity)
            for t in trade_records
        )
        friction_bps = round((total_friction / total_traded_volume) * 10000.0, 1) if total_traded_volume > 0 else 0.0

        # Turnover calculation: Total Volume / (2 * Avg NAV) * (252 / Trading Days)
        avg_nav = float(df_nav["nav"].mean())
        if avg_nav > 0 and trading_days > 0:
            turnover_ann = (total_traded_volume / (2.0 * avg_nav)) * (252.0 / trading_days) * 100.0
            turnover_pct = round(turnover_ann, 2)
        else:
            turnover_pct = 0.0

        # ── 2. Daily Returns & Volatility ──────────────────────────────────
        df_nav["daily_return"] = df_nav["nav"].pct_change().fillna(0.0)
        returns = df_nav["daily_return"].to_numpy()

        ann_factor = 252.0
        daily_rf = self.risk_free_rate / ann_factor
        mean_ret = float(np.mean(returns)) if len(returns) > 0 else 0.0
        daily_vol = float(np.std(returns, ddof=1)) if len(returns) > 1 else 0.0
        ann_vol_pct = round(daily_vol * math.sqrt(ann_factor) * 100.0, 2)

        # Sharpe
        sharpe = math.sqrt(ann_factor) * (mean_ret - daily_rf) / daily_vol if daily_vol > 1e-8 else 0.0
        sharpe_ratio = round(sharpe, 2)

        # Sortino
        downside = returns[returns < daily_rf]
        downside_std = float(np.std(downside, ddof=1)) if len(downside) > 1 else 0.0
        downside_vol_pct = round(downside_std * math.sqrt(ann_factor) * 100.0, 2)
        sortino = math.sqrt(ann_factor) * (mean_ret - daily_rf) / downside_std if downside_std > 1e-8 else 0.0
        sortino_ratio = round(sortino, 2)

        # ── 3. Drawdowns ───────────────────────────────────────────────────
        df_nav["cum_max"] = df_nav["nav"].cummax()
        df_nav["drawdown_amount"] = df_nav["cum_max"] - df_nav["nav"]
        df_nav["drawdown_pct"] = (df_nav["drawdown_amount"] / df_nav["cum_max"]) * 100.0

        max_dd_pct = round(float(df_nav["drawdown_pct"].max()), 2)
        max_dd_amount = round(float(df_nav["drawdown_amount"].max()), 2)
        current_dd_pct = round(float(df_nav["drawdown_pct"].iloc[-1]), 2)

        max_dd_dur = 0
        curr_dd_dur = 0
        for dd in df_nav["drawdown_pct"]:
            if dd > 0:
                curr_dd_dur += 1
                if curr_dd_dur > max_dd_dur:
                    max_dd_dur = curr_dd_dur
            else:
                curr_dd_dur = 0

        calmar = round(net_cagr_pct / max_dd_pct, 2) if max_dd_pct > 0 else 0.0
        recovery = round(net_profit / max_dd_amount, 2) if max_dd_amount > 0 else 0.0

        # ── 4. Benchmark Relative Ratios (Beta, Alpha, IR, Treynor) ─────────
        bench_returns = np.array(benchmark.daily_returns)
        if len(returns) > 1 and len(bench_returns) == len(returns):
            # Covariance matrix between strategy and benchmark returns
            cov_matrix = np.cov(returns, bench_returns)
            cov_sb = cov_matrix[0, 1]
            var_b = cov_matrix[1, 1]

            beta_val = float(cov_sb / var_b) if var_b > 1e-10 else 1.0
            beta = round(beta_val, 2)

            # Jensen's Alpha annualized: (CAGR_s - Rf) - Beta * (CAGR_b - Rf)
            alpha_val = (net_cagr_pct - self.risk_free_rate * 100.0) - beta_val * (benchmark.cagr_pct - self.risk_free_rate * 100.0)
            alpha_pct = round(alpha_val, 2)

            # Information Ratio: mean(excess returns) / std(excess returns) * sqrt(252)
            diff_returns = returns - bench_returns
            diff_std = float(np.std(diff_returns, ddof=1))
            diff_mean = float(np.mean(diff_returns))
            info_ratio = round((diff_mean / diff_std) * math.sqrt(ann_factor), 2) if diff_std > 1e-8 else 0.0

            # Treynor Ratio: (CAGR_s - Rf) / Beta
            treynor = round((net_cagr_pct - self.risk_free_rate * 100.0) / beta_val, 2) if abs(beta_val) > 1e-4 else 0.0
        else:
            beta = 1.0
            alpha_pct = round(net_cagr_pct - benchmark.cagr_pct, 2)
            info_ratio = 0.0
            treynor = 0.0

        excess_ret_pct = round(net_return_pct - benchmark.total_return_pct, 2)

        # ── 5. Trade Statistics & Streaks ───────────────────────────────────
        total_trades = len(trade_records)
        wins = [t for t in trade_records if t.net_pnl > 0]
        losses = [t for t in trade_records if t.net_pnl < 0]
        breakevens = [t for t in trade_records if t.net_pnl == 0]

        win_count = len(wins)
        loss_count = len(losses)
        be_count = len(breakevens)
        win_rate = round((win_count / total_trades) * 100.0, 2) if total_trades > 0 else 0.0

        gross_gains = sum(t.net_pnl for t in wins)
        gross_losses_abs = abs(sum(t.net_pnl for t in losses))

        profit_factor = round(gross_gains / gross_losses_abs, 2) if gross_losses_abs > 0 else (999.99 if gross_gains > 0 else 0.0)
        avg_trade = round(sum(t.net_pnl for t in trade_records) / total_trades, 2) if total_trades > 0 else 0.0
        avg_win = round(gross_gains / win_count, 2) if win_count > 0 else 0.0
        avg_loss = round(gross_losses_abs / loss_count, 2) if loss_count > 0 else 0.0
        win_loss_rat = round(avg_win / avg_loss, 2) if avg_loss > 0 else 0.0

        largest_w = round(max((t.net_pnl for t in wins), default=0.0), 2)
        largest_l = round(min((t.net_pnl for t in losses), default=0.0), 2)

        # Streaks
        max_w_streak = 0
        curr_w_streak = 0
        max_l_streak = 0
        curr_l_streak = 0
        for t in trade_records:
            if t.net_pnl > 0:
                curr_w_streak += 1
                curr_l_streak = 0
                if curr_w_streak > max_w_streak:
                    max_w_streak = curr_w_streak
            elif t.net_pnl < 0:
                curr_l_streak += 1
                curr_w_streak = 0
                if curr_l_streak > max_l_streak:
                    max_l_streak = curr_l_streak
            else:
                curr_w_streak = 0
                curr_l_streak = 0

        # Expectancy
        if total_trades > 0:
            p_win = win_count / total_trades
            p_loss = loss_count / total_trades
            expectancy_val = round((p_win * avg_win) - (p_loss * avg_loss), 2)
        else:
            expectancy_val = 0.0

        avg_hold_days = round(sum(t.hold_duration_days for t in trade_records) / total_trades, 1) if total_trades > 0 else 0.0

        # Exposure
        if "open_positions_count" in df_nav.columns:
            exposed_days = int((df_nav["open_positions_count"] > 0).sum())
            exposure_pct = round((exposed_days / trading_days) * 100.0, 2) if trading_days > 0 else 0.0
        else:
            exposure_pct = 0.0

        return ComprehensiveMetrics(
            market=market,
            start_date=start_date_str,
            end_date=end_date_str,
            calendar_days=calendar_days,
            trading_days=trading_days,
            initial_capital=initial_capital,
            ending_nav=ending_nav,
            peak_nav=peak_nav,
            gross_profit=round(gross_profit, 2),
            gross_return_pct=gross_return_pct,
            gross_cagr_pct=gross_cagr_pct,
            net_profit=net_profit,
            net_return_pct=net_return_pct,
            net_cagr_pct=net_cagr_pct,
            annualized_volatility_pct=ann_vol_pct,
            downside_volatility_pct=downside_vol_pct,
            sharpe_ratio=sharpe_ratio,
            sortino_ratio=sortino_ratio,
            calmar_ratio=calmar,
            beta=beta,
            alpha_pct=alpha_pct,
            information_ratio=info_ratio,
            treynor_ratio=treynor,
            max_drawdown_amount=max_dd_amount,
            max_drawdown_pct=max_dd_pct,
            max_drawdown_duration_days=max_dd_dur,
            current_drawdown_pct=current_dd_pct,
            recovery_factor=recovery,
            total_brokerage_fees=round(total_brokerage, 2),
            total_slippage_cost=round(total_slippage, 2),
            total_transaction_friction=round(total_friction, 2),
            cost_drag_pct=cost_drag_pct,
            friction_bps_on_volume=friction_bps,
            total_traded_volume=round(total_traded_volume, 2),
            annualized_turnover_pct=turnover_pct,
            market_exposure_pct=exposure_pct,
            avg_hold_duration_days=avg_hold_days,
            total_trades=total_trades,
            winning_trades=win_count,
            losing_trades=loss_count,
            breakeven_trades=be_count,
            win_rate_pct=win_rate,
            profit_factor=profit_factor,
            expectancy=expectancy_val,
            avg_trade_pnl=avg_trade,
            avg_win_amount=avg_win,
            avg_loss_amount=avg_loss,
            win_loss_ratio=win_loss_rat,
            largest_win=largest_w,
            largest_loss=largest_l,
            max_consecutive_wins=max_w_streak,
            max_consecutive_losses=max_l_streak,
            benchmark_symbol=benchmark.symbol,
            benchmark_name=benchmark.name,
            benchmark_return_pct=benchmark.total_return_pct,
            benchmark_cagr_pct=benchmark.cagr_pct,
            benchmark_volatility_pct=benchmark.annualized_volatility_pct,
            benchmark_sharpe_ratio=benchmark.sharpe_ratio,
            benchmark_max_drawdown_pct=benchmark.max_drawdown_pct,
            excess_return_pct=excess_ret_pct,
            daily_returns=returns.tolist(),
            daily_nav=daily_nav_history,
        )

    def _empty_metrics(
        self,
        initial_capital: float,
        benchmark: BenchmarkSummary,
        market: Literal["india", "us"],
    ) -> ComprehensiveMetrics:
        return ComprehensiveMetrics(
            market=market,
            start_date="",
            end_date="",
            calendar_days=0,
            trading_days=0,
            initial_capital=initial_capital,
            ending_nav=initial_capital,
            peak_nav=initial_capital,
            gross_profit=0.0,
            gross_return_pct=0.0,
            gross_cagr_pct=0.0,
            net_profit=0.0,
            net_return_pct=0.0,
            net_cagr_pct=0.0,
            annualized_volatility_pct=0.0,
            downside_volatility_pct=0.0,
            sharpe_ratio=0.0,
            sortino_ratio=0.0,
            calmar_ratio=0.0,
            beta=1.0,
            alpha_pct=0.0,
            information_ratio=0.0,
            treynor_ratio=0.0,
            max_drawdown_amount=0.0,
            max_drawdown_pct=0.0,
            max_drawdown_duration_days=0,
            current_drawdown_pct=0.0,
            recovery_factor=0.0,
            total_brokerage_fees=0.0,
            total_slippage_cost=0.0,
            total_transaction_friction=0.0,
            cost_drag_pct=0.0,
            friction_bps_on_volume=0.0,
            total_traded_volume=0.0,
            annualized_turnover_pct=0.0,
            market_exposure_pct=0.0,
            avg_hold_duration_days=0.0,
            total_trades=0,
            winning_trades=0,
            losing_trades=0,
            breakeven_trades=0,
            win_rate_pct=0.0,
            profit_factor=0.0,
            expectancy=0.0,
            avg_trade_pnl=0.0,
            avg_win_amount=0.0,
            avg_loss_amount=0.0,
            win_loss_ratio=0.0,
            largest_win=0.0,
            largest_loss=0.0,
            max_consecutive_wins=0,
            max_consecutive_losses=0,
            benchmark_symbol=benchmark.symbol,
            benchmark_name=benchmark.name,
            benchmark_return_pct=benchmark.total_return_pct,
            benchmark_cagr_pct=benchmark.cagr_pct,
            benchmark_volatility_pct=benchmark.annualized_volatility_pct,
            benchmark_sharpe_ratio=benchmark.sharpe_ratio,
            benchmark_max_drawdown_pct=benchmark.max_drawdown_pct,
            excess_return_pct=0.0,
        )
