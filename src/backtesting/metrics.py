"""
Performance Metrics Calculation for Backtesting.

Computes institutional-grade quantitative trading metrics:
  - Total Return, CAGR, Benchmark Comparison
  - Sharpe Ratio, Sortino Ratio, Calmar Ratio
  - Max Drawdown, Drawdown Duration
  - Win Rate, Profit Factor, Expectancy, Win/Loss Ratio
  - Fees, Slippage Impact, Exposure
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from datetime import date, datetime
from typing import Any, Optional

import numpy as np
import pandas as pd

from src.backtesting.execution_simulator import SimulatedTradeRecord


@dataclass
class PerformanceMetrics:
    """Quantitative performance summary of a backtest run."""

    # ── Capital & Returns ──────────────────────────────────────────────────
    initial_capital: float
    ending_nav: float
    peak_nav: float
    total_net_profit: float
    total_return_pct: float
    cagr_pct: float
    benchmark_return_pct: Optional[float] = None
    alpha_pct: Optional[float] = None

    # ── Risk-Adjusted Ratios ───────────────────────────────────────────────
    annualized_volatility_pct: float = 0.0
    sharpe_ratio: float = 0.0
    sortino_ratio: float = 0.0
    calmar_ratio: float = 0.0

    # ── Drawdowns ──────────────────────────────────────────────────────────
    max_drawdown_pct: float = 0.0
    max_drawdown_amount: float = 0.0
    max_drawdown_duration_days: int = 0

    # ── Trade Statistics ───────────────────────────────────────────────────
    total_trades: int = 0
    winning_trades: int = 0
    losing_trades: int = 0
    breakeven_trades: int = 0
    win_rate_pct: float = 0.0
    profit_factor: float = 0.0
    expectancy: float = 0.0
    avg_trade_pnl: float = 0.0
    avg_win_amount: float = 0.0
    avg_loss_amount: float = 0.0
    win_loss_ratio: float = 0.0
    largest_win_amount: float = 0.0
    largest_loss_amount: float = 0.0
    avg_hold_duration_days: float = 0.0

    # ── Costs & Exposure ───────────────────────────────────────────────────
    total_fees_paid: float = 0.0
    total_slippage_cost: float = 0.0
    market_exposure_pct: float = 0.0   # % of days with active open positions

    # ── Time Horizon ───────────────────────────────────────────────────────
    start_date: str = ""
    end_date: str = ""
    total_calendar_days: int = 0
    trading_days_count: int = 0

    # ── Series data (optional for plotting/export) ──────────────────────────
    daily_returns: list[float] = field(default_factory=list)
    daily_nav: list[dict[str, Any]] = field(default_factory=list)


def calculate_metrics(
    daily_nav_history: list[dict[str, Any]],
    trade_records: list[SimulatedTradeRecord],
    initial_capital: float,
    market: str = "india",
    risk_free_rate: float = 0.05,
    benchmark_nav_series: Optional[list[float]] = None,
) -> PerformanceMetrics:
    """
    Calculate full quantitative metrics from NAV history and completed trade records.
    """
    if not daily_nav_history:
        return PerformanceMetrics(
            initial_capital=initial_capital,
            ending_nav=initial_capital,
            peak_nav=initial_capital,
            total_net_profit=0.0,
            total_return_pct=0.0,
            cagr_pct=0.0,
        )

    df_nav = pd.DataFrame(daily_nav_history)
    df_nav["nav"] = df_nav["nav"].astype(float)
    df_nav["date"] = pd.to_datetime(df_nav["date"])
    df_nav = df_nav.sort_values("date").reset_index(drop=True)

    start_date_str = df_nav["date"].iloc[0].strftime("%Y-%m-%d")
    end_date_str = df_nav["date"].iloc[-1].strftime("%Y-%m-%d")
    calendar_days = max(1, (df_nav["date"].iloc[-1] - df_nav["date"].iloc[0]).days)
    trading_days = len(df_nav)

    ending_nav = float(df_nav["nav"].iloc[-1])
    peak_nav = float(df_nav["nav"].max())
    total_net_profit = round(ending_nav - initial_capital, 2)
    total_return_pct = round(((ending_nav - initial_capital) / initial_capital) * 100, 2)

    # CAGR calculation
    years = calendar_days / 365.25
    if years > 0 and ending_nav > 0 and initial_capital > 0:
        cagr = ((ending_nav / initial_capital) ** (1.0 / years) - 1.0) * 100
        cagr_pct = round(cagr, 2)
    else:
        cagr_pct = 0.0

    # ── Daily returns & Drawdowns ──────────────────────────────────────────
    df_nav["daily_return"] = df_nav["nav"].pct_change().fillna(0.0)
    returns = df_nav["daily_return"].to_numpy()

    # Peak-to-trough Drawdowns
    df_nav["cum_max"] = df_nav["nav"].cummax()
    df_nav["drawdown_amount"] = df_nav["cum_max"] - df_nav["nav"]
    df_nav["drawdown_pct"] = (df_nav["drawdown_amount"] / df_nav["cum_max"]) * 100

    max_dd_pct = round(float(df_nav["drawdown_pct"].max()), 2)
    max_dd_amount = round(float(df_nav["drawdown_amount"].max()), 2)

    # Drawdown duration calculation
    max_dd_duration = 0
    current_dd_duration = 0
    for dd in df_nav["drawdown_pct"]:
        if dd > 0:
            current_dd_duration += 1
            if current_dd_duration > max_dd_duration:
                max_dd_duration = current_dd_duration
        else:
            current_dd_duration = 0

    # ── Risk-adjusted metrics ──────────────────────────────────────────────
    ann_factor = 252.0
    daily_rf = risk_free_rate / ann_factor

    mean_daily_return = float(np.mean(returns)) if len(returns) > 0 else 0.0
    daily_vol = float(np.std(returns, ddof=1)) if len(returns) > 1 else 0.0
    ann_vol_pct = round(daily_vol * math.sqrt(ann_factor) * 100, 2)

    # Sharpe Ratio
    if daily_vol > 1e-8:
        sharpe = math.sqrt(ann_factor) * (mean_daily_return - daily_rf) / daily_vol
        sharpe_ratio = round(sharpe, 2)
    else:
        sharpe_ratio = 0.0

    # Sortino Ratio (downside deviation)
    downside_returns = returns[returns < daily_rf]
    if len(downside_returns) > 1:
        downside_std = float(np.std(downside_returns, ddof=1))
        if downside_std > 1e-8:
            sortino = math.sqrt(ann_factor) * (mean_daily_return - daily_rf) / downside_std
            sortino_ratio = round(sortino, 2)
        else:
            sortino_ratio = 0.0
    else:
        sortino_ratio = 0.0

    # Calmar Ratio
    calmar_ratio = round(cagr_pct / max_dd_pct, 2) if max_dd_pct > 0 else 0.0

    # ── Benchmark comparison (if provided) ─────────────────────────────────
    benchmark_return_pct: Optional[float] = None
    alpha_pct: Optional[float] = None
    if benchmark_nav_series and len(benchmark_nav_series) == len(df_nav):
        b_init = benchmark_nav_series[0]
        b_end = benchmark_nav_series[-1]
        if b_init > 0:
            benchmark_return_pct = round(((b_end - b_init) / b_init) * 100, 2)
            alpha_pct = round(total_return_pct - benchmark_return_pct, 2)

    # ── Trade Analytics ────────────────────────────────────────────────────
    total_trades = len(trade_records)
    winning_trades = [t for t in trade_records if t.net_pnl > 0]
    losing_trades = [t for t in trade_records if t.net_pnl < 0]
    breakeven_trades = [t for t in trade_records if t.net_pnl == 0]

    win_count = len(winning_trades)
    loss_count = len(losing_trades)
    be_count = len(breakeven_trades)

    win_rate = round((win_count / total_trades) * 100, 2) if total_trades > 0 else 0.0

    gross_profits = sum(t.net_pnl for t in winning_trades)
    gross_losses = abs(sum(t.net_pnl for t in losing_trades))

    if gross_losses > 0:
        profit_factor = round(gross_profits / gross_losses, 2)
    elif gross_profits > 0:
        profit_factor = 999.99
    else:
        profit_factor = 0.0

    avg_trade_pnl = round(sum(t.net_pnl for t in trade_records) / total_trades, 2) if total_trades > 0 else 0.0
    avg_win = round(gross_profits / win_count, 2) if win_count > 0 else 0.0
    avg_loss = round(gross_losses / loss_count, 2) if loss_count > 0 else 0.0
    win_loss_ratio = round(avg_win / avg_loss, 2) if avg_loss > 0 else 0.0

    largest_win = round(max((t.net_pnl for t in winning_trades), default=0.0), 2)
    largest_loss = round(min((t.net_pnl for t in losing_trades), default=0.0), 2)

    avg_hold_days = round(sum(t.hold_duration_days for t in trade_records) / total_trades, 1) if total_trades > 0 else 0.0

    # Expectancy: (Win Rate * Avg Win) - (Loss Rate * Avg Loss)
    if total_trades > 0:
        win_prob = win_count / total_trades
        loss_prob = loss_count / total_trades
        expectancy = round((win_prob * avg_win) - (loss_prob * avg_loss), 2)
    else:
        expectancy = 0.0

    # Total costs
    total_fees = round(sum(t.total_fees for t in trade_records), 2)
    total_slippage = round(
        sum(
            abs(t.entry_fill_price - t.entry_price) * t.quantity +
            abs(t.exit_fill_price - t.exit_price_raw) * t.quantity
            for t in trade_records
        ),
        2,
    )

    # Market exposure (% of trading days holding >= 1 position)
    if "open_positions_count" in df_nav.columns:
        exposed_days = int((df_nav["open_positions_count"] > 0).sum())
        exposure_pct = round((exposed_days / trading_days) * 100, 2) if trading_days > 0 else 0.0
    else:
        exposure_pct = 0.0

    return PerformanceMetrics(
        initial_capital=initial_capital,
        ending_nav=ending_nav,
        peak_nav=peak_nav,
        total_net_profit=total_net_profit,
        total_return_pct=total_return_pct,
        cagr_pct=cagr_pct,
        benchmark_return_pct=benchmark_return_pct,
        alpha_pct=alpha_pct,
        annualized_volatility_pct=ann_vol_pct,
        sharpe_ratio=sharpe_ratio,
        sortino_ratio=sortino_ratio,
        calmar_ratio=calmar_ratio,
        max_drawdown_pct=max_dd_pct,
        max_drawdown_amount=max_dd_amount,
        max_drawdown_duration_days=max_dd_duration,
        total_trades=total_trades,
        winning_trades=win_count,
        losing_trades=loss_count,
        breakeven_trades=be_count,
        win_rate_pct=win_rate,
        profit_factor=profit_factor,
        expectancy=expectancy,
        avg_trade_pnl=avg_trade_pnl,
        avg_win_amount=avg_win,
        avg_loss_amount=avg_loss,
        win_loss_ratio=win_loss_ratio,
        largest_win_amount=largest_win,
        largest_loss_amount=largest_loss,
        avg_hold_duration_days=avg_hold_days,
        total_fees_paid=total_fees,
        total_slippage_cost=total_slippage,
        market_exposure_pct=exposure_pct,
        start_date=start_date_str,
        end_date=end_date_str,
        total_calendar_days=calendar_days,
        trading_days_count=trading_days,
        daily_returns=returns.tolist(),
        daily_nav=daily_nav_history,
    )
