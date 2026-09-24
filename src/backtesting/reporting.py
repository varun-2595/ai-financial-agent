"""
Backtest Reporting and Visualization.

Generates rich ASCII terminal summaries, Markdown reports, and CSV exports.
"""
from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Optional

from src.backtesting.engine import BacktestResult


def print_backtest_summary(result: BacktestResult) -> None:
    """Print a clean, formatted ASCII performance summary to stdout."""
    m = result.metrics
    cfg = result.config
    curr = "₹" if cfg.market == "india" else "$"

    sep = "═" * 70
    sub_sep = "─" * 70

    print(f"\n{sep}")
    print(f"📊 AEGIS BACKTEST REPORT — {cfg.market.upper()} ({cfg.strategy.upper()})")
    print(f"📅 Period: {m.start_date} to {m.end_date} ({m.trading_days_count} trading days)")
    print(f"{sep}")

    print("\n💰 PORTFOLIO PERFORMANCE")
    print(sub_sep)
    print(f"  Initial Capital:       {curr}{m.initial_capital:>12,.2f}")
    print(f"  Ending NAV:            {curr}{m.ending_nav:>12,.2f}")
    print(f"  Peak NAV:              {curr}{m.peak_nav:>12,.2f}")
    print(f"  Total Net Profit:      {curr}{m.total_net_profit:>12,.2f} ({m.total_return_pct:>+6.2f}%)")
    print(f"  CAGR:                               {m.cagr_pct:>+6.2f}%")
    if m.benchmark_return_pct is not None:
        print(f"  Benchmark Return:                   {m.benchmark_return_pct:>+6.2f}%")
        print(f"  Alpha vs Benchmark:                 {m.alpha_pct:>+6.2f}%")

    print("\n🛡️ RISK & DRAWDOWN")
    print(sub_sep)
    print(f"  Annualized Volatility:              {m.annualized_volatility_pct:>6.2f}%")
    print(f"  Sharpe Ratio (Rf=5%):               {m.sharpe_ratio:>6.2f}")
    print(f"  Sortino Ratio:                      {m.sortino_ratio:>6.2f}")
    print(f"  Calmar Ratio:                       {m.calmar_ratio:>6.2f}")
    print(f"  Max Drawdown:                      -{m.max_drawdown_pct:>6.2f}% ({curr}{m.max_drawdown_amount:,.2f})")
    print(f"  Max Drawdown Duration:              {m.max_drawdown_duration_days} days")
    print(f"  Market Exposure:                    {m.market_exposure_pct:>6.2f}%")

    print("\n🎯 TRADE ANALYTICS")
    print(sub_sep)
    print(f"  Total Closed Trades:                {m.total_trades:>6}")
    print(f"  Win / Loss / BE:                    {m.winning_trades} / {m.losing_trades} / {m.breakeven_trades}")
    print(f"  Win Rate:                           {m.win_rate_pct:>6.2f}%")
    print(f"  Profit Factor:                      {m.profit_factor:>6.2f}")
    print(f"  Expectancy:            {curr}{m.expectancy:>12,.2f} per trade")
    print(f"  Average Trade P&L:     {curr}{m.avg_trade_pnl:>12,.2f}")
    print(f"  Average Win / Loss:    {curr}{m.avg_win_amount:>8,.2f} / {curr}{m.avg_loss_amount:>8,.2f} (Ratio: {m.win_loss_ratio:.2f})")
    print(f"  Largest Win / Loss:    {curr}{m.largest_win_amount:>8,.2f} / {curr}{m.largest_loss_amount:>8,.2f}")
    print(f"  Average Hold Duration:              {m.avg_hold_duration_days:>6.1f} days")

    print("\n💸 EXECUTION COSTS")
    print(sub_sep)
    print(f"  Total Brokerage & Fees Paid: {curr}{m.total_fees_paid:>8,.2f}")
    print(f"  Total Slippage Cost:         {curr}{m.total_slippage_cost:>8,.2f}")

    if result.trades:
        print("\n📋 RECENT TRADES (Sample)")
        print(sub_sep)
        print(f"  {'TICKER':<12} {'DIR':<4} {'QTY':<4} {'ENTRY':<8} {'EXIT':<8} {'NET PNL':<10} {'RET%':<7} {'REASON'}")
        for t in result.trades[-8:]:
            pnl_sign = "+" if t.net_pnl >= 0 else ""
            print(
                f"  {t.ticker:<12} {t.direction:<4} {t.quantity:<4} "
                f"{t.entry_fill_price:<8.2f} {t.exit_fill_price:<8.2f} "
                f"{pnl_sign + f'{curr}{t.net_pnl:.2f}':<10} {t.pnl_pct:>+6.2f}% {t.exit_reason}"
            )
    print(f"{sep}\n")


def generate_markdown_report(result: BacktestResult, output_path: Optional[Path] = None) -> str:
    """
    Generate a full institutional Markdown report for the backtest.
    """
    m = result.metrics
    cfg = result.config
    curr = "₹" if cfg.market == "india" else "$"

    lines = [
        f"# Aegis Backtest Report — {cfg.market.upper()} ({cfg.strategy.capitalize()})",
        "",
        f"**Date Range:** `{m.start_date}` to `{m.end_date}` ({m.trading_days_count} trading days)  ",
        f"**Strategy Mode:** `{cfg.strategy}` | **Leverage:** `{result.config.leverage or 1.0}x` | **Max Positions:** `{cfg.max_positions}`  ",
        f"**Generated At:** `{result.metrics.end_date}`  ",
        "",
        "---",
        "",
        "## Executive Summary",
        "",
        "| Metric | Value |",
        "|---|---|",
        f"| **Initial Capital** | {curr}{m.initial_capital:,.2f} |",
        f"| **Ending NAV** | **{curr}{m.ending_nav:,.2f}** |",
        f"| **Total Net Profit** | **{curr}{m.total_net_profit:,.2f} ({m.total_return_pct:+.2f}%)** |",
        f"| **CAGR** | {m.cagr_pct:+.2f}% |",
        f"| **Sharpe Ratio** | {m.sharpe_ratio:.2f} |",
        f"| **Sortino Ratio** | {m.sortino_ratio:.2f} |",
        f"| **Max Drawdown** | **-{m.max_drawdown_pct:.2f}%** ({curr}{m.max_drawdown_amount:,.2f}) |",
        f"| **Win Rate** | **{m.win_rate_pct:.1f}%** ({m.winning_trades}W / {m.losing_trades}L) |",
        f"| **Profit Factor** | {m.profit_factor:.2f} |",
        f"| **Expectancy** | {curr}{m.expectancy:.2f} / trade |",
        "",
        "---",
        "",
        "## Risk & Capital Statistics",
        "",
        "| Risk Metric | Value | Description |",
        "|---|---|---|",
        f"| **Annualized Volatility** | {m.annualized_volatility_pct:.2f}% | Standard deviation of daily returns annualized |",
        f"| **Calmar Ratio** | {m.calmar_ratio:.2f} | CAGR / Max Drawdown ratio |",
        f"| **Max Drawdown Duration** | {m.max_drawdown_duration_days} days | Consecutive trading days below high-water mark |",
        f"| **Market Exposure** | {m.market_exposure_pct:.1f}% | Percentage of time capital was actively deployed |",
        f"| **Brokerage & Fees** | {curr}{m.total_fees_paid:,.2f} | Total transaction costs (entry + exit) |",
        f"| **Slippage Impact** | {curr}{m.total_slippage_cost:,.2f} | Total adverse execution slippage cost |",
        "",
        "---",
        "",
        "## Trade Analytics",
        "",
        "| Trade Metric | Value |",
        "|---|---|",
        f"| **Total Closed Trades** | {m.total_trades} |",
        f"| **Average Trade Return** | {curr}{m.avg_trade_pnl:,.2f} |",
        f"| **Average Winning Trade** | +{curr}{m.avg_win_amount:,.2f} |",
        f"| **Average Losing Trade** | -{curr}{m.avg_loss_amount:,.2f} |",
        f"| **Win / Loss Ratio** | {m.win_loss_ratio:.2f} |",
        f"| **Largest Win** | +{curr}{m.largest_win_amount:,.2f} |",
        f"| **Largest Loss** | -{curr}{m.largest_loss_amount:,.2f} |",
        f"| **Average Holding Time** | {m.avg_hold_duration_days:.1f} days |",
        "",
        "---",
        "",
        "## Complete Trade Log",
        "",
        "| # | Ticker | Strategy | Dir | Qty | Entry Fill | Exit Fill | Gross PnL | Fees | Net PnL | Return % | Reason | Duration |",
        "|---|---|---|---|---|---|---|---|---|---|---|---|---|",
    ]

    for i, t in enumerate(result.trades, 1):
        pnl_sign = "+" if t.net_pnl >= 0 else ""
        lines.append(
            f"| {i} | `{t.ticker}` | {t.strategy} | {t.direction} | {t.quantity} | "
            f"{curr}{t.entry_fill_price:.2f} | {curr}{t.exit_fill_price:.2f} | "
            f"{curr}{t.gross_pnl:.2f} | {curr}{t.total_fees:.2f} | "
            f"**{pnl_sign}{curr}{t.net_pnl:.2f}** | {t.pnl_pct:+.2f}% | "
            f"`{t.exit_reason}` | {t.hold_duration_days:.1f}d |"
        )

    md_content = "\n".join(lines)

    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            f.write(md_content)

    return md_content


def export_trades_csv(result: BacktestResult, output_path: Path) -> None:
    """Export trade log to CSV."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "trade_id", "ticker", "market", "strategy", "direction", "quantity",
            "entry_fill_price", "exit_fill_price", "entry_fees", "exit_fees",
            "total_fees", "gross_pnl", "net_pnl", "pnl_pct", "entry_time",
            "exit_time", "hold_duration_days", "exit_reason",
        ])
        for t in result.trades:
            writer.writerow([
                t.trade_id, t.ticker, t.market, t.strategy, t.direction, t.quantity,
                t.entry_fill_price, t.exit_fill_price, t.entry_fees, t.exit_fees,
                t.total_fees, t.gross_pnl, t.net_pnl, t.pnl_pct, t.entry_time.isoformat(),
                t.exit_time.isoformat(), t.hold_duration_days, t.exit_reason,
            ])


def export_nav_csv(result: BacktestResult, output_path: Path) -> None:
    """Export daily NAV equity curve to CSV."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["date", "cash", "reserved_margin", "unrealized_pnl", "nav", "total_return_pct", "open_positions_count"])
        for item in result.daily_nav_history:
            writer.writerow([
                item["date"], item["cash"], item["reserved_margin"],
                item["unrealized_pnl"], item["nav"], item["total_return_pct"],
                item["open_positions_count"],
            ])
