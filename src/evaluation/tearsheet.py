"""
Strategy vs Benchmark Performance Tearsheet & Comparative Report.

Formats and exports institutional-grade comparative evaluations clearly distinguishing:
  1. Gross Return vs Net Return
  2. Risk-Adjusted Performance
  3. Drawdown Analysis
  4. Transaction Costs & Slippage Impact
  5. Turnover & Capital Efficiency
  6. Trade Distribution Analytics
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

from src.evaluation.evaluator import ComprehensiveMetrics


def print_tearsheet(m: ComprehensiveMetrics) -> None:
    """Print an institutional terminal tearsheet comparing Strategy vs Benchmark."""
    curr = "₹" if m.market == "india" else "$"
    sep = "═" * 78
    sub_sep = "─" * 78

    print(f"\n{sep}")
    print(f"📊 AEGIS INSTITUTIONAL EVALUATION TEARSHEET — {m.market.upper()}")
    print(f"📅 Period: {m.start_date} to {m.end_date} ({m.trading_days} trading days / {m.calendar_days} calendar days)")
    print(f"🎯 Strategy vs Benchmark: {m.benchmark_name} ({m.benchmark_symbol})")
    print(f"{sep}")

    print("\n1️⃣  RETURNS BREAKDOWN: GROSS VS NET")
    print(sub_sep)
    print(f"  {'Metric':<32} {'Strategy (Gross)':<16} {'Strategy (Net)':<16} {'Benchmark'}")
    print(
        f"  {'Total Return':<32} "
        f"{m.gross_return_pct:>+14.2f}% "
        f"{m.net_return_pct:>+14.2f}% "
        f"{m.benchmark_return_pct:>+14.2f}%"
    )
    print(
        f"  {'CAGR (Annualized Return)':<32} "
        f"{m.gross_cagr_pct:>+14.2f}% "
        f"{m.net_cagr_pct:>+14.2f}% "
        f"{m.benchmark_cagr_pct:>+14.2f}%"
    )
    print(
        f"  {'Net Profit / (Loss)':<32} "
        f"{curr + f'{m.gross_profit:,.2f}':>15} "
        f"{curr + f'{m.net_profit:,.2f}':>15} "
        f"{'N/A':>15}"
    )
    print(f"  {'Excess Return (Alpha vs Bench)':<32} {'':>15} {m.excess_return_pct:>+14.2f}% {'':>15}")

    print("\n2️⃣  RISK-ADJUSTED PERFORMANCE")
    print(sub_sep)
    print(f"  {'Risk Metric':<32} {'Strategy':<16} {'Benchmark':<16} {'Spread / Delta'}")
    print(
        f"  {'Annualized Volatility':<32} "
        f"{m.annualized_volatility_pct:>14.2f}% "
        f"{m.benchmark_volatility_pct:>14.2f}% "
        f"{m.annualized_volatility_pct - m.benchmark_volatility_pct:>+14.2f}%"
    )
    print(
        f"  {'Sharpe Ratio (Rf=5%)':<32} "
        f"{m.sharpe_ratio:>15.2f} "
        f"{m.benchmark_sharpe_ratio:>15.2f} "
        f"{m.sharpe_ratio - m.benchmark_sharpe_ratio:>+15.2f}"
    )
    print(
        f"  {'Sortino Ratio (Downside)':<32} "
        f"{m.sortino_ratio:>15.2f} "
        f"{'N/A':>15} "
        f"{'':>15}"
    )
    print(f"  {'Calmar Ratio (CAGR/MaxDD)':<32} {m.calmar_ratio:>15.2f} {'N/A':>15}")
    print(f"  {'Market Beta (β)':<32} {m.beta:>15.2f} {1.0:>15.2f}")
    print(f"  {'Jensen Alpha (α Annualized)':<32} {m.alpha_pct:>+14.2f}% {'0.00%':>15}")
    print(f"  {'Information Ratio (IR)':<32} {m.information_ratio:>15.2f} {'0.00':>15}")
    print(f"  {'Treynor Ratio':<32} {m.treynor_ratio:>15.2f} {'N/A':>15}")

    print("\n3️⃣  DRAWDOWN & TAIL RISK")
    print(sub_sep)
    print(f"  {'Drawdown Metric':<32} {'Strategy':<16} {'Benchmark'}")
    print(
        f"  {'Max Drawdown (%)':<32} "
        f"-{m.max_drawdown_pct:>13.2f}% "
        f"-{m.benchmark_max_drawdown_pct:>13.2f}%"
    )
    print(f"  {'Max Drawdown Amount':<32} {curr + f'{m.max_drawdown_amount:,.2f}':>15}")
    print(f"  {'Max Drawdown Duration':<32} {f'{m.max_drawdown_duration_days} days':>15}")
    print(f"  {'Recovery Factor':<32} {m.recovery_factor:>15.2f}")

    print("\n4️⃣  TRANSACTION COSTS & SLIPPAGE IMPACT")
    print(sub_sep)
    print(f"  Total Brokerage & Regulatory Fees:   {curr}{m.total_brokerage_fees:>10,.2f}")
    print(f"  Total Adverse Slippage Impact:       {curr}{m.total_slippage_cost:>10,.2f}")
    print(f"  Total Execution Friction:            {curr}{m.total_transaction_friction:>10,.2f}")
    print(f"  Performance Cost Drag:                           {m.cost_drag_pct:>6.2f}%")
    print(f"  Friction Drag on Traded Volume:                 {m.friction_bps_on_volume:>6.1f} bps")

    print("\n5️⃣  PORTFOLIO TURNOVER & EXPOSURE")
    print(sub_sep)
    print(f"  Total Traded Notional Volume:        {curr}{m.total_traded_volume:>10,.2f}")
    print(f"  Annualized Portfolio Turnover:                  {m.annualized_turnover_pct:>6.2f}%")
    print(f"  Market Time Exposure:                           {m.market_exposure_pct:>6.2f}%")
    print(f"  Average Holding Duration:                       {m.avg_hold_duration_days:>6.1f} days")

    print("\n6️⃣  TRADE DISTRIBUTION & EXPECTANCY")
    print(sub_sep)
    print(f"  Total Closed Trades:                 {m.total_trades:>6} (Win: {m.win_rate_pct:.1f}%)")
    print(f"  Winning / Losing / Breakeven Trades: {m.winning_trades} / {m.losing_trades} / {m.breakeven_trades}")
    print(f"  Profit Factor:                       {m.profit_factor:>6.2f}")
    print(f"  Trade Expectancy:                    {curr}{m.expectancy:>8,.2f} per trade")
    print(f"  Average Win / Average Loss:          {curr}{m.avg_win_amount:>6,.2f} / {curr}{m.avg_loss_amount:>6,.2f} (Payoff: {m.win_loss_ratio:.2f})")
    print(f"  Largest Win / Largest Loss:          {curr}{m.largest_win:>6,.2f} / {curr}{m.largest_loss:>6,.2f}")
    print(f"  Max Win Streak / Loss Streak:        {m.max_consecutive_wins} wins / {m.max_consecutive_losses} losses")
    print(f"{sep}\n")


def generate_strategy_vs_benchmark_report(
    m: ComprehensiveMetrics,
    output_path: Optional[Path] = None,
) -> str:
    """
    Generate complete institutional Markdown report for Strategy vs Benchmark.
    """
    curr = "₹" if m.market == "india" else "$"

    lines = [
        f"# Aegis Performance Evaluation: Strategy vs. Benchmark ({m.market.upper()})",
        "",
        f"**Evaluation Period:** `{m.start_date}` to `{m.end_date}` ({m.trading_days} trading days)  ",
        f"**Market Benchmark:** `{m.benchmark_name}` (`{m.benchmark_symbol}`)  ",
        f"**Initial Capital:** `{curr}{m.initial_capital:,.2f}` | **Ending NAV:** `{curr}{m.ending_nav:,.2f}`  ",
        "",
        "---",
        "",
        "## 1. Executive Comparison: Strategy vs. Benchmark",
        "",
        "| Performance Metric | Strategy (Gross) | Strategy (Net) | Benchmark (" + m.benchmark_name + ") | Alpha / Spread |",
        "|---|---|---|---|---|",
        f"| **Total Return (%)** | **{m.gross_return_pct:+.2f}%** | **{m.net_return_pct:+.2f}%** | {m.benchmark_return_pct:+.2f}% | **{m.excess_return_pct:+.2f}%** |",
        f"| **CAGR (%)** | {m.gross_cagr_pct:+.2f}% | {m.net_cagr_pct:+.2f}% | {m.benchmark_cagr_pct:+.2f}% | {m.net_cagr_pct - m.benchmark_cagr_pct:+.2f}% |",
        f"| **Sharpe Ratio (Rf=5%)** | — | **{m.sharpe_ratio:.2f}** | {m.benchmark_sharpe_ratio:.2f} | {m.sharpe_ratio - m.benchmark_sharpe_ratio:+.2f} |",
        f"| **Sortino Ratio** | — | **{m.sortino_ratio:.2f}** | {m.benchmark_sharpe_ratio:.2f} | — |",
        f"| **Annualized Volatility** | — | {m.annualized_volatility_pct:.2f}% | {m.benchmark_volatility_pct:.2f}% | {m.annualized_volatility_pct - m.benchmark_volatility_pct:+.2f}% |",
        f"| **Maximum Drawdown** | — | **-{m.max_drawdown_pct:.2f}%** | -{m.benchmark_max_drawdown_pct:.2f}% | {m.benchmark_max_drawdown_pct - m.max_drawdown_pct:+.2f}% |",
        f"| **Calmar Ratio** | — | {m.calmar_ratio:.2f} | — | — |",
        "",
        "---",
        "",
        "## 2. Risk-Adjusted & Factor Analysis",
        "",
        "| Factor Metric | Value | Interpretation |",
        "|---|---|---|",
        f"| **Market Beta (β)** | `{m.beta:.2f}` | Sensitivity relative to {m.benchmark_name} (1.0 = neutral) |",
        f"| **Jensen's Alpha (α)** | `{m.alpha_pct:+.2f}%` | Annualized risk-adjusted excess return over CAPM expectation |",
        f"| **Information Ratio (IR)** | `{m.information_ratio:.2f}` | Consistency of active excess returns per unit of tracking risk |",
        f"| **Treynor Ratio** | `{m.treynor_ratio:.2f}` | Return earned in excess of risk-free rate per unit of systematic risk |",
        f"| **Downside Volatility** | `{m.downside_volatility_pct:.2f}%` | Standard deviation of negative excess returns |",
        "",
        "---",
        "",
        "## 3. Execution Costs & Slippage Drag",
        "",
        "| Friction Component | Amount | % of Capital / Volume |",
        "|---|---|---|",
        f"| **Gross Total Return** | {curr}{m.gross_profit:,.2f} | {m.gross_return_pct:+.2f}% |",
        f"| **Brokerage & Regulatory Fees** | -{curr}{m.total_brokerage_fees:,.2f} | -{(m.total_brokerage_fees/m.initial_capital)*100:.2f}% of capital |",
        f"| **Adverse Slippage Cost** | -{curr}{m.total_slippage_cost:,.2f} | -{(m.total_slippage_cost/m.initial_capital)*100:.2f}% of capital |",
        f"| **Total Friction Drag** | **-{curr}{m.total_transaction_friction:,.2f}** | **-{m.cost_drag_pct:.2f}% drag on return** ({m.friction_bps_on_volume:.1f} bps on volume) |",
        f"| **Net Realized Return** | **{curr}{m.net_profit:,.2f}** | **{m.net_return_pct:+.2f}%** |",
        "",
        "---",
        "",
        "## 4. Drawdown & Tail Risk Profile",
        "",
        "| Metric | Strategy | Benchmark |",
        "|---|---|---|",
        f"| **Maximum Drawdown (%)** | **-{m.max_drawdown_pct:.2f}%** | -{m.benchmark_max_drawdown_pct:.2f}% |",
        f"| **Maximum Drawdown ($ / ₹)** | {curr}{m.max_drawdown_amount:,.2f} | — |",
        f"| **Max Drawdown Duration** | {m.max_drawdown_duration_days} days | — |",
        f"| **Current Drawdown (%)** | {m.current_drawdown_pct:.2f}% | — |",
        f"| **Recovery Factor** | {m.recovery_factor:.2f} | — |",
        "",
        "---",
        "",
        "## 5. Portfolio Turnover & Capital Efficiency",
        "",
        "| Metric | Value | Description |",
        "|---|---|---|",
        f"| **Total Traded Notional Volume** | {curr}{m.total_traded_volume:,.2f} | Sum of entry and exit filled notionals |",
        f"| **Annualized Portfolio Turnover** | {m.annualized_turnover_pct:.2f}% | Volume / (2 × Avg NAV) normalized to 252 days |",
        f"| **Market Time Exposure** | {m.market_exposure_pct:.1f}% | Percentage of days with active open positions |",
        f"| **Average Position Holding Period** | {m.avg_hold_duration_days:.1f} days | Mean duration from fill to exit |",
        "",
        "---",
        "",
        "## 6. Trade Analytics & Statistical Distribution",
        "",
        "| Trade Analytics | Metric Value |",
        "|---|---|",
        f"| **Total Closed Trades** | `{m.total_trades}` |",
        f"| **Winning Trades (Count / %)** | `{m.winning_trades}` ({m.win_rate_pct:.1f}%) |",
        f"| **Losing Trades (Count / %)** | `{m.losing_trades}` ({(m.losing_trades/m.total_trades*100 if m.total_trades else 0):.1f}%) |",
        f"| **Profit Factor** | **`{m.profit_factor:.2f}`** |",
        f"| **Mathematical Expectancy** | **`{curr}{m.expectancy:.2f}` per trade** |",
        f"| **Average Trade Net P&L** | `{curr}{m.avg_trade_pnl:,.2f}` |",
        f"| **Average Winning Trade** | `+{curr}{m.avg_win_amount:,.2f}` |",
        f"| **Average Losing Trade** | `-{curr}{m.avg_loss_amount:,.2f}` |",
        f"| **Payoff Ratio (Avg Win / Avg Loss)** | `{m.win_loss_ratio:.2f}` |",
        f"| **Largest Winning Trade** | `+{curr}{m.largest_win:,.2f}` |",
        f"| **Largest Losing Trade** | `-{curr}{m.largest_loss:,.2f}` |",
        f"| **Max Consecutive Wins / Losses** | `{m.max_consecutive_wins}` wins / `{m.max_consecutive_losses}` losses |",
    ]

    md_content = "\n".join(lines)
    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            f.write(md_content)

    return md_content
