"""
CLI Entry Point for Aegis Historical Backtesting Engine.

Usage:
  python -m src.backtesting --market india --start 2025-01-01 --end 2025-12-31
  python -m src.backtesting --market us --start 2025-01-01 --end 2025-12-31 --strategy swing
  python -m src.backtesting --market us --tickers AAPL,NVDA,MSFT,TSLA --start 2025-01-01 --end 2025-12-31 --capital 5000
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from src.backtesting.engine import BacktestConfig, BacktestEngine
from src.backtesting.reporting import (
    export_nav_csv,
    export_trades_csv,
    generate_markdown_report,
    print_backtest_summary,
)
from src.evaluation.tearsheet import (
    generate_strategy_vs_benchmark_report,
    print_tearsheet,
)
from src.utils.logger import logger


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Aegis Deterministic Historical Backtesting Engine",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--market",
        type=str,
        choices=["india", "us"],
        default="india",
        help="Market to backtest ('india' for NSE, 'us' for NYSE/NASDAQ)",
    )
    parser.add_argument(
        "--start",
        type=str,
        default="2025-01-01",
        help="Backtest start date (YYYY-MM-DD)",
    )
    parser.add_argument(
        "--end",
        type=str,
        default="2025-12-31",
        help="Backtest end date (YYYY-MM-DD)",
    )
    parser.add_argument(
        "--strategy",
        type=str,
        choices=["scalping", "intraday", "swing", "positional"],
        default="swing",
        help="Trading strategy to backtest",
    )
    parser.add_argument(
        "--capital",
        type=float,
        default=None,
        help="Initial capital (defaults to configured paper trading capital)",
    )
    parser.add_argument(
        "--tickers",
        type=str,
        default=None,
        help="Comma-separated list of tickers (e.g. 'RELIANCE.NS,TCS.NS' or 'AAPL,MSFT,NVDA')",
    )
    parser.add_argument(
        "--leverage",
        type=float,
        default=None,
        help="Leverage multiplier (default: 3x for intraday/scalp, 1x for swing/positional)",
    )
    parser.add_argument(
        "--max-positions",
        type=int,
        default=5,
        help="Maximum simultaneous open positions",
    )
    parser.add_argument(
        "--report",
        type=str,
        default=None,
        help="Path to save generated Markdown report (e.g. 'reports/backtest.md')",
    )
    parser.add_argument(
        "--csv-dir",
        type=str,
        default=None,
        help="Directory to export trades.csv and nav.csv",
    )
    parser.add_argument(
        "--no-cache",
        action="store_true",
        help="Disable disk caching of historical data",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    ticker_list = [t.strip() for t in args.tickers.split(",") if t.strip()] if args.tickers else None

    config = BacktestConfig(
        market=args.market,
        start_date=args.start,
        end_date=args.end,
        strategy=args.strategy,
        initial_capital=args.capital,
        tickers=ticker_list,
        leverage=args.leverage,
        max_positions=args.max_positions,
        use_cache=not args.no_cache,
    )

    engine = BacktestEngine(config=config)
    result = engine.run()

    # Print summary and tearsheet to stdout
    if result.evaluation is not None:
        print_tearsheet(result.evaluation)
    else:
        print_backtest_summary(result)

    # Save markdown report if requested or default to reports/
    report_path = Path(args.report) if args.report else Path("reports") / f"backtest_{args.market}_{args.strategy}_{args.start}_{args.end}.md"
    if result.evaluation is not None:
        generate_strategy_vs_benchmark_report(result.evaluation, output_path=report_path)
    else:
        generate_markdown_report(result, output_path=report_path)
    logger.info(f"[Backtest] Markdown report saved to {report_path}")

    # Export CSVs if directory provided
    if args.csv_dir:
        csv_dir = Path(args.csv_dir)
        export_trades_csv(result, csv_dir / "trades.csv")
        export_nav_csv(result, csv_dir / "nav.csv")
        logger.info(f"[Backtest] CSV exports saved to {csv_dir}")


if __name__ == "__main__":
    main()

