"""
CLI Entry Point for Aegis Historical Backtesting & Walk-Forward Validation Engine.

Usage:
  # Single Backtest
  python -m src.backtesting --market india --start 2025-01-01 --end 2025-12-31
  python -m src.backtesting --market us --start 2025-01-01 --end 2025-12-31 --strategy swing
  python -m src.backtesting --market us --tickers AAPL,NVDA,MSFT,TSLA --start 2025-01-01 --end 2025-12-31 --capital 5000

  # Rolling Walk-Forward Validation
  python -m src.backtesting --walk-forward --market india --wf-start-year 2020 --wf-end-year 2026
  python -m src.backtesting --walk-forward --market us --strategy swing --wf-train-years 2 --wf-test-years 1
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
from src.evaluation.walk_forward import WalkForwardConfig, WalkForwardValidator
from src.utils.logger import logger


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Aegis Deterministic Historical Backtesting & Walk-Forward Validation Engine",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    # Mode selection
    parser.add_argument(
        "--walk-forward",
        action="store_true",
        help="Run Rolling Walk-Forward Out-of-Sample Validation across rolling windows",
    )
    parser.add_argument(
        "--wf-start-year",
        type=int,
        default=2020,
        help="Start year for walk-forward validation",
    )
    parser.add_argument(
        "--wf-end-year",
        type=int,
        default=2026,
        help="End year for walk-forward validation",
    )
    parser.add_argument(
        "--wf-train-years",
        type=int,
        default=2,
        help="In-sample training length in years per window",
    )
    parser.add_argument(
        "--wf-test-years",
        type=int,
        default=1,
        help="Out-of-sample test length in years per window",
    )
    parser.add_argument(
        "--wf-step-years",
        type=int,
        default=1,
        help="Step stride in years between consecutive rolling windows",
    )

    # Standard Backtest Arguments
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
        help="Directory to export CSV reports",
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

    # ── 1. Walk-Forward Mode ─────────────────────────────────────────────────
    if args.walk_forward:
        wf_config = WalkForwardConfig(
            market=args.market,
            start_year=args.wf_start_year,
            end_year=args.wf_end_year,
            train_years=args.wf_train_years,
            test_years=args.wf_test_years,
            step_years=args.wf_step_years,
            strategy=args.strategy,
            tickers=ticker_list,
            initial_capital=args.capital,
            leverage=args.leverage,
            max_positions=args.max_positions,
            use_cache=not args.no_cache,
        )

        validator = WalkForwardValidator()
        report = validator.run(wf_config)

        # Print markdown summary to stdout
        print("\n" + report.to_markdown() + "\n")

        # Save markdown report
        report_path = Path(args.report) if args.report else Path("reports") / f"walk_forward_{args.market}_{args.strategy}_{args.wf_start_year}_{args.wf_end_year}.md"
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(report.to_markdown(), encoding="utf-8")
        logger.info(f"[WalkForward] Robustness report saved to {report_path}")

        # Export CSV if requested
        if args.csv_dir:
            csv_path = Path(args.csv_dir) / f"walk_forward_{args.market}_{args.strategy}.csv"
            report.export_csv(csv_path)
            logger.info(f"[WalkForward] CSV metrics saved to {csv_path}")
        return

    # ── 2. Standard Single Period Backtest ──────────────────────────────────
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
