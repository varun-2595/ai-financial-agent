"""
Core Backtesting Engine for Aegis AI Trading System.

Orchestrates point-in-time market simulation, screening, signal generation,
risk sizing, order execution, and portfolio accounting.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import date, datetime, time, timezone
from pathlib import Path
from typing import Any, Literal, Optional

import pandas as pd

from src.analyst.engine import AnalystEngine
from src.backtesting.data_provider import BacktestDataProvider
from src.backtesting.execution_simulator import (
    ExecutionSimulator,
    SimulatedPosition,
    SimulatedTradeRecord,
)
from src.backtesting.metrics import calculate_metrics, PerformanceMetrics
from src.data.models import StockSnapshot, Strategy, TradeSignal
from src.evaluation.benchmark import BenchmarkProvider, BenchmarkSummary
from src.evaluation.evaluator import ComprehensiveMetrics, PerformanceEvaluator
from src.risk.position_sizer import RiskEngine
from src.screener.universe import get_apple_ecosystem, get_rotated_universe
from src.signals.generator import SignalGenerator
from src.trading.fees import DEFAULT_FEE_SCHEDULE, FeeSchedule
from src.utils.config import get_config
from src.utils.logger import logger


@dataclass
class BacktestConfig:
    """Configuration parameters for a backtest run."""

    market: Literal["india", "us"] = "india"
    start_date: str = "2025-01-01"
    end_date: str = "2025-12-31"
    initial_capital: Optional[float] = None
    strategy: Strategy = "swing"
    tickers: Optional[list[str]] = None
    leverage: Optional[float] = None
    max_positions: int = 5
    risk_per_trade_pct: float = 0.03
    max_position_pct: float = 0.35
    fee_schedule: Optional[FeeSchedule] = None
    use_cache: bool = True
    benchmark_ticker: Optional[str] = None


@dataclass
class BacktestResult:
    """Complete result packet from a completed backtest run."""

    config: BacktestConfig
    metrics: PerformanceMetrics
    trades: list[SimulatedTradeRecord]
    daily_nav_history: list[dict[str, Any]]
    ledger: list[dict[str, Any]]
    positions_history: list[dict[str, Any]]
    evaluation: Optional[ComprehensiveMetrics] = None
    benchmark: Optional[BenchmarkSummary] = None



class BacktestEngine:
    """
    Deterministic historical backtesting engine with exact portfolio accounting.
    """

    def __init__(self, config: BacktestConfig, data_provider: Optional[BacktestDataProvider] = None):
        self.config = config
        self.app_cfg = get_config()
        self.data_provider = data_provider or BacktestDataProvider(use_cache=config.use_cache)
        self.fees = config.fee_schedule or DEFAULT_FEE_SCHEDULE
        self.execution = ExecutionSimulator(fee_schedule=self.fees)

        # Capital configuration
        if config.initial_capital is not None:
            self.initial_capital = config.initial_capital
        elif config.market == "india":
            self.initial_capital = self.app_cfg.paper_trading.virtual_capital_inr
        else:
            self.initial_capital = self.app_cfg.paper_trading.virtual_capital_usd

        # Leverage configuration
        if config.leverage is not None:
            self.leverage = config.leverage
        elif config.strategy in ("scalping", "intraday"):
            self.leverage = self.app_cfg.paper_trading.intraday_leverage_multiplier
        else:
            self.leverage = 1.0

        # Risk engine & signal generator (with offline heuristic analyst)
        self.risk_engine = RiskEngine()
        # Force offline heuristic analyst to prevent live Gemini calls during backtest
        offline_analyst = AnalystEngine(api_key=None, force_heuristic=True)
        self.signal_generator = SignalGenerator(analyst=offline_analyst, risk=self.risk_engine)

        # Portfolio state tracking
        self.cash: float = self.initial_capital
        self.reserved_margin: float = 0.0
        self.open_positions: dict[str, SimulatedPosition] = {}
        self.closed_trades: list[SimulatedTradeRecord] = []
        self.daily_nav_history: list[dict[str, Any]] = []
        self.ledger: list[dict[str, Any]] = []

    def _append_ledger(
        self,
        entry_type: str,
        amount: float,
        description: str,
        timestamp: datetime,
        ref_id: Optional[str] = None,
    ) -> None:
        self.ledger.append({
            "entry_type": entry_type,
            "amount": amount,
            "balance_after": round(self.cash, 2),
            "reserved_margin_after": round(self.reserved_margin, 2),
            "description": description,
            "ref_id": ref_id,
            "timestamp": timestamp.isoformat(),
        })

    def _resolve_tickers(self) -> list[str]:
        """Resolve universe of candidate tickers for the backtest."""
        if self.config.tickers and len(self.config.tickers) > 0:
            return self.config.tickers

        if self.config.market == "india":
            # Default representative NSE liquid universe
            return [
                "RELIANCE.NS", "TCS.NS", "HDFCBANK.NS", "INFY.NS", "ICICIBANK.NS",
                "BHARTIARTL.NS", "SBIN.NS", "LICI.NS", "ITC.NS", "LT.NS",
                "TATAMOTORS.NS", "SUNPHARMA.NS", "TITAN.NS", "BAJFINANCE.NS", "MARUTI.NS",
            ]
        else:
            # Default representative US liquid universe
            return [
                "AAPL", "MSFT", "NVDA", "AMZN", "GOOGL",
                "META", "TSLA", "AVGO", "AMD", "QCOM",
                "TSM", "TXN", "MU", "CRUS", "SWKS",
            ]

    def run(self) -> BacktestResult:
        """
        Execute full backtest over configured date range.
        """
        tickers = self._resolve_tickers()
        logger.info(
            f"[Backtest] Starting {self.config.market.upper()} backtest "
            f"({self.config.start_date} to {self.config.end_date}) "
            f"| Capital: {self.initial_capital:,.2f} | Strategy: {self.config.strategy} "
            f"| Tickers: {len(tickers)}"
        )

        # 1. Preload historical data
        self.data_provider.preload_universe(tickers, self.config.start_date, self.config.end_date)

        # Also load benchmark if configured
        benchmark_ticker = self.config.benchmark_ticker or ("^NSEI" if self.config.market == "india" else "^GSPC")
        self.data_provider.fetch_history(benchmark_ticker, self.config.start_date, self.config.end_date)

        # 2. Extract chronological trading days
        trading_days = self.data_provider.get_trading_days(
            self.config.start_date, self.config.end_date, self.config.market
        )

        if not trading_days:
            logger.error("[Backtest] No trading days found in specified date range.")
            empty_metrics = calculate_metrics([], [], self.initial_capital, self.config.market)
            return BacktestResult(
                config=self.config,
                metrics=empty_metrics,
                trades=[],
                daily_nav_history=[],
                ledger=[],
                positions_history=[],
            )

        # Record initial ledger entry
        init_ts = datetime.combine(trading_days[0], time(9, 0), tzinfo=timezone.utc)
        self._append_ledger("INITIAL_DEPOSIT", self.initial_capital, "Initial Capital Funding", init_ts)

        # 3. Main bar-by-bar progression loop
        for current_date in trading_days:
            current_time = datetime.combine(current_date, time(15, 30), tzinfo=timezone.utc)

            # ── A. Evaluate & update existing open positions ───────────────
            positions_to_remove: list[str] = []

            for ticker, pos in list(self.open_positions.items()):
                df = self.data_provider._memory_cache.get(ticker)
                if df is None or df.empty:
                    continue

                # Get the bar for current_date
                day_df = df[df.index.date == current_date]
                if day_df.empty:
                    continue

                bar = day_df.iloc[-1]
                bar_open = float(bar["open"])
                bar_high = float(bar["high"])
                bar_low = float(bar["low"])
                bar_close = float(bar["close"])

                # Check if position triggers exit during this bar
                is_eod = (self.config.strategy in ("scalping", "intraday"))
                exit_res = self.execution.check_and_execute_exit(
                    position=pos,
                    bar_open=bar_open,
                    bar_high=bar_high,
                    bar_low=bar_low,
                    bar_close=bar_close,
                    timestamp=current_time,
                    is_eod_square_off=is_eod,
                )

                if exit_res is not None:
                    trade_record, cash_returned = exit_res
                    self.closed_trades.append(trade_record)
                    positions_to_remove.append(ticker)

                    # Accounting updates: return cash, release margin
                    self.cash += cash_returned
                    self.reserved_margin -= pos.margin_blocked

                    # Ledger entries
                    self._append_ledger(
                        "MARGIN_RELEASE", pos.margin_blocked,
                        f"Close {pos.quantity}x {pos.ticker} ({trade_record.exit_reason})",
                        current_time, ref_id=trade_record.trade_id,
                    )
                    self._append_ledger(
                        "REALIZED_PNL", trade_record.net_pnl,
                        f"Net PnL {pos.ticker}: {trade_record.net_pnl:+.2f} ({trade_record.pnl_pct:+.2f}%)",
                        current_time, ref_id=trade_record.trade_id,
                    )

            for t in positions_to_remove:
                self.open_positions.pop(t, None)

            # ── B. Scan for new entries ────────────────────────────────────
            if len(self.open_positions) < self.config.max_positions:
                for ticker in tickers:
                    if ticker in self.open_positions:
                        continue
                    if len(self.open_positions) >= self.config.max_positions:
                        break

                    # Point-in-time snapshot up to current_time
                    snapshot = self.data_provider.get_point_in_time_snapshot(
                        ticker=ticker,
                        current_time=current_time,
                        market=self.config.market,
                    )
                    if snapshot is None:
                        continue

                    # Current portfolio NAV for position sizer
                    current_unrealized = sum(p.unrealized_pnl for p in self.open_positions.values())
                    current_portfolio_nav = self.cash + self.reserved_margin + current_unrealized

                    # Generate signal with offline rules
                    signal = self.signal_generator.generate_signal(
                        snapshot=snapshot,
                        strategy=self.config.strategy,
                        current_cash=self.cash,
                        portfolio_val=current_portfolio_nav,
                        skip_db_check=True,
                        force_heuristic=True,
                    )

                    if signal and signal.quantity > 0:
                        # Attempt simulated entry execution at bar close
                        pos, cash_deducted = self.execution.execute_entry(
                            signal=signal,
                            fill_price_raw=snapshot.current_price,
                            timestamp=current_time,
                            leverage=self.leverage,
                        )

                        if self.cash >= cash_deducted:
                            self.cash -= cash_deducted
                            self.reserved_margin += pos.margin_blocked
                            self.open_positions[ticker] = pos

                            # Ledger entries
                            self._append_ledger(
                                "MARGIN_BLOCK", -pos.margin_blocked,
                                f"BUY {pos.quantity}x {pos.ticker} @ {pos.avg_cost:.2f}",
                                current_time, ref_id=pos.position_id,
                            )
                            self._append_ledger(
                                "FEE", -pos.fees_paid,
                                f"Entry fees {pos.ticker}",
                                current_time, ref_id=pos.position_id,
                            )

            # ── C. Record daily portfolio NAV & mark-to-market snapshot ────
            unrealized_pnl = sum(p.unrealized_pnl for p in self.open_positions.values())
            nav = round(self.cash + self.reserved_margin + unrealized_pnl, 2)
            ret_pct = round(((nav - self.initial_capital) / self.initial_capital) * 100, 2)

            self.daily_nav_history.append({
                "date": current_date.strftime("%Y-%m-%d"),
                "cash": round(self.cash, 2),
                "reserved_margin": round(self.reserved_margin, 2),
                "unrealized_pnl": round(unrealized_pnl, 2),
                "nav": nav,
                "total_return_pct": ret_pct,
                "open_positions_count": len(self.open_positions),
                "open_tickers": list(self.open_positions.keys()),
            })

        # 4. Close any remaining open positions on the final date at last close
        if self.open_positions and trading_days:
            final_date = trading_days[-1]
            final_time = datetime.combine(final_date, time(16, 0), tzinfo=timezone.utc)
            for ticker, pos in list(self.open_positions.items()):
                exit_price = pos.current_price
                exit_res = self.execution.check_and_execute_exit(
                    position=pos,
                    bar_open=exit_price,
                    bar_high=exit_price,
                    bar_low=exit_price,
                    bar_close=exit_price,
                    timestamp=final_time,
                    is_eod_square_off=True,
                )
                if exit_res:
                    trade_record, cash_returned = exit_res
                    trade_record.exit_reason = "BACKTEST_END"
                    self.closed_trades.append(trade_record)
                    self.cash += cash_returned
                    self.reserved_margin -= pos.margin_blocked

            self.open_positions.clear()

        # 5. Extract benchmark summary & performance
        bench_provider = BenchmarkProvider(cache_dir=self.data_provider.cache_dir, use_cache=self.config.use_cache)
        benchmark_summary = bench_provider.compute_summary(
            market=self.config.market,
            strategy_dates=trading_days,
            risk_free_rate=0.05,
        )

        # 6. Compute legacy & comprehensive evaluation metrics
        metrics = calculate_metrics(
            daily_nav_history=self.daily_nav_history,
            trade_records=self.closed_trades,
            initial_capital=self.initial_capital,
            market=self.config.market,
            benchmark_nav_series=benchmark_summary.price_series if benchmark_summary.price_series else None,
        )

        evaluator = PerformanceEvaluator(risk_free_rate=0.05)
        comprehensive_eval = evaluator.evaluate(
            daily_nav_history=self.daily_nav_history,
            trade_records=self.closed_trades,
            initial_capital=self.initial_capital,
            benchmark=benchmark_summary,
            market=self.config.market,
        )

        logger.success(
            f"[Backtest] Finished {self.config.market.upper()} backtest | "
            f"Ending NAV: {metrics.ending_nav:,.2f} ({metrics.total_return_pct:+.2f}%) | "
            f"Sharpe: {metrics.sharpe_ratio:.2f} | MaxDD: -{metrics.max_drawdown_pct:.2f}% | "
            f"Trades: {metrics.total_trades} (Win: {metrics.win_rate_pct:.1f}%)"
        )

        return BacktestResult(
            config=self.config,
            metrics=metrics,
            trades=self.closed_trades,
            daily_nav_history=self.daily_nav_history,
            ledger=self.ledger,
            positions_history=[],
            evaluation=comprehensive_eval,
            benchmark=benchmark_summary,
        )

