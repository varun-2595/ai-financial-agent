"""
Benchmark Data & Analytics Provider.

Handles loading, aligning, and calculating baseline statistics for:
  - India: NIFTY 50 (^NSEI)
  - US: S&P 500 (^GSPC / SPY)
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Literal, Optional

import numpy as np
import pandas as pd

from src.utils.logger import logger

CACHE_DIR = Path(__file__).parent.parent.parent / "data" / "backtest_cache"


@dataclass
class BenchmarkSummary:
    """Quantitative performance profile for a market benchmark index."""
    symbol: str
    name: str
    market: Literal["india", "us"]
    start_date: str
    end_date: str
    trading_days: int
    initial_price: float
    ending_price: float
    total_return_pct: float
    cagr_pct: float
    annualized_volatility_pct: float
    sharpe_ratio: float
    sortino_ratio: float
    max_drawdown_pct: float
    daily_returns: list[float]
    price_series: list[float]


class BenchmarkProvider:
    """
    Fetches and aligns benchmark index data to strategy dates.
    """

    BENCHMARK_SYMBOLS = {
        "india": {"symbol": "^NSEI", "name": "NIFTY 50", "alt": "NIFTYBEES.NS"},
        "us": {"symbol": "^GSPC", "name": "S&P 500", "alt": "SPY"},
    }

    def __init__(self, cache_dir: Optional[Path] = None, use_cache: bool = True):
        self.cache_dir = cache_dir or CACHE_DIR
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.use_cache = use_cache

    def _generate_synthetic_benchmark(
        self,
        market: Literal["india", "us"],
        dates: list[date],
    ) -> pd.DataFrame:
        """Deterministic synthetic benchmark series when offline."""
        base_price = 24000.0 if market == "india" else 5800.0
        drift = 0.04 / 252.0  # ~10% annual drift
        vol = 0.12 / math.sqrt(252.0)  # ~12% annualized vol

        # Seed based on market name
        np.random.seed(42 if market == "india" else 84)
        prices = [base_price]
        for _ in range(1, len(dates)):
            ret = np.random.normal(drift, vol)
            prices.append(prices[-1] * (1.0 + ret))

        utc_dates = [datetime.combine(d, datetime.min.time(), tzinfo=timezone.utc) for d in dates]
        return pd.DataFrame({"close": prices}, index=utc_dates)

    def get_benchmark_history(
        self,
        market: Literal["india", "us"],
        strategy_dates: list[date],
    ) -> pd.DataFrame:
        """
        Fetch benchmark history aligned exactly to the strategy's trading dates.
        """
        if not strategy_dates:
            return pd.DataFrame()

        info = self.BENCHMARK_SYMBOLS.get(market, self.BENCHMARK_SYMBOLS["india"])
        symbol = info["symbol"]
        start_str = strategy_dates[0].strftime("%Y-%m-%d")
        end_str = strategy_dates[-1].strftime("%Y-%m-%d")

        cache_file = self.cache_dir / f"bench_{symbol}_{start_str}_{end_str}.parquet"

        df: Optional[pd.DataFrame] = None
        if self.use_cache and cache_file.exists():
            try:
                df = pd.read_parquet(cache_file)
            except Exception as e:
                logger.warning(f"[Benchmark] Failed reading cache: {e}")

        if df is None or df.empty:
            try:
                import yfinance as yf
                buffered_start = (strategy_dates[0] - timedelta(days=10)).strftime("%Y-%m-%d")
                download_end = (strategy_dates[-1] + timedelta(days=2)).strftime("%Y-%m-%d")
                df = yf.download(symbol, start=buffered_start, end=download_end, auto_adjust=True, progress=False)

                if df is None or df.empty:
                    # Try alternate ETF symbol (e.g. SPY or NIFTYBEES)
                    alt_sym = info.get("alt", symbol)
                    df = yf.download(alt_sym, start=buffered_start, end=download_end, auto_adjust=True, progress=False)

                if df is not None and not df.empty:
                    if isinstance(df.columns, pd.MultiIndex):
                        df.columns = [col[0] for col in df.columns]
                    df = df.rename(columns={"Close": "close", "Open": "open", "High": "high", "Low": "low"})
                    if not isinstance(df.index, pd.DatetimeIndex):
                        df.index = pd.to_datetime(df.index)
                    if df.index.tz is None:
                        df.index = df.index.tz_localize(timezone.utc)
                    else:
                        df.index = df.index.tz_convert(timezone.utc)
                    df = df.dropna(subset=["close"])
                    if self.use_cache and not df.empty:
                        try:
                            df.to_parquet(cache_file)
                        except Exception:
                            pass
            except Exception as exc:
                logger.warning(f"[Benchmark] Could not fetch live benchmark {symbol} ({exc}), using synthetic.")
                df = self._generate_synthetic_benchmark(market, strategy_dates)

        if df is None or df.empty:
            df = self._generate_synthetic_benchmark(market, strategy_dates)

        # Align exactly with strategy_dates
        aligned_rows = []
        last_price = float(df["close"].iloc[0]) if not df.empty else 100.0
        for d in strategy_dates:
            sub = df[df.index.date == d]
            if not sub.empty:
                last_price = float(sub.iloc[-1]["close"])
            aligned_rows.append({"date": d, "close": last_price})

        aligned_df = pd.DataFrame(aligned_rows)
        return aligned_df

    def compute_summary(
        self,
        market: Literal["india", "us"],
        strategy_dates: list[date],
        risk_free_rate: float = 0.05,
    ) -> BenchmarkSummary:
        """
        Compute full statistical profile for the market benchmark.
        """
        info = self.BENCHMARK_SYMBOLS.get(market, self.BENCHMARK_SYMBOLS["india"])
        aligned_df = self.get_benchmark_history(market, strategy_dates)

        if aligned_df.empty or len(aligned_df) < 2:
            return BenchmarkSummary(
                symbol=info["symbol"],
                name=info["name"],
                market=market,
                start_date=strategy_dates[0].strftime("%Y-%m-%d") if strategy_dates else "",
                end_date=strategy_dates[-1].strftime("%Y-%m-%d") if strategy_dates else "",
                trading_days=len(strategy_dates),
                initial_price=100.0,
                ending_price=100.0,
                total_return_pct=0.0,
                cagr_pct=0.0,
                annualized_volatility_pct=0.0,
                sharpe_ratio=0.0,
                sortino_ratio=0.0,
                max_drawdown_pct=0.0,
                daily_returns=[],
                price_series=[],
            )

        prices = aligned_df["close"].to_numpy()
        returns = np.diff(prices) / prices[:-1]

        init_p = float(prices[0])
        end_p = float(prices[-1])
        tot_ret = ((end_p - init_p) / init_p) * 100.0

        days_count = len(strategy_dates)
        years = days_count / 252.0
        cagr = ((end_p / init_p) ** (1.0 / years) - 1.0) * 100.0 if (years > 0 and init_p > 0 and end_p > 0) else 0.0

        ann_factor = 252.0
        daily_rf = risk_free_rate / ann_factor
        vol = float(np.std(returns, ddof=1)) * math.sqrt(ann_factor) * 100.0 if len(returns) > 1 else 0.0

        mean_ret = float(np.mean(returns)) if len(returns) > 0 else 0.0
        daily_vol = float(np.std(returns, ddof=1)) if len(returns) > 1 else 0.0
        sharpe = (math.sqrt(ann_factor) * (mean_ret - daily_rf) / daily_vol) if daily_vol > 1e-8 else 0.0

        downside = returns[returns < daily_rf]
        downside_std = float(np.std(downside, ddof=1)) if len(downside) > 1 else 0.0
        sortino = (math.sqrt(ann_factor) * (mean_ret - daily_rf) / downside_std) if downside_std > 1e-8 else 0.0

        # Drawdowns
        cummax = np.maximum.accumulate(prices)
        drawdowns = (cummax - prices) / cummax * 100.0
        max_dd = float(np.max(drawdowns))

        return BenchmarkSummary(
            symbol=info["symbol"],
            name=info["name"],
            market=market,
            start_date=strategy_dates[0].strftime("%Y-%m-%d"),
            end_date=strategy_dates[-1].strftime("%Y-%m-%d"),
            trading_days=days_count,
            initial_price=round(init_p, 2),
            ending_price=round(end_p, 2),
            total_return_pct=round(tot_ret, 2),
            cagr_pct=round(cagr, 2),
            annualized_volatility_pct=round(vol, 2),
            sharpe_ratio=round(sharpe, 2),
            sortino_ratio=round(sortino, 2),
            max_drawdown_pct=round(max_dd, 2),
            daily_returns=returns.tolist(),
            price_series=prices.tolist(),
        )
