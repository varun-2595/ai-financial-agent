"""
Historical Data Provider for Backtesting.

Handles fetching, caching, and strict point-in-time slicing of historical price data.
Prevents look-ahead bias by ensuring the engine only sees bars up to the simulated current time.
"""
from __future__ import annotations

import json
from datetime import date, datetime, time as dtime, timedelta, timezone
from pathlib import Path
from typing import Literal, Optional

import pandas as pd

from src.data.models import Fundamentals, Quote, StockSnapshot
from src.utils.logger import logger

CACHE_DIR = Path(__file__).parent.parent.parent / "data" / "backtest_cache"


class BacktestDataProvider:
    """
    Manages historical market data with local caching and point-in-time access.
    """

    def __init__(self, cache_dir: Optional[Path] = None, use_cache: bool = True):
        self.cache_dir = cache_dir or CACHE_DIR
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.use_cache = use_cache
        self._memory_cache: dict[str, pd.DataFrame] = {}

    def _generate_fallback_history(
        self,
        ticker: str,
        start_date: str,
        end_date: str,
    ) -> pd.DataFrame:
        """Generate deterministic synthetic daily OHLCV series when offline or network unavailable."""
        import numpy as np
        # Seed deterministic series based on ticker string
        seed_val = sum(ord(c) for c in ticker) % 10000
        np.random.seed(seed_val)

        start_dt = datetime.strptime(start_date, "%Y-%m-%d").replace(tzinfo=timezone.utc) - timedelta(days=90)
        end_dt = datetime.strptime(end_date, "%Y-%m-%d").replace(tzinfo=timezone.utc)
        dates = pd.bdate_range(start=start_dt, end=end_dt, tz=timezone.utc)

        base_price = 100.0 if not ticker.endswith(".NS") else 1000.0
        if "RELIANCE" in ticker:
            base_price = 2500.0
        elif "TCS" in ticker:
            base_price = 3500.0
        elif "AAPL" in ticker:
            base_price = 200.0
        elif "NVDA" in ticker:
            base_price = 130.0

        price = base_price
        data = []
        for i, d in enumerate(dates):
            # Upward trending momentum with oscillations
            drift = 0.05 + 0.1 * np.sin(i / 7.0)
            open_p = price * (1.0 + np.random.normal(0, 0.005))
            high_p = open_p * (1.0 + abs(np.random.normal(0.008, 0.004)))
            low_p = open_p * (1.0 - abs(np.random.normal(0.008, 0.004)))
            close_p = open_p * (1.0 + np.random.normal(drift / 100.0, 0.01))
            vol = int(50_000 + abs(np.random.normal(100_000, 20_000)))
            price = close_p
            data.append({
                "open": round(open_p, 2),
                "high": round(max(open_p, high_p, close_p), 2),
                "low": round(min(open_p, low_p, close_p), 2),
                "close": round(close_p, 2),
                "volume": vol,
            })
        return pd.DataFrame(data, index=dates)

    def fetch_history(
        self,
        ticker: str,
        start_date: str,
        end_date: str,
        interval: str = "1d",
    ) -> pd.DataFrame:
        """
        Fetch historical OHLCV data for a ticker across a date range.
        Uses disk/memory cache to minimize network calls.
        """
        cache_key = f"{ticker}_{start_date}_{end_date}_{interval}.parquet"
        cache_file = self.cache_dir / cache_key

        if ticker in self._memory_cache:
            df = self._memory_cache[ticker]
            if not df.empty:
                return df

        if self.use_cache and cache_file.exists():
            try:
                df = pd.read_parquet(cache_file)
                self._memory_cache[ticker] = df
                return df
            except Exception as e:
                logger.warning(f"[DataProvider] Failed reading cache for {ticker}: {e}")

        logger.info(f"[DataProvider] Loading {ticker} ({start_date} to {end_date}, {interval})...")
        try:
            import yfinance as yf

            # Add small buffer to start_date for indicator warm-up (e.g., 60 days prior)
            start_dt = datetime.strptime(start_date, "%Y-%m-%d")
            buffered_start = (start_dt - timedelta(days=90)).strftime("%Y-%m-%d")

            df = yf.download(
                ticker,
                start=buffered_start,
                end=end_date,
                interval=interval,
                auto_adjust=True,
                progress=False,
            )

            if df.empty:
                logger.warning(f"[DataProvider] No data returned from yfinance for {ticker}, using fallback")
                df = self._generate_fallback_history(ticker, start_date, end_date)
            else:
                # Flatten MultiIndex columns if present (yfinance >= 0.2.x)
                if isinstance(df.columns, pd.MultiIndex):
                    df.columns = [col[0] for col in df.columns]

                # Standardize column names to lower case
                df = df.rename(columns={
                    "Open": "open",
                    "High": "high",
                    "Low": "low",
                    "Close": "close",
                    "Volume": "volume",
                })

                # Ensure index is datetime with UTC timezone
                if not isinstance(df.index, pd.DatetimeIndex):
                    df.index = pd.to_datetime(df.index)
                if df.index.tz is None:
                    df.index = df.index.tz_localize(timezone.utc)
                else:
                    df.index = df.index.tz_convert(timezone.utc)

                # Drop missing rows
                df = df.dropna(subset=["open", "high", "low", "close"])

        except Exception as exc:
            logger.warning(f"[DataProvider] yfinance download unavailable for {ticker} ({exc}), using fallback data.")
            df = self._generate_fallback_history(ticker, start_date, end_date)

        if self.use_cache and not df.empty:
            try:
                df.to_parquet(cache_file)
            except Exception as e:
                logger.warning(f"[DataProvider] Could not write cache for {ticker}: {e}")

        self._memory_cache[ticker] = df
        return df


    def get_point_in_time_snapshot(
        self,
        ticker: str,
        current_time: datetime,
        market: Literal["india", "us"],
        lookback_bars: int = 60,
    ) -> Optional[StockSnapshot]:
        """
        Build a StockSnapshot using ONLY data up to `current_time`.
        Strictly prevents look-ahead bias by filtering `df.index <= current_time`.
        """
        df = self._memory_cache.get(ticker)
        if df is None or df.empty:
            return None

        # Filter strictly up to current_time (Point-in-time)
        pit_df = df[df.index <= current_time]
        if pit_df.empty:
            return None

        # Slice recent lookback window for indicator computation
        history_window = pit_df.tail(lookback_bars)
        if len(history_window) < 10:
            return None

        quotes: list[Quote] = []
        for ts, row in history_window.iterrows():
            quotes.append(
                Quote(
                    timestamp=ts.to_pydatetime() if isinstance(ts, pd.Timestamp) else ts,
                    open=float(row["open"]),
                    high=float(row["high"]),
                    low=float(row["low"]),
                    close=float(row["close"]),
                    volume=int(row["volume"]) if not pd.isna(row.get("volume")) else 0,
                )
            )

        latest_bar = pit_df.iloc[-1]
        current_price = float(latest_bar["close"])
        currency = "INR" if market == "india" else "USD"

        # Baseline point-in-time fundamentals (neutral/mocked)
        fundamentals = Fundamentals(
            market_cap=None,
            pe_ratio=None,
            roe=None,
            debt_to_equity=None,
            profit_margin=None,
            sector="General",
            industry="General",
        )

        return StockSnapshot(
            ticker=ticker,
            market=market,
            currency=currency,
            current_price=current_price,
            history=quotes,
            fundamentals=fundamentals,
            timestamp=quotes[-1].timestamp,
        )

    def get_trading_days(self, start_date: str, end_date: str, market: Literal["india", "us"]) -> list[date]:
        """
        Extract unique sorted trading days present in the dataset across fetched symbols.
        """
        all_dates: set[date] = set()
        for df in self._memory_cache.values():
            if df.empty:
                continue
            # Sliced by start and end
            start_dt = datetime.strptime(start_date, "%Y-%m-%d").replace(tzinfo=timezone.utc)
            end_dt = datetime.strptime(end_date, "%Y-%m-%d").replace(tzinfo=timezone.utc) + timedelta(days=1)
            sub = df[(df.index >= start_dt) & (df.index < end_dt)]
            for ts in sub.index:
                all_dates.add(ts.date())

        if not all_dates:
            # Fallback to business days if no memory cache yet
            b_days = pd.bdate_range(start_date, end_date)
            return [d.date() for d in b_days]

        return sorted(list(all_dates))

    def preload_universe(
        self,
        tickers: list[str],
        start_date: str,
        end_date: str,
        interval: str = "1d",
    ) -> None:
        """Pre-fetch and cache all tickers for the backtest universe."""
        logger.info(f"[DataProvider] Preloading {len(tickers)} tickers...")
        for ticker in tickers:
            self.fetch_history(ticker, start_date, end_date, interval)
