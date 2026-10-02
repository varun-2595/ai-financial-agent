"""
Market Regime & Benchmark Index Gatekeeper.

Evaluates benchmark indices (NIFTY 50 for India, S&P 500 / SPY for US)
to ensure directional trades align with broader market momentum and trend structure.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Literal, Optional

from src.utils.logger import logger


@dataclass
class MarketRegime:
    market: Literal["india", "us"]
    benchmark_symbol: str
    current_price: float
    ema_20: float
    ema_50: float
    rsi_14: float
    is_above_ema20: bool
    is_above_ema50: bool
    regime: Literal["BULLISH_TREND", "BULLISH_PULLBACK", "BEARISH_DOWNTREND", "CHOPPY_SIDEWAYS"]
    allows_long: bool
    allows_short: bool
    summary: str


class MarketRegimeDetector:
    """Detects broad market regime from benchmark index trend and momentum."""

    BENCHMARK_MAPPING = {
        "india": {"primary": "^NSEI", "alt": "NIFTYBEES.NS", "name": "NIFTY 50"},
        "us": {"primary": "SPY", "alt": "^GSPC", "name": "S&P 500"},
    }

    def __init__(self):
        self._cache: dict[str, tuple[float, MarketRegime]] = {}  # market -> (timestamp, regime)
        self.cache_ttl_seconds = 300  # Cache regime evaluation for 5 minutes

    def _compute_ema(self, series: list[float], period: int) -> float:
        if not series or len(series) < period:
            return series[-1] if series else 0.0
        k = 2 / (period + 1)
        ema = series[0]
        for p in series[1:]:
            ema = p * k + ema * (1 - k)
        return ema

    def _compute_rsi(self, series: list[float], period: int = 14) -> float:
        if len(series) < period + 1:
            return 50.0
        deltas = [series[i] - series[i - 1] for i in range(1, len(series))]
        gains = [d for d in deltas[-period:] if d > 0]
        losses = [-d for d in deltas[-period:] if d < 0]
        avg_gain = sum(gains) / period if gains else 0.0001
        avg_loss = sum(losses) / period if losses else 0.0001
        rs = avg_gain / avg_loss
        return 100.0 - (100.0 / (1.0 + rs))

    def get_market_regime(self, market: Literal["india", "us"], force_refresh: bool = False) -> MarketRegime:
        """
        Calculates and returns the current market regime for India or US.
        Cached for 5 minutes to avoid redundant API queries.
        """
        now = datetime.now(timezone.utc).timestamp()
        if not force_refresh and market in self._cache:
            cached_time, cached_regime = self._cache[market]
            if now - cached_time < self.cache_ttl_seconds:
                return cached_regime

        bench_info = self.BENCHMARK_MAPPING.get(market, self.BENCHMARK_MAPPING["india"])
        symbol = bench_info["primary"]
        closes: list[float] = []

        try:
            import yfinance as yf
            ticker_obj = yf.Ticker(symbol)
            df = ticker_obj.history(period="1mo", interval="1d")
            if df.empty and bench_info.get("alt"):
                symbol = bench_info["alt"]
                ticker_obj = yf.Ticker(symbol)
                df = ticker_obj.history(period="1mo", interval="1d")
            if not df.empty and "Close" in df.columns:
                closes = [float(x) for x in df["Close"].dropna().tolist()]
        except Exception as e:
            logger.debug(f"[Regime] Live benchmark fetch error for {symbol}: {e}")

        # Fallback heuristic if live fetch failed
        if len(closes) < 20:
            regime = MarketRegime(
                market=market,
                benchmark_symbol=symbol,
                current_price=100.0,
                ema_20=100.0,
                ema_50=100.0,
                rsi_14=50.0,
                is_above_ema20=True,
                is_above_ema50=True,
                regime="BULLISH_TREND",
                allows_long=True,
                allows_short=False,
                summary=f"Neutral/Permissive fallback regime for {market.upper()}",
            )
            self._cache[market] = (now, regime)
            return regime

        curr_price = closes[-1]
        ema20 = self._compute_ema(closes, 20)
        ema50 = self._compute_ema(closes, min(len(closes), 50))
        rsi = self._compute_rsi(closes, 14)

        is_above_20 = curr_price >= ema20
        is_above_50 = curr_price >= ema50

        if is_above_20 and is_above_50:
            regime_type = "BULLISH_TREND"
            allows_long = True
            allows_short = False
            desc = f"Bullish Trend ({symbol} ₹{curr_price:,.2f} > EMA20 ₹{ema20:,.2f})" if market == "india" else f"Bullish Trend ({symbol} ${curr_price:,.2f} > EMA20 ${ema20:,.2f})"
        elif (not is_above_20) and is_above_50 and rsi > 45:
            regime_type = "BULLISH_PULLBACK"
            allows_long = True  # Pullback in uptrend allows selective high-conviction buys
            allows_short = False
            desc = f"Bullish Pullback ({symbol} resting above EMA50, RSI {rsi:.1f})"
        elif (not is_above_20) and (not is_above_50):
            regime_type = "BEARISH_DOWNTREND"
            allows_long = False  # Block LONG entries in downtrend
            allows_short = True
            desc = f"Bearish Downtrend ({symbol} below EMA20 & EMA50) - LONGs Blocked"
        else:
            regime_type = "CHOPPY_SIDEWAYS"
            allows_long = is_above_20
            allows_short = not is_above_20
            desc = f"Choppy Sideways Consolidation ({symbol} RSI {rsi:.1f})"

        regime_obj = MarketRegime(
            market=market,
            benchmark_symbol=symbol,
            current_price=round(curr_price, 2),
            ema_20=round(ema20, 2),
            ema_50=round(ema50, 2),
            rsi_14=round(rsi, 1),
            is_above_ema20=is_above_20,
            is_above_ema50=is_above_50,
            regime=regime_type,
            allows_long=allows_long,
            allows_short=allows_short,
            summary=desc,
        )
        self._cache[market] = (now, regime_obj)
        logger.info(f"[Market Regime] {market.upper()}: {regime_obj.summary}")
        return regime_obj

    def is_market_bullish(self, market: Literal["india", "us"]) -> bool:
        """Returns True if the broad market supports taking new LONG positions."""
        regime = self.get_market_regime(market)
        return regime.allows_long
