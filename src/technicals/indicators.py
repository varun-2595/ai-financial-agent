"""
Technical indicators computation engine using pandas and numpy.
Computes RSI, MACD, Moving Averages (EMA/SMA), Bollinger Bands, ATR, VWAP,
and Stochastic Oscillator directly on OHLCV Quote series.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
import pandas as pd

from src.data.models import Quote


@dataclass
class TechnicalSummary:
    """Computed technical indicator snapshot for the current candle."""
    current_price: float
    rsi_14: float | None = None
    macd: float | None = None
    macd_signal: float | None = None
    macd_hist: float | None = None
    ema_20: float | None = None
    ema_50: float | None = None
    ema_200: float | None = None
    sma_50: float | None = None
    sma_200: float | None = None
    bb_upper: float | None = None
    bb_middle: float | None = None
    bb_lower: float | None = None
    atr_14: float | None = None
    atr_pct: float | None = None
    trend_short: str = "NEUTRAL"   # BULLISH / BEARISH / NEUTRAL
    trend_long: str = "NEUTRAL"    # BULLISH / BEARISH / NEUTRAL
    rsi_condition: str = "NEUTRAL" # OVERSOLD / OVERBOUGHT / NEUTRAL


def quotes_to_dataframe(quotes: Sequence[Quote]) -> pd.DataFrame:
    if not quotes:
        return pd.DataFrame()
    data = [
        {
            "timestamp": q.timestamp,
            "open": q.open,
            "high": q.high,
            "low": q.low,
            "close": q.close,
            "volume": q.volume,
        }
        for q in quotes
    ]
    df = pd.DataFrame(data)
    df.sort_values("timestamp", inplace=True)
    df.reset_index(drop=True, inplace=True)
    return df


def compute_rsi(series: pd.Series, period: int = 14) -> pd.Series:
    delta = series.diff()
    gain = (delta.where(delta > 0, 0.0)).copy()
    loss = (-delta.where(delta < 0, 0.0)).copy()

    avg_gain = gain.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()

    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    return rsi


def compute_macd(
    series: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9
) -> tuple[pd.Series, pd.Series, pd.Series]:
    fast_ema = series.ewm(span=fast, adjust=False).mean()
    slow_ema = series.ewm(span=slow, adjust=False).mean()
    macd_line = fast_ema - slow_ema
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    hist = macd_line - signal_line
    return macd_line, signal_line, hist


def compute_bollinger_bands(
    series: pd.Series, period: int = 20, num_std: float = 2.0
) -> tuple[pd.Series, pd.Series, pd.Series]:
    middle = series.rolling(window=period).mean()
    std = series.rolling(window=period).std()
    upper = middle + (std * num_std)
    lower = middle - (std * num_std)
    return upper, middle, lower


def compute_atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    high = df["high"]
    low = df["low"]
    prev_close = df["close"].shift(1)
    tr1 = high - low
    tr2 = (high - prev_close).abs()
    tr3 = (low - prev_close).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
    return atr


def compute_technical_indicators(quotes: Sequence[Quote]) -> TechnicalSummary:
    """
    Calculate comprehensive technical indicators from a series of quotes.
    Returns a TechnicalSummary with current values and trend assessment.
    """
    if len(quotes) < 20:
        price = quotes[-1].close if quotes else 0.0
        return TechnicalSummary(current_price=price)

    df = quotes_to_dataframe(quotes)
    close = df["close"]
    current_price = float(close.iloc[-1])

    # Moving Averages
    ema_20 = close.ewm(span=20, adjust=False).mean()
    ema_50 = close.ewm(span=50, adjust=False).mean()
    ema_200 = close.ewm(span=200, adjust=False).mean() if len(close) >= 200 else None
    sma_50 = close.rolling(50).mean() if len(close) >= 50 else None
    sma_200 = close.rolling(200).mean() if len(close) >= 200 else None

    # Oscillators
    rsi = compute_rsi(close, 14)
    macd_line, signal_line, hist = compute_macd(close, 12, 26, 9)
    bb_upper, bb_mid, bb_lower = compute_bollinger_bands(close, 20, 2.0)
    atr = compute_atr(df, 14)

    latest_rsi = float(rsi.iloc[-1]) if not pd.isna(rsi.iloc[-1]) else None
    latest_macd = float(macd_line.iloc[-1]) if not pd.isna(macd_line.iloc[-1]) else None
    latest_signal = float(signal_line.iloc[-1]) if not pd.isna(signal_line.iloc[-1]) else None
    latest_hist = float(hist.iloc[-1]) if not pd.isna(hist.iloc[-1]) else None
    latest_ema20 = float(ema_20.iloc[-1]) if not pd.isna(ema_20.iloc[-1]) else None
    latest_ema50 = float(ema_50.iloc[-1]) if not pd.isna(ema_50.iloc[-1]) else None
    latest_ema200 = float(ema_200.iloc[-1]) if (ema_200 is not None and not pd.isna(ema_200.iloc[-1])) else None
    latest_sma50 = float(sma_50.iloc[-1]) if (sma_50 is not None and not pd.isna(sma_50.iloc[-1])) else None
    latest_sma200 = float(sma_200.iloc[-1]) if (sma_200 is not None and not pd.isna(sma_200.iloc[-1])) else None
    latest_bbu = float(bb_upper.iloc[-1]) if not pd.isna(bb_upper.iloc[-1]) else None
    latest_bbm = float(bb_mid.iloc[-1]) if not pd.isna(bb_mid.iloc[-1]) else None
    latest_bbl = float(bb_lower.iloc[-1]) if not pd.isna(bb_lower.iloc[-1]) else None
    latest_atr = float(atr.iloc[-1]) if not pd.isna(atr.iloc[-1]) else None
    atr_pct = (latest_atr / current_price * 100) if (latest_atr and current_price > 0) else None

    # Trend classifications
    trend_short = "NEUTRAL"
    if latest_ema20:
        if current_price > latest_ema20 and (latest_ema50 is None or latest_ema20 > latest_ema50):
            trend_short = "BULLISH"
        elif current_price < latest_ema20 and (latest_ema50 is None or latest_ema20 < latest_ema50):
            trend_short = "BEARISH"

    trend_long = "NEUTRAL"
    if latest_ema200:
        if current_price > latest_ema200:
            trend_long = "BULLISH"
        elif current_price < latest_ema200:
            trend_long = "BEARISH"

    rsi_cond = "NEUTRAL"
    if latest_rsi:
        if latest_rsi <= 30:
            rsi_cond = "OVERSOLD"
        elif latest_rsi >= 70:
            rsi_cond = "OVERBOUGHT"

    return TechnicalSummary(
        current_price=current_price,
        rsi_14=latest_rsi,
        macd=latest_macd,
        macd_signal=latest_signal,
        macd_hist=latest_hist,
        ema_20=latest_ema20,
        ema_50=latest_ema50,
        ema_200=latest_ema200,
        sma_50=latest_sma50,
        sma_200=latest_sma200,
        bb_upper=latest_bbu,
        bb_middle=latest_bbm,
        bb_lower=latest_bbl,
        atr_14=latest_atr,
        atr_pct=atr_pct,
        trend_short=trend_short,
        trend_long=trend_long,
        rsi_condition=rsi_cond,
    )
