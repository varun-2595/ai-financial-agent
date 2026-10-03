"""
Crypto market data fetcher — Live public 24/7 market data for crypto pairs (BTC, ETH, SOL, etc.).
Uses public exchange REST endpoints (Binance & OKX) with zero authentication required.
Returns standard StockSnapshot objects identical in interface to India and US fetchers.
"""
from __future__ import annotations

import json
import re
import urllib.error
import urllib.request
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from src.data.coalescer import market_coalescer
from src.data.models import Fundamentals, Quote, StockSnapshot
from src.utils.logger import logger

# Core high-volume, liquid crypto universe
CORE_CRYPTO_WATCHLIST = [
    "BTC/USDT",
    "ETH/USDT",
    "SOL/USDT",
    "XRP/USDT",
    "DOGE/USDT",
    "AVAX/USDT",
    "NEAR/USDT",
    "SUI/USDT",
]


def normalize_crypto_ticker(ticker: str) -> str:
    """Normalize symbols like 'BTC', 'BTCUSDT', 'BTC-USDT' -> 'BTC/USDT'."""
    clean = ticker.upper().strip()
    if "/" in clean:
        return clean
    if "-" in clean:
        parts = clean.split("-")
        return f"{parts[0]}/{parts[1]}"
    if clean.endswith("USDT") and len(clean) > 4:
        base = clean[:-4]
        return f"{base}/USDT"
    return f"{clean}/USDT"


def to_exchange_symbol(ticker: str) -> str:
    """Normalize 'BTC/USDT' -> 'BTCUSDT' for exchange APIs."""
    norm = normalize_crypto_ticker(ticker)
    return norm.replace("/", "").replace("-", "")


import ssl

def _get_ssl_context() -> ssl.SSLContext:
    try:
        import certifi
        return ssl.create_default_context(cafile=certifi.where())
    except Exception:
        pass
    try:
        ctx = ssl.create_default_context()
        ctx.check_hostname = False
        ctx.verify_mode = ssl.CERT_NONE
        return ctx
    except Exception:
        return ssl._create_unverified_context()


def fetch_crypto_price(ticker: str) -> Optional[float]:
    """
    Ultra-fast current price fetcher for the 3-second Fast Tick Monitor.
    Returns float price or None.
    """
    symbol = to_exchange_symbol(ticker)
    url = f"https://api.binance.com/api/v3/ticker/price?symbol={symbol}"

    def _fetch() -> Optional[float]:
        try:
            req = urllib.request.Request(
                url,
                headers={"User-Agent": "AegisAgent/1.0 (CryptoFeed)"},
            )
            with urllib.request.urlopen(req, timeout=2.5, context=_get_ssl_context()) as resp:
                if resp.status == 200:
                    data = json.loads(resp.read().decode("utf-8"))
                    return float(data["price"])
        except Exception as e:
            logger.debug(f"[Crypto Fast Price] Binance fetch failed for {symbol}: {e}")
        return None

    return market_coalescer.coalesce(
        key=f"crypto:fast_price:{symbol}",
        fetch_fn=_fetch,
        ttl_seconds=1.5,
    )


def fetch_crypto_snapshot(ticker: str, limit: int = 100) -> Optional[StockSnapshot]:
    """
    Fetches full live OHLCV history and 24h ticker metrics for a crypto asset.
    Returns a standard StockSnapshot.
    """
    norm_ticker = normalize_crypto_ticker(ticker)
    symbol = to_exchange_symbol(norm_ticker)

    def _fetch_snapshot() -> Optional[StockSnapshot]:
        try:
            # 1. Fetch 24hr ticker stats (price, 24h change, volume)
            ticker_url = f"https://api.binance.com/api/v3/ticker/24hr?symbol={symbol}"
            t_req = urllib.request.Request(ticker_url, headers={"User-Agent": "AegisAgent/1.0"})
            with urllib.request.urlopen(t_req, timeout=3.5, context=_get_ssl_context()) as resp:
                if resp.status != 200:
                    return None
                t_data = json.loads(resp.read().decode("utf-8"))

            curr_price = float(t_data["lastPrice"])
            chg_24h = float(t_data["priceChangePercent"])
            vol_24h = float(t_data["quoteVolume"])  # Volume in USDT

            # 2. Fetch recent 15m candles
            kline_url = f"https://api.binance.com/api/v3/klines?symbol={symbol}&interval=15m&limit={limit}"
            k_req = urllib.request.Request(kline_url, headers={"User-Agent": "AegisAgent/1.0"})
            with urllib.request.urlopen(k_req, timeout=4.0, context=_get_ssl_context()) as k_resp:
                if k_resp.status != 200:
                    return None
                k_data = json.loads(k_resp.read().decode("utf-8"))

            quotes: List[Quote] = []
            for row in k_data:
                # Binance kline format: [open_time, open, high, low, close, volume, ...]
                ts = datetime.fromtimestamp(row[0] / 1000, tz=timezone.utc)
                o, h, l, c, v = float(row[1]), float(row[2]), float(row[3]), float(row[4]), int(float(row[5]))
                quotes.append(Quote(
                    timestamp=ts,
                    open=o,
                    high=h,
                    low=l,
                    close=c,
                    volume=v,
                ))

            # Base coin name
            base_coin = norm_ticker.split("/")[0]

            return StockSnapshot(
                ticker=norm_ticker,
                name=f"{base_coin} / Tether USD",
                market="crypto",
                sector="Crypto Assets",
                currency="USDT",
                is_etf=False,
                current_price=curr_price,
                price_change_pct_1d=chg_24h,
                history=quotes,
                fundamentals=Fundamentals(
                    market_cap=None,
                    avg_volume_30d=int(vol_24h),
                    beta=1.5,
                ),
                data_source="binance_public",
            )
        except Exception as err:
            logger.warning(f"[Crypto Fetcher] Failed to fetch live snapshot for {norm_ticker}: {err}")
            return None

    return market_coalescer.coalesce(
        key=f"crypto:snapshot:{symbol}",
        fetch_fn=_fetch_snapshot,
        ttl_seconds=3.0,
    )


def fetch_crypto_batch(tickers: Optional[List[str]] = None) -> Dict[str, StockSnapshot]:
    """
    Fetch a batch of crypto snapshots. Defaults to CORE_CRYPTO_WATCHLIST.
    """
    targets = tickers or CORE_CRYPTO_WATCHLIST
    results: Dict[str, StockSnapshot] = {}

    for t in targets:
        snap = fetch_crypto_snapshot(t)
        if snap:
            results[snap.ticker] = snap

    return results
