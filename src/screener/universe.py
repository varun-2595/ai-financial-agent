"""
Universe fetcher — pulls the full investable universe for India (NSE 500)
and US (S&P 500).

Fixes applied:
  - Cache universe in-memory with 1-day TTL
  - Corrected ticker typos (BAJAJ-AUTO.NS, removed defunct tickers)
  - Cleaned up unused imports
"""
from __future__ import annotations

import io
import time

import pandas as pd
import requests

from src.utils.logger import logger

_NSE500_CSV_URL = (
    "https://www.niftyindices.com/IndexConstituent/ind_nifty500list.csv"
)

_NSE_FALLBACK = [
    "RELIANCE.NS", "TCS.NS", "INFY.NS", "HDFCBANK.NS", "ICICIBANK.NS",
    "HINDUNILVR.NS", "SBIN.NS", "BAJFINANCE.NS", "BHARTIARTL.NS", "KOTAKBANK.NS",
    "WIPRO.NS", "LT.NS", "AXISBANK.NS", "ASIANPAINT.NS", "MARUTI.NS",
    "SUNPHARMA.NS", "TITAN.NS", "ULTRACEMCO.NS", "NESTLEIND.NS", "POWERGRID.NS",
    "NTPC.NS", "ONGC.NS", "TATAMOTORS.NS", "ADANIENT.NS", "JSWSTEEL.NS",
    "TATASTEEL.NS", "TECHM.NS", "HCLTECH.NS", "DIVISLAB.NS", "DRREDDY.NS",
    "CIPLA.NS", "BAJAJ-AUTO.NS", "EICHERMOT.NS", "BRITANNIA.NS", "HAVELLS.NS",
    "PIDILITIND.NS", "BERGEPAINT.NS", "MUTHOOTFIN.NS", "BALKRISIND.NS",
    "PERSISTENT.NS", "LTIM.NS", "POLYCAB.NS", "DIXON.NS", "TRENT.NS",
    "ZOMATO.NS", "NYKAA.NS", "INDHOTEL.NS", "IRCTC.NS",
    # ETFs
    "NIFTYBEES.NS", "BANKBEES.NS", "GOLDBEES.NS", "ITBEES.NS", "JUNIORBEES.NS",
]

_SP500_WIKI_URL = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"

_SP500_FALLBACK = [
    "AAPL", "MSFT", "NVDA", "GOOGL", "AMZN", "META", "TSLA", "BRK-B", "JPM",
    "V", "UNH", "XOM", "JNJ", "MA", "PG", "HD", "AVGO", "CVX", "MRK", "ABBV",
    "COST", "PEP", "KO", "TMO", "ADBE", "ACN", "CRM", "MCD", "NKE", "LIN",
    "DHR", "ORCL", "CSCO", "INTC", "AMD", "QCOM", "TXN", "AMGN", "CAT",
    "GS", "BA", "HON", "SYK", "SPGI", "BLK", "AXP", "DE", "RTX", "NOW",
    "ISRG", "REGN", "VRTX", "CI", "CB", "ZTS", "ADI", "MCHP", "KLAC", "AMAT",
    "LRCX", "SNPS", "CDNS", "PANW", "CRWD", "FTNT", "NET", "DDOG", "SNOW",
    # ETFs
    "SPY", "QQQ", "VTI", "GLD", "XLK", "XLF", "XLE", "XLV", "XLP", "XLI",
]

_UNIVERSE_CACHE: dict[str, tuple[list[str], float]] = {}
_UNIVERSE_TTL = 86400  # 24 hours


def fetch_nse500(use_cache: bool = True) -> list[str]:
    now = time.time()
    if use_cache and "nse500" in _UNIVERSE_CACHE:
        tickers, ts = _UNIVERSE_CACHE["nse500"]
        if now - ts < _UNIVERSE_TTL:
            return tickers

    logger.info("[Universe] Fetching NSE 500 constituent list...")
    try:
        headers = {
            "User-Agent": (
                "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0 Safari/537.36"
            ),
            "Referer": "https://www.niftyindices.com/",
        }
        resp = requests.get(_NSE500_CSV_URL, headers=headers, timeout=15)
        resp.raise_for_status()

        df = pd.read_csv(io.StringIO(resp.text))
        symbol_col = next(
            (c for c in df.columns if "symbol" in c.lower()), None
        )
        if symbol_col is None:
            raise ValueError(f"No 'Symbol' column found. Columns: {list(df.columns)}")

        tickers = [f"{s.strip()}.NS" for s in df[symbol_col].dropna().unique()]
        logger.success(f"[Universe] NSE 500: fetched {len(tickers)} tickers")
        _UNIVERSE_CACHE["nse500"] = (tickers, now)
        return tickers

    except Exception as exc:
        logger.warning(
            f"[Universe] NSE 500 fetch failed ({exc}). "
            f"Using fallback list of {len(_NSE_FALLBACK)} tickers."
        )
        _UNIVERSE_CACHE["nse500"] = (_NSE_FALLBACK, now)
        return _NSE_FALLBACK


def fetch_sp500(use_cache: bool = True) -> list[str]:
    now = time.time()
    if use_cache and "sp500" in _UNIVERSE_CACHE:
        tickers, ts = _UNIVERSE_CACHE["sp500"]
        if now - ts < _UNIVERSE_TTL:
            return tickers

    logger.info("[Universe] Fetching S&P 500 constituent list...")
    try:
        tables = pd.read_html(_SP500_WIKI_URL, attrs={"id": "constituents"})
        df = tables[0]

        symbol_col = next(
            (c for c in df.columns if "symbol" in c.lower() or "ticker" in c.lower()),
            None,
        )
        if symbol_col is None:
            raise ValueError(f"No symbol column found. Columns: {list(df.columns)}")

        tickers = [
            s.strip().replace(".", "-")
            for s in df[symbol_col].dropna().unique()
        ]
        logger.success(f"[Universe] S&P 500: fetched {len(tickers)} tickers")
        _UNIVERSE_CACHE["sp500"] = (tickers, now)
        return tickers

    except Exception as exc:
        logger.warning(
            f"[Universe] S&P 500 fetch failed ({exc}). "
            f"Using fallback list of {len(_SP500_FALLBACK)} tickers."
        )
        _UNIVERSE_CACHE["sp500"] = (_SP500_FALLBACK, now)
        return _SP500_FALLBACK


def fetch_full_universe() -> dict[str, list[str]]:
    india = fetch_nse500()
    us = fetch_sp500()
    return {"india": india, "us": us}
