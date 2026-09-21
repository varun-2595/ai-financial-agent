"""
India market data fetcher — NSE/BSE stocks and ETFs via yfinance.
Uses .NS suffix for NSE and .BO suffix for BSE.
Returns StockSnapshot objects identical in shape to the US fetcher.

Fixes applied:
  - Thread-safe cache with Lock
  - Retry logic via tenacity (3 attempts, exponential backoff)
  - BSE auto-fallback if NSE ticker fails
  - is_etf detected from yfinance quoteType field
  - Skip rate limit sleep on cache hits
"""
from __future__ import annotations

import threading
import time
from datetime import datetime, timezone
from typing import Any

import yfinance as yf
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

from src.data.models import (
    Fundamentals,
    NewsItem,
    Quote,
    StockSnapshot,
)
from src.utils.logger import logger

_CACHE: dict[str, tuple[StockSnapshot, float]] = {}
_CACHE_LOCK = threading.Lock()
_CACHE_TTL_SECONDS = 300  # 5 minutes


def normalize_india_ticker(ticker: str) -> str:
    """
    Ensure the ticker has a valid India suffix.
    - "RELIANCE"    → "RELIANCE.NS"  (default NSE)
    - "RELIANCE.NS" → "RELIANCE.NS"  (unchanged)
    - "RELIANCE.BO" → "RELIANCE.BO"  (BSE kept)
    """
    ticker = ticker.upper().strip()
    if ticker.endswith(".NS") or ticker.endswith(".BO"):
        return ticker
    return f"{ticker}.NS"


def _parse_fundamentals(info: dict[str, Any]) -> Fundamentals:
    return Fundamentals(
        market_cap=info.get("marketCap"),
        pe_ratio=info.get("trailingPE"),
        pb_ratio=info.get("priceToBook"),
        eps=info.get("trailingEps"),
        revenue_ttm=info.get("totalRevenue"),
        net_income_ttm=info.get("netIncomeToCommon"),
        ebitda=info.get("ebitda"),
        debt_to_equity=info.get("debtToEquity"),
        free_cash_flow=info.get("freeCashflow"),
        dividend_yield=info.get("dividendYield"),
        beta=info.get("beta"),
        week_52_high=info.get("fiftyTwoWeekHigh"),
        week_52_low=info.get("fiftyTwoWeekLow"),
        avg_volume_30d=info.get("averageVolume"),
        shares_outstanding=info.get("sharesOutstanding"),
        return_on_equity=info.get("returnOnEquity"),
        profit_margin=info.get("profitMargins"),
    )


def _parse_news(raw_news: list[dict]) -> list[NewsItem]:
    items = []
    for article in (raw_news or [])[:10]:
        try:
            published = article.get("providerPublishTime")
            items.append(NewsItem(
                title=article.get("title", ""),
                publisher=article.get("publisher"),
                published_at=datetime.fromtimestamp(published, tz=timezone.utc) if published else None,
                url=article.get("link"),
            ))
        except Exception as exc:
            logger.debug(f"Skipping malformed news item: {exc}")
    return items


def _parse_history(hist) -> list[Quote]:
    quotes = []
    for ts, row in hist.iterrows():
        try:
            quotes.append(Quote(
                timestamp=ts.to_pydatetime(),
                open=float(row["Open"]),
                high=float(row["High"]),
                low=float(row["Low"]),
                close=float(row["Close"]),
                volume=int(row["Volume"]),
            ))
        except Exception as exc:
            logger.debug(f"Skipping malformed OHLCV row: {exc}")
    return quotes


def _pct_change(history: list[Quote], days: int) -> float | None:
    if len(history) < days + 1:
        return None
    current = history[-1].close
    past = history[-(days + 1)].close
    if past == 0:
        return None
    return ((current - past) / past) * 100


@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=2, max=10),
    retry=retry_if_exception_type(Exception),
    reraise=True,
)
def _fetch_yf_data(ticker: str) -> tuple[dict, Any, list]:
    t = yf.Ticker(ticker)
    info = t.info or {}
    hist = t.history(period="1y", interval="1d", auto_adjust=True)
    news = t.news or []
    return info, hist, news


def fetch_india_stock(ticker: str, use_cache: bool = True) -> StockSnapshot:
    """
    Fetch a complete StockSnapshot for an NSE/BSE-listed ticker.
    Automatically tries BSE (.BO) if NSE (.NS) yields no data.
    """
    ticker = normalize_india_ticker(ticker)

    # Thread-safe cache check
    if use_cache:
        with _CACHE_LOCK:
            if ticker in _CACHE:
                snapshot, ts = _CACHE[ticker]
                if time.time() - ts < _CACHE_TTL_SECONDS:
                    logger.debug(f"[IN] Cache hit for {ticker}")
                    return snapshot

    logger.info(f"[IN] Fetching data for {ticker}")

    def _attempt_fetch(sym: str):
        info, hist, news = _fetch_yf_data(sym)
        current_price = (
            info.get("currentPrice")
            or info.get("regularMarketPrice")
            or info.get("previousClose")
        )
        if (not info) or (current_price is None):
            raise ValueError(f"No price data for ticker '{sym}'")
        return info, hist, news, current_price

    try:
        active_ticker = ticker
        try:
            info, hist, news, current_price = _attempt_fetch(active_ticker)
        except Exception as primary_err:
            if active_ticker.endswith(".NS"):
                bse_ticker = active_ticker[:-3] + ".BO"
                logger.warning(f"[IN] Failed {active_ticker}, trying BSE fallback {bse_ticker}: {primary_err}")
                info, hist, news, current_price = _attempt_fetch(bse_ticker)
                active_ticker = bse_ticker
            else:
                raise primary_err

        history = _parse_history(hist)
        exchange = "NSE" if active_ticker.endswith(".NS") else "BSE"
        is_etf = str(info.get("quoteType", "")).upper() in ("ETF", "MUTUALFUND")

        snapshot = StockSnapshot(
            ticker=active_ticker,
            name=info.get("longName") or info.get("shortName") or active_ticker,
            market="india",
            sector=info.get("sector") or info.get("category"),
            currency="INR",
            is_etf=is_etf,
            current_price=float(current_price),
            price_change_pct_1d=_pct_change(history, 1),
            price_change_pct_1w=_pct_change(history, 5),
            price_change_pct_1m=_pct_change(history, 21),
            price_change_pct_3m=_pct_change(history, 63),
            price_change_pct_1y=_pct_change(history, 252),
            history=history,
            fundamentals=_parse_fundamentals(info),
            recent_news=_parse_news(news),
            data_source=f"yfinance/{exchange}",
        )

        with _CACHE_LOCK:
            _CACHE[ticker] = (snapshot, time.time())
            if active_ticker != ticker:
                _CACHE[active_ticker] = (snapshot, time.time())

        logger.success(f"[IN] ✓ {snapshot.price_summary()}")
        return snapshot

    except Exception as exc:
        logger.error(f"[IN] Failed to fetch {ticker}: {exc}")
        raise


def fetch_india_batch(
    tickers: list[str], delay_seconds: float = 0.5
) -> dict[str, StockSnapshot]:
    results: dict[str, StockSnapshot] = {}
    for ticker in tickers:
        try:
            normalized = normalize_india_ticker(ticker)
            was_cached = normalized in _CACHE and (time.time() - _CACHE[normalized][1] < _CACHE_TTL_SECONDS)
            results[normalized] = fetch_india_stock(normalized)
            if not was_cached:
                time.sleep(delay_seconds)
        except Exception as exc:
            logger.warning(f"[IN] Skipping {ticker}: {exc}")
    return results
