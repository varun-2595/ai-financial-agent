"""
US market data fetcher — yfinance wrapper for NYSE/NASDAQ stocks and ETFs.
Returns StockSnapshot objects ready for the analyst engine.

Fixes applied:
  - Thread-safe cache using threading.Lock
  - Retry logic via tenacity (3 attempts, exponential backoff)
  - Sleep skipped on cache hits
  - Explicit parentheses in price validation
  - is_etf detected from yfinance quoteType field
"""
from __future__ import annotations

import threading
import time
from datetime import datetime, timezone
from typing import Any

import yfinance as yf
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

from src.data.models import Fundamentals, NewsItem, Quote, StockSnapshot
from src.utils.logger import logger

_CACHE: dict[str, tuple[StockSnapshot, float]] = {}
_CACHE_LOCK = threading.Lock()
_CACHE_TTL_SECONDS = 300  # 5 minutes


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
    """Fetch info, history, and news from yfinance with retry."""
    t = yf.Ticker(ticker)
    info = t.info or {}
    hist = t.history(period="1y", interval="1d", auto_adjust=True)
    news = t.news or []
    return info, hist, news


def fetch_us_stock(ticker: str, use_cache: bool = True) -> StockSnapshot:
    """
    Fetch a complete StockSnapshot for a US-listed ticker (stocks or ETFs).

    Args:
        ticker: e.g. "AAPL", "SPY", "NVDA"
        use_cache: if True, returns cached data if < 5 minutes old

    Returns:
        StockSnapshot populated with price, fundamentals, history, news

    Raises:
        ValueError: if the ticker is invalid or data cannot be fetched
    """
    ticker = ticker.upper().strip()

    # Thread-safe cache check
    if use_cache:
        with _CACHE_LOCK:
            if ticker in _CACHE:
                snapshot, ts = _CACHE[ticker]
                if time.time() - ts < _CACHE_TTL_SECONDS:
                    logger.debug(f"[US] Cache hit for {ticker}")
                    return snapshot

    logger.info(f"[US] Fetching data for {ticker}")

    try:
        info, hist, news = _fetch_yf_data(ticker)

        current_price = (
            info.get("currentPrice")
            or info.get("regularMarketPrice")
            or info.get("previousClose")
        )

        # Explicit check: both info empty AND no price
        if (not info) or (current_price is None):
            raise ValueError(
                f"No price data found for US ticker '{ticker}'. Check the symbol."
            )

        history = _parse_history(hist)
        is_etf = str(info.get("quoteType", "")).upper() in ("ETF", "MUTUALFUND")

        snapshot = StockSnapshot(
            ticker=ticker,
            name=info.get("longName") or info.get("shortName") or ticker,
            market="us",
            sector=info.get("sector") or info.get("category"),
            currency="USD",
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
            data_source="yfinance",
        )

        with _CACHE_LOCK:
            _CACHE[ticker] = (snapshot, time.time())

        logger.success(f"[US] ✓ {snapshot.price_summary()}")
        return snapshot

    except Exception as exc:
        logger.error(f"[US] Failed to fetch {ticker}: {exc}")
        raise


def fetch_us_batch(tickers: list[str], delay_seconds: float = 0.5) -> dict[str, StockSnapshot]:
    """
    Fetch multiple US tickers sequentially with rate-limiting delay.
    Delay is skipped when data comes from cache.
    """
    results: dict[str, StockSnapshot] = {}
    for ticker in tickers:
        try:
            ticker = ticker.upper().strip()
            was_cached = ticker in _CACHE and (time.time() - _CACHE[ticker][1] < _CACHE_TTL_SECONDS)
            results[ticker] = fetch_us_stock(ticker)
            if not was_cached:
                time.sleep(delay_seconds)
        except Exception as exc:
            logger.warning(f"[US] Skipping {ticker}: {exc}")
    return results
