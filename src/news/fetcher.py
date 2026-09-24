"""
Real-Time Financial News & Catalyst Fetcher.

Fetches breaking financial news, corporate disclosures, and ticker-specific headlines
from multiple fast zero-rate-limit sources (Google News Financial RSS, yfinance, Finnhub fallback).
Maintains in-memory TTL caching.
"""
from __future__ import annotations

import html
import os
import re
import threading
import time
import xml.etree.ElementTree as ET
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
from typing import Literal, Optional

import requests

from src.data.models import NewsItem
from src.utils.logger import logger

_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36"
    ),
    "Accept": "application/rss+xml, application/xml, text/xml, */*",
}

_CACHE: dict[str, tuple[list[NewsItem], float]] = {}
_CACHE_LOCK = threading.Lock()
_CACHE_TTL_SECONDS = 600  # 10 minutes TTL


def _clean_html(raw_html: str) -> str:
    """Removes HTML tags and decodes entities."""
    clean = re.sub(r"<[^>]+>", "", raw_html)
    return html.unescape(clean).strip()


class RealTimeNewsFetcher:
    """Multi-source real-time news fetcher for US and Indian equity markets."""

    def __init__(self, cache_ttl: int = _CACHE_TTL_SECONDS, session: Optional[requests.Session] = None):
        self.cache_ttl = cache_ttl
        self.session = session or requests.Session()
        self.finnhub_api_key = os.getenv("FINNHUB_API_KEY", "")

    def fetch_news_for_ticker(
        self,
        ticker: str,
        market: Literal["india", "us"] = "us",
        max_articles: int = 5,
        use_cache: bool = True,
    ) -> list[NewsItem]:
        """
        Fetch recent breaking headlines for a specific stock.
        Cleans ticker suffix for search query (e.g., RELIANCE.NS -> RELIANCE).
        """
        norm_ticker = ticker.upper().strip()
        cache_key = f"{market}:{norm_ticker}"

        if use_cache:
            with _CACHE_LOCK:
                if cache_key in _CACHE:
                    items, ts = _CACHE[cache_key]
                    if time.time() - ts < self.cache_ttl:
                        return items[:max_articles]

        query_symbol = norm_ticker.replace(".NS", "").replace(".BO", "")
        items = self._fetch_google_news_rss(query_symbol, market=market, max_articles=max_articles)

        # Fallback to Finnhub for US stocks if RSS returns few results and API key exists
        if len(items) < 2 and market == "us" and self.finnhub_api_key:
            finnhub_items = self._fetch_finnhub_news(query_symbol, max_articles=max_articles)
            items.extend(finnhub_items)

        # Cache results
        with _CACHE_LOCK:
            _CACHE[cache_key] = (items, time.time())

        return items[:max_articles]

    def fetch_market_breaking_news(
        self,
        market: Literal["india", "us"] = "us",
        max_articles: int = 10,
        use_cache: bool = True,
    ) -> list[NewsItem]:
        """Fetch broad market macroeconomic and index-level breaking headlines."""
        cache_key = f"market:{market}"

        if use_cache:
            with _CACHE_LOCK:
                if cache_key in _CACHE:
                    items, ts = _CACHE[cache_key]
                    if time.time() - ts < self.cache_ttl:
                        return items[:max_articles]

        query = "Nifty Sensex stock market India" if market == "india" else "Stock Market S&P 500 Nasdaq Wall Street"
        items = self._fetch_google_news_rss(query, market=market, max_articles=max_articles)

        with _CACHE_LOCK:
            _CACHE[cache_key] = (items, time.time())

        return items[:max_articles]

    def _fetch_google_news_rss(
        self,
        query: str,
        market: Literal["india", "us"] = "us",
        max_articles: int = 5,
    ) -> list[NewsItem]:
        """Fetches and parses Google News RSS feed for a specific query."""
        if market == "india":
            url = f"https://news.google.com/rss/search?q={query}+stock+when:3d&hl=en-IN&gl=IN&ceid=IN:en"
        else:
            url = f"https://news.google.com/rss/search?q={query}+stock+when:3d&hl=en-US&gl=US&ceid=US:en"

        try:
            resp = self.session.get(url, headers=_HEADERS, timeout=10)
            if resp.status_code != 200:
                logger.warning(f"[NewsFetcher] RSS request returned status {resp.status_code} for query '{query}'")
                return []

            root = ET.fromstring(resp.content)
            items: list[NewsItem] = []

            for item_elem in root.findall(".//item"):
                title_elem = item_elem.find("title")
                link_elem = item_elem.find("link")
                pubdate_elem = item_elem.find("pubDate")
                source_elem = item_elem.find("source")

                if title_elem is None or not title_elem.text:
                    continue

                full_title = title_elem.text.strip()
                publisher = source_elem.text.strip() if (source_elem is not None and source_elem.text) else None

                # Google News RSS title format is usually "Headline - Publisher"
                if not publisher and " - " in full_title:
                    parts = full_title.rsplit(" - ", 1)
                    title = parts[0].strip()
                    publisher = parts[1].strip()
                else:
                    title = full_title

                published_at = None
                if pubdate_elem is not None and pubdate_elem.text:
                    try:
                        published_at = parsedate_to_datetime(pubdate_elem.text)
                    except Exception:
                        published_at = datetime.now(timezone.utc)

                link = link_elem.text.strip() if (link_elem is not None and link_elem.text) else None

                items.append(
                    NewsItem(
                        title=title,
                        publisher=publisher,
                        published_at=published_at,
                        url=link,
                    )
                )
                if len(items) >= max_articles:
                    break

            return items

        except Exception as exc:
            logger.warning(f"[NewsFetcher] Error fetching Google News RSS for '{query}': {exc}")
            return []

    def _fetch_finnhub_news(self, symbol: str, max_articles: int = 5) -> list[NewsItem]:
        """Fetch US company news from Finnhub API if token is provided."""
        if not self.finnhub_api_key:
            return []

        today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        url = f"https://finnhub.io/api/v1/company-news?symbol={symbol}&from={today}&to={today}&token={self.finnhub_api_key}"
        try:
            resp = self.session.get(url, timeout=8)
            if resp.status_code != 200:
                return []
            data = resp.json()
            items: list[NewsItem] = []
            for entry in data[:max_articles]:
                pub_ts = entry.get("datetime")
                pub_dt = datetime.fromtimestamp(pub_ts, tz=timezone.utc) if pub_ts else datetime.now(timezone.utc)
                items.append(
                    NewsItem(
                        title=entry.get("headline", ""),
                        publisher=entry.get("source"),
                        published_at=pub_dt,
                        url=entry.get("url"),
                        summary=entry.get("summary"),
                    )
                )
            return items
        except Exception as exc:
            logger.debug(f"[NewsFetcher] Finnhub error for {symbol}: {exc}")
            return []
