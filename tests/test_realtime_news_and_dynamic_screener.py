"""
Unit and Integration Tests for Real-Time News & Catalyst Engine + Dynamic Opportunity Scanner.
"""
from datetime import datetime, timezone
from unittest.mock import MagicMock, patch

import pytest

from src.data.models import NewsItem, Quote, StockSnapshot
from src.news.catalyst_engine import CatalystReport, NewsCatalystEngine
from src.news.fetcher import RealTimeNewsFetcher
from src.screener.dynamic_scanner import DynamicOpportunityScanner, ScannedOpportunity
from src.signals.generator import SignalGenerator


def _sample_quote(price: float, volume: int = 1000000) -> Quote:
    return Quote(
        timestamp=datetime.now(timezone.utc),
        open=price * 0.99,
        high=price * 1.01,
        low=price * 0.98,
        close=price,
        volume=volume,
    )


def _make_snapshot(ticker: str, market: str, price: float, chg_1d: float, news: list[NewsItem]) -> StockSnapshot:
    history = [_sample_quote(price * (1.0 + 0.005 * i), volume=1000000) for i in range(30)]
    return StockSnapshot(
        ticker=ticker,
        market=market,  # type: ignore
        currency="USD" if market == "us" else "INR",
        current_price=price,
        price_change_pct_1d=chg_1d,
        price_change_pct_1w=chg_1d * 1.5,
        history=history,
        recent_news=news,
    )


# ── 1. RealTimeNewsFetcher Tests ───────────────────────────────────────────────

def test_realtime_news_fetcher_rss_parsing():
    sample_xml = b"""<?xml version="1.0" encoding="UTF-8"?>
    <rss version="2.0">
      <channel>
        <title>Google News</title>
        <item>
          <title>Nvidia Beats Q4 Earnings with Record AI Chip Demand - Reuters</title>
          <link>https://news.google.com/articles/123</link>
          <pubDate>Thu, 24 Sep 2026 14:00:00 GMT</pubDate>
          <source url="https://reuters.com">Reuters</source>
        </item>
        <item>
          <title>Nvidia Partners with Sovereign Cloud Providers - Bloomberg</title>
          <link>https://news.google.com/articles/124</link>
          <pubDate>Thu, 24 Sep 2026 12:00:00 GMT</pubDate>
          <source url="https://bloomberg.com">Bloomberg</source>
        </item>
      </channel>
    </rss>
    """
    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.content = sample_xml

    with patch("requests.Session.get", return_value=mock_resp):
        fetcher = RealTimeNewsFetcher()
        items = fetcher.fetch_news_for_ticker("NVDA", market="us", max_articles=2, use_cache=False)

        assert len(items) == 2
        assert "Nvidia Beats Q4 Earnings" in items[0].title
        assert items[0].publisher == "Reuters"
        assert items[0].url == "https://news.google.com/articles/123"


def test_realtime_news_fetcher_caching():
    fetcher = RealTimeNewsFetcher(cache_ttl=60)
    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.content = b"""<rss><channel><item><title>Test - Source</title><link>http://a.com</link></item></channel></rss>"""

    with patch("requests.Session.get", return_value=mock_resp) as mock_get:
        # First call fetches
        fetcher.fetch_news_for_ticker("AAPL", market="us", use_cache=False)
        # Second call uses cache
        cached = fetcher.fetch_news_for_ticker("AAPL", market="us", use_cache=True)
        assert len(cached) == 1
        assert cached[0].title == "Test"


# ── 2. NewsCatalystEngine Tests ────────────────────────────────────────────────

def test_catalyst_engine_bullish_earnings_heuristic():
    engine = NewsCatalystEngine()
    news = [
        NewsItem(title="Tech Mahindra beats estimates with 40% profit jump", publisher="Economic Times"),
        NewsItem(title="Brokerages raise price target to ₹1800", publisher="Moneycontrol"),
    ]
    report = engine.evaluate_catalysts("TECHM.NS", market="india", news_items=news, force_heuristic=True)

    assert report.sentiment_label == "BULLISH_CATALYST"
    assert report.catalyst_category in ("EARNINGS_BEAT", "UPGRADE_PRICE_TARGET")
    assert report.sentiment_score > 0.5
    assert not report.has_headline_risk


def test_catalyst_engine_severe_headline_risk_heuristic():
    engine = NewsCatalystEngine()
    news = [
        NewsItem(title="SEC launches fraud investigation into accounting irregularities", publisher="WSJ"),
    ]
    report = engine.evaluate_catalysts("RISK_STOCK", market="us", news_items=news, force_heuristic=True)

    assert report.has_headline_risk is True
    assert report.sentiment_label == "BEARISH_RISK"
    assert report.sentiment_score < -0.5
    assert report.catalyst_category == "LAWSUIT_INVESTIGATION"


def test_catalyst_engine_empty_news():
    engine = NewsCatalystEngine()
    report = engine.evaluate_catalysts("QUIET_STOCK", market="us", news_items=[])

    assert report.catalyst_category == "NO_CATALYST"
    assert report.sentiment_label == "NEUTRAL"
    assert report.sentiment_score == 0.0
    assert not report.has_headline_risk


# ── 3. DynamicOpportunityScanner Tests ─────────────────────────────────────────

def test_dynamic_opportunity_scanner_scoring():
    mock_fetcher = MagicMock(spec=RealTimeNewsFetcher)
    mock_cat_engine = MagicMock(spec=NewsCatalystEngine)

    # Return positive catalyst report
    mock_cat_engine.evaluate_catalysts.return_value = CatalystReport(
        ticker="TSLA",
        market="us",
        sentiment_score=0.8,
        sentiment_label="BULLISH_CATALYST",
        catalyst_category="CONTRACT_DEAL",
        has_headline_risk=False,
        catalyst_summary="Signed mega fleet contract.",
    )

    scanner = DynamicOpportunityScanner(news_fetcher=mock_fetcher, catalyst_engine=mock_cat_engine)

    snap = _make_snapshot(
        ticker="TSLA",
        market="us",
        price=220.0,
        chg_1d=3.5,
        news=[NewsItem(title="Mega Fleet Deal", publisher="Reuters")],
    )

    with patch("src.screener.dynamic_scanner.fetch_us_batch", return_value={"TSLA": snap}), \
         patch("src.screener.dynamic_scanner.get_rotated_universe", return_value=["TSLA"]):

        opps = scanner.scan_market_opportunities(market="us", candidate_pool_size=1, top_picks_limit=5)
        assert len(opps) == 1
        assert opps[0].ticker == "TSLA"
        assert opps[0].composite_score > 20.0
        assert opps[0].catalyst_category == "CONTRACT_DEAL"


def test_dynamic_opportunity_scanner_skips_headline_risk():
    mock_fetcher = MagicMock(spec=RealTimeNewsFetcher)
    mock_cat_engine = MagicMock(spec=NewsCatalystEngine)

    mock_cat_engine.evaluate_catalysts.return_value = CatalystReport(
        ticker="FRAUD_CORP",
        market="us",
        sentiment_score=-0.9,
        sentiment_label="BEARISH_RISK",
        catalyst_category="LAWSUIT_INVESTIGATION",
        has_headline_risk=True,
        catalyst_summary="Raid by authorities.",
    )

    scanner = DynamicOpportunityScanner(news_fetcher=mock_fetcher, catalyst_engine=mock_cat_engine)

    snap = _make_snapshot(
        ticker="FRAUD_CORP",
        market="us",
        price=50.0,
        chg_1d=-5.0,
        news=[NewsItem(title="Raid by authorities", publisher="WSJ")],
    )

    with patch("src.screener.dynamic_scanner.fetch_us_batch", return_value={"FRAUD_CORP": snap}), \
         patch("src.screener.dynamic_scanner.get_rotated_universe", return_value=["FRAUD_CORP"]):

        opps = scanner.scan_market_opportunities(market="us", candidate_pool_size=1, top_picks_limit=5)
        assert len(opps) == 0  # Filtered out due to headline risk


# ── 4. SignalGenerator News Catalyst Gatekeeper Test ─────────────────────────

def test_signal_generator_blocks_on_headline_risk():
    mock_analyst = MagicMock()
    # Analyst would otherwise say Good Buy
    mock_analysis = MagicMock()
    mock_analysis.action = "Good Buy"
    mock_analysis.confidence_score = 0.85
    mock_analysis.reasons_attractive = ["Strong momentum"]
    mock_analyst.analyze_stock.return_value = mock_analysis

    mock_cat_engine = MagicMock()
    mock_cat_engine.evaluate_catalysts.return_value = CatalystReport(
        ticker="BAD_CORP",
        market="us",
        sentiment_score=-0.8,
        sentiment_label="BEARISH_RISK",
        catalyst_category="LAWSUIT_INVESTIGATION",
        has_headline_risk=True,
        catalyst_summary="SEC Subpoena Issued",
    )

    sig_gen = SignalGenerator(analyst=mock_analyst, catalyst_engine=mock_cat_engine)

    snap = _make_snapshot(
        ticker="BAD_CORP",
        market="us",
        price=100.0,
        chg_1d=-1.0,
        news=[NewsItem(title="SEC Subpoena Issued", publisher="Bloomberg")],
    )

    sig = sig_gen.generate_signal(snapshot=snap, strategy="scalping")
    assert sig is None  # Blocked by catalyst engine headline risk gatekeeper
