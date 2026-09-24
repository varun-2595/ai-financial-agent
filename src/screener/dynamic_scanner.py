"""
Dynamic Opportunity Scanner.

Continuously discovers breakout runners and volume shockers across the entire
investable universe (NSE 500 + Mid/Small Caps for India; S&P 500 + Growth/Tech + Apple Ecosystem for US).
Bypasses hardcoded watchlists by actively ranking momentum, relative volume spikes,
and real-time news catalysts to inject live potential winners into the trading engine.
"""
from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Literal, Optional

from src.data.fetcher_india import fetch_india_batch, normalize_india_ticker
from src.data.fetcher_us import fetch_us_batch
from src.data.models import StockSnapshot
from src.news.catalyst_engine import CatalystReport, NewsCatalystEngine
from src.news.fetcher import RealTimeNewsFetcher
from src.screener.universe import fetch_nse500, fetch_sp500, get_rotated_universe
from src.screener.watchlist_manager import add_ticker, get_active_tickers
from src.utils.logger import logger


@dataclass
class ScannedOpportunity:
    ticker: str
    market: Literal["india", "us"]
    current_price: float
    change_1d_pct: float
    relative_volume: float
    catalyst_category: str
    catalyst_score: float
    composite_score: float
    summary: str
    headline_risk: bool = False


class DynamicOpportunityScanner:
    """Discovers and filters high-velocity breakout candidates dynamically."""

    def __init__(
        self,
        news_fetcher: Optional[RealTimeNewsFetcher] = None,
        catalyst_engine: Optional[NewsCatalystEngine] = None,
    ):
        self.news_fetcher = news_fetcher or RealTimeNewsFetcher()
        self.catalyst_engine = catalyst_engine or NewsCatalystEngine()

    def scan_market_opportunities(
        self,
        market: Literal["india", "us"],
        candidate_pool_size: int = 35,
        top_picks_limit: int = 15,
    ) -> list[ScannedOpportunity]:
        """
        Scans a rotated candidate pool of the broader universe for real-time momentum,
        volume spikes, and news catalysts.
        """
        logger.info(f"[DynamicScanner] Initiating dynamic opportunity scan for {market.upper()} universe...")

        # 1. Fetch rotated universe candidates
        candidate_tickers = get_rotated_universe(market, max_candidates=candidate_pool_size)

        # 2. Fetch fresh snapshots in batch
        if market == "india":
            snapshots = fetch_india_batch(candidate_tickers, delay_seconds=0.15)
        else:
            snapshots = fetch_us_batch(candidate_tickers, delay_seconds=0.15)

        if not snapshots:
            logger.warning(f"[DynamicScanner] No snapshots returned for {market.upper()} dynamic scan.")
            return []

        opportunities: list[ScannedOpportunity] = []

        # 3. Filter and score each candidate
        for ticker, snap in snapshots.items():
            if snap.current_price <= 0:
                continue

            # Compute relative volume vs 30D average
            hist = snap.history
            f = snap.fundamentals
            rel_vol = 1.0
            if hist and len(hist) >= 5:
                recent_vol = hist[-1].volume
                avg_vol = f.avg_volume_30d or (sum(q.volume for q in hist[-20:]) / max(len(hist[-20:]), 1))
                if avg_vol and avg_vol > 0:
                    rel_vol = round(recent_vol / avg_vol, 2)

            chg_1d = snap.price_change_pct_1d or 0.0

            # Momentum filter: positive 1D or weekly momentum
            chg_1w = snap.price_change_pct_1w or 0.0
            
            # Fetch real-time news & evaluate catalysts
            news_items = snap.recent_news or self.news_fetcher.fetch_news_for_ticker(ticker, market=market, max_articles=4)
            cat_report = self.catalyst_engine.evaluate_catalysts(ticker, market, news_items)

            # Skip severe headline risks immediately
            if cat_report.has_headline_risk:
                logger.warning(f"[DynamicScanner] Skipping {ticker} due to headline risk: {cat_report.catalyst_summary}")
                continue

            # Scoring algorithm:
            # - Momentum: 0-40 pts (favoring positive 1D & 1W price momentum)
            mom_score = max(min((chg_1d * 4.0) + (chg_1w * 1.5), 40.0), -20.0)

            # - Volume surge: 0-35 pts (rewarding 1.2x to 3.0x relative volume)
            vol_score = min(max((rel_vol - 0.8) * 20.0, 0.0), 35.0)

            # - News Catalyst: 0-25 pts (rewarding positive catalyst sentiment)
            cat_score = max(cat_report.sentiment_score * 25.0, 0.0)

            composite = mom_score + vol_score + cat_score

            # Retain viable candidates
            if composite > 15.0 or chg_1d > 0.8 or rel_vol > 1.3 or cat_report.sentiment_score > 0.3:
                opp = ScannedOpportunity(
                    ticker=ticker,
                    market=market,
                    current_price=snap.current_price,
                    change_1d_pct=chg_1d,
                    relative_volume=rel_vol,
                    catalyst_category=cat_report.catalyst_category,
                    catalyst_score=cat_report.sentiment_score,
                    composite_score=round(composite, 2),
                    summary=cat_report.catalyst_summary,
                    headline_risk=cat_report.has_headline_risk,
                )
                opportunities.append(opp)

        # Sort by highest composite score
        opportunities.sort(key=lambda x: x.composite_score, reverse=True)
        top_picks = opportunities[:top_picks_limit]

        logger.success(
            f"[DynamicScanner] {market.upper()} Scan Complete: Found {len(top_picks)} top dynamic opportunities."
        )
        return top_picks

    def sync_dynamic_opportunities_to_watchlist(
        self,
        market: Literal["india", "us"],
        top_n: int = 15,
    ) -> list[str]:
        """
        Discovers top dynamic opportunities and registers them in the active watchlist
        with scalping, intraday, and swing strategies enabled.
        """
        top_opps = self.scan_market_opportunities(market=market, top_picks_limit=top_n)
        added_tickers: list[str] = []

        existing_active = {t["ticker"] for t in get_active_tickers(market=market)}

        for opp in top_opps:
            if opp.ticker not in existing_active:
                add_ticker(
                    ticker=opp.ticker,
                    market=market,
                    strategies=["scalping", "intraday", "swing"],
                    pinned=False,
                    score=min(opp.composite_score, 99.0),
                    notes=f"Dynamic Runner ({opp.catalyst_category} | RVol: {opp.relative_volume}x)",
                )
                added_tickers.append(opp.ticker)

        if added_tickers:
            logger.success(f"[DynamicScanner] Added {len(added_tickers)} dynamic runners to active {market.upper()} trading queue: {added_tickers}")
        return added_tickers
