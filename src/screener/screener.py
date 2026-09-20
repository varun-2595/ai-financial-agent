"""
Autonomous Market Screener — the agent's self-directed ticker discovery engine.

This is what makes the agent truly autonomous. It runs on a schedule and:

  1. Fetches the full investable universe (NSE 500 + S&P 500)
  2. Applies fast quantitative filters (liquidity, price, trend, volatility)
  3. Shortlists top N candidates per market
  4. Fetches full data for the shortlist
  5. Scores each candidate on a multi-factor model
  6. Adds high-conviction picks to the live watchlist
  7. Reviews and prunes stale/underperforming tickers already on the watchlist

The user can always:
  - Add any ticker manually (bypasses all filters, immediately active)
  - Pin a ticker (agent can never remove it)
  - Ban a ticker (agent will never trade it)

Scoring model (out of 100):
  ┌───────────────────────────────────────┬────────┐
  │ Factor                                │ Weight │
  ├───────────────────────────────────────┼────────┤
  │ Price momentum (1M + 3M)              │  25%   │
  │ Relative volume (vs 30D avg)          │  20%   │
  │ Trend health (price vs EMA50)         │  20%   │
  │ Fundamental quality (PE, ROE, margin) │  20%   │
  │ Volatility fit (ATR-based)            │  15%   │
  └───────────────────────────────────────┴────────┘
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Literal

from src.data.fetcher_india import fetch_india_batch, normalize_india_ticker
from src.data.fetcher_us import fetch_us_batch
from src.data.models import StockSnapshot
from src.screener.filters import DEFAULT_FILTERS, FilterConfig, filter_batch
from src.screener.universe import fetch_full_universe
from src.screener.watchlist_manager import (
    add_ticker,
    get_active_tickers,
    get_watchlist_summary,
    init_watchlist,
    remove_ticker,
)
from src.utils.logger import logger


@dataclass
class ScreenerConfig:
    """Tunable screener parameters."""

    # How many candidates to pull full data for (after fast filters)
    shortlist_size_india: int = 40
    shortlist_size_us: int = 40

    # How many top picks to add to watchlist per run
    max_new_picks_india: int = 10
    max_new_picks_us: int = 10

    # Minimum score (out of 100) to be added to the watchlist
    min_score_to_add: float = 55.0

    # Score below which an existing auto-added ticker gets removed
    remove_score_threshold: float = 35.0

    # Default strategy to assign to auto-discovered tickers
    default_strategies: list[str] = field(default_factory=lambda: ["swing", "positional"])

    # Delay between yfinance calls during bulk fetch (seconds)
    fetch_delay: float = 0.3


DEFAULT_SCREENER_CONFIG = ScreenerConfig()


# ── Scoring ────────────────────────────────────────────────────────────────────

def _ema(values: list[float], period: int) -> float | None:
    """Compute EMA of a price series."""
    if len(values) < period:
        return None
    k = 2 / (period + 1)
    ema = values[0]
    for v in values[1:]:
        ema = v * k + ema * (1 - k)
    return ema


def score_snapshot(snap: StockSnapshot) -> float:
    """
    Score a stock snapshot on 0–100.
    Higher = more attractive for trading.
    """
    score = 0.0
    h = snap.history
    f = snap.fundamentals

    # ── 1. Price Momentum (25 pts) ─────────────────────────────────────────
    momentum_score = 0.0
    m1 = snap.price_change_pct_1m
    m3 = snap.price_change_pct_3m
    if m1 is not None:
        # +5pts for each 5% gain in 1M, capped at 15pts
        momentum_score += min(max(m1 / 5 * 5, -15), 15)
    if m3 is not None:
        # +5pts for each 10% gain in 3M, capped at 10pts
        momentum_score += min(max(m3 / 10 * 5, -10), 10)
    score += max(momentum_score, 0)  # don't go negative on this factor

    # ── 2. Relative Volume (20 pts) ────────────────────────────────────────
    if h and f.avg_volume_30d and f.avg_volume_30d > 0:
        recent_vol = h[-1].volume if h else 0
        rel_vol = recent_vol / f.avg_volume_30d
        # 1.5x average volume = 10pts, 2x = 15pts, 3x+ = 20pts
        vol_score = min(rel_vol / 3.0 * 20, 20)
        score += vol_score
    else:
        score += 10  # neutral if no volume data

    # ── 3. Trend Health — price vs EMA50 (20 pts) ──────────────────────────
    if len(h) >= 50:
        closes = [q.close for q in h]
        ema50 = _ema(closes, 50)
        if ema50 and ema50 > 0:
            pct_above_ema = (snap.current_price - ema50) / ema50 * 100
            # +20 if significantly above EMA50, 0 if at EMA, negative if below
            trend_score = min(max(pct_above_ema / 5 * 10, 0), 20)
            score += trend_score

    # ── 4. Fundamental Quality (20 pts) ────────────────────────────────────
    fund_score = 10.0  # neutral baseline for stocks without fundamentals (ETFs)

    if f.pe_ratio is not None and f.pe_ratio > 0:
        # Reasonable PE (10–30) = good, very high PE = penalty
        if 10 <= f.pe_ratio <= 30:
            fund_score += 5
        elif 30 < f.pe_ratio <= 50:
            fund_score += 2
        elif f.pe_ratio > 50:
            fund_score -= 2

    if f.return_on_equity is not None:
        roe_pct = f.return_on_equity * 100
        if roe_pct > 20:
            fund_score += 5
        elif roe_pct > 10:
            fund_score += 2

    if f.profit_margin is not None:
        margin = f.profit_margin * 100
        if margin > 20:
            fund_score += 5
        elif margin > 10:
            fund_score += 2

    if f.debt_to_equity is not None:
        if f.debt_to_equity < 0.5:
            fund_score += 5
        elif f.debt_to_equity > 2.0:
            fund_score -= 3

    score += min(max(fund_score, 0), 20)

    # ── 5. Volatility Fit for swing trading (15 pts) ───────────────────────
    if len(h) >= 14:
        # Compute ATR %
        true_ranges = []
        for i in range(1, min(15, len(h))):
            tr = max(
                h[-i].high - h[-i].low,
                abs(h[-i].high - h[-i - 1].close),
                abs(h[-i].low  - h[-i - 1].close),
            )
            true_ranges.append(tr)
        if true_ranges and snap.current_price > 0:
            atr = sum(true_ranges) / len(true_ranges)
            atr_pct = atr / snap.current_price * 100
            # Sweet spot for swing: 1–4% ATR
            if 1.0 <= atr_pct <= 4.0:
                score += 15
            elif 0.5 <= atr_pct < 1.0 or 4.0 < atr_pct <= 6.0:
                score += 8
            else:
                score += 3

    return round(min(score, 100.0), 2)


# ── Screener ───────────────────────────────────────────────────────────────────

class MarketScreener:
    """
    Autonomous market screener. Discovers, scores, and manages the watchlist.

    Usage:
        screener = MarketScreener()
        result = screener.run()
        print(result)
    """

    def __init__(self, cfg: ScreenerConfig = DEFAULT_SCREENER_CONFIG):
        self.cfg = cfg
        init_watchlist()

    def _fetch_and_score(
        self,
        tickers: list[str],
        market: Literal["india", "us"],
        strategy: str,
        limit: int,
    ) -> list[tuple[str, StockSnapshot, float]]:
        """
        Fetch data for a list of tickers, apply filters, score, and return top N.

        Returns:
            List of (ticker, snapshot, score) sorted by score descending.
        """
        logger.info(
            f"[Screener] Fetching {len(tickers)} {market.upper()} candidates..."
        )

        # Batch fetch
        if market == "india":
            snapshots = fetch_india_batch(tickers, delay_seconds=self.cfg.fetch_delay)
        else:
            snapshots = fetch_us_batch(tickers, delay_seconds=self.cfg.fetch_delay)

        # Apply quantitative filters
        filtered = filter_batch(snapshots, strategy=strategy)

        # Score each passing ticker
        scored = []
        for ticker, snap in filtered.items():
            s = score_snapshot(snap)
            scored.append((ticker, snap, s))

        # Sort by score descending, take top N
        scored.sort(key=lambda x: x[2], reverse=True)
        return scored[:limit]

    def _update_watchlist(
        self,
        scored: list[tuple[str, StockSnapshot, float]],
        market: Literal["india", "us"],
        max_new: int,
    ) -> dict[str, int]:
        """Add high-conviction picks, return counts."""
        added = 0
        skipped = 0

        for ticker, snap, score in scored[:max_new]:
            if score < self.cfg.min_score_to_add:
                logger.debug(
                    f"[Screener] {ticker} score {score:.1f} < "
                    f"threshold {self.cfg.min_score_to_add} — skipping"
                )
                skipped += 1
                continue

            result = add_ticker(
                ticker=ticker,
                market=market,
                strategies=self.cfg.default_strategies,
                source="auto",
                reason=f"Auto-screened: score {score:.1f}/100",
            )
            if result:
                added += 1
                logger.success(
                    f"[Screener] ✅ Added {ticker} to watchlist "
                    f"(score={score:.1f}/100)"
                )

        return {"added": added, "skipped": skipped}

    def _prune_stale_tickers(self) -> int:
        """
        Review existing auto-added tickers.
        Remove any whose score has fallen below the removal threshold.
        Returns count of tickers removed.
        """
        removed = 0
        active = get_active_tickers()
        auto_tickers = [t for t in active if t["source"] == "auto"]

        if not auto_tickers:
            return 0

        logger.info(
            f"[Screener] Reviewing {len(auto_tickers)} auto-added tickers for pruning..."
        )

        india_auto = [t["ticker"] for t in auto_tickers if t["market"] == "india"]
        us_auto    = [t["ticker"] for t in auto_tickers if t["market"] == "us"]

        def prune_batch(tickers, market):
            nonlocal removed
            if not tickers:
                return
            if market == "india":
                snaps = fetch_india_batch(tickers, delay_seconds=self.cfg.fetch_delay)
            else:
                snaps = fetch_us_batch(tickers, delay_seconds=self.cfg.fetch_delay)

            for ticker, snap in snaps.items():
                score = score_snapshot(snap)
                if score < self.cfg.remove_score_threshold:
                    ok = remove_ticker(
                        ticker,
                        reason=f"Score dropped to {score:.1f}/100 (threshold: {self.cfg.remove_score_threshold})",
                    )
                    if ok:
                        removed += 1
                        logger.info(
                            f"[Screener] 🗑️ Pruned {ticker} (score={score:.1f}/100)"
                        )

        prune_batch(india_auto, "india")
        prune_batch(us_auto, "us")
        return removed

    def run(
        self,
        prune: bool = True,
        india: bool = True,
        us: bool = True,
    ) -> dict:
        """
        Run the full autonomous screening cycle.

        Args:
            prune: if True, also reviews and removes stale tickers
            india: if True, screen Indian market
            us: if True, screen US market

        Returns:
            Summary dict with counts and top picks
        """
        logger.info("=" * 60)
        logger.info("[Screener] 🔍 Starting autonomous market screening...")
        logger.info("=" * 60)

        result = {
            "india_added": 0,
            "us_added": 0,
            "pruned": 0,
            "top_india": [],
            "top_us": [],
        }

        # ── Fetch universe ─────────────────────────────────────────────────
        universe = fetch_full_universe()

        # ── Screen India ───────────────────────────────────────────────────
        if india and universe.get("india"):
            # Take a random spread to keep variety (not just same top-50 every time)
            candidates = universe["india"][:200]  # first 200 for speed
            scored_india = self._fetch_and_score(
                tickers=candidates,
                market="india",
                strategy="swing",
                limit=self.cfg.shortlist_size_india,
            )
            counts = self._update_watchlist(
                scored_india, market="india", max_new=self.cfg.max_new_picks_india
            )
            result["india_added"] = counts["added"]
            result["top_india"] = [
                {"ticker": t, "score": s}
                for t, _, s in scored_india[:5]
            ]

        # ── Screen US ──────────────────────────────────────────────────────
        if us and universe.get("us"):
            candidates = universe["us"][:200]
            scored_us = self._fetch_and_score(
                tickers=candidates,
                market="us",
                strategy="swing",
                limit=self.cfg.shortlist_size_us,
            )
            counts = self._update_watchlist(
                scored_us, market="us", max_new=self.cfg.max_new_picks_us
            )
            result["us_added"] = counts["added"]
            result["top_us"] = [
                {"ticker": t, "score": s}
                for t, _, s in scored_us[:5]
            ]

        # ── Prune stale tickers ────────────────────────────────────────────
        if prune:
            result["pruned"] = self._prune_stale_tickers()

        # ── Summary ────────────────────────────────────────────────────────
        summary = get_watchlist_summary()
        result["watchlist_total"] = summary["total"]
        result["watchlist_india"] = summary["india"]
        result["watchlist_us"]    = summary["us"]

        logger.info("=" * 60)
        logger.info(
            f"[Screener] ✅ Done. Added {result['india_added']} India + "
            f"{result['us_added']} US tickers. Pruned {result['pruned']}. "
            f"Total watchlist: {summary['total']}"
        )
        logger.info("=" * 60)

        return result
