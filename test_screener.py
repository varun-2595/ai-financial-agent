"""
Quick test script for the autonomous screener.
Run: python test_screener.py

Tests:
  1. Universe fetch (NSE 500 + S&P 500)
  2. Watchlist seeding from watchlist.yaml
  3. Manual ticker add / pin / ban
  4. Mini screen run (first 20 tickers only for speed)
  5. Watchlist summary
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from src.screener.universe import fetch_nse500, fetch_sp500
from src.screener.watchlist_manager import (
    add_ticker,
    ban_ticker,
    get_active_tickers,
    get_watchlist_summary,
    init_watchlist,
    pin_ticker,
)
from src.screener.screener import MarketScreener, ScreenerConfig
from src.utils.logger import logger


def separator(title: str) -> None:
    print(f"\n{'═'*60}")
    print(f"  {title}")
    print(f"{'═'*60}")


if __name__ == "__main__":
    # ── 1. Fetch universes ─────────────────────────────────────────────────
    separator("TEST 1: Universe Fetch")
    nse = fetch_nse500()
    sp  = fetch_sp500()
    print(f"  NSE universe: {len(nse)} tickers  (first 5: {nse[:5]})")
    print(f"  S&P universe: {len(sp)} tickers   (first 5: {sp[:5]})")

    # ── 2. Watchlist init + seed from YAML ────────────────────────────────
    separator("TEST 2: Watchlist Init + YAML Seed")
    init_watchlist()
    summary = get_watchlist_summary()
    print(f"  After init: {summary['total']} tickers "
          f"({summary['india']} India, {summary['us']} US)")

    # ── 3. Manual operations ───────────────────────────────────────────────
    separator("TEST 3: Manual Add / Pin / Ban")

    add_ticker("TATASTEEL.NS", market="india", strategies=["swing"], source="manual",
               reason="Test manual add")
    print("  ✓ Manually added TATASTEEL.NS")

    pin_ticker("RELIANCE.NS", reason="Core holding — never remove")
    print("  ✓ Pinned RELIANCE.NS")

    ban_ticker("PAYTM.NS", reason="Test ban — avoid speculative names")
    print("  ✓ Banned PAYTM.NS")

    # Verify ban works
    result = add_ticker("PAYTM.NS", market="india", strategies=["swing"], source="auto")
    print(f"  ✓ Re-add PAYTM.NS returned: {result} (should be False — banned)")

    # ── 4. Mini screen (fast — only 20 tickers) ───────────────────────────
    separator("TEST 4: Mini Autonomous Screen (20 India + 20 US)")

    cfg = ScreenerConfig(
        shortlist_size_india=10,
        shortlist_size_us=10,
        max_new_picks_india=5,
        max_new_picks_us=5,
        min_score_to_add=40.0,   # lower threshold for test
        fetch_delay=0.3,
        default_strategies=["swing"],
    )
    screener = MarketScreener(cfg=cfg)

    # Override universe to only first 20 for test speed
    from src.screener import universe as u
    import unittest.mock as mock
    with mock.patch.object(u, "fetch_nse500", return_value=nse[:20]), \
         mock.patch.object(u, "fetch_sp500",  return_value=sp[:20]):
        result = screener.run(prune=False)

    print(f"\n  India added:   {result['india_added']}")
    print(f"  US added:      {result['us_added']}")
    if result["top_india"]:
        print(f"\n  Top India picks (scored):")
        for item in result["top_india"]:
            print(f"    {item['ticker']:20} score={item['score']:.1f}/100")
    if result["top_us"]:
        print(f"\n  Top US picks (scored):")
        for item in result["top_us"]:
            print(f"    {item['ticker']:20} score={item['score']:.1f}/100")

    # ── 5. Final watchlist state ───────────────────────────────────────────
    separator("TEST 5: Final Watchlist State")
    summary = get_watchlist_summary()
    print(f"  Total tickers:  {summary['total']}")
    print(f"  India:          {summary['india']}")
    print(f"  US:             {summary['us']}")
    print(f"  Auto-added:     {summary['auto']}")
    print(f"  Manual:         {summary['manual']}")

    active = get_active_tickers()
    print(f"\n  All active tickers:")
    for t in active:
        pin = "📌" if t["pinned"] else "  "
        src = "👤 manual" if t["source"] == "manual" else "🤖 auto  "
        print(f"    {pin} {src}  {t['ticker']:20}  {t['strategies']}")

    print("\n  ✅ All screener tests passed.\n")
