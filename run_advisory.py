"""
CLI entry point for Financial Advisory actions.

Usage:
    python run_advisory.py --monthly-scan       # Run monthly screening for stocks & ETFs
    python run_advisory.py --sell-scan          # Run weekly review of open advisory holdings
    python run_advisory.py --portfolio          # View current advisory portfolio and returns
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))
load_dotenv(ROOT / ".env")

from src.advisory.recommender import AdvisoryRecommender
from src.advisory.sell_signal_engine import AdvisorySellEngine
from src.data.fetcher_india import fetch_india_stock
from src.data.fetcher_us import fetch_us_stock
from src.db.advisory_store import get_active_advisory_holdings, init_advisory_db
from src.utils.logger import logger


def display_portfolio() -> None:
    init_advisory_db()
    holdings = get_active_advisory_holdings()

    if not holdings:
        print("\n  No active advisory holdings found. Run --monthly-scan to generate recommendations.\n")
        return

    print("\n" + "═" * 70)
    print("  📊 ACTIVE FINANCIAL ADVISORY PORTFOLIO")
    print("═" * 70)

    for h in holdings:
        sym = "₹" if h["market"] == "india" else "$"
        curr = h["current_price"] or h["entry_price"]
        ret = ((curr - h["entry_price"]) / h["entry_price"]) * 100
        print(f"  [{h['horizon'].upper():10}] {h['ticker']:15} {h['name'] or '':20}")
        print(f"    Entry: {sym}{h['entry_price']:.2f} | Current: {sym}{curr:.2f} | PnL: {ret:+.2f}%")
        print(f"    Stop Loss: {sym}{h['stop_loss']:.2f} | Target: {sym}{h['target_price']:.2f}")
        print(f"    Thesis: {h['thesis']}")
        print("  " + "─" * 66)
    print()


def run_sell_scan() -> None:
    init_advisory_db()
    holdings = get_active_advisory_holdings()
    if not holdings:
        print("\n  No active holdings to scan.\n")
        return

    print(f"\n  Scanning {len(holdings)} advisory holdings for exit / trim signals...")
    snapshots = {}
    for h in holdings:
        t = h["ticker"]
        try:
            if h["market"] == "india":
                snapshots[t] = fetch_india_stock(t)
            else:
                snapshots[t] = fetch_us_stock(t)
        except Exception as exc:
            logger.warning(f"Could not fetch {t}: {exc}")

    engine = AdvisorySellEngine()
    alerts = engine.scan_holdings(snapshots)

    if not alerts:
        print("\n  ✅ All holdings intact. No stop loss or target exits triggered today.\n")
    else:
        print("\n  🚨 ADVISORY ACTIONS REQUIRED:")
        for a in alerts:
            print(f"    - {a.action} {a.ticker}: {a.reason}")
        print()


def main():
    parser = argparse.ArgumentParser(description="Financial Advisory CLI")
    parser.add_argument("--monthly-scan", action="store_true", help="Run monthly screening")
    parser.add_argument("--sell-scan", action="store_true", help="Run weekly sell signal scan")
    parser.add_argument("--portfolio", action="store_true", help="View active advisory portfolio")

    args = parser.parse_args()

    if args.monthly_scan:
        recommender = AdvisoryRecommender()
        recommender.run_monthly_advisory()
        display_portfolio()
    elif args.sell_scan:
        run_sell_scan()
    elif args.portfolio:
        display_portfolio()
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
