"""
CLI entry point for manual analysis of any ticker.

Usage:
    python run_analysis.py --ticker RELIANCE.NS
    python run_analysis.py --ticker AAPL --market us
    python run_analysis.py --watchlist          # runs all tickers in watchlist.yaml
    python run_analysis.py --ticker INFY.NS --json   # output raw JSON
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import yaml
from dotenv import load_dotenv

# ── Bootstrap ─────────────────────────────────────────────────────────────────
ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))
load_dotenv(ROOT / ".env")

from src.utils.logger import logger
from src.utils.market_hours import market_status
from src.data.fetcher_us import fetch_us_stock
from src.data.fetcher_india import fetch_india_stock, normalize_india_ticker


# ── Helpers ───────────────────────────────────────────────────────────────────

def detect_market(ticker: str) -> str:
    """Guess market from ticker format."""
    t = ticker.upper()
    if t.endswith(".NS") or t.endswith(".BO"):
        return "india"
    # Common India tickers without suffix — heuristic
    india_hints = {"NIFTY", "SENSEX", "RELIANCE", "INFY", "TCS", "HDFC", "ICICI", "WIPRO"}
    if any(h in t for h in india_hints):
        return "india"
    return "us"


def format_snapshot_table(snapshot) -> str:
    """Pretty-print a StockSnapshot to the console."""
    f = snapshot.fundamentals
    currency = "₹" if snapshot.currency == "INR" else "$"

    lines = [
        "",
        "═" * 60,
        f"  {snapshot.ticker}  —  {snapshot.name or 'N/A'}",
        f"  Market: {snapshot.market.upper()}  |  Sector: {snapshot.sector or 'N/A'}",
        "═" * 60,
        "",
        "  PRICE",
        f"    Current : {currency}{snapshot.current_price:>12,.2f}",
        f"    1D      : {snapshot.price_change_pct_1d:>+8.2f}%" if snapshot.price_change_pct_1d is not None else "    1D      :      N/A",
        f"    1W      : {snapshot.price_change_pct_1w:>+8.2f}%" if snapshot.price_change_pct_1w is not None else "    1W      :      N/A",
        f"    1M      : {snapshot.price_change_pct_1m:>+8.2f}%" if snapshot.price_change_pct_1m is not None else "    1M      :      N/A",
        f"    3M      : {snapshot.price_change_pct_3m:>+8.2f}%" if snapshot.price_change_pct_3m is not None else "    3M      :      N/A",
        f"    1Y      : {snapshot.price_change_pct_1y:>+8.2f}%" if snapshot.price_change_pct_1y is not None else "    1Y      :      N/A",
        f"    52W H   : {currency}{f.week_52_high:>12,.2f}" if f.week_52_high else "    52W H   :      N/A",
        f"    52W L   : {currency}{f.week_52_low:>12,.2f}" if f.week_52_low else "    52W L   :      N/A",
        "",
        "  FUNDAMENTALS",
        f"    Market Cap  : {currency}{f.market_cap/1e9:>8,.2f}B" if f.market_cap else "    Market Cap  :      N/A",
        f"    P/E Ratio   : {f.pe_ratio:>12.2f}" if f.pe_ratio else "    P/E Ratio   :      N/A",
        f"    P/B Ratio   : {f.pb_ratio:>12.2f}" if f.pb_ratio else "    P/B Ratio   :      N/A",
        f"    EPS         : {currency}{f.eps:>11.2f}" if f.eps else "    EPS         :      N/A",
        f"    Debt/Equity : {f.debt_to_equity:>12.2f}" if f.debt_to_equity else "    Debt/Equity :      N/A",
        f"    Beta        : {f.beta:>12.2f}" if f.beta else "    Beta        :      N/A",
        f"    ROE         : {f.return_on_equity*100:>11.1f}%" if f.return_on_equity else "    ROE         :      N/A",
        f"    Profit Margin: {f.profit_margin*100:>10.1f}%" if f.profit_margin else "    Profit Margin:      N/A",
        f"    Div Yield   : {f.dividend_yield*100:>11.2f}%" if f.dividend_yield else "    Div Yield   :      N/A",
        f"    Avg Vol 30D : {f.avg_volume_30d:>12,}" if f.avg_volume_30d else "    Avg Vol 30D :      N/A",
        "",
        f"  PRICE HISTORY : {len(snapshot.history)} daily candles",
        f"  DATA SOURCE   : {snapshot.data_source}",
        f"  FETCHED AT    : {snapshot.fetched_at.strftime('%Y-%m-%d %H:%M:%S UTC')}",
    ]

    if snapshot.recent_news:
        lines += ["", "  RECENT NEWS"]
        for i, n in enumerate(snapshot.recent_news[:5], 1):
            pub = n.published_at.strftime("%b %d") if n.published_at else "?"
            lines.append(f"    {i}. [{pub}] {n.title[:65]}…" if len(n.title) > 65 else f"    {i}. [{pub}] {n.title}")

    lines += ["", "═" * 60, ""]
    return "\n".join(lines)


def run_single(ticker: str, market: str | None, as_json: bool) -> None:
    """Fetch and display data for a single ticker."""
    if market is None:
        market = detect_market(ticker)

    fetch_fn = fetch_india_stock if market == "india" else fetch_us_stock
    snapshot = fetch_fn(ticker)

    if as_json:
        print(snapshot.model_dump_json(indent=2))
    else:
        print(format_snapshot_table(snapshot))


def run_watchlist(as_json: bool) -> None:
    """Fetch and display all tickers from watchlist.yaml."""
    watchlist_path = ROOT / "config" / "watchlist.yaml"
    if not watchlist_path.exists():
        logger.error("watchlist.yaml not found at config/watchlist.yaml")
        sys.exit(1)

    with open(watchlist_path) as f:
        config = yaml.safe_load(f)

    wl = config.get("watchlist", {})
    india_tickers = [item["ticker"] for item in wl.get("india", [])]
    us_tickers    = [item["ticker"] for item in wl.get("us", [])]

    print(f"\n  Running watchlist scan — {len(india_tickers)} India + {len(us_tickers)} US tickers\n")

    for ticker in india_tickers:
        try:
            snap = fetch_india_stock(ticker)
            if as_json:
                print(snap.model_dump_json(indent=2))
            else:
                print(format_snapshot_table(snap))
        except Exception as exc:
            logger.warning(f"Skipping {ticker}: {exc}")

    for ticker in us_tickers:
        try:
            snap = fetch_us_stock(ticker)
            if as_json:
                print(snap.model_dump_json(indent=2))
            else:
                print(format_snapshot_table(snap))
        except Exception as exc:
            logger.warning(f"Skipping {ticker}: {exc}")


# ── CLI ───────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="AI Financial Agent — Manual Analysis CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run_analysis.py --ticker RELIANCE.NS
  python run_analysis.py --ticker AAPL --market us
  python run_analysis.py --watchlist
  python run_analysis.py --ticker NVDA --json
  python run_analysis.py --market-status
        """,
    )
    parser.add_argument("--ticker",        type=str, help="Ticker symbol (e.g. RELIANCE.NS or AAPL)")
    parser.add_argument("--market",        type=str, choices=["india", "us"], help="Override market detection")
    parser.add_argument("--watchlist",     action="store_true", help="Run analysis on all watchlist tickers")
    parser.add_argument("--json",          action="store_true", help="Output raw JSON instead of table")
    parser.add_argument("--market-status", action="store_true", help="Show current market open/closed status")

    args = parser.parse_args()

    if args.market_status:
        status = market_status()
        nse  = status["NSE"]
        nyse = status["NYSE"]
        print(f"\n  NSE  : {nse['status']:6}  {nse['local_time']}  |  opens {nse['opens_at_ist']} – {nse['closes_at_ist']}")
        print(f"  NYSE : {nyse['status']:6}  {nyse['local_time_ist']}  |  opens {nyse['opens_at_ist']} – {nyse['closes_at_ist']} IST  [{nyse['dst_note']}]\n")
        return

    if args.watchlist:
        run_watchlist(as_json=args.json)
        return

    if args.ticker:
        run_single(ticker=args.ticker, market=args.market, as_json=args.json)
        return

    parser.print_help()


if __name__ == "__main__":
    main()
