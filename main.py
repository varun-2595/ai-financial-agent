"""
Aegis — Autonomous AI Financial & Trading Agent.
Primary entry point to start the continuous trading and advisory daemon.

Usage:
  python main.py                  # Start continuous daemon (scheduler + trading + advisory)
  python main.py --dry-run        # Run one immediate test cycle without waiting
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))
load_dotenv(ROOT / ".env")

from src.db.advisory_store import init_advisory_db
from src.db.trading_store import init_trading_db
from src.notifier.telegram_bot import TelegramNotifier
from src.scheduler.jobs import job_market_intraday_scan, job_screen_universe
from src.scheduler.runner import build_scheduler, run_scheduler
from src.screener.watchlist_manager import init_watchlist
from src.utils.logger import logger
from src.utils.market_hours import todays_agenda


def dry_run() -> None:
    logger.info("⚡ Executing instant dry-run verification cycle...")
    init_watchlist()
    init_trading_db()
    init_advisory_db()

    print(todays_agenda())

    # Run instant scan on active markets
    job_market_intraday_scan("india")
    job_market_intraday_scan("us")
    logger.success("⚡ Dry-run finished successfully.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Aegis AI Financial Agent Daemon")
    parser.add_argument("--dry-run", action="store_true", help="Execute single immediate test cycle")
    args = parser.parse_args()

    # Bootstrap databases and state
    init_watchlist()
    init_trading_db()
    init_advisory_db()

    if args.dry_run:
        dry_run()
        return

    # Notify Telegram bot on startup
    tg = TelegramNotifier()
    tg.send_message("🤖 <b>Aegis AI Financial Agent Started</b>\nDual-market monitoring active (NSE + NYSE).")

    # Start continuous scheduler
    run_scheduler()


if __name__ == "__main__":
    main()
