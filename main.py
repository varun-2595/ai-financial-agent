"""
Aegis — Autonomous AI Financial & Trading Agent.
Primary entry point to start the continuous trading and advisory daemon.

Usage:
  python main.py                  # Start continuous daemon (scheduler + trading + advisory)
  python main.py --dry-run        # Run one immediate test cycle without waiting
"""
from __future__ import annotations

import argparse
import os
import signal
import sys
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))
load_dotenv(ROOT / ".env")

from src.db.advisory_store import init_advisory_db
from src.db.trading_store import init_trading_db
from src.notifier.telegram_bot import AegisTelegramBot, TelegramNotifier
from src.scheduler.jobs import job_market_intraday_scan, job_screen_universe
from src.scheduler.runner import run_scheduler, start_background_scheduler
from src.screener.watchlist_manager import init_watchlist
from src.utils.logger import logger
from src.utils.market_hours import todays_agenda
from src.utils.state_manager import init_state_table


def dry_run() -> None:
    logger.info("⚡ Executing instant dry-run verification cycle...")
    init_watchlist()
    init_trading_db()
    init_advisory_db()
    init_state_table()

    from src.trading.paper_engine import PaperTradingEngine
    engine = PaperTradingEngine()
    engine.reset_account_balances()

    summary_inr = engine.get_portfolio_summary("india")
    summary_usd = engine.get_portfolio_summary("us")
    cfg_paper = engine.config.paper_trading

    print(todays_agenda())
    print("=" * 60)
    print("💰 REALISTIC PAPER TRADING CAPITAL & DAILY TARGETS")
    print("=" * 60)
    print(f"  🇮🇳 India Capital: ₹{summary_inr['cash']:,.2f} INR | Daily Target: ₹{cfg_paper.daily_profit_target_inr:,.2f} (10%)")
    print(f"  🇺🇸 US Capital:    ${summary_usd['cash']:,.2f} USD  | Daily Target: ${cfg_paper.daily_profit_target_usd:,.2f} (15%)")
    print(f"  ⚡ Strategies:    Scalping, Intraday, Swing, Positional")
    print(f"  📊 Universe:      High-Velocity Mid-Caps, Small-Caps & Momentum Runners")
    print("=" * 60)

    # Run instant scan on active markets
    job_market_intraday_scan("india")
    job_market_intraday_scan("us")
    logger.success("⚡ Dry-run finished successfully.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Aegis AI Financial Agent Daemon")
    parser.add_argument("--dry-run", action="store_true", help="Execute single immediate test cycle")
    parser.add_argument("--reset", action="store_true", help="Reset all trades, positions, journal entries, and reset accounts from scratch")
    args = parser.parse_args()

    # Bootstrap databases and state
    init_watchlist()
    init_trading_db()
    init_advisory_db()
    init_state_table()

    if args.reset:
        from src.trading.paper_engine import PaperTradingEngine
        from src.db.advisory_store import _conn as adv_conn
        engine = PaperTradingEngine()
        res = engine.full_reset(hard_wipe=True)
        # Also clean advisory holdings
        with adv_conn() as aconn:
            aconn.execute("DELETE FROM advisory_holdings")
            aconn.execute("DELETE FROM advisory_alerts")
            aconn.commit()
        logger.success(
            f"✅ [Aegis Reset] All trades and journals wiped. "
            f"Starting capital: ₹{res['new_balance_inr']:,.2f} INR | ${res['new_balance_usd']:,.2f} USD."
        )
        return

    if args.dry_run:
        dry_run()
        return


    # Check LLM Gateway status on startup
    from src.gateway.gemini_provider import GeminiProvider
    from src.gateway.local_provider import LocalModelProvider
    gemini_prov = GeminiProvider()
    local_prov = LocalModelProvider()

    local_ok = local_prov.is_available()
    gemini_ok, gemini_status = gemini_prov.validate_connection()

    if not local_ok and not gemini_ok:
        llm_status_msg = (
            "⚠️ <b>WARNING: AI Reasoning Providers Offline!</b>\n"
            "• Local Ollama: 🔴 OFFLINE\n"
            f"• Cloud Gemini: 🔴 ERROR ({gemini_status})\n"
            "<i>Trading intelligence is degraded. Update GEMINI_API_KEY in .env to restore full AI analysis.</i>"
        )
        logger.error(f"[Main Startup] AI Providers Offline: {gemini_status}")
    else:
        active_llms = []
        if local_ok:
            active_llms.append("M5 Local Node (Ollama)")
        if gemini_ok:
            active_llms.append("Google Gemini Cloud")
        llm_status_msg = f"🧠 AI Reasoning Active: <b>{', '.join(active_llms)}</b>"

    # Notify Telegram on startup
    tg = TelegramNotifier()
    tg.send_message(
        "🤖 <b>Aegis AI Financial Agent Started</b>\n"
        "Tri-Market 24/7 Monitoring (NSE + NYSE + Crypto).\n"
        f"{llm_status_msg}\n"
        "Interactive commands enabled: send <code>/help</code> for options."
    )

    # Start background scheduler
    scheduler = start_background_scheduler()

    # Start Fast Tick Position Monitor (BeeBots 3s high-frequency SL/TP loop)
    from src.trading.fast_tick_monitor import FastTickMonitor
    tick_monitor = FastTickMonitor(tick_interval_seconds=3.0)
    tick_monitor.start()

    # Launch interactive Telegram bot if token is configured
    bot_token = os.getenv("TELEGRAM_BOT_TOKEN")
    if bot_token:
        try:
            bot = AegisTelegramBot(token=bot_token)
            logger.info("🤖 Interactive Telegram Bot is listening for commands...")
            bot.run_polling()
        except (KeyboardInterrupt, SystemExit):
            logger.info("Shutting down agent...")
        finally:
            tick_monitor.stop()
            if scheduler.running:
                scheduler.shutdown(wait=False)
            logger.info("Aegis agent shutdown cleanly.")
    else:
        logger.warning("TELEGRAM_BOT_TOKEN not found; interactive bot disabled. Running scheduler only.")
        try:
            signal.pause()
        except (KeyboardInterrupt, SystemExit):
            tick_monitor.stop()
            scheduler.shutdown(wait=False)


if __name__ == "__main__":
    main()
