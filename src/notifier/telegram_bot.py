"""
Telegram Bot Notifier and Two-Way Interactive Command Center for Aegis.
Handles outgoing alerts and incoming commands (/status, /pause, /resume, /stop, /positions, /watchlist, /include, /exclude, /analyze).
"""
from __future__ import annotations

import asyncio
import html
import os
from datetime import datetime, timezone
from typing import Optional

import requests
from telegram import Update
from telegram.ext import Application, ApplicationBuilder, CommandHandler, ContextTypes

from src.db.trading_store import get_open_positions
from src.screener.watchlist_manager import add_ticker, get_active_tickers, remove_ticker
from src.trading.paper_engine import PaperTradingEngine
from src.utils.logger import logger
from src.utils.market_hours import is_nse_open, is_nyse_open
from src.utils.state_manager import is_trading_paused, set_trading_paused


# ── Outgoing Notification Dispatcher (Sync) ───────────────────────────────────

class TelegramNotifier:
    """
    Lightweight synchronous dispatcher for sending system notifications.
    Used by scheduled jobs and background monitors.
    """
    def __init__(self, token: Optional[str] = None, chat_id: Optional[str] = None):
        self.token = token or os.getenv("TELEGRAM_BOT_TOKEN")
        self.chat_id = chat_id or os.getenv("TELEGRAM_CHAT_ID")
        self.base_url = f"https://api.telegram.org/bot{self.token}" if self.token else None

    def send_message(self, text: str, parse_mode: str = "HTML") -> bool:
        if not self.base_url or not self.chat_id:
            logger.debug(f"[Telegram] Credentials missing; message not sent: {text[:50]}...")
            return False

        try:
            resp = requests.post(
                f"{self.base_url}/sendMessage",
                json={
                    "chat_id": self.chat_id,
                    "text": text,
                    "parse_mode": parse_mode,
                    "disable_web_page_preview": True,
                },
                timeout=10,
            )
            resp.raise_for_status()
            logger.info("[Telegram] Message delivered successfully.")
            return True
        except Exception as exc:
            logger.error(f"[Telegram] Failed to send message: {exc}")
            return False


# ── Interactive Command Center (Async) ─────────────────────────────────────────

def _is_authorized(update: Update) -> bool:
    """Strict security check: only authorized TELEGRAM_CHAT_ID can execute commands."""
    if not update.effective_chat:
        return False
    authorized = os.getenv("TELEGRAM_CHAT_ID")
    if not authorized:
        return True
    return str(update.effective_chat.id).strip() == str(authorized).strip()


async def cmd_start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if not _is_authorized(update):
        return
    text = (
        "🤖 <b>Aegis AI Financial Agent — Command Center</b>\n\n"
        "<b>Trading Controls:</b>\n"
        "• <code>/status</code> — System health, markets, balances, and trading state\n"
        "• <code>/pause [india|us|all]</code> — Pause new trade entries (stops stay active)\n"
        "• <code>/resume [india|us|all]</code> — Resume autonomous scanning\n"
        "• <code>/stop [india|us|all]</code> — 🚨 Emergency square-off all positions & pause\n\n"
        "<b>Portfolio & Watchlist:</b>\n"
        "• <code>/positions [india|us]</code> — View active open positions & P&L\n"
        "• <code>/watchlist [india|us]</code> — List active screened tickers\n"
        "• <code>/include &lt;TICKER&gt; [india|us]</code> — Add & pin stock permanently\n"
        "• <code>/exclude &lt;TICKER&gt;</code> — Remove stock from active watchlist\n\n"
        "<b>Intelligence On-Demand:</b>\n"
        "• <code>/analyze &lt;TICKER&gt;</code> — Real-time Gemini + technical evaluation"
    )
    await update.message.reply_text(text, parse_mode="HTML")


async def cmd_help(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    await cmd_start(update, context)


async def cmd_status(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if not _is_authorized(update):
        return

    # Check market sessions
    nse_status = "🟢 OPEN" if is_nse_open() else "🔴 CLOSED"
    nyse_status = "🟢 OPEN" if is_nyse_open() else "🔴 CLOSED"

    # Check pause state
    pause_india = "⏸️ PAUSED" if is_trading_paused("india") else "▶️ ACTIVE"
    pause_us = "⏸️ PAUSED" if is_trading_paused("us") else "▶️ ACTIVE"

    # Fetch account balances
    engine = PaperTradingEngine()
    summary_inr = engine.get_portfolio_summary("india")
    summary_usd = engine.get_portfolio_summary("us")

    text = (
        "📊 <b>AEGIS SYSTEM STATUS</b>\n"
        "──────────────────────────────\n"
        f"🇮🇳 <b>NSE/BSE (India):</b> {nse_status}\n"
        f"   • Trading State: <b>{pause_india}</b>\n"
        f"   • Cash: ₹{summary_inr['cash']:,.2f} | Invested: ₹{summary_inr['invested']:,.2f}\n"
        f"   • Total Value: ₹{summary_inr['total_value']:,.2f}\n"
        f"   • Open Positions: {summary_inr['open_positions_count']}\n\n"
        f"🇺🇸 <b>NYSE/NASDAQ (US):</b> {nyse_status}\n"
        f"   • Trading State: <b>{pause_us}</b>\n"
        f"   • Cash: ${summary_usd['cash']:,.2f} | Invested: ${summary_usd['invested']:,.2f}\n"
        f"   • Total Value: ${summary_usd['total_value']:,.2f}\n"
        f"   • Open Positions: {summary_usd['open_positions_count']}\n"
        "──────────────────────────────\n"
        "⚙️ Scheduler: <b>RUNNING (24×7 Daemon)</b>"
    )
    await update.message.reply_text(text, parse_mode="HTML")


async def cmd_pause(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if not _is_authorized(update):
        return

    target = context.args[0].lower() if context.args else "all"
    if target not in ("india", "us", "all"):
        await update.message.reply_text("Usage: <code>/pause [india|us|all]</code>", parse_mode="HTML")
        return

    set_trading_paused(target, True)
    await update.message.reply_text(
        f"⏸️ <b>Trading PAUSED for {target.upper()}</b>\n"
        "• New signal entries are halted.\n"
        "• Existing trailing stops and target monitors remain actively monitored.",
        parse_mode="HTML"
    )


async def cmd_resume(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if not _is_authorized(update):
        return

    target = context.args[0].lower() if context.args else "all"
    if target not in ("india", "us", "all"):
        await update.message.reply_text("Usage: <code>/resume [india|us|all]</code>", parse_mode="HTML")
        return

    set_trading_paused(target, False)
    await update.message.reply_text(
        f"▶️ <b>Trading RESUMED for {target.upper()}</b>\n"
        "Autonomous 15-minute signal scans and order execution are now active.",
        parse_mode="HTML"
    )


async def cmd_stop(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if not _is_authorized(update):
        return

    target = context.args[0].lower() if context.args else "all"
    if target not in ("india", "us", "all"):
        await update.message.reply_text("Usage: <code>/stop [india|us|all]</code>", parse_mode="HTML")
        return

    # 1. Pause immediately
    set_trading_paused(target, True)

    # 2. Square off positions
    engine = PaperTradingEngine()
    reports: list[str] = []
    if target in ("india", "all"):
        reports.extend(engine.close_all_positions("india"))
    if target in ("us", "all"):
        reports.extend(engine.close_all_positions("us"))

    report_lines = "\n".join(reports) if reports else "No active positions were open."

    await update.message.reply_text(
        f"🚨 <b>EMERGENCY STOP EXECUTED ({target.upper()})</b>\n\n"
        f"{report_lines}\n\n"
        f"⏸️ <i>New trading has been paused. Use /resume to re-enable when ready.</i>",
        parse_mode="HTML"
    )


async def cmd_positions(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if not _is_authorized(update):
        return

    target = context.args[0].lower() if context.args else None
    positions = get_open_positions(market=target)

    if not positions:
        market_label = f" in {target.upper()}" if target else ""
        await update.message.reply_text(f"📭 No open positions{market_label}.", parse_mode="HTML")
        return

    lines = ["📋 <b>ACTIVE OPEN POSITIONS</b>\n──────────────────────────────"]
    for p in positions:
        cost = p["avg_cost"]
        curr = p["current_price"] or cost
        pnl_pct = ((curr - cost) / cost * 100) if cost else 0.0
        pnl_emoji = "🟢" if pnl_pct >= 0 else "🔴"
        currency = "₹" if p["market"] == "india" else "$"

        lines.append(
            f"{pnl_emoji} <b>{p['ticker']}</b> ({p['market'].upper()} • {p['strategy']})\n"
            f"   Qty: {p['quantity']} | Avg: {currency}{cost:,.2f} | Now: {currency}{curr:,.2f}\n"
            f"   P&L: <b>{pnl_pct:+.2f}%</b>\n"
            f"   SL: {currency}{p['stop_loss'] or 0:,.2f} | TP: {currency}{p['target_price'] or 0:,.2f}\n"
        )

    lines.append("──────────────────────────────")
    await update.message.reply_text("\n".join(lines), parse_mode="HTML")


async def cmd_watchlist(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if not _is_authorized(update):
        return

    target = context.args[0].lower() if context.args else None
    tickers = get_active_tickers(market=target)

    if not tickers:
        await update.message.reply_text("📭 Active watchlist is currently empty.", parse_mode="HTML")
        return

    lines = [f"👁️ <b>ACTIVE WATCHLIST ({len(tickers)} Tickers)</b>\n──────────────────────────────"]
    for t in tickers:
        pin_icon = "📌" if t.get("pinned") else "🔹"
        source = t.get("source", "auto")
        strats = ", ".join(t.get("strategies", []))
        lines.append(f"{pin_icon} <b>{t['ticker']}</b> [{t['market'].upper()}] — {strats} <i>({source})</i>")

    lines.append("──────────────────────────────")
    lines.append("<i>📌 = Pinned permanently (never auto-pruned)</i>")
    await update.message.reply_text("\n".join(lines), parse_mode="HTML")


async def cmd_include(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if not _is_authorized(update):
        return

    if not context.args:
        await update.message.reply_text(
            "Usage: <code>/include &lt;TICKER&gt; [india|us]</code>\n"
            "Example: <code>/include TATAPOWER.NS</code> or <code>/include TSLA us</code>",
            parse_mode="HTML"
        )
        return

    ticker = context.args[0].upper()
    market = context.args[1].lower() if len(context.args) > 1 else None

    if not market:
        market = "india" if (".NS" in ticker or ".BO" in ticker) else "us"

    success = add_ticker(
        ticker=ticker,
        market=market,
        strategies=["swing", "positional"],
        source="manual",
        pinned=True,
        reason="Added via Telegram /include command",
    )

    if success:
        await update.message.reply_text(
            f"📌 <b>Stock Included & Pinned:</b> <code>{ticker}</code>\n"
            f"• Market: <b>{market.upper()}</b>\n"
            f"• Strategies: Swing, Positional\n"
            f"• Protected from autonomous screener pruning.",
            parse_mode="HTML"
        )
    else:
        await update.message.reply_text(f"⚠️ Failed to add <code>{ticker}</code>.", parse_mode="HTML")


async def cmd_exclude(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if not _is_authorized(update):
        return

    if not context.args:
        await update.message.reply_text("Usage: <code>/exclude &lt;TICKER&gt;</code>", parse_mode="HTML")
        return

    ticker = context.args[0].upper()
    removed = remove_ticker(ticker, reason="Removed via Telegram /exclude command", force=True)

    if removed:
        await update.message.reply_text(
            f"➖ <b>Stock Excluded:</b> <code>{ticker}</code> has been removed from the active watchlist.",
            parse_mode="HTML"
        )
    else:
        await update.message.reply_text(f"Ticker <code>{ticker}</code> was not found in active watchlist.", parse_mode="HTML")


async def cmd_analyze(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if not _is_authorized(update):
        return

    if not context.args:
        await update.message.reply_text(
            "Usage: <code>/analyze &lt;TICKER&gt;</code>\n"
            "Example: <code>/analyze INFY.NS</code> or <code>/analyze NVDA</code>",
            parse_mode="HTML"
        )
        return

    ticker = context.args[0].upper()
    await update.message.reply_text(f"🔍 <i>Analyzing {ticker}... fetching live data & running Gemini evaluation...</i>", parse_mode="HTML")

    from src.analyst.engine import AnalystEngine
    from src.data.fetcher_india import fetch_india_stock
    from src.data.fetcher_us import fetch_us_stock

    # Fetch market data
    is_india = (".NS" in ticker or ".BO" in ticker)
    try:
        snapshot = fetch_india_stock(ticker) if is_india else fetch_us_stock(ticker)
    except Exception as exc:
        await update.message.reply_text(f"❌ Failed to fetch data for <code>{ticker}</code>: {html.escape(str(exc))}", parse_mode="HTML")
        return

    if not snapshot:
        await update.message.reply_text(f"❌ Could not retrieve market data for <code>{ticker}</code>.", parse_mode="HTML")
        return

    # Run Analyst Engine
    try:
        engine = AnalystEngine()
        analysis = engine.analyze_stock(snapshot)
    except Exception as exc:
        await update.message.reply_text(f"❌ Analysis failed: {html.escape(str(exc))}", parse_mode="HTML")
        return

    action_emoji = "🟢" if "Buy" in analysis.action else ("🔴" if "Avoid" in analysis.action else "🟡")
    currency = "₹" if is_india else "$"

    attractive_bullets = "\n".join(f"  • {html.escape(r)}" for r in analysis.reasons_attractive)
    risk_bullets = "\n".join(f"  • {html.escape(r)}" for r in analysis.risks)

    text = (
        f"{action_emoji} <b>ANALYSIS: {analysis.ticker}</b>\n"
        f"──────────────────────────────\n"
        f"<b>Action:</b> {analysis.action} (Confidence: {analysis.confidence_score * 100:.0f}%)\n"
        f"<b>Risk Level:</b> {analysis.risk_level}\n"
        f"<b>Current Price:</b> {currency}{snapshot.current_price:,.2f}\n"
    )

    if analysis.entry_price_hint:
        text += f"<b>Entry:</b> {currency}{analysis.entry_price_hint:,.2f} | <b>SL:</b> {currency}{analysis.stop_loss_hint or 0:,.2f} | <b>TP:</b> {currency}{analysis.target_price_hint or 0:,.2f}\n"

    text += (
        f"──────────────────────────────\n"
        f"<b>Why Attractive:</b>\n{attractive_bullets}\n\n"
        f"<b>Downside Risks:</b>\n{risk_bullets}\n\n"
        f"<b>Approach:</b> {html.escape(analysis.suggested_approach)}\n"
        f"<b>Invalidation:</b> {html.escape(analysis.invalidation_trigger)}"
    )

    await update.message.reply_text(text, parse_mode="HTML")


# ── Bot Server Application ───────────────────────────────────────────────────

class AegisTelegramBot:
    """
    Long-polling interactive Telegram bot instance.
    Runs concurrently with the APScheduler.
    """
    def __init__(self, token: Optional[str] = None):
        self.token = token or os.getenv("TELEGRAM_BOT_TOKEN")
        if not self.token:
            raise ValueError("TELEGRAM_BOT_TOKEN is required to run AegisTelegramBot.")

        self.app: Application = ApplicationBuilder().token(self.token).build()
        self._register_handlers()

    def _register_handlers(self) -> None:
        self.app.add_handler(CommandHandler("start", cmd_start))
        self.app.add_handler(CommandHandler("help", cmd_help))
        self.app.add_handler(CommandHandler("status", cmd_status))
        self.app.add_handler(CommandHandler("pause", cmd_pause))
        self.app.add_handler(CommandHandler("resume", cmd_resume))
        self.app.add_handler(CommandHandler("stop", cmd_stop))
        self.app.add_handler(CommandHandler("positions", cmd_positions))
        self.app.add_handler(CommandHandler("watchlist", cmd_watchlist))
        self.app.add_handler(CommandHandler("include", cmd_include))
        self.app.add_handler(CommandHandler("exclude", cmd_exclude))
        self.app.add_handler(CommandHandler("analyze", cmd_analyze))

    def run_polling(self) -> None:
        """Starts the long-polling loop (blocking the calling thread)."""
        logger.info("[Telegram] Starting interactive Telegram command bot...")
        self.app.run_polling(drop_pending_updates=True)
