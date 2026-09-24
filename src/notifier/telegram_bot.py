"""
Telegram Bot Notifier and Two-Way Interactive Command Center for Aegis.
Handles outgoing alerts and incoming commands (/status, /pause, /resume, /stop, /positions, /portfolio, /watchlist, /include, /exclude, /analyze, /journal, /scorecard).
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
        "<b>Trading & Portfolio Controls:</b>\n"
        "• <code>/status</code> — System health, active LLM, market states & NAV\n"
        "• <code>/positions</code> (or <code>/portfolio</code>) — Live active positions & mark-to-market P&L\n"
        "• <code>/pause [india|us|all]</code> — Pause new entries (trailing stops stay active)\n"
        "• <code>/resume [india|us|all]</code> — Resume autonomous scanning & execution\n"
        "• <code>/stop [india|us|all]</code> — 🚨 Emergency square-off all positions & pause\n"
        "• <code>/reset</code> — 🔄 Clean wipe of positions & fresh capital restart\n\n"
        "<b>Watchlist Management:</b>\n"
        "• <code>/watchlist [india|us]</code> — List active screened tickers\n"
        "• <code>/include &lt;TICKER&gt; [india|us]</code> — Pin stock permanently to watchlist\n"
        "• <code>/exclude &lt;TICKER&gt;</code> — Remove stock from active watchlist\n\n"
        "<b>Observability & Multi-Agent Intelligence:</b>\n"
        "• <code>/journal</code> — View recent immutable trade decision entries\n"
        "• <code>/scorecard</code> — View multi-agent calibration & accuracy scorecards\n"
        "• <code>/analyze &lt;TICKER&gt;</code> — Real-time multi-agent + technical evaluation\n"
        "• <code>/catalysts &lt;TICKER&gt;</code> — Real-time news catalyst & headline risk intelligence"
    )
    await update.message.reply_text(text, parse_mode="HTML")


async def cmd_help(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    await cmd_start(update, context)


async def cmd_status(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if not _is_authorized(update):
        return

    nse_status = "🟢 OPEN" if is_nse_open() else "🔴 CLOSED"
    nyse_status = "🟢 OPEN" if is_nyse_open() else "🔴 CLOSED"
    pause_india = "⏸️ PAUSED" if is_trading_paused("india") else "▶️ ACTIVE"
    pause_us = "⏸️ PAUSED" if is_trading_paused("us") else "▶️ ACTIVE"

    try:
        from src.gateway.router import ModelRouter
        router = ModelRouter()
        llm_engine_str = "🍏 Local M5 14B Qwen2.5 (Online)" if router.local.is_available() else "☁️ Gemini Cloud"
    except Exception:
        llm_engine_str = "☁️ Gemini Cloud"

    engine = PaperTradingEngine()
    si = engine.get_portfolio_summary("india")
    su = engine.get_portfolio_summary("us")
    di = engine.get_daily_stats("india")
    du = engine.get_daily_stats("us")

    def pnl_line(stats: dict, cur: str) -> str:
        r = stats["realized_pnl"]
        u = stats["unrealized_pnl"]
        t = stats["daily_target"]
        emoji = "✅" if stats["target_met"] else ("🟡" if r > 0 else "🔴")
        return (
            f"   • Realized P&L: <b>{cur}{r:+,.2f}</b> / Target: {cur}{t:,.0f} {emoji}\n"
            f"   • Unrealized: {cur}{u:+,.2f} | Closed Trades: {stats['total_trades']} "
            f"(W:{stats['wins']} L:{stats['losses']})"
        )

    text = (
        "📊 <b>AEGIS SYSTEM STATUS</b>\n"
        "──────────────────────────────\n"
        f"🧠 <b>LLM Inference:</b> <b>{llm_engine_str}</b>\n"
        "──────────────────────────────\n"
        f"🇮🇳 <b>NSE/BSE (India):</b> {nse_status}\n"
        f"   • State: <b>{pause_india}</b> | True NAV: <b>₹{si['total_value']:,.2f}</b>\n"
        f"   • Free Cash: ₹{si['cash']:,.2f} | Margin Blocked: ₹{si['reserved_margin']:,.2f}\n"
        f"   • Active Positions: <b>{si['open_positions_count']}</b>\n"
        f"{pnl_line(di, '₹')}\n\n"
        f"🇺🇸 <b>NYSE/NASDAQ (US):</b> {nyse_status}\n"
        f"   • State: <b>{pause_us}</b> | True NAV: <b>${su['total_value']:,.2f}</b>\n"
        f"   • Free Cash: ${su['cash']:,.2f} | Margin Blocked: ${su['reserved_margin']:,.2f}\n"
        f"   • Active Positions: <b>{su['open_positions_count']}</b>\n"
        f"{pnl_line(du, '$')}\n"
        "──────────────────────────────\n"
        f"⏱️ <i>Updated: {datetime.now(timezone.utc).strftime('%H:%M:%S UTC')}</i>\n"
        "<i>Use /positions to view active trades or /reset to start fresh.</i>"
    )
    await update.message.reply_text(text, parse_mode="HTML")


async def cmd_reset(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if not _is_authorized(update):
        return

    engine = PaperTradingEngine()
    result = engine.full_reset(hard_wipe=True)
    text = (
        "🔄 <b>PAPER TRADING RESET COMPLETE</b>\n"
        "──────────────────────────────\n"
        f"• All active/closed positions & trade history wiped cleanly.\n"
        f"• 🇮🇳 India starting capital: <b>₹{result['new_balance_inr']:,.2f} INR</b>\n"
        f"• 🇺🇸 US starting capital: <b>${result['new_balance_usd']:,.2f} USD</b>\n"
        "──────────────────────────────\n"
        "<i>Clean slate active. The agent will evaluate new setups on the next 15-minute cycle.</i>"
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

    # Fetch real-time market prices to calculate current live mark-to-market P&L
    tickers_in = [p["ticker"] for p in positions if p["market"] == "india"]
    tickers_us = [p["ticker"] for p in positions if p["market"] == "us"]
    live_prices: dict[str, float] = {}

    try:
        if tickers_in:
            from src.data.fetcher_india import fetch_india_batch
            snaps_in = fetch_india_batch(tickers_in, delay_seconds=0.1)
            for t, s in snaps_in.items():
                live_prices[t] = s.current_price
        if tickers_us:
            from src.data.fetcher_us import fetch_us_batch
            snaps_us = fetch_us_batch(tickers_us, delay_seconds=0.1)
            for t, s in snaps_us.items():
                live_prices[t] = s.current_price
    except Exception as exc:
        logger.warning(f"[Telegram] Live price refresh error: {exc}")

    def render_market_block(label: str, flag: str, cur: str, market_positions: list) -> str:
        if not market_positions:
            return ""

        total_margin = 0.0
        total_pnl_abs = 0.0
        block_lines = [f"{flag} <b>{label}</b>  ({len(market_positions)} active)\n──────────────"]

        for p in market_positions:
            cost = p["avg_cost"]
            curr = live_prices.get(p["ticker"], p["current_price"] or cost)
            qty = p["quantity"]
            direction = p.get("direction", "LONG")
            margin = p.get("margin_blocked", cost * qty)
            total_margin += margin

            if direction == "SHORT":
                pnl_abs = (cost - curr) * qty
                pnl_pct = ((cost - curr) / cost * 100) if cost else 0.0
            else:
                pnl_abs = (curr - cost) * qty
                pnl_pct = ((curr - cost) / cost * 100) if cost else 0.0

            total_pnl_abs += pnl_abs
            pnl_emoji = "🟢" if pnl_pct >= 0 else "🔴"
            strat_tag = p["strategy"].upper()

            block_lines.append(
                f"{pnl_emoji} <b>{p['ticker']}</b> <i>[{direction} · {strat_tag}]</i>\n"
                f"   Qty: {qty} | Entry: {cur}{cost:,.2f} → Live: <b>{cur}{curr:,.2f}</b>\n"
                f"   P&L: <b>{cur}{pnl_abs:+,.2f} ({pnl_pct:+.2f}%)</b>\n"
                f"   SL: {cur}{p['stop_loss'] or 0:,.2f} | TGT: {cur}{p['target_price'] or 0:,.2f} | Margin: {cur}{margin:,.2f}"
            )

        total_pnl_emoji = "🟢" if total_pnl_abs >= 0 else "🔴"
        block_lines.append(
            f"──────────────\n"
            f"{total_pnl_emoji} <b>Unrealized P&amp;L: {cur}{total_pnl_abs:+,.2f}</b>  "
            f"| Margin Blocked: {cur}{total_margin:,.2f}"
        )
        return "\n".join(block_lines)

    india_pos = [p for p in positions if p["market"] == "india"]
    us_pos = [p for p in positions if p["market"] == "us"]

    parts = []
    if not target or target == "india":
        block = render_market_block("NSE/BSE  🇮🇳 India", "🇮🇳", "₹", india_pos)
        if block:
            parts.append(block)
    if not target or target == "us":
        block = render_market_block("NYSE/NASDAQ  🇺🇸 US", "🇺🇸", "$", us_pos)
        if block:
            parts.append(block)

    header = "📋 <b>ACTIVE OPEN POSITIONS</b>"
    body = ("\n\n" if len(parts) > 1 else "\n").join(parts)
    await update.message.reply_text(f"{header}\n{body}", parse_mode="HTML")


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
            "Example: <code>/include INFY.NS india</code> or <code>/include NVDA us</code>",
            parse_mode="HTML"
        )
        return

    ticker = context.args[0].upper()
    market = "india" if (".NS" in ticker or ".BO" in ticker) else "us"
    if len(context.args) > 1:
        market = context.args[1].lower()

    if market not in ("india", "us"):
        await update.message.reply_text("❌ Market must be <code>india</code> or <code>us</code>.", parse_mode="HTML")
        return

    added = add_ticker(
        ticker=ticker,
        market=market,
        strategies=["swing", "intraday"],
        source="telegram_command",
        pinned=True,
    )

    if added:
        await update.message.reply_text(f"✅ <b>{ticker}</b> added & pinned to {market.upper()} watchlist.", parse_mode="HTML")
    else:
        await update.message.reply_text(f"ℹ️ <b>{ticker}</b> is already on the {market.upper()} watchlist.", parse_mode="HTML")


async def cmd_exclude(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if not _is_authorized(update):
        return

    if not context.args:
        await update.message.reply_text("Usage: <code>/exclude &lt;TICKER&gt;</code>\nExample: <code>/exclude TSLA</code>", parse_mode="HTML")
        return

    ticker = context.args[0].upper()
    removed = remove_ticker(ticker)

    if removed:
        await update.message.reply_text(f"🗑️ <b>{ticker}</b> removed from watchlist.", parse_mode="HTML")
    else:
        await update.message.reply_text(f"⚠️ <b>{ticker}</b> was not found in the active watchlist.", parse_mode="HTML")


async def cmd_journal(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """View recent immutable decision journal entries."""
    if not _is_authorized(update):
        return

    try:
        from src.journal.journal_store import DecisionJournalStore
        store = DecisionJournalStore()
        entries = store.get_recent_decisions(limit=5)

        if not entries:
            await update.message.reply_text("📭 No decision journal entries recorded yet.", parse_mode="HTML")
            return

        lines = ["🏛️ <b>RECENT DECISION JOURNAL ENTRIES</b>\n──────────────────────────────"]
        for e in entries:
            ts = e.timestamp[:16].replace("T", " ")
            lines.append(
                f"• <b>{e.symbol}</b> [{e.market.upper()} · {e.strategy}] — <b>{e.direction}</b>\n"
                f"  Conviction: <code>{e.confidence:.2f}</code> | Risk Decision: <b>{e.risk_decision}</b>\n"
                f"  Thesis: <i>{html.escape(e.final_thesis[:90])}...</i>\n"
                f"  Time: <code>{ts} UTC</code>\n"
            )
        await update.message.reply_text("\n".join(lines), parse_mode="HTML")
    except Exception as exc:
        await update.message.reply_text(f"❌ Could not load decision journal: {html.escape(str(exc))}", parse_mode="HTML")


async def cmd_scorecard(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """View multi-agent calibration and directional accuracy scorecards."""
    if not _is_authorized(update):
        return

    try:
        from src.journal.journal_store import DecisionJournalStore
        store = DecisionJournalStore()
        scorecards = store.get_agent_scorecards()

        if not scorecards:
            await update.message.reply_text("📭 No agent scorecards computed yet (minimum closed trades required).", parse_mode="HTML")
            return

        lines = ["📊 <b>MULTI-AGENT CALIBRATION SCORECARDS</b>\n──────────────────────────────"]
        for sc in scorecards:
            lines.append(
                f"🤖 <b>{sc.agent_name}</b>\n"
                f"   • Directional Accuracy: <b>{sc.directional_accuracy_pct:.1f}%</b> ({sc.correct_direction_count}/{sc.total_evaluations})\n"
                f"   • Thesis Accuracy: <b>{sc.thesis_accuracy_pct:.1f}%</b>\n"
                f"   • Avg Conviction: <code>{sc.average_conviction:.2f}</code> | Brier Score: <code>{sc.brier_score:.3f}</code>\n"
            )
        await update.message.reply_text("\n".join(lines), parse_mode="HTML")
    except Exception as exc:
        await update.message.reply_text(f"❌ Could not load scorecards: {html.escape(str(exc))}", parse_mode="HTML")


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
    await update.message.reply_text(f"🔍 <i>Analyzing {ticker}... fetching live data & running multi-agent evaluation...</i>", parse_mode="HTML")

    from src.analyst.engine import AnalystEngine
    from src.data.fetcher_india import fetch_india_stock
    from src.data.fetcher_us import fetch_us_stock

    is_india = (".NS" in ticker or ".BO" in ticker)
    try:
        snapshot = fetch_india_stock(ticker) if is_india else fetch_us_stock(ticker)
    except Exception as exc:
        await update.message.reply_text(f"❌ Failed to fetch data for <code>{ticker}</code>: {html.escape(str(exc))}", parse_mode="HTML")
        return

    if not snapshot:
        await update.message.reply_text(f"❌ Could not retrieve market data for <code>{ticker}</code>.", parse_mode="HTML")
        return

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


async def cmd_catalysts(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if not _is_authorized(update):
        return

    if not context.args:
        await update.message.reply_text("ℹ️ <b>Usage:</b> <code>/catalysts &lt;TICKER&gt;</code>\nExample: <code>/catalysts NVDA</code> or <code>/catalysts RELIANCE.NS</code>", parse_mode="HTML")
        return

    raw_ticker = context.args[0].upper().strip()
    is_india = raw_ticker.endswith(".NS") or raw_ticker.endswith(".BO")
    market = "india" if is_india else "us"
    ticker = raw_ticker

    await update.message.reply_text(f"🔍 Fetching live breaking news & analyzing catalysts for <code>{ticker}</code>...", parse_mode="HTML")

    from src.news.catalyst_engine import NewsCatalystEngine
    from src.news.fetcher import RealTimeNewsFetcher

    fetcher = RealTimeNewsFetcher()
    engine = NewsCatalystEngine()

    news_items = fetcher.fetch_news_for_ticker(ticker, market=market, max_articles=5)
    if not news_items:
        await update.message.reply_text(f"📰 No recent breaking headlines found for <code>{ticker}</code>.", parse_mode="HTML")
        return

    cat_report = engine.evaluate_catalysts(ticker, market=market, news_items=news_items)

    sentiment_emoji = "🚀" if cat_report.sentiment_label == "BULLISH_CATALYST" else ("🛑" if cat_report.has_headline_risk else "⚖️")
    risk_badge = "⚠️ <b>HEADLINE RISK DETECTED</b>\n" if cat_report.has_headline_risk else ""

    news_list = "\n".join([
        f"• <a href=\"{html.escape(item.url or '')}\">{html.escape(item.title)}</a> <i>({html.escape(item.publisher or 'News')})</i>"
        if item.url else f"• {html.escape(item.title)} <i>({html.escape(item.publisher or 'News')})</i>"
        for item in news_items[:4]
    ])

    text = (
        f"{sentiment_emoji} <b>CATALYST INTELLIGENCE: {ticker}</b>\n"
        f"──────────────────────────────\n"
        f"{risk_badge}"
        f"<b>Category:</b> {cat_report.catalyst_category}\n"
        f"<b>Sentiment Score:</b> <code>{cat_report.sentiment_score:+.2f}</code> ({cat_report.sentiment_label})\n"
        f"<b>Summary:</b> {html.escape(cat_report.catalyst_summary)}\n\n"
        f"📰 <b>Recent Headlines:</b>\n{news_list}"
    )

    await update.message.reply_text(text, parse_mode="HTML", disable_web_page_preview=True)


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
        self.app.add_handler(CommandHandler("nav", cmd_status))
        self.app.add_handler(CommandHandler("pause", cmd_pause))
        self.app.add_handler(CommandHandler("resume", cmd_resume))
        self.app.add_handler(CommandHandler("stop", cmd_stop))
        self.app.add_handler(CommandHandler("reset", cmd_reset))
        self.app.add_handler(CommandHandler("positions", cmd_positions))
        self.app.add_handler(CommandHandler("portfolio", cmd_positions))
        self.app.add_handler(CommandHandler("journal", cmd_journal))
        self.app.add_handler(CommandHandler("scorecard", cmd_scorecard))
        self.app.add_handler(CommandHandler("watchlist", cmd_watchlist))
        self.app.add_handler(CommandHandler("include", cmd_include))
        self.app.add_handler(CommandHandler("exclude", cmd_exclude))
        self.app.add_handler(CommandHandler("analyze", cmd_analyze))
        self.app.add_handler(CommandHandler("catalysts", cmd_catalysts))
        self.app.add_handler(CommandHandler("news", cmd_catalysts))

    def run_polling(self) -> None:
        """Starts the long-polling loop (blocking the calling thread)."""
        logger.info("[Telegram] Starting interactive Telegram command bot...")
        self.app.run_polling(drop_pending_updates=True)
