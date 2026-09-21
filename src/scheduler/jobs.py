"""
Scheduled Automated Trading and Advisory Jobs.
Executed periodically by the APScheduler runner across India and US sessions.
"""
from __future__ import annotations

from typing import Literal

from src.advisory.recommender import AdvisoryRecommender
from src.advisory.sell_signal_engine import AdvisorySellEngine
from src.data.fetcher_india import fetch_india_batch, fetch_india_stock
from src.data.fetcher_us import fetch_us_batch, fetch_us_stock
from src.notifier.formatter import (
    format_eod_email_html,
    format_eod_report_telegram,
    format_order_fill_telegram,
    format_trade_signal_telegram,
)
from src.notifier.email_sender import EmailNotifier
from src.notifier.telegram_bot import TelegramNotifier
from src.screener.screener import MarketScreener
from src.screener.watchlist_manager import get_active_tickers
from src.signals.generator import SignalGenerator
from src.trading.paper_engine import PaperTradingEngine
from src.utils.logger import logger
from src.utils.market_hours import is_nse_open, is_nyse_open
from src.utils.state_manager import is_trading_paused


def job_screen_universe() -> None:
    """05:30 AM IST: Screener runs across NSE 500 & S&P 500, updates dynamic watchlist."""
    logger.info("[Job] Executing daily universe screening...")
    screener = MarketScreener()
    res = screener.run(prune=True)
    tg = TelegramNotifier()
    tg.send_message(
        f"🔍 <b>Market Screening Complete</b>\n"
        f"Added: {res['india_added']} India, {res['us_added']} US | Pruned: {res['pruned']}\n"
        f"Total Active Watchlist: {res['watchlist_total']}"
    )


def job_market_intraday_scan(market: Literal["india", "us"]) -> None:
    """Runs every 15 minutes during active trading hours for the respective market."""
    if market == "india" and not is_nse_open():
        return
    if market == "us" and not is_nyse_open():
        return

    logger.info(f"[Job] Running {market.upper()} intraday signal scan...")
    engine = PaperTradingEngine()
    sig_gen = SignalGenerator()
    tg = TelegramNotifier()

    # 1. Fetch current active tickers for this market
    tickers_info = get_active_tickers(market=market)
    tickers = [t["ticker"] for t in tickers_info]

    if market == "india":
        snapshots = fetch_india_batch(tickers, delay_seconds=0.2)
    else:
        snapshots = fetch_us_batch(tickers, delay_seconds=0.2)

    # 2. Check open positions for Stop-Loss or Target-Profit triggers
    closed_reports = engine.evaluate_open_positions(market, snapshots)
    for rep in closed_reports:
        tg.send_message(f"🔔 <b>Trade Exit:</b>\n{rep}")

    # If new trades are paused for this market, stop here (do not scan for new entries)
    if is_trading_paused(market):
        logger.info(f"[Job] {market.upper()} new entries PAUSED by user. Evaluated open positions only.")
        return

    # Check daily profit target lock & max daily loss guardrail
    daily_pnl = engine.get_daily_realized_pnl(market)
    cfg_paper = engine.config.paper_trading
    target = cfg_paper.daily_profit_target_inr if market == "india" else cfg_paper.daily_profit_target_usd
    max_loss = cfg_paper.daily_max_loss_inr if market == "india" else cfg_paper.daily_max_loss_usd
    currency_sym = "₹" if market == "india" else "$"

    if daily_pnl >= target:
        logger.success(f"[Job] 🎯 {market.upper()} Daily Profit Target Reached ({currency_sym}{daily_pnl:,.2f} >= {currency_sym}{target:,.2f})! Preserving day's profit.")
        return

    if daily_pnl <= -max_loss:
        logger.warning(f"[Job] 🛑 {market.upper()} Daily Max Loss Limit Reached ({currency_sym}{daily_pnl:,.2f} <= -{currency_sym}{max_loss:,.2f})! Halting new entries.")
        return

    # 3. Scan for new trading signals
    portfolio = engine.get_portfolio_summary(market)
    cash = portfolio["cash"]
    port_val = portfolio["total_value"]

    for t_info in tickers_info:
        ticker = t_info["ticker"]
        if ticker not in snapshots:
            continue
        snap = snapshots[ticker]

        for strat in t_info["strategies"]:
            sig = sig_gen.generate_signal(
                snapshot=snap,
                strategy=strat,
                current_cash=cash,
                portfolio_val=port_val,
            )
            if sig:
                tg.send_message(format_trade_signal_telegram(sig))
                # Execute order
                order = engine.execute_signal(sig)
                if order:
                    tg.send_message(format_order_fill_telegram(order))
                    cash -= (order.filled_price * order.quantity)


def job_intraday_square_off(market: Literal["india", "us"]) -> None:
    """3:20 PM IST (India) / 3:55 PM EST (US): Mandatory square-off for intraday & scalping trades."""
    logger.info(f"[Job] Executing {market.upper()} intraday square-off...")
    engine = PaperTradingEngine()
    tg = TelegramNotifier()

    tickers = [t["ticker"] for t in get_active_tickers(market=market)]
    snapshots = fetch_india_batch(tickers) if market == "india" else fetch_us_batch(tickers)

    closed = engine.square_off_intraday(market, snapshots)
    for c in closed:
        tg.send_message(f"🔔 <b>Intraday Auto Square-Off:</b>\n{c}")


def job_eod_report(market: Literal["india", "us"]) -> None:
    """Delivers EOD report with learning autopsies via Telegram and Gmail."""
    logger.info(f"[Job] Generating {market.upper()} EOD report & running trade learning engine...")
    engine = PaperTradingEngine()
    summary = engine.get_portfolio_summary(market)

    from src.analyst.learning_engine import LearningEngine
    learning = LearningEngine()
    eod_learning = learning.run_daily_eod_learning(market)

    tg = TelegramNotifier()
    em = EmailNotifier()

    # Build Telegram EOD Report with Profit Goal and Mistake Lessons
    curr_sym = "₹" if market == "india" else "$"
    target_emoji = "🎯 Goal Achieved!" if eod_learning["target_met"] else "⏳ In Progress"
    report_text = (
        f"📊 <b>AEGIS {market.upper()} EOD REPORT & LEARNINGS</b>\n"
        f"──────────────────────────────\n"
        f"💰 <b>Daily Realized P&L:</b> <b>{'+' if eod_learning['total_pnl'] >= 0 else ''}{curr_sym}{eod_learning['total_pnl']:,.2f}</b>\n"
        f"🎯 <b>Daily Target:</b> {curr_sym}{eod_learning['daily_target']:,.2f} ({target_emoji})\n"
        f"📈 <b>Trades:</b> {eod_learning['total_trades']} (Wins: {eod_learning['wins']} | Losses: {eod_learning['losses']})\n"
        f"🏆 <b>Win Rate:</b> {eod_learning['win_rate_pct']}%\n"
        f"💼 <b>Total Portfolio Value:</b> {curr_sym}{summary['total_value']:,.2f}\n"
        f"──────────────────────────────\n"
    )

    if eod_learning["autopsies"]:
        report_text += "🧠 <b>EOD Loss Autopsies & Lessons Learned:</b>\n"
        for idx, a in enumerate(eod_learning["autopsies"][:3], 1):
            report_text += (
                f"<b>{idx}. {a.ticker}</b> ({a.strategy}) — PnL: {a.realized_pnl:+.2f} ({a.realized_pnl_pct:+.1f}%)\n"
                f"   • Flaw: <i>{a.failure_category}</i>\n"
                f"   • Lesson: <b>{a.prescriptive_lesson}</b>\n"
            )
        report_text += "──────────────────────────────\n"
        report_text += "<i>Lessons saved to trade playbook and injected into tomorrow's prompts.</i>"
    else:
        report_text += "✨ <i>Zero losing trades today! Risk execution was flawless.</i>"

    tg.send_message(report_text)
    email_html = format_eod_email_html(summary, [])
    em.send_email(f"Daily Portfolio & Learning Report — {market.upper()}", email_html)


def job_monthly_advisory() -> None:
    """1st of every month: Runs full financial advisory screening."""
    logger.info("[Job] Executing monthly advisory run...")
    recommender = AdvisoryRecommender()
    recs = recommender.run_monthly_advisory()
    tg = TelegramNotifier()
    tg.send_message(
        f"🌟 <b>Monthly Advisory Recommendations</b>\n"
        f"India Long-Term: {len(recs['india_long_term'])}\n"
        f"India Short-Term: {len(recs['india_short_term'])}\n"
        f"US Long-Term: {len(recs['us_long_term'])}\n"
        f"US Short-Term: {len(recs['us_short_term'])}\n"
        f"Review full portfolio via <code>python run_advisory.py --portfolio</code>"
    )
