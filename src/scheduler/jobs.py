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
    """3:20 PM IST (India) / 3:55 PM EST (US): Mandatory square-off for intraday trades."""
    logger.info(f"[Job] Executing {market.upper()} intraday square-off...")
    engine = PaperTradingEngine()
    tg = TelegramNotifier()

    tickers = [t["ticker"] for t in get_active_tickers(market=market)]
    snapshots = fetch_india_batch(tickers) if market == "india" else fetch_us_batch(tickers)

    closed = engine.square_off_intraday(market, snapshots)
    for c in closed:
        tg.send_message(f"🔔 <b>Intraday Auto Square-Off:</b>\n{c}")


def job_eod_report(market: Literal["india", "us"]) -> None:
    """Delivers EOD report via Telegram and Gmail."""
    logger.info(f"[Job] Generating {market.upper()} EOD report...")
    engine = PaperTradingEngine()
    summary = engine.get_portfolio_summary(market)

    tg = TelegramNotifier()
    em = EmailNotifier()

    tg.send_message(format_eod_report_telegram(summary, []))
    email_html = format_eod_email_html(summary, [])
    em.send_email(f"Daily Portfolio Summary — {market.upper()}", email_html)


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
