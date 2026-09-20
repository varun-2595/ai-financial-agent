"""
Master Scheduler Runner utilizing APScheduler.
Coordinates the entire dual-market daily schedule across IST and EST/EDT timezones.
"""
from __future__ import annotations

from zoneinfo import ZoneInfo
from apscheduler.schedulers.blocking import BlockingScheduler

from src.scheduler.jobs import (
    job_eod_report,
    job_intraday_square_off,
    job_market_intraday_scan,
    job_monthly_advisory,
    job_screen_universe,
)
from src.utils.logger import logger

IST = ZoneInfo("Asia/Kolkata")
EST = ZoneInfo("America/New_York")


def build_scheduler() -> BlockingScheduler:
    sched = BlockingScheduler()

    # 1. Daily Universe Screening at 05:30 AM IST
    sched.add_job(
        job_screen_universe,
        "cron",
        hour=5,
        minute=30,
        timezone=IST,
        id="daily_universe_screen",
    )

    # 2. India (NSE) Session
    # Intraday scan every 15 mins between 09:15 and 15:30 IST on weekdays
    sched.add_job(
        lambda: job_market_intraday_scan("india"),
        "cron",
        day_of_week="mon-fri",
        hour="9-15",
        minute="*/15",
        timezone=IST,
        id="india_intraday_scan",
    )
    # India Intraday Square-off at 15:20 IST
    sched.add_job(
        lambda: job_intraday_square_off("india"),
        "cron",
        day_of_week="mon-fri",
        hour=15,
        minute=20,
        timezone=IST,
        id="india_square_off",
    )
    # India EOD Report at 15:45 IST
    sched.add_job(
        lambda: job_eod_report("india"),
        "cron",
        day_of_week="mon-fri",
        hour=15,
        minute=45,
        timezone=IST,
        id="india_eod_report",
    )

    # 3. US (NYSE/NASDAQ) Session (uses America/New_York to auto-handle DST shifts)
    # Intraday scan every 15 mins between 09:30 and 16:00 EST on weekdays
    sched.add_job(
        lambda: job_market_intraday_scan("us"),
        "cron",
        day_of_week="mon-fri",
        hour="9-16",
        minute="*/15",
        timezone=EST,
        id="us_intraday_scan",
    )
    # US Intraday Square-off at 15:55 EST
    sched.add_job(
        lambda: job_intraday_square_off("us"),
        "cron",
        day_of_week="mon-fri",
        hour=15,
        minute=55,
        timezone=EST,
        id="us_square_off",
    )
    # US EOD Report (delivered at 08:00 AM IST next morning)
    sched.add_job(
        lambda: job_eod_report("us"),
        "cron",
        day_of_week="tue-sat",
        hour=8,
        minute=0,
        timezone=IST,
        id="us_overnight_eod_report",
    )

    # 4. Monthly Advisory Run (1st of month at 09:00 AM IST)
    sched.add_job(
        job_monthly_advisory,
        "cron",
        day=1,
        hour=9,
        minute=0,
        timezone=IST,
        id="monthly_advisory_run",
    )

    return sched


def run_scheduler() -> None:
    scheduler = build_scheduler()
    logger.info("=" * 60)
    logger.info("⏱️  AEGIS DUAL-MARKET SCHEDULER STARTED")
    logger.info("=" * 60)
    for j in scheduler.get_jobs():
        logger.info(f"  • Job: {j.id} -> {j.trigger}")
    logger.info("=" * 60)
    try:
        scheduler.start()
    except (KeyboardInterrupt, SystemExit):
        logger.info("Scheduler stopped by user.")
