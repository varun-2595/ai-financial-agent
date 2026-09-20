"""
Market hours utilities for NSE (India) and NYSE/NASDAQ (US).

Uses the `exchange_calendars` library for authoritative, multi-year holiday
calendars — no hardcoded holiday lists that expire annually.

DST handling:
  zoneinfo / America/New_York handles EST↔EDT automatically.
  NYSE open at 9:30 AM EST = 8:00 PM IST (Nov–Mar, EST UTC-5)
  NYSE open at 9:30 AM EDT = 7:00 PM IST (Mar–Nov, EDT UTC-4)
  The 1-hour IST shift is handled automatically by timezone conversion.
"""
from __future__ import annotations

from datetime import datetime, time, timedelta
from zoneinfo import ZoneInfo

import exchange_calendars as xcals

IST = ZoneInfo("Asia/Kolkata")
EST = ZoneInfo("America/New_York")

# ── Exchange hours (in local exchange time) ───────────────────────────────────

NSE_OPEN        = time(9, 15)
NSE_CLOSE       = time(15, 30)
NSE_SQUARE_OFF  = time(15, 20)   # intraday auto-close

NYSE_OPEN       = time(9, 30)
NYSE_CLOSE      = time(16, 0)
NYSE_SQUARE_OFF = time(15, 55)

# ── Exchange calendar objects (authoritative multi-year holiday data) ──────────
# Loaded once at import; exchange_calendars caches internally.
# India uses XBOM (Bombay / India national holidays) and US uses XNYS (NYSE)
_XNSE = xcals.get_calendar("XBOM")   # BSE/NSE India
_XNYS = xcals.get_calendar("XNYS")   # NYSE


def _is_weekday(dt: datetime) -> bool:
    return dt.weekday() < 5


def _nse_is_session(date_str: str) -> bool:
    """True if the given YYYY-MM-DD is a valid NSE trading session."""
    try:
        return _XNSE.is_session(date_str)
    except Exception:
        # Fallback: weekday only (better than crashing)
        return datetime.strptime(date_str, "%Y-%m-%d").weekday() < 5


def _nyse_is_session(date_str: str) -> bool:
    """True if the given YYYY-MM-DD is a valid NYSE trading session."""
    try:
        return _XNYS.is_session(date_str)
    except Exception:
        return datetime.strptime(date_str, "%Y-%m-%d").weekday() < 5


# ── NSE helpers ───────────────────────────────────────────────────────────────

def is_nse_trading_day(dt: datetime | None = None) -> bool:
    now = (dt or datetime.now(IST)).astimezone(IST)
    return _nse_is_session(now.strftime("%Y-%m-%d"))


def is_nse_open(dt: datetime | None = None) -> bool:
    now = (dt or datetime.now(IST)).astimezone(IST)
    if not _nse_is_session(now.strftime("%Y-%m-%d")):
        return False
    return NSE_OPEN <= now.time() < NSE_CLOSE


def is_nse_square_off_time(dt: datetime | None = None) -> bool:
    """True at/after 3:20 PM IST on NSE trading days only."""
    now = (dt or datetime.now(IST)).astimezone(IST)
    if not _nse_is_session(now.strftime("%Y-%m-%d")):
        return False
    return now.time() >= NSE_SQUARE_OFF


# ── NYSE helpers ──────────────────────────────────────────────────────────────

def is_nyse_trading_day(dt: datetime | None = None) -> bool:
    now = (dt or datetime.now(EST)).astimezone(EST)
    return _nyse_is_session(now.strftime("%Y-%m-%d"))


def is_nyse_open(dt: datetime | None = None) -> bool:
    now = (dt or datetime.now(EST)).astimezone(EST)
    if not _nyse_is_session(now.strftime("%Y-%m-%d")):
        return False
    return NYSE_OPEN <= now.time() < NYSE_CLOSE


def is_nyse_square_off_time(dt: datetime | None = None) -> bool:
    """True at/after 3:55 PM EST on NYSE trading days only."""
    now = (dt or datetime.now(EST)).astimezone(EST)
    if not _nyse_is_session(now.strftime("%Y-%m-%d")):
        return False
    return now.time() >= NYSE_SQUARE_OFF


# ── DST-aware IST conversions ─────────────────────────────────────────────────

def nyse_open_in_ist(date: datetime | None = None) -> datetime:
    """Today's NYSE 9:30 AM open expressed in IST (DST-aware)."""
    base = (date or datetime.now(EST)).astimezone(EST)
    open_est = base.replace(hour=9, minute=30, second=0, microsecond=0)
    return open_est.astimezone(IST)


def nyse_close_in_ist(date: datetime | None = None) -> datetime:
    """Today's NYSE 4:00 PM close expressed in IST (DST-aware)."""
    base = (date or datetime.now(EST)).astimezone(EST)
    close_est = base.replace(hour=16, minute=0, second=0, microsecond=0)
    return close_est.astimezone(IST)


# ── Dual-market status ────────────────────────────────────────────────────────

def market_status() -> dict:
    """Live status of both markets, IST + exchange local times."""
    now_ist = datetime.now(IST)
    now_est = datetime.now(EST)
    nyse_open_ist  = nyse_open_in_ist()
    nyse_close_ist = nyse_close_in_ist()
    dst_label = "EDT (UTC-4)" if now_est.dst() and now_est.dst().total_seconds() > 0 else "EST (UTC-5)"

    return {
        "NSE": {
            "status":        "OPEN" if is_nse_open() else "CLOSED",
            "local_time":    now_ist.strftime("%H:%M IST"),
            "opens_at_ist":  "09:15 IST",
            "closes_at_ist": "15:30 IST",
            "trading_day":   is_nse_trading_day(),
        },
        "NYSE": {
            "status":          "OPEN" if is_nyse_open() else "CLOSED",
            "local_time_est":  now_est.strftime("%H:%M EST/EDT"),
            "local_time_ist":  now_ist.strftime("%H:%M IST"),
            "opens_at_ist":    nyse_open_ist.strftime("%I:%M %p IST"),
            "closes_at_ist":   nyse_close_ist.strftime("%I:%M %p IST"),
            "dst_note":        dst_label,
            "trading_day":     is_nyse_trading_day(),
        },
    }


def todays_agenda() -> str:
    """Full dual-market trading agenda for today, expressed in IST."""
    now          = datetime.now(IST)
    nyse_open    = nyse_open_in_ist()
    nyse_close   = nyse_close_in_ist()
    nyse_pre     = nyse_open  - timedelta(minutes=30)
    nyse_sqoff   = nyse_close - timedelta(minutes=5)
    nyse_eod     = nyse_close + timedelta(minutes=30)

    nse_ok  = is_nse_trading_day()
    nyse_ok = is_nyse_trading_day()
    tz_label = "EDT" if (datetime.now(EST).dst() and datetime.now(EST).dst().total_seconds() > 0) else "EST"

    lines = [
        f"\n📅 TRADING AGENDA — {now.strftime('%A, %d %b %Y')}",
        "",
        f"  🇮🇳 NSE  {'✅ Trading Day' if nse_ok  else '🔴 Market Holiday'}",
        f"  🇺🇸 NYSE {'✅ Trading Day' if nyse_ok else '🔴 Market Holiday'}",
        "",
        "  TIME (IST)    EVENT",
        "  " + "─" * 54,
        "  05:30 AM      🔍 Universe screening (NSE 500 + S&P 500)",
        "  08:00 AM      📊 US overnight report delivery",
        "  09:00 AM      🇮🇳 India pre-market scan",
        "  09:15 AM      🟢 NSE OPENS",
        "  03:20 PM      🔔 India intraday square-off",
        "  03:30 PM      🔴 NSE CLOSES",
        "  03:45 PM      📊 India EOD report → Telegram + Gmail",
        f"  {nyse_pre.strftime('%I:%M %p')}      🇺🇸 US pre-market scan",
        f"  {nyse_open.strftime('%I:%M %p')}      🟢 NYSE OPENS  (9:30 AM {tz_label})",
        f"  {nyse_sqoff.strftime('%I:%M %p')}      🔔 US intraday square-off",
        f"  {nyse_close.strftime('%I:%M %p')}      🔴 NYSE CLOSES (4:00 PM {tz_label})",
        f"  {nyse_eod.strftime('%I:%M %p')}      📝 US EOD generated (delivered 8 AM)",
        "  " + "─" * 54,
        f"\n  Timezone: {tz_label} — NYSE open shifts by 1 hour between",
        "  summer (EDT, UTC-4) and winter (EST, UTC-5).",
        "  Holiday data: exchange_calendars (multi-year, authoritative)\n",
    ]
    return "\n".join(lines)


if __name__ == "__main__":
    import json
    print(todays_agenda())
    print(json.dumps(market_status(), indent=2))
