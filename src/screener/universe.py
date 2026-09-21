"""
Universe fetcher — pulls the full investable universe for India (NSE 500)
and US (S&P 500).

Fixes applied:
  - Cache universe in-memory with 1-day TTL
  - Corrected ticker typos (BAJAJ-AUTO.NS, removed defunct tickers)
  - Cleaned up unused imports
"""
from __future__ import annotations

import io
import time

import pandas as pd
import requests

from src.utils.logger import logger

_NSE500_CSV_URL = (
    "https://www.niftyindices.com/IndexConstituent/ind_nifty500list.csv"
)

_NSE_FALLBACK = [
    "RELIANCE.NS", "TCS.NS", "INFY.NS", "HDFCBANK.NS", "ICICIBANK.NS",
    "HINDUNILVR.NS", "SBIN.NS", "BAJFINANCE.NS", "BHARTIARTL.NS", "KOTAKBANK.NS",
    "WIPRO.NS", "LT.NS", "AXISBANK.NS", "ASIANPAINT.NS", "MARUTI.NS",
    "SUNPHARMA.NS", "TITAN.NS", "ULTRACEMCO.NS", "NESTLEIND.NS", "POWERGRID.NS",
    "NTPC.NS", "ONGC.NS", "TATAMOTORS.NS", "ADANIENT.NS", "JSWSTEEL.NS",
    "TATASTEEL.NS", "TECHM.NS", "HCLTECH.NS", "DIVISLAB.NS", "DRREDDY.NS",
    "CIPLA.NS", "BAJAJ-AUTO.NS", "EICHERMOT.NS", "BRITANNIA.NS", "HAVELLS.NS",
    "PIDILITIND.NS", "BERGEPAINT.NS", "MUTHOOTFIN.NS", "BALKRISIND.NS",
    "PERSISTENT.NS", "LTIM.NS", "POLYCAB.NS", "DIXON.NS", "TRENT.NS",
    "ZOMATO.NS", "NYKAA.NS", "INDHOTEL.NS", "IRCTC.NS",
    # ETFs
    "NIFTYBEES.NS", "BANKBEES.NS", "GOLDBEES.NS", "ITBEES.NS", "JUNIORBEES.NS",
]

_NSE_MID_SMALL_CAPS = [
    # High-beta momentum, defence, railway, green energy, electronics
    "SUZLON.NS", "RVNL.NS", "IREDA.NS", "BSE.NS", "COCHINSHIP.NS", "MAZDOCK.NS",
    "KALYANKJIL.NS", "POLYCAB.NS", "DIXON.NS", "HUDCO.NS", "NBCC.NS", "RAILTEL.NS",
    "IRFC.NS", "MOTHERSON.NS", "TITAGARH.NS", "TEJASNET.NS", "HFCL.NS", "RITES.NS",
    "GPIL.NS", "HEG.NS", "GRAPHITE.NS", "CDSL.NS", "ANGELONE.NS", "MCX.NS",
    "EXIDEIND.NS", "AMARAJABAT.NS", "FEDERALBNK.NS", "IDFCFIRSTB.NS", "PNB.NS",
    "IOB.NS", "UCOBANK.NS", "CENTRALBK.NS", "ENGINERSIN.NS", "SJVN.NS", "NHPC.NS",
    "TATACHEM.NS", "TATAPOWER.NS", "VOLTAS.NS", "DEEPAKNTR.NS", "TATAELXSI.NS",
    "KPITTECH.NS", "COFORGE.NS", "MPHASIS.NS", "PERSISTENT.NS", "CYIENT.NS",
    "ZOMATO.NS", "PAYTM.NS", "POLICYBZR.NS", "DELHIVERY.NS", "NYKAA.NS",
    "TRENT.NS", "ABFRL.NS", "DMART.NS", "DEVYANI.NS", "SAPPHIRE.NS",
]

_SP500_WIKI_URL = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"

_SP500_FALLBACK = [
    "AAPL", "MSFT", "NVDA", "GOOGL", "AMZN", "META", "TSLA", "BRK-B", "JPM",
    "V", "UNH", "XOM", "JNJ", "MA", "PG", "HD", "AVGO", "CVX", "MRK", "ABBV",
    "COST", "PEP", "KO", "TMO", "ADBE", "ACN", "CRM", "MCD", "NKE", "LIN",
    "DHR", "ORCL", "CSCO", "INTC", "AMD", "QCOM", "TXN", "AMGN", "CAT",
    "GS", "BA", "HON", "SYK", "SPGI", "BLK", "AXP", "DE", "RTX", "NOW",
    "ISRG", "REGN", "VRTX", "CI", "CB", "ZTS", "ADI", "MCHP", "KLAC", "AMAT",
    "LRCX", "SNPS", "CDNS", "PANW", "CRWD", "FTNT", "NET", "DDOG", "SNOW",
    # ETFs
    "SPY", "QQQ", "VTI", "GLD", "XLK", "XLF", "XLE", "XLV", "XLP", "XLI",
]

_US_MID_SMALL_CAPS = [
    # High-momentum growth, AI, crypto, fintech, quantum, space
    "PLTR", "SOFI", "MARA", "RIOT", "COIN", "HOOD", "ARM", "SMCI", "AFRM", "CELH",
    "DKNG", "APP", "RBLX", "IONQ", "RIVN", "LCID", "HIMS", "PATH", "UPST", "DUOL",
    "MSTR", "CLSK", "SYM", "JOBY", "ACHR", "ASTS", "RGTI", "QUBT", "BBAI", "SOUN",
    "PLUG", "FCEL", "RUN", "ENPH", "FSLR", "SEDG", "CHPT", "BLNK", "QS",
    "CRWD", "NET", "DDOG", "SNOW", "ZS", "MDB", "CFLT", "IOT", "S", "GTLB",
]

# ── Apple Ecosystem / Supply-Chain Plays ─────────────────────────────────────
# User-curated list of stocks tied to Apple's supply chain:
# chip fab, RF modules, memory, sensors, contract manufacturing, optics.
# Sourced from research — suitable for both daily trading and monthly advisory.
_APPLE_ECOSYSTEM_US = [
    # Chip Fabrication & Design
    "TSM",   # TSMC — fabricates A-series/M-series chips
    "AVGO",  # Broadcom — RF front-end & custom ASICs
    "QCOM",  # Qualcomm — modem & 5G connectivity (iPhone cellular)
    "TXN",   # Texas Instruments — analog & power management
    "ADI",   # Analog Devices — mixed-signal & power chips
    "NXPI",  # NXP Semiconductors — NFC (Apple Pay) & secure element
    "ON",    # ON Semiconductor — power ICs & image sensors
    # RF / Wireless Front-End
    "QRVO",  # Qorvo — RF amplifiers & wireless front-end
    "SWKS",  # Skyworks Solutions — RF front-end for cellular
    # Memory & Storage
    "MU",    # Micron — DRAM & NAND flash
    "WDC",   # Western Digital — NAND flash & storage
    # Optics / Sensing / Laser
    "LITE",  # Lumentum — VCSEL lasers for Face ID & 3D sensing
    "COHR",  # Coherent Corp — photonics for Face ID & data links
    "SONY",  # Sony Group — camera image sensors for iPhone
    # Display / Glass
    "GLW",   # Corning — Gorilla Glass / Ceramic Shield
    "CRUS",  # Cirrus Logic — audio codecs & haptic/battery ICs
    # Semiconductor Equipment
    "AMAT",  # Applied Materials — fab equipment used by Apple's chip suppliers
    # Connectors & PCB
    "APH",   # Amphenol — electrical, fiber-optic & high-speed connectors
    "STM",   # STMicroelectronics — gyro/accelerometer sensors & power ICs
    # Contract Manufacturing
    "JBL",   # Jabil — enclosures & sub-assemblies (incl. AirPods)
]

_UNIVERSE_CACHE: dict[str, tuple[list[str], float]] = {}
_UNIVERSE_TTL = 86400  # 24 hours


def fetch_nse500(use_cache: bool = True) -> list[str]:
    now = time.time()
    if use_cache and "nse500" in _UNIVERSE_CACHE:
        tickers, ts = _UNIVERSE_CACHE["nse500"]
        if now - ts < _UNIVERSE_TTL:
            return tickers

    logger.info("[Universe] Fetching NSE 500 constituent list...")
    try:
        headers = {
            "User-Agent": (
                "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0 Safari/537.36"
            ),
            "Referer": "https://www.niftyindices.com/",
        }
        resp = requests.get(_NSE500_CSV_URL, headers=headers, timeout=15)
        resp.raise_for_status()

        df = pd.read_csv(io.StringIO(resp.text))
        symbol_col = next(
            (c for c in df.columns if "symbol" in c.lower()), None
        )
        if symbol_col is None:
            raise ValueError(f"No 'Symbol' column found. Columns: {list(df.columns)}")

        tickers = [f"{s.strip()}.NS" for s in df[symbol_col].dropna().unique()]
        combined = list(dict.fromkeys(tickers + _NSE_MID_SMALL_CAPS))
        logger.success(f"[Universe] NSE Universe: fetched {len(combined)} tickers (including mid & small caps)")
        _UNIVERSE_CACHE["nse500"] = (combined, now)
        return combined

    except Exception as exc:
        logger.warning(
            f"[Universe] NSE fetch failed ({exc}). "
            f"Using fallback list of {len(_NSE_FALLBACK) + len(_NSE_MID_SMALL_CAPS)} tickers."
        )
        combined_fallback = list(dict.fromkeys(_NSE_FALLBACK + _NSE_MID_SMALL_CAPS))
        _UNIVERSE_CACHE["nse500"] = (combined_fallback, now)
        return combined_fallback


def fetch_sp500(use_cache: bool = True) -> list[str]:
    now = time.time()
    if use_cache and "sp500" in _UNIVERSE_CACHE:
        tickers, ts = _UNIVERSE_CACHE["sp500"]
        if now - ts < _UNIVERSE_TTL:
            return tickers

    logger.info("[Universe] Fetching US constituent list...")
    try:
        tables = pd.read_html(_SP500_WIKI_URL, attrs={"id": "constituents"})
        df = tables[0]

        symbol_col = next(
            (c for c in df.columns if "symbol" in c.lower() or "ticker" in c.lower()),
            None,
        )
        if symbol_col is None:
            raise ValueError(f"No symbol column found. Columns: {list(df.columns)}")

        tickers = [
            s.strip().replace(".", "-")
            for s in df[symbol_col].dropna().unique()
        ]
        combined = list(dict.fromkeys(tickers + _US_MID_SMALL_CAPS + _APPLE_ECOSYSTEM_US))
        logger.success(f"[Universe] US Universe: {len(combined)} tickers (SP500 + growth + Apple ecosystem)")
        _UNIVERSE_CACHE["sp500"] = (combined, now)
        return combined

    except Exception as exc:
        logger.warning(
            f"[Universe] US fetch failed ({exc}). "
            f"Using fallback of {len(_SP500_FALLBACK) + len(_US_MID_SMALL_CAPS) + len(_APPLE_ECOSYSTEM_US)} tickers."
        )
        combined_fallback = list(dict.fromkeys(_SP500_FALLBACK + _US_MID_SMALL_CAPS + _APPLE_ECOSYSTEM_US))
        _UNIVERSE_CACHE["sp500"] = (combined_fallback, now)
        return combined_fallback


def fetch_full_universe() -> dict[str, list[str]]:
    india = fetch_nse500()
    us = fetch_sp500()
    return {"india": india, "us": us}


def get_apple_ecosystem() -> list[str]:
    """Returns the curated Apple supply-chain ecosystem ticker list for advisory or focused scans."""
    return list(_APPLE_ECOSYSTEM_US)


def get_rotated_universe(market: Literal["india", "us"], max_candidates: int = 40) -> list[str]:
    """
    Returns a dynamically rotated candidate pool based on day-of-year.
    Guarantees Apple-ecosystem stocks appear in the US rotation every few days.
    """
    from datetime import datetime, timezone
    all_tickers = fetch_nse500() if market == "india" else fetch_sp500()
    day_offset = int(datetime.now(timezone.utc).strftime("%j"))  # day of year 1-366
    total = len(all_tickers)
    if total <= max_candidates:
        return all_tickers

    # Shift start index daily by prime step 17 to cycle across mid/small caps
    start_idx = (day_offset * 17) % total
    rotated = all_tickers[start_idx:] + all_tickers[:start_idx]
    base = rotated[:max_candidates]

    # Every 3 days, inject a fresh batch of Apple ecosystem picks into the rotation
    if market == "us" and (day_offset % 3) == 0:
        eco_batch = _APPLE_ECOSYSTEM_US[(day_offset // 3) % len(_APPLE_ECOSYSTEM_US):][:6]
        base = list(dict.fromkeys(eco_batch + base))[:max_candidates]

    return base

