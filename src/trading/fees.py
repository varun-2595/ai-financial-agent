"""
Fee Schedule — simulates realistic brokerage, exchange, and regulatory fees.

Paper trading default: flat 0.05 % per side (approximates discount broker + exchange charges).
NSE real fees breakdown (for future live trading reference):
    - STT (Securities Transaction Tax): 0.1 % on sell side only (equity delivery)
    - SEBI turnover fee: 0.0001 %
    - NSE exchange transaction charges: 0.00335 %
    - IPFT: 0.0001 %
    - Stamp duty: 0.015 % on buy side only
    - GST 18 % on (brokerage + exchange charges)

US real fees breakdown (for future live trading reference):
    - SEC fee: $0.0000278 per $ of sell-side notional
    - FINRA TAF: $0.000145 per share sold (capped at $7.27 per trade)
    - Exchange fees: ~$0.0030 per share (varies)
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal


@dataclass
class FeeSchedule:
    """
    Configurable fee rates for paper and live trading.

    All rates are expressed as a decimal fraction of notional value unless
    otherwise noted (e.g. per_share_usd).

    For paper trading only `paper_per_side_pct` is used. Live rates are
    broken out individually to support accurate real-money cost accounting.
    """

    # ── Paper trading (simplified flat rate) ──────────────────────────────
    paper_per_side_pct: float = 0.0005   # 0.05 % per side (entry AND exit)

    # ── Slippage ──────────────────────────────────────────────────────────
    slippage_pct: float = 0.0005         # 0.05 % applied to fill price on top of fees

    # ── India / NSE real-money rates (for reference; not used in paper) ──
    india_stt_sell_pct: float = 0.001    # 0.10 % on sell notional
    india_exchange_pct: float = 0.0000335
    india_sebi_pct: float = 0.000001
    india_stamp_buy_pct: float = 0.00015 # 0.015 % on buy notional
    india_gst_rate: float = 0.18         # 18 % on (brokerage + exchange)

    # ── US / NYSE-NASDAQ real-money rates ─────────────────────────────────
    us_sec_sell_pct: float = 0.0000278   # SEC fee on sell notional
    us_finra_per_share: float = 0.000145 # FINRA TAF per share sold
    us_finra_cap: float = 7.27           # FINRA TAF cap per transaction


    # ── Public API ────────────────────────────────────────────────────────

    def compute_fees(
        self,
        market: Literal["india", "us"],
        direction: Literal["BUY", "SELL"],
        filled_price: float,
        quantity: int,
        is_paper: bool = True,
    ) -> float:
        """
        Return total fees in local currency for a single fill.

        For paper trading this is simply:
            notional × paper_per_side_pct

        This is charged on BOTH entry (BUY) and exit (SELL).
        """
        notional = filled_price * quantity
        if notional <= 0 or quantity <= 0:
            return 0.0

        if is_paper:
            return round(notional * self.paper_per_side_pct, 4)

        # ── Live India fees (for future use) ──────────────────────────────
        if market == "india":
            brokerage = 0.0         # most discount brokers: flat ₹20/trade or 0
            exchange = notional * self.india_exchange_pct
            sebi = notional * self.india_sebi_pct
            gst = (brokerage + exchange) * self.india_gst_rate

            if direction == "BUY":
                stamp = notional * self.india_stamp_buy_pct
                return round(brokerage + exchange + sebi + gst + stamp, 4)
            else:  # SELL
                stt = notional * self.india_stt_sell_pct
                return round(brokerage + exchange + sebi + gst + stt, 4)

        # ── Live US fees (for future use) ─────────────────────────────────
        if direction == "SELL":
            sec = notional * self.us_sec_sell_pct
            finra = min(quantity * self.us_finra_per_share, self.us_finra_cap)
            return round(sec + finra, 4)
        return 0.0   # US buy side: no regulatory fees (exchange fees billed by broker)

    def apply_slippage(
        self,
        price: float,
        direction: Literal["BUY", "SELL"],
    ) -> float:
        """
        Return the fill price after applying realistic market-impact slippage.
        BUY fills slightly above signal price; SELL fills slightly below.
        """
        if direction == "BUY":
            return round(price * (1 + self.slippage_pct), 6)
        return round(price * (1 - self.slippage_pct), 6)


# ── Module-level default instance ─────────────────────────────────────────────
DEFAULT_FEE_SCHEDULE = FeeSchedule()
