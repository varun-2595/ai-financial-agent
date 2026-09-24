"""
Statutory Fee Engine — accurately models Indian and US stock exchange regulatory friction,
brokerage, taxes, and adverse execution slippage.

Indian Market Statutory Friction:
    - Securities Transaction Tax (STT):
        • Equity Delivery (CNC): 0.10% (0.001) on both BUY and SELL sides.
        • Equity Intraday (MIS/Scalp): 0.025% (0.00025) on SELL side only (0% on BUY).
    - Exchange Turnover Charges: 0.00297% (0.0000297) on NSE.
    - SEBI Charges: ₹10 per crore (0.0000010 = 0.0001%).
    - Stamp Duty: 0.015% (0.00015) on BUY orders only.
    - GST: 18% levied on (Brokerage + Exchange Turnover Charges + SEBI Charges).

US Market Statutory Friction:
    - SEC Fee: $0.0000278 per $ of sell notional.
    - FINRA TAF: $0.000166 per share sold (capped at $8.30 per trade).
    - Clearing / Pass-through fee: $0.0005 per share.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Optional


@dataclass
class FeeSchedule:
    """Configurable statutory and regulatory fee schedule."""

    # ── Paper Trading Flat Rate ──────────────────────────────────────────
    paper_per_side_pct: float = 0.0005   # 0.05% flat per side for simplified paper accounting

    # ── Slippage ──────────────────────────────────────────────────────────
    slippage_pct: float = 0.0005         # 0.05% adverse slippage on fill prices

    # ── India / NSE Statutory Rates ──────────────────────────────────────
    india_stt_delivery_pct: float = 0.0010    # 0.10% on Delivery (both Buy and Sell)
    india_stt_intraday_sell_pct: float = 0.00025 # 0.025% on Intraday MIS (Sell side only)
    india_exchange_turnover_pct: float = 0.0000297 # 0.00297% on NSE
    india_sebi_turnover_pct: float = 0.0000010     # ₹10 per crore (0.0001%)
    india_stamp_duty_buy_pct: float = 0.00015      # 0.015% on Buy orders only
    india_gst_rate: float = 0.18                   # 18% on (Brokerage + Exchange + SEBI)

    # ── US Statutory & Regulatory Rates ──────────────────────────────────
    us_sec_sell_pct: float = 0.0000278             # SEC fee on sell notional
    us_finra_per_share: float = 0.000166           # FINRA TAF per share sold
    us_finra_cap: float = 8.30                     # FINRA TAF cap per trade
    us_clearing_per_share: float = 0.0005          # Clearing pass-through fee

    def compute_fees(
        self,
        market: Literal["india", "us"],
        direction: Literal["BUY", "SELL"],
        filled_price: float,
        quantity: int,
        is_paper: bool = True,
        strategy: Optional[str] = None,
    ) -> float:
        """
        Calculates exact statutory and regulatory transaction friction.
        """
        notional = filled_price * quantity
        if notional <= 0 or quantity <= 0:
            return 0.0

        if is_paper:
            return round(notional * self.paper_per_side_pct, 4)

        # ── Real / Live Indian Fee Calculation ────────────────────────────
        if market == "india":
            brokerage = min(20.0, notional * 0.0003)  # standard discount broker ₹20 cap
            exchange_turnover = notional * self.india_exchange_turnover_pct
            sebi_charges = notional * self.india_sebi_turnover_pct
            gst = (brokerage + exchange_turnover + sebi_charges) * self.india_gst_rate

            is_intraday = strategy in ("scalping", "intraday") if strategy else False

            if is_intraday:
                stt = (notional * self.india_stt_intraday_sell_pct) if direction == "SELL" else 0.0
            else:
                stt = notional * self.india_stt_delivery_pct

            stamp_duty = (notional * self.india_stamp_duty_buy_pct) if direction == "BUY" else 0.0

            total = brokerage + exchange_turnover + sebi_charges + gst + stt + stamp_duty
            return round(total, 4)

        # ── Real / Live US Fee Calculation ────────────────────────────────
        clearing = quantity * self.us_clearing_per_share
        if direction == "SELL":
            sec_fee = notional * self.us_sec_sell_pct
            finra_taf = min(quantity * self.us_finra_per_share, self.us_finra_cap)
            total = sec_fee + finra_taf + clearing
        else:
            total = clearing

        return round(total, 4)

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


DEFAULT_FEE_SCHEDULE = FeeSchedule()
