"""
Sell and Exit Signal Engine for Financial Advisory Holdings.
Enforces scheduled weekly portfolio reviews and triggers:
  - Stop Loss hit (-8% short-term, -20% long-term)
  - Target Profit hit (+20% short-term, +80% long-term trim alert)
  - Time-based expiration (e.g. short-term held > 180 days with no momentum)
  - Thesis invalidation
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Literal

from src.data.models import StockSnapshot
from src.db.advisory_store import get_active_advisory_holdings, log_advisory_alert, _conn
from src.utils.logger import logger


@dataclass
class AdvisorySellAlert:
    holding_id: int
    ticker: str
    action: Literal["SELL", "TRIM", "HOLD"]
    reason: str
    current_price: float
    entry_price: float
    return_pct: float
    days_held: int


class AdvisorySellEngine:
    def scan_holdings(self, latest_snapshots: dict[str, StockSnapshot]) -> list[AdvisorySellAlert]:
        """
        Scans all active advisory holdings against live market prices.
        Generates actionable sell or trim alerts.
        """
        alerts = []
        holdings = get_active_advisory_holdings()

        for h in holdings:
            hid = h["id"]
            ticker = h["ticker"]
            entry_p = h["entry_price"]
            sl = h["stop_loss"]
            tgt = h["target_price"]
            horizon = h["horizon"]
            rec_at = datetime.fromisoformat(h["recommended_at"])

            if ticker not in latest_snapshots:
                continue

            curr_p = latest_snapshots[ticker].current_price
            return_pct = round((curr_p - entry_p) / entry_p * 100, 2)
            days_held = (datetime.now(timezone.utc) - rec_at).days

            # Update holding current price in DB
            with _conn() as conn:
                conn.execute("UPDATE advisory_holdings SET current_price = ? WHERE id = ?", (curr_p, hid))
                conn.commit()

            alert_action: Literal["SELL", "TRIM", "HOLD"] = "HOLD"
            reason = ""

            # 1. Stop-Loss Trigger
            if sl and curr_p <= sl:
                alert_action = "SELL"
                reason = f"Hard Stop-Loss breached ({curr_p:.2f} <= {sl:.2f}, Return: {return_pct:+.2f}%)"

            # 2. Target Profit Trigger
            elif tgt and curr_p >= tgt:
                if horizon == "short_term":
                    alert_action = "SELL"
                    reason = f"Short-term target achieved (+20% gain, Return: {return_pct:+.2f}%)"
                else:
                    alert_action = "TRIM"
                    reason = f"Long-term compounder reached +80% trim milestone. Suggest locking in 50% profits."

            # 3. Time Stop-Loss for Short-Term picks (held > 180 days with flat/negative return)
            elif horizon == "short_term" and days_held >= 180 and return_pct < 5.0:
                alert_action = "SELL"
                reason = f"Time-stop triggered (Held {days_held} days with stagnant momentum: {return_pct:+.2f}%)"

            if alert_action != "HOLD":
                log_advisory_alert(hid, ticker, alert_action, reason, curr_p)
                alert = AdvisorySellAlert(
                    holding_id=hid,
                    ticker=ticker,
                    action=alert_action,
                    reason=reason,
                    current_price=curr_p,
                    entry_price=entry_p,
                    return_pct=return_pct,
                    days_held=days_held,
                )
                logger.warning(f"[Advisory Sell Engine] 🚨 {alert.action} {alert.ticker}: {alert.reason}")
                alerts.append(alert)

        return alerts
