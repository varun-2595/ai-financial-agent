"""
Message formatting utilities for Telegram (HTML formatted) and Email (rich HTML tables).
Formats trade signals, fills, EOD summaries, and advisory recommendations.
"""
from __future__ import annotations

from typing import Any, Sequence

from src.data.models import Order, TradeSignal


def format_trade_signal_telegram(sig: TradeSignal) -> str:
    sym = "₹" if sig.market == "india" else "$"
    flag = "🇮🇳" if sig.market == "india" else "🇺🇸"
    return f"""
🔥 <b>NEW TRADE SIGNAL</b> {flag}

<b>Ticker:</b> <code>{sig.ticker}</code>
<b>Action:</b> <b>{sig.direction}</b> ({sig.strategy.upper()})
<b>Quantity:</b> {sig.quantity} shares
<b>Entry Price:</b> {sym}{sig.entry_price:,.2f}
<b>Stop Loss:</b> {sym}{sig.stop_loss:,.2f}
<b>Target:</b> {sym}{sig.target_price:,.2f}
<b>Confidence:</b> {int(sig.confidence * 100)}%

<b>Reasoning:</b>
<i>{sig.reasoning}</i>
""".strip()


def format_order_fill_telegram(order: Order) -> str:
    flag = "🇮🇳" if order.market == "india" else "🇺🇸"
    return f"""
🚀 <b>ORDER EXECUTED (PAPER)</b> {flag}

<b>Order ID:</b> <code>{order.order_id}</code>
<b>Ticker:</b> <code>{order.ticker}</code>
<b>Direction:</b> {order.direction}
<b>Quantity:</b> {order.quantity}
<b>Filled Price:</b> {order.filled_price}
<b>Strategy:</b> {order.strategy.capitalize()}
""".strip()


def format_eod_report_telegram(summary: dict[str, Any], closed_trades: Sequence[str]) -> str:
    flag = "🇮🇳" if summary["market"] == "india" else "🇺🇸"
    sym = summary["currency"]

    trades_text = "No closed positions today."
    if closed_trades:
        trades_text = "\n".join([f"• {t}" for t in closed_trades[:10]])

    return f"""
📊 <b>EOD TRADING REPORT</b> {flag}

<b>Cash Available:</b> {sym}{summary['cash']:,.2f}
<b>Invested Value:</b> {sym}{summary['invested']:,.2f}
<b>Total Portfolio:</b> {sym}{summary['total_value']:,.2f}
<b>Open Positions:</b> {summary['open_positions_count']}

<b>Today's Exits / Realized Trades:</b>
{trades_text}
""".strip()


def format_eod_email_html(summary: dict[str, Any], closed_trades: Sequence[str]) -> str:
    sym = summary["currency"]
    market_name = "India (NSE)" if summary["market"] == "india" else "US (NYSE/NASDAQ)"

    trades_rows = "<tr><td colspan='4'>No closed trades today.</td></tr>"
    if closed_trades:
        trades_rows = "".join([f"<tr><td colspan='4'>{t}</td></tr>" for t in closed_trades])

    positions_rows = "<tr><td colspan='5'>No open positions.</td></tr>"
    if summary.get("positions"):
        positions_rows = "".join([
            f"<tr><td>{p['ticker']}</td><td>{p['quantity']}</td><td>{sym}{p['avg_cost']:.2f}</td><td>{sym}{p['current_price'] or p['avg_cost']:.2f}</td><td>{p['strategy']}</td></tr>"
            for p in summary["positions"]
        ])

    return f"""
<!DOCTYPE html>
<html>
<head>
<style>
  body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; color: #333; line-height: 1.6; }}
  .card {{ background: #f8fafc; border: 1px solid #e2e8f0; border-radius: 8px; padding: 16px; margin-bottom: 20px; }}
  table {{ width: 100%; border-collapse: collapse; margin-top: 12px; }}
  th, td {{ padding: 8px 12px; border: 1px solid #cbd5e1; text-align: left; font-size: 14px; }}
  th {{ background: #f1f5f9; }}
  .header {{ font-size: 20px; font-weight: bold; margin-bottom: 12px; }}
</style>
</head>
<body>
  <div class="header">Daily Portfolio Report — {market_name}</div>
  <div class="card">
    <b>Cash Balance:</b> {sym}{summary['cash']:,.2f} | <b>Invested:</b> {sym}{summary['invested']:,.2f} | <b>Total Value:</b> {sym}{summary['total_value']:,.2f}
  </div>

  <h3>Open Positions ({summary['open_positions_count']})</h3>
  <table>
    <thead><tr><th>Ticker</th><th>Qty</th><th>Avg Cost</th><th>Current Price</th><th>Strategy</th></tr></thead>
    <tbody>{positions_rows}</tbody>
  </table>

  <h3>Today's Trade Exits</h3>
  <table>
    <thead><tr><th colspan='4'>Details</th></tr></thead>
    <tbody>{trades_rows}</tbody>
  </table>
</body>
</html>
""".strip()
