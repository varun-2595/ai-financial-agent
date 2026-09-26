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


def format_eod_email_html(
    summary: dict[str, Any],
    closed_trades: Sequence[dict[str, Any] | str] | None = None,
    eod_learning: dict[str, Any] | None = None,
) -> str:
    market = summary.get("market", "india")
    sym = summary.get("currency", "₹" if market == "india" else "$")
    market_name = "🇮🇳 India (NSE / BSE)" if market == "india" else "🇺🇸 US (NYSE / NASDAQ)"
    flag = "🇮🇳" if market == "india" else "🇺🇸"

    cash = summary.get("cash", 0.0)
    margin = summary.get("reserved_margin", summary.get("invested", 0.0))
    nav = summary.get("nav", summary.get("total_value", cash + margin))
    initial_cap = summary.get("initial_capital", nav)
    total_ret_pct = summary.get("total_return_pct", 0.0)
    unrealized_pnl = summary.get("unrealized_pnl", 0.0)
    open_count = summary.get("open_positions_count", len(summary.get("positions", [])))

    # Learning & daily metrics
    realized_pnl = 0.0
    daily_target = 1000.0 if market == "india" else 150.0
    target_met = False
    total_trades = 0
    wins = 0
    losses = 0
    win_rate = 0.0
    autopsies = []

    if eod_learning:
        realized_pnl = float(eod_learning.get("total_pnl", 0.0))
        daily_target = float(eod_learning.get("daily_target", daily_target))
        target_met = bool(eod_learning.get("target_met", False))
        total_trades = int(eod_learning.get("total_trades", 0))
        wins = int(eod_learning.get("wins", 0))
        losses = int(eod_learning.get("losses", 0))
        win_rate = float(eod_learning.get("win_rate_pct", 0.0))
        autopsies = eod_learning.get("autopsies", [])

    # Color helpers
    pnl_color = "#10b981" if realized_pnl >= 0 else "#ef4444"
    unrealized_color = "#10b981" if unrealized_pnl >= 0 else "#ef4444"
    target_badge = (
        '<span style="background-color: #dcfce7; color: #15803d; padding: 4px 10px; border-radius: 9999px; font-weight: 600; font-size: 12px;">🎯 Target Achieved</span>'
        if target_met
        else '<span style="background-color: #fef3c7; color: #b45309; padding: 4px 10px; border-radius: 9999px; font-weight: 600; font-size: 12px;">⏳ In Progress</span>'
    )

    # 1. Open Positions Rows
    positions_rows = '<tr><td colspan="7" style="text-align: center; color: #64748b; padding: 18px;">No open positions held.</td></tr>'
    positions = summary.get("positions", [])
    if positions:
        pos_list = []
        for p in positions:
            upnl = p.get("unrealized_pnl", 0.0)
            upnl_pct = p.get("return_pct", 0.0)
            u_color = "#10b981" if upnl >= 0 else "#ef4444"
            dir_badge = (
                '<span style="background-color: #dbeafe; color: #1d4ed8; padding: 2px 6px; border-radius: 4px; font-weight: 600; font-size: 11px;">BUY</span>'
                if p.get("direction", "LONG").upper() in ("BUY", "LONG")
                else '<span style="background-color: #fef2f2; color: #b91c1c; padding: 2px 6px; border-radius: 4px; font-weight: 600; font-size: 11px;">SELL</span>'
            )
            pos_list.append(f"""
            <tr>
              <td style="font-weight: 600; color: #0f172a;">{p.get('ticker')}</td>
              <td>{dir_badge}</td>
              <td style="color: #475569; font-size: 12px; text-transform: uppercase;">{p.get('strategy', 'intraday')}</td>
              <td>{p.get('quantity')}</td>
              <td>{sym}{p.get('avg_cost', 0.0):,.2f}</td>
              <td>{sym}{p.get('current_price', p.get('avg_cost', 0.0)):,.2f}</td>
              <td style="font-weight: 600; color: {u_color};">{'+' if upnl >= 0 else ''}{sym}{upnl:,.2f} ({'+' if upnl_pct >= 0 else ''}{upnl_pct:.2f}%)</td>
            </tr>
            """)
        positions_rows = "".join(pos_list)

    # 2. Closed Trades Rows
    trades_rows = '<tr><td colspan="7" style="text-align: center; color: #64748b; padding: 18px;">No closed trades recorded today.</td></tr>'
    if closed_trades:
        trade_list = []
        for t in closed_trades:
            if isinstance(t, dict):
                r_pnl = t.get("realized_pnl", 0.0)
                r_pct = t.get("return_pct", 0.0)
                t_color = "#10b981" if r_pnl >= 0 else "#ef4444"
                dur_sec = t.get("holding_period_seconds", 0.0)
                dur_str = f"{int(dur_sec//60)}m" if dur_sec >= 60 else f"{int(dur_sec)}s"
                dir_badge = (
                    '<span style="background-color: #dbeafe; color: #1d4ed8; padding: 2px 6px; border-radius: 4px; font-weight: 600; font-size: 11px;">BUY</span>'
                    if str(t.get("direction", "BUY")).upper() in ("BUY", "LONG")
                    else '<span style="background-color: #fef2f2; color: #b91c1c; padding: 2px 6px; border-radius: 4px; font-weight: 600; font-size: 11px;">SELL</span>'
                )
                driver = t.get("pm_driver") or t.get("exit_reason") or "Rule Exit"
                trade_list.append(f"""
                <tr>
                  <td style="font-weight: 600; color: #0f172a;">{t.get('symbol', 'N/A')}</td>
                  <td>{dir_badge}</td>
                  <td style="color: #475569; font-size: 12px; text-transform: uppercase;">{t.get('strategy', 'scalping')}</td>
                  <td>{sym}{t.get('entry_price', 0.0):,.2f} &rarr; {sym}{t.get('exit_price', 0.0):,.2f}</td>
                  <td style="font-weight: 600; color: {t_color};">{'+' if r_pnl >= 0 else ''}{sym}{r_pnl:,.2f} ({'+' if r_pct >= 0 else ''}{r_pct:.2f}%)</td>
                  <td style="color: #64748b; font-size: 12px;">{dur_str}</td>
                  <td style="color: #334155; font-size: 12px; max-width: 200px;"><i>{driver[:80]}</i></td>
                </tr>
                """)
            else:
                trade_list.append(f"<tr><td colspan='7' style='padding: 8px 12px;'>{t}</td></tr>")
        trades_rows = "".join(trade_list)

    # 3. Autopsies / Lessons Learned Block
    autopsy_html = ""
    if autopsies:
        autopsy_cards = []
        for a in autopsies[:3]:
            ticker = getattr(a, "ticker", "N/A")
            strat = getattr(a, "strategy", "intraday")
            pnl = getattr(a, "realized_pnl", 0.0)
            flaw = getattr(a, "failure_category", "Execution Risk")
            lesson = getattr(a, "prescriptive_lesson", "Follow disciplined stop-loss rules.")
            autopsy_cards.append(f"""
            <div style="background-color: #fff1f2; border: 1px solid #fecdd3; border-radius: 6px; padding: 12px; margin-bottom: 8px;">
              <div style="display: flex; justify-content: space-between; font-weight: 600; color: #9f1239; margin-bottom: 4px;">
                <span>{ticker} ({strat}) &bull; Loss: {sym}{pnl:,.2f}</span>
                <span style="font-size: 12px; text-transform: uppercase; background: #ffe4e6; padding: 2px 6px; border-radius: 4px;">{flaw}</span>
              </div>
              <div style="color: #881337; font-size: 13px;">
                <b>Lesson Learned:</b> {lesson}
              </div>
            </div>
            """)
        autopsy_html = f"""
        <div style="margin-top: 24px;">
          <h3 style="color: #0f172a; margin-bottom: 8px; font-size: 16px;">🧠 Trade Autopsy & Prescriptive Lessons Learned</h3>
          {''.join(autopsy_cards)}
        </div>
        """
    else:
        autopsy_html = """
        <div style="margin-top: 24px; background-color: #f0fdf4; border: 1px solid #bbf7d0; border-radius: 6px; padding: 14px; color: #166534; font-size: 14px;">
          ✨ <b>Flawless Session:</b> Zero losing trades recorded today. Risk guardrails and entry filters performed perfectly.
        </div>
        """

    return f"""
<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<style>
  body {{
    font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Helvetica, Arial, sans-serif;
    color: #1e293b;
    background-color: #f8fafc;
    margin: 0;
    padding: 24px;
    line-height: 1.5;
  }}
  .container {{
    max-width: 780px;
    margin: 0 auto;
    background: #ffffff;
    border: 1px solid #e2e8f0;
    border-radius: 10px;
    padding: 28px;
    box-shadow: 0 1px 3px rgba(0,0,0,0.05);
  }}
  .header {{
    border-bottom: 1px solid #f1f5f9;
    padding-bottom: 16px;
    margin-bottom: 20px;
    display: flex;
    justify-content: space-between;
    align-items: center;
  }}
  .grid {{
    display: table;
    width: 100%;
    margin-bottom: 24px;
  }}
  .grid-row {{
    display: table-row;
  }}
  .grid-cell {{
    display: table-cell;
    width: 50%;
    padding: 6px;
    box-sizing: border-box;
  }}
  .card {{
    background: #f8fafc;
    border: 1px solid #e2e8f0;
    border-radius: 8px;
    padding: 14px 18px;
  }}
  .card-label {{
    font-size: 12px;
    font-weight: 600;
    color: #64748b;
    text-transform: uppercase;
    letter-spacing: 0.5px;
    margin-bottom: 4px;
  }}
  .card-value {{
    font-size: 20px;
    font-weight: 700;
    color: #0f172a;
  }}
  .card-sub {{
    font-size: 12px;
    color: #64748b;
    margin-top: 2px;
  }}
  table {{
    width: 100%;
    border-collapse: collapse;
    margin-top: 8px;
    font-size: 13px;
  }}
  th {{
    background-color: #f8fafc;
    color: #475569;
    font-weight: 600;
    text-align: left;
    padding: 10px 12px;
    border-bottom: 1px solid #e2e8f0;
    text-transform: uppercase;
    font-size: 11px;
    letter-spacing: 0.5px;
  }}
  td {{
    padding: 10px 12px;
    border-bottom: 1px solid #f1f5f9;
    color: #334155;
  }}
  .footer {{
    margin-top: 32px;
    padding-top: 16px;
    border-top: 1px solid #f1f5f9;
    font-size: 12px;
    color: #94a3b8;
    text-align: center;
  }}
</style>
</head>
<body>
  <div class="container">
    <div class="header">
      <div>
        <h2 style="margin: 0; color: #0f172a; font-size: 20px;">🏛️ Aegis Daily Portfolio & Learning Report</h2>
        <div style="color: #64748b; font-size: 13px; margin-top: 4px;">Market: <b>{market_name}</b> | Dual-Session Multi-Agent Intelligence</div>
      </div>
      <div>
        {target_badge}
      </div>
    </div>

    <!-- Executive Metrics Grid -->
    <div class="grid">
      <div class="grid-row">
        <div class="grid-cell">
          <div class="card">
            <div class="card-label">True Portfolio NAV</div>
            <div class="card-value">{sym}{nav:,.2f}</div>
            <div class="card-sub">Initial: {sym}{initial_cap:,.2f} &bull; Return: <span style="color: {'#10b981' if total_ret_pct >= 0 else '#ef4444'}; font-weight: 600;">{'+' if total_ret_pct >= 0 else ''}{total_ret_pct:.2f}%</span></div>
          </div>
        </div>
        <div class="grid-cell">
          <div class="card">
            <div class="card-label">Daily Realized P&L</div>
            <div class="card-value" style="color: {pnl_color};">{'+' if realized_pnl >= 0 else ''}{sym}{realized_pnl:,.2f}</div>
            <div class="card-sub">Daily Target: {sym}{daily_target:,.2f} ({'Goal Met' if target_met else 'In Progress'})</div>
          </div>
        </div>
      </div>
      <div class="grid-row">
        <div class="grid-cell">
          <div class="card">
            <div class="card-label">Free Cash & Margin</div>
            <div class="card-value">{sym}{cash:,.2f}</div>
            <div class="card-sub">Margin Blocked: {sym}{margin:,.2f} &bull; Unrealized: <span style="color: {unrealized_color}; font-weight: 600;">{'+' if unrealized_pnl >= 0 else ''}{sym}{unrealized_pnl:,.2f}</span></div>
          </div>
        </div>
        <div class="grid-cell">
          <div class="card">
            <div class="card-label">Performance & Win Rate</div>
            <div class="card-value">{win_rate:.1f}%</div>
            <div class="card-sub">Trades Today: {total_trades} (Wins: {wins} &bull; Losses: {losses})</div>
          </div>
        </div>
      </div>
    </div>

    <!-- Active Open Positions -->
    <div style="margin-top: 20px;">
      <h3 style="color: #0f172a; margin-bottom: 4px; font-size: 16px;">Active Open Positions ({open_count})</h3>
      <table>
        <thead>
          <tr>
            <th>Ticker</th>
            <th>Type</th>
            <th>Strategy</th>
            <th>Qty</th>
            <th>Avg Cost</th>
            <th>Current</th>
            <th>Unrealized P&L</th>
          </tr>
        </thead>
        <tbody>
          {positions_rows}
        </tbody>
      </table>
    </div>

    <!-- Closed Trades Today -->
    <div style="margin-top: 24px;">
      <h3 style="color: #0f172a; margin-bottom: 4px; font-size: 16px;">Session Exited Trades</h3>
      <table>
        <thead>
          <tr>
            <th>Ticker</th>
            <th>Type</th>
            <th>Strategy</th>
            <th>Entry &rarr; Exit</th>
            <th>Realized P&L</th>
            <th>Duration</th>
            <th>Consensus Driver</th>
          </tr>
        </thead>
        <tbody>
          {trades_rows}
        </tbody>
      </table>
    </div>

    <!-- Trade Autopsy & Prescriptive Lessons -->
    {autopsy_html}

    <div class="footer">
      Generated autonomously by <b>Aegis AI Multi-Agent Network</b> &bull; Risk Gate & Volatility Parity Enforced &bull; PostgreSQL Persistence
    </div>
  </div>
</body>
</html>
""".strip()
