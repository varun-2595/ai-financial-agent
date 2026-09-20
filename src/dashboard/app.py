"""
Aegis Interactive Financial Dashboard — FastAPI Backend.
Serves a sleek, real-time trading terminal at http://localhost:8000.
"""
from __future__ import annotations

import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse

from src.db.trading_store import get_open_positions
from src.screener.watchlist_manager import get_active_tickers
from src.technicals.indicators import compute_technical_indicators
from src.technicals.levels import compute_support_resistance
from src.trading.paper_engine import PaperTradingEngine
from src.utils.logger import logger
from src.utils.market_hours import is_nse_open, is_nyse_open
from src.utils.state_manager import is_trading_paused

app = FastAPI(title="Aegis Financial Terminal", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── API Endpoints ─────────────────────────────────────────────────────────────

@app.get("/api/status")
def api_status():
    """Returns dual-market status, trading pause states, and account equity summaries."""
    engine = PaperTradingEngine()
    summary_inr = engine.get_portfolio_summary("india")
    summary_usd = engine.get_portfolio_summary("us")

    return {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "markets": {
            "india": {
                "name": "NSE / BSE",
                "is_open": is_nse_open(),
                "is_paused": is_trading_paused("india"),
                "summary": summary_inr,
            },
            "us": {
                "name": "NYSE / NASDAQ",
                "is_open": is_nyse_open(),
                "is_paused": is_trading_paused("us"),
                "summary": summary_usd,
            },
        },
        "is_paused_all": is_trading_paused("all"),
    }


@app.get("/api/positions")
def api_positions(market: Optional[str] = None):
    """Returns currently open positions with live calculations."""
    positions = get_open_positions(market=market)
    enhanced = []
    for p in positions:
        cost = p["avg_cost"]
        curr = p["current_price"] or cost
        pnl = (curr - cost) * p["quantity"]
        pnl_pct = ((curr - cost) / cost * 100) if cost else 0.0
        enhanced.append({
            **p,
            "current_price": round(curr, 2),
            "unrealized_pnl": round(pnl, 2),
            "unrealized_pnl_pct": round(pnl_pct, 2),
        })
    return {"positions": enhanced, "count": len(enhanced)}


@app.get("/api/watchlist")
def api_watchlist(market: Optional[str] = None):
    """Returns active tickers in dynamic watchlist."""
    tickers = get_active_tickers(market=market)
    return {"watchlist": tickers, "count": len(tickers)}


@app.get("/api/inspect/{ticker}")
def api_inspect(ticker: str):
    """Fetches live price, technical indicators, and pivot levels for a ticker."""
    t = ticker.upper().strip()
    is_india = (".NS" in t or ".BO" in t)

    from dataclasses import asdict
    from src.data.fetcher_india import fetch_india_stock
    from src.data.fetcher_us import fetch_us_stock

    try:
        snapshot = fetch_india_stock(t) if is_india else fetch_us_stock(t)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Failed to fetch data for {t}: {exc}")

    if not snapshot:
        raise HTTPException(status_code=404, detail=f"No data available for {t}")

    technicals = compute_technical_indicators(snapshot.history)
    levels = compute_support_resistance(snapshot.history)

    vol = snapshot.history[-1].volume if snapshot.history else 0
    change = round(snapshot.price_change_pct_1d, 2) if snapshot.price_change_pct_1d is not None else 0.0

    return {
        "ticker": t,
        "name": snapshot.name or t,
        "market": snapshot.market,
        "current_price": round(snapshot.current_price, 2),
        "change_pct": change,
        "volume": vol,
        "currency": "₹" if is_india else "$",
        "technicals": asdict(technicals),
        "levels": asdict(levels),
    }


# ── Single-Page Dashboard HTML ────────────────────────────────────────────────

@app.get("/", response_class=HTMLResponse)
def serve_dashboard():
    return """<!DOCTYPE html>
<html lang="en" class="dark">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>Aegis — Autonomous AI Financial Terminal</title>
  <script src="https://cdn.tailwindcss.com"></script>
  <script>
    tailwind.config = {
      darkMode: 'class',
      theme: {
        extend: {
          colors: {
            brand: { 500: '#3b82f6', 600: '#2563eb' }
          }
        }
      }
    }
  </script>
  <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.5.1/css/all.min.css">
  <style>
    body { background-color: #0b0f19; font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif; }
    .card { background: rgba(18, 24, 38, 0.95); border: 1px solid rgba(255, 255, 255, 0.08); }
    .card-hover:hover { border-color: rgba(59, 130, 246, 0.4); }
    ::-webkit-scrollbar { width: 6px; height: 6px; }
    ::-webkit-scrollbar-track { background: #0b0f19; }
    ::-webkit-scrollbar-thumb { background: #1e293b; border-radius: 4px; }
  </style>
</head>
<body class="text-slate-100 min-h-screen">

  <!-- Top Navigation -->
  <header class="border-b border-slate-800 bg-slate-900/80 backdrop-blur sticky top-0 z-50">
    <div class="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 h-16 flex items-center justify-between">
      <div class="flex items-center space-x-3">
        <div class="w-9 h-9 rounded-lg bg-blue-600/20 border border-blue-500/40 flex items-center justify-center text-blue-400 font-bold text-lg">
          🛡️
        </div>
        <div>
          <h1 class="text-base font-bold tracking-tight text-white flex items-center space-x-2">
            <span>AEGIS</span>
            <span class="text-xs font-medium px-2 py-0.5 rounded bg-blue-500/10 text-blue-400 border border-blue-500/20">AI Terminal</span>
          </h1>
          <p class="text-xs text-slate-400">Autonomous Financial Analyst & Dual-Market Trading Agent</p>
        </div>
      </div>

      <!-- Live Session Status Badges -->
      <div class="flex items-center space-x-4 text-xs">
        <div id="badge-india" class="flex items-center space-x-1.5 px-3 py-1.5 rounded-full border border-slate-700 bg-slate-800/80">
          <span class="w-2 h-2 rounded-full bg-slate-500 animate-pulse"></span>
          <span class="font-medium text-slate-300">🇮🇳 NSE: Checking...</span>
        </div>
        <div id="badge-us" class="flex items-center space-x-1.5 px-3 py-1.5 rounded-full border border-slate-700 bg-slate-800/80">
          <span class="w-2 h-2 rounded-full bg-slate-500 animate-pulse"></span>
          <span class="font-medium text-slate-300">🇺🇸 NYSE: Checking...</span>
        </div>
        <div id="badge-pause" class="flex items-center space-x-1.5 px-3 py-1.5 rounded-full border border-emerald-500/30 bg-emerald-950/40 text-emerald-400">
          <i class="fa-solid fa-play text-[10px]"></i>
          <span class="font-medium" id="pause-text">ACTIVE</span>
        </div>
        <div class="text-slate-500 text-xs pl-2 border-l border-slate-800" id="live-clock">
          --:--:--
        </div>
      </div>
    </div>
  </header>

  <!-- Main Container -->
  <main class="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-6 space-y-6">

    <!-- Section 1: Executive Account Cards -->
    <div class="grid grid-cols-1 md:grid-cols-2 gap-5">
      <!-- India Card -->
      <div class="card rounded-xl p-5 shadow-lg relative overflow-hidden card-hover transition duration-200">
        <div class="flex items-center justify-between mb-4">
          <div class="flex items-center space-x-2">
            <span class="text-2xl">🇮🇳</span>
            <div>
              <h2 class="text-sm font-semibold text-slate-200">India Paper Portfolio (NSE/BSE)</h2>
              <p class="text-xs text-slate-400">Currency: INR (₹)</p>
            </div>
          </div>
          <span id="inr-pos-count" class="text-xs px-2.5 py-1 rounded bg-slate-800 text-slate-300 border border-slate-700">0 Positions</span>
        </div>
        <div class="grid grid-cols-3 gap-4 pt-2 border-t border-slate-800/80">
          <div>
            <p class="text-xs text-slate-400 mb-1">Total Equity</p>
            <p id="inr-total" class="text-xl font-bold text-white">₹--</p>
          </div>
          <div>
            <p class="text-xs text-slate-400 mb-1">Available Cash</p>
            <p id="inr-cash" class="text-sm font-medium text-slate-300">₹--</p>
          </div>
          <div>
            <p class="text-xs text-slate-400 mb-1">Invested Capital</p>
            <p id="inr-invested" class="text-sm font-medium text-slate-300">₹--</p>
          </div>
        </div>
      </div>

      <!-- US Card -->
      <div class="card rounded-xl p-5 shadow-lg relative overflow-hidden card-hover transition duration-200">
        <div class="flex items-center justify-between mb-4">
          <div class="flex items-center space-x-2">
            <span class="text-2xl">🇺🇸</span>
            <div>
              <h2 class="text-sm font-semibold text-slate-200">US Paper Portfolio (NYSE/NASDAQ)</h2>
              <p class="text-xs text-slate-400">Currency: USD ($)</p>
            </div>
          </div>
          <span id="usd-pos-count" class="text-xs px-2.5 py-1 rounded bg-slate-800 text-slate-300 border border-slate-700">0 Positions</span>
        </div>
        <div class="grid grid-cols-3 gap-4 pt-2 border-t border-slate-800/80">
          <div>
            <p class="text-xs text-slate-400 mb-1">Total Equity</p>
            <p id="usd-total" class="text-xl font-bold text-white">$--</p>
          </div>
          <div>
            <p class="text-xs text-slate-400 mb-1">Available Cash</p>
            <p id="usd-cash" class="text-sm font-medium text-slate-300">$--</p>
          </div>
          <div>
            <p class="text-xs text-slate-400 mb-1">Invested Capital</p>
            <p id="usd-invested" class="text-sm font-medium text-slate-300">$--</p>
          </div>
        </div>
      </div>
    </div>

    <!-- Section 2: Active Open Positions Table -->
    <div class="card rounded-xl shadow-lg overflow-hidden">
      <div class="px-5 py-4 border-b border-slate-800/80 flex items-center justify-between">
        <div class="flex items-center space-x-2">
          <i class="fa-solid fa-chart-line text-blue-400"></i>
          <h2 class="text-sm font-semibold text-white">Active Open Positions</h2>
          <span id="pos-badge-total" class="text-xs px-2 py-0.5 rounded bg-blue-500/20 text-blue-400 font-medium">0</span>
        </div>
        <span class="text-xs text-slate-400"><i class="fa-solid fa-arrows-rotate animate-spin mr-1"></i> Auto-refreshes every 10s</span>
      </div>

      <div class="overflow-x-auto">
        <table class="w-full text-left text-xs">
          <thead class="bg-slate-900/50 text-slate-400 uppercase tracking-wider text-[11px] border-b border-slate-800">
            <tr>
              <th class="px-5 py-3 font-semibold">Ticker</th>
              <th class="px-4 py-3 font-semibold">Market</th>
              <th class="px-4 py-3 font-semibold">Strategy</th>
              <th class="px-4 py-3 font-semibold">Side</th>
              <th class="px-4 py-3 font-semibold text-right">Qty</th>
              <th class="px-4 py-3 font-semibold text-right">Avg Cost</th>
              <th class="px-4 py-3 font-semibold text-right">Current Price</th>
              <th class="px-4 py-3 font-semibold text-right">Unrealized P&L</th>
              <th class="px-4 py-3 font-semibold text-right">Stop Loss</th>
              <th class="px-4 py-3 font-semibold text-right">Target</th>
            </tr>
          </thead>
          <tbody id="positions-tbody" class="divide-y divide-slate-800/60 font-mono">
            <tr>
              <td colspan="10" class="text-center py-8 text-slate-500 font-sans">
                <i class="fa-regular fa-folder-open text-2xl mb-2 block"></i>
                No active paper positions right now. The 15-minute scanner runs when markets open.
              </td>
            </tr>
          </tbody>
        </table>
      </div>
    </div>

    <!-- Section 3: Dual Column (Watchlist Explorer & Ticker Technical Inspector) -->
    <div class="grid grid-cols-1 lg:grid-cols-12 gap-5">

      <!-- Column 1: Active Watchlist (7 cols) -->
      <div class="lg:col-span-7 card rounded-xl shadow-lg p-5">
        <div class="flex items-center justify-between mb-4">
          <div class="flex items-center space-x-2">
            <i class="fa-solid fa-list-check text-blue-400"></i>
            <h2 class="text-sm font-semibold text-white">Active Dynamic Watchlist</h2>
            <span id="watchlist-count" class="text-xs px-2 py-0.5 rounded bg-slate-800 text-slate-400">0</span>
          </div>
          <input type="text" id="watchlist-search" placeholder="Filter tickers..." onkeyup="filterWatchlist()"
            class="bg-slate-900 border border-slate-700 text-xs rounded-lg px-3 py-1.5 focus:outline-none focus:border-blue-500 w-40 text-slate-200 placeholder-slate-500">
        </div>

        <div class="overflow-y-auto max-h-[380px] rounded-lg border border-slate-800/80">
          <table class="w-full text-left text-xs">
            <thead class="bg-slate-900 text-slate-400 text-[11px] sticky top-0 border-b border-slate-800">
              <tr>
                <th class="px-4 py-2.5 font-semibold">Ticker</th>
                <th class="px-3 py-2.5 font-semibold">Market</th>
                <th class="px-3 py-2.5 font-semibold">Strategies</th>
                <th class="px-3 py-2.5 font-semibold">Source</th>
                <th class="px-3 py-2.5 font-semibold text-right">Action</th>
              </tr>
            </thead>
            <tbody id="watchlist-tbody" class="divide-y divide-slate-800/60">
              <tr><td colspan="5" class="text-center py-6 text-slate-500">Loading active watchlist...</td></tr>
            </tbody>
          </table>
        </div>
      </div>

      <!-- Column 2: Quick Ticker Technical Inspector (5 cols) -->
      <div class="lg:col-span-5 card rounded-xl shadow-lg p-5 flex flex-col justify-between">
        <div>
          <div class="flex items-center justify-between mb-3">
            <div class="flex items-center space-x-2">
              <i class="fa-solid fa-magnifying-glass-chart text-blue-400"></i>
              <h2 class="text-sm font-semibold text-white">Technical Inspector</h2>
            </div>
            <span class="text-xs text-slate-400">Live indicators & pivots</span>
          </div>

          <!-- Search Input -->
          <div class="flex space-x-2 mb-4">
            <input type="text" id="inspect-input" value="RELIANCE.NS" placeholder="e.g. INFY.NS or NVDA"
              class="flex-1 bg-slate-900 border border-slate-700 text-xs rounded-lg px-3 py-2 focus:outline-none focus:border-blue-500 font-mono text-slate-100">
            <button onclick="inspectTicker()" class="px-4 py-2 bg-blue-600 hover:bg-blue-500 text-xs font-semibold rounded-lg transition duration-150 flex items-center space-x-1.5">
              <span>Inspect</span>
            </button>
          </div>

          <!-- Inspector Results Card -->
          <div id="inspect-result" class="bg-slate-900/60 border border-slate-800 rounded-lg p-4 space-y-3">
            <div class="flex items-center justify-between border-b border-slate-800 pb-2">
              <div>
                <h3 id="inspect-ticker" class="font-bold text-base text-white">RELIANCE.NS</h3>
                <p id="inspect-name" class="text-xs text-slate-400">Reliance Industries Ltd</p>
              </div>
              <div class="text-right">
                <p id="inspect-price" class="text-base font-bold text-white">₹--</p>
                <p id="inspect-change" class="text-xs font-medium text-slate-400">--%</p>
              </div>
            </div>

            <!-- Technical Grid -->
            <div class="grid grid-cols-2 gap-2 text-xs pt-1">
              <div class="p-2 rounded bg-slate-800/50 border border-slate-700/50">
                <span class="text-slate-400 block text-[11px]">RSI (14)</span>
                <span id="inspect-rsi" class="font-bold text-sm text-slate-200">--</span>
              </div>
              <div class="p-2 rounded bg-slate-800/50 border border-slate-700/50">
                <span class="text-slate-400 block text-[11px]">MACD Status</span>
                <span id="inspect-macd" class="font-bold text-sm text-slate-200">--</span>
              </div>
              <div class="p-2 rounded bg-slate-800/50 border border-slate-700/50">
                <span class="text-slate-400 block text-[11px]">EMA 20 / 50</span>
                <span id="inspect-ema" class="font-bold text-xs text-slate-200">--</span>
              </div>
              <div class="p-2 rounded bg-slate-800/50 border border-slate-700/50">
                <span class="text-slate-400 block text-[11px]">ATR (14)</span>
                <span id="inspect-atr" class="font-bold text-xs text-slate-200">--</span>
              </div>
            </div>

            <!-- Key Levels -->
            <div class="pt-2 border-t border-slate-800 text-[11px]">
              <span class="text-slate-400 block mb-1.5 font-medium">Daily Key Levels</span>
              <div class="grid grid-cols-5 gap-1 text-center font-mono">
                <div class="p-1 rounded bg-rose-950/30 border border-rose-900/40 text-rose-300">
                  <div class="text-[9px] text-rose-400">S2</div>
                  <div id="level-s2" class="font-bold">--</div>
                </div>
                <div class="p-1 rounded bg-rose-950/20 border border-rose-800/40 text-rose-200">
                  <div class="text-[9px] text-rose-400">S1</div>
                  <div id="level-s1" class="font-bold">--</div>
                </div>
                <div class="p-1 rounded bg-blue-950/30 border border-blue-800/40 text-blue-300">
                  <div class="text-[9px] text-blue-400">PIVOT</div>
                  <div id="level-p" class="font-bold">--</div>
                </div>
                <div class="p-1 rounded bg-emerald-950/20 border border-emerald-800/40 text-emerald-200">
                  <div class="text-[9px] text-emerald-400">R1</div>
                  <div id="level-r1" class="font-bold">--</div>
                </div>
                <div class="p-1 rounded bg-emerald-950/30 border border-emerald-900/40 text-emerald-300">
                  <div class="text-[9px] text-emerald-400">R2</div>
                  <div id="level-r2" class="font-bold">--</div>
                </div>
              </div>
            </div>
          </div>
        </div>

        <p class="text-[11px] text-slate-500 mt-4 text-center">
          💡 Tip: Click "Inspect" on any ticker in the watchlist table to load its indicators instantly.
        </p>
      </div>

    </div>

  </main>

  <!-- Dashboard JavaScript Logic -->
  <script>
    let watchlistData = [];

    // Live Clock
    function updateClock() {
      const now = new Date();
      document.getElementById('live-clock').innerText = now.toLocaleTimeString('en-US', { hour12: false });
    }
    setInterval(updateClock, 1000);
    updateClock();

    // Fetch System Status & Accounts
    async function loadStatus() {
      try {
        const res = await fetch('/api/status');
        const data = await res.json();

        // Markets
        const india = data.markets.india;
        const us = data.markets.us;

        const badgeIndia = document.getElementById('badge-india');
        badgeIndia.innerHTML = india.is_open 
          ? '<span class="w-2 h-2 rounded-full bg-emerald-500 animate-pulse"></span><span class="font-medium text-emerald-400">🇮🇳 NSE: OPEN</span>'
          : '<span class="w-2 h-2 rounded-full bg-slate-500"></span><span class="font-medium text-slate-400">🇮🇳 NSE: CLOSED</span>';

        const badgeUs = document.getElementById('badge-us');
        badgeUs.innerHTML = us.is_open 
          ? '<span class="w-2 h-2 rounded-full bg-emerald-500 animate-pulse"></span><span class="font-medium text-emerald-400">🇺🇸 NYSE: OPEN</span>'
          : '<span class="w-2 h-2 rounded-full bg-slate-500"></span><span class="font-medium text-slate-400">🇺🇸 NYSE: CLOSED</span>';

        // Pause state
        const badgePause = document.getElementById('badge-pause');
        const pauseText = document.getElementById('pause-text');
        if (data.is_paused_all) {
          badgePause.className = 'flex items-center space-x-1.5 px-3 py-1.5 rounded-full border border-amber-500/30 bg-amber-950/40 text-amber-400';
          badgePause.innerHTML = '<i class="fa-solid fa-pause text-[10px]"></i><span class="font-medium">TRADING PAUSED</span>';
        } else {
          badgePause.className = 'flex items-center space-x-1.5 px-3 py-1.5 rounded-full border border-emerald-500/30 bg-emerald-950/40 text-emerald-400';
          badgePause.innerHTML = '<i class="fa-solid fa-play text-[10px]"></i><span class="font-medium">ACTIVE</span>';
        }

        // Account Summaries
        document.getElementById('inr-total').innerText = '₹' + india.summary.total_value.toLocaleString('en-IN');
        document.getElementById('inr-cash').innerText = '₹' + india.summary.cash.toLocaleString('en-IN');
        document.getElementById('inr-invested').innerText = '₹' + india.summary.invested.toLocaleString('en-IN');
        document.getElementById('inr-pos-count').innerText = india.summary.open_positions_count + ' Positions';

        document.getElementById('usd-total').innerText = '$' + us.summary.total_value.toLocaleString('en-US');
        document.getElementById('usd-cash').innerText = '$' + us.summary.cash.toLocaleString('en-US');
        document.getElementById('usd-invested').innerText = '$' + us.summary.invested.toLocaleString('en-US');
        document.getElementById('usd-pos-count').innerText = us.summary.open_positions_count + ' Positions';
      } catch (err) {
        console.error('Failed to load status:', err);
      }
    }

    // Fetch Open Positions
    async function loadPositions() {
      try {
        const res = await fetch('/api/positions');
        const data = await res.json();
        const tbody = document.getElementById('positions-tbody');
        document.getElementById('pos-badge-total').innerText = data.count;

        if (data.positions.length === 0) {
          tbody.innerHTML = `
            <tr>
              <td colspan="10" class="text-center py-8 text-slate-500 font-sans">
                <i class="fa-regular fa-folder-open text-2xl mb-2 block"></i>
                No active paper positions right now. The 15-minute scanner runs when markets open.
              </td>
            </tr>`;
          return;
        }

        tbody.innerHTML = data.positions.map(p => {
          const currSign = p.market === 'india' ? '₹' : '$';
          const pnlColor = p.unrealized_pnl >= 0 ? 'text-emerald-400 font-bold' : 'text-rose-400 font-bold';
          const pnlSign = p.unrealized_pnl >= 0 ? '+' : '';
          return `
            <tr class="hover:bg-slate-900/40 transition duration-150">
              <td class="px-5 py-3 font-bold text-white font-sans flex items-center space-x-2">
                <span>${p.ticker}</span>
              </td>
              <td class="px-4 py-3 text-slate-300 uppercase">${p.market}</td>
              <td class="px-4 py-3 text-slate-400">${p.strategy}</td>
              <td class="px-4 py-3"><span class="px-2 py-0.5 rounded text-[10px] ${p.direction === 'LONG' ? 'bg-emerald-500/20 text-emerald-400' : 'bg-rose-500/20 text-rose-400'}">${p.direction}</span></td>
              <td class="px-4 py-3 text-right text-slate-200">${p.quantity}</td>
              <td class="px-4 py-3 text-right text-slate-300">${currSign}${p.avg_cost.toFixed(2)}</td>
              <td class="px-4 py-3 text-right text-white font-semibold">${currSign}${p.current_price.toFixed(2)}</td>
              <td class="px-4 py-3 text-right ${pnlColor}">${pnlSign}${currSign}${p.unrealized_pnl.toFixed(2)} (${pnlSign}${p.unrealized_pnl_pct.toFixed(2)}%)</td>
              <td class="px-4 py-3 text-right text-slate-400">${currSign}${p.stop_loss ? p.stop_loss.toFixed(2) : '-'}</td>
              <td class="px-4 py-3 text-right text-slate-400">${currSign}${p.target_price ? p.target_price.toFixed(2) : '-'}</td>
            </tr>`;
        }).join('');
      } catch (err) {
        console.error('Failed to load positions:', err);
      }
    }

    // Fetch Watchlist
    async function loadWatchlist() {
      try {
        const res = await fetch('/api/watchlist');
        const data = await res.json();
        watchlistData = data.watchlist;
        document.getElementById('watchlist-count').innerText = data.count;
        renderWatchlist(watchlistData);
      } catch (err) {
        console.error('Failed to load watchlist:', err);
      }
    }

    function renderWatchlist(items) {
      const tbody = document.getElementById('watchlist-tbody');
      if (items.length === 0) {
        tbody.innerHTML = `<tr><td colspan="5" class="text-center py-6 text-slate-500">No matching tickers found.</td></tr>`;
        return;
      }

      tbody.innerHTML = items.map(t => {
        const isPinned = t.pinned ? '📌' : '';
        const strats = (t.strategies || []).map(s => `<span class="px-1.5 py-0.5 rounded text-[10px] bg-slate-800 text-slate-400 border border-slate-700">${s}</span>`).join(' ');
        return `
          <tr class="hover:bg-slate-900/50 transition">
            <td class="px-4 py-2.5 font-bold text-white flex items-center space-x-1.5">
              <span>${t.ticker}</span>
              <span class="text-xs" title="Pinned permanently">${isPinned}</span>
            </td>
            <td class="px-3 py-2.5 uppercase text-slate-400 text-[11px]">${t.market}</td>
            <td class="px-3 py-2.5 space-x-1">${strats}</td>
            <td class="px-3 py-2.5 text-slate-500 text-[11px]">${t.source || 'auto'}</td>
            <td class="px-3 py-2.5 text-right">
              <button onclick="setAndInspect('${t.ticker}')" class="px-2 py-1 bg-slate-800 hover:bg-blue-600 text-slate-300 hover:text-white rounded text-[10px] transition">
                Inspect
              </button>
            </td>
          </tr>`;
      }).join('');
    }

    function filterWatchlist() {
      const query = document.getElementById('watchlist-search').value.toUpperCase().trim();
      const filtered = watchlistData.filter(t => t.ticker.includes(query) || t.market.toUpperCase().includes(query));
      renderWatchlist(filtered);
    }

    function setAndInspect(ticker) {
      document.getElementById('inspect-input').value = ticker;
      inspectTicker();
    }

    // Inspect Ticker Technicals
    async function inspectTicker() {
      const ticker = document.getElementById('inspect-input').value.trim().toUpperCase();
      if (!ticker) return;

      const btn = event ? event.target : null;
      try {
        const res = await fetch(`/api/inspect/${encodeURIComponent(ticker)}`);
        if (!res.ok) {
          alert('Could not fetch data for ' + ticker);
          return;
        }
        const data = await res.json();

        document.getElementById('inspect-ticker').innerText = data.ticker;
        document.getElementById('inspect-name').innerText = data.name || data.ticker;
        document.getElementById('inspect-price').innerText = `${data.currency}${data.current_price.toLocaleString()}`;

        const chgEl = document.getElementById('inspect-change');
        const sign = data.change_pct >= 0 ? '+' : '';
        chgEl.innerText = `${sign}${data.change_pct.toFixed(2)}%`;
        chgEl.className = `text-xs font-semibold ${data.change_pct >= 0 ? 'text-emerald-400' : 'text-rose-400'}`;

        // Technicals
        const rsiVal = data.technicals.rsi_14 ? data.technicals.rsi_14.toFixed(1) : '--';
        const rsiEl = document.getElementById('inspect-rsi');
        let rsiTag = (data.technicals.rsi_14 >= 70) ? ' (Overbought)' : (data.technicals.rsi_14 <= 30 ? ' (Oversold)' : ' (Neutral)');
        rsiEl.innerText = `${rsiVal}${rsiTag}`;
        rsiEl.className = `font-bold text-sm ${data.technicals.rsi_14 >= 70 ? 'text-rose-400' : (data.technicals.rsi_14 <= 30 ? 'text-emerald-400' : 'text-slate-200')}`;

        document.getElementById('inspect-macd').innerText = data.technicals.trend_short || 'NEUTRAL';
        document.getElementById('inspect-ema').innerText = data.technicals.ema_20 ? `EMA20: ${data.technicals.ema_20.toFixed(1)}` : 'NEUTRAL';
        document.getElementById('inspect-atr').innerText = data.technicals.atr_14 ? `${data.currency}${data.technicals.atr_14.toFixed(2)}` : '--';

        // Pivot levels
        document.getElementById('level-s2').innerText = data.levels.support_2 ? `${data.currency}${data.levels.support_2.toFixed(1)}` : '--';
        document.getElementById('level-s1').innerText = data.levels.support_1 ? `${data.currency}${data.levels.support_1.toFixed(1)}` : '--';
        document.getElementById('level-p').innerText = data.levels.pivot_point ? `${data.currency}${data.levels.pivot_point.toFixed(1)}` : '--';
        document.getElementById('level-r1').innerText = data.levels.resistance_1 ? `${data.currency}${data.levels.resistance_1.toFixed(1)}` : '--';
        document.getElementById('level-r2').innerText = data.levels.resistance_2 ? `${data.currency}${data.levels.resistance_2.toFixed(1)}` : '--';

      } catch (err) {
        console.error('Failed to inspect ticker:', err);
      }
    }

    // Initial Load & Auto-Refresh Cycle
    loadStatus();
    loadPositions();
    loadWatchlist();
    inspectTicker();

    setInterval(loadStatus, 10000);
    setInterval(loadPositions, 10000);
  </script>
</body>
</html>
"""
