"""
Unit tests for Institutional HTML Email Formatter.
"""
from __future__ import annotations

import pytest
from src.notifier.formatter import format_eod_email_html


def test_format_eod_email_html_full_render():
    summary = {
        "market": "india",
        "currency": "₹",
        "cash": 9500.0,
        "reserved_margin": 500.0,
        "nav": 10250.0,
        "initial_capital": 10000.0,
        "total_return_pct": 2.5,
        "unrealized_pnl": 250.0,
        "open_positions_count": 1,
        "positions": [
            {
                "ticker": "TATAMOTORS.NS",
                "quantity": 10,
                "avg_cost": 600.0,
                "current_price": 625.0,
                "direction": "LONG",
                "strategy": "scalping",
                "unrealized_pnl": 250.0,
                "return_pct": 4.17,
                "margin_blocked": 1200.0,
            }
        ],
    }

    closed_trades = [
        {
            "symbol": "ZOMATO.NS",
            "market": "india",
            "strategy": "scalping",
            "direction": "BUY",
            "entry_price": 240.0,
            "exit_price": 244.5,
            "realized_pnl": 45.0,
            "return_pct": 1.88,
            "holding_period_seconds": 450.0,
            "pm_driver": "Breakout above 15m consolidation with volume",
            "learning_note": "Target reached cleanly",
        }
    ]

    eod_learning = {
        "total_pnl": 45.0,
        "daily_target": 1000.0,
        "target_met": False,
        "total_trades": 1,
        "wins": 1,
        "losses": 0,
        "win_rate_pct": 100.0,
        "autopsies": [],
    }

    html = format_eod_email_html(
        summary=summary,
        closed_trades=closed_trades,
        eod_learning=eod_learning,
    )

    assert "TATAMOTORS.NS" in html
    assert "ZOMATO.NS" in html
    assert "₹10,250.00" in html
    assert "45.00" in html
    assert "100.0%" in html
    assert "Breakout above 15m consolidation" in html
    assert "Flawless Session" in html


def test_format_eod_email_html_empty_state():
    summary = {
        "market": "us",
        "currency": "$",
        "cash": 1000.0,
        "reserved_margin": 0.0,
        "nav": 1000.0,
        "initial_capital": 1000.0,
        "total_return_pct": 0.0,
        "unrealized_pnl": 0.0,
        "open_positions_count": 0,
        "positions": [],
    }

    html = format_eod_email_html(summary)
    assert "No open positions held" in html
    assert "No closed trades recorded today" in html
    assert "$1,000.00" in html
