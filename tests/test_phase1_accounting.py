"""
Phase 1 Accounting Tests — Aegis Paper Trading Engine.

Covers all 12 Phase 1 requirements:
  1.  Correct portfolio accounting (NAV formula)
  2.  Cash balance tracking (deduction on buy, credit on sell)
  3.  Reserved margin (blocked on open, released on close)
  4.  Buying power (cash, equity BP, intraday BP)
  5.  Realized P&L (net of fees, direction-aware)
  6.  Unrealized P&L (mark-to-market on open positions)
  7.  Portfolio NAV / equity (cash + reserved_margin + unrealized_pnl)
  8.  Position average price (avg_cost stored correctly)
  9.  Fees (entry and exit fees deducted from P&L and cash)
  10. Slippage (applied symmetrically on entry and exit)
  11. Leverage accounting (3x intraday; NAV not inflated)
  12. Transaction ledger (every cash event logged)

Additional tests:
  - Buy then profit exit
  - Buy then loss exit
  - Partial sell
  - Multiple simultaneous positions
  - Intraday square-off
  - Stop-loss and target-price triggers
  - Full reset / account reset
  - Config path and initial capital verification
"""
from __future__ import annotations

import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal
from unittest.mock import patch

import pytest

from src.data.models import StockSnapshot, TradeSignal
from src.db.trading_store import _conn, init_trading_db, get_ledger
from src.trading.fees import FeeSchedule
from src.trading.paper_engine import PaperTradingEngine
from src.utils.config import get_config


# ── Fixtures ───────────────────────────────────────────────────────────────────

@pytest.fixture(autouse=True)
def fresh_engine():
    """Each test gets a clean engine with freshly reset account balances and clean tables."""
    init_trading_db()
    with _conn() as conn:
        conn.execute("DELETE FROM positions")
        conn.execute("DELETE FROM orders")
        conn.execute("DELETE FROM signals")
        conn.execute("DELETE FROM ledger")
        conn.commit()
    engine = PaperTradingEngine()
    engine.reset_account_balances()
    return engine


@pytest.fixture
def engine(fresh_engine):
    return fresh_engine


@pytest.fixture
def zero_fee_engine(fresh_engine):
    """Engine with fees and slippage zeroed out — isolates price-only accounting."""
    fees = FeeSchedule(paper_per_side_pct=0.0, slippage_pct=0.0)
    return PaperTradingEngine(fee_schedule=fees)



def _signal(
    ticker: str = "TEST",
    market: Literal["india", "us"] = "india",
    strategy: str = "swing",
    entry: float = 100.0,
    sl: float = 95.0,
    tp: float = 115.0,
    qty: int = 10,
) -> TradeSignal:
    return TradeSignal(
        ticker=ticker, market=market, strategy=strategy,
        direction="BUY", entry_price=entry,
        stop_loss=sl, target_price=tp,
        quantity=qty, confidence=0.85, reasoning="test",
    )


def _snap(ticker: str, price: float, market: str = "india") -> StockSnapshot:
    return StockSnapshot(
        ticker=ticker, market=market,  # type: ignore
        currency="INR" if market == "india" else "USD",
        current_price=price,
    )


# ── 1. Config path & initial capital ──────────────────────────────────────────

def test_config_loads_correctly():
    """Verify config.py loads settings.yaml from project root (not src/)."""
    cfg = get_config(reload=True)
    # These values come from settings.yaml — if path bug were present they'd be
    # Pydantic defaults (10_000 / 1_000) but with no YAML they'd be the same,
    # so also verify a non-default key like daily_profit_target_inr
    assert cfg.paper_trading.virtual_capital_inr == 10_000.0
    assert cfg.paper_trading.virtual_capital_usd == 1_000.0
    assert cfg.paper_trading.daily_profit_target_inr == 1_000.0
    assert cfg.paper_trading.intraday_leverage_multiplier == 3.0


def test_initial_capital_after_reset(engine):
    """After full_reset, INR account = 10,000 and USD account = 1,000."""
    assert engine.get_account_balance("india") == 10_000.0
    assert engine.get_account_balance("us") == 1_000.0


# ── 2. Cash balance ────────────────────────────────────────────────────────────

def test_cash_deducted_on_buy(zero_fee_engine):
    """Cash decreases by exactly margin_blocked on BUY (swing = 1x leverage)."""
    e = zero_fee_engine
    before = e.get_account_balance("india")

    order = e.execute_signal(_signal(entry=100.0, qty=10, strategy="swing"))
    assert order is not None

    after = e.get_account_balance("india")
    expected_margin = 100.0 * 10 / 1.0   # swing, leverage=1
    assert abs(before - after - expected_margin) < 0.01


def test_cash_restored_on_profitable_close(zero_fee_engine):
    """After profitable close, cash = initial + realized_pnl."""
    e = zero_fee_engine
    initial = e.get_account_balance("india")

    e.execute_signal(_signal(entry=100.0, qty=10, strategy="swing"))
    cash_after_buy = e.get_account_balance("india")

    # Simulate price hitting target
    snaps = {"TEST": _snap("TEST", price=115.0)}
    e.evaluate_open_positions("india", snaps)

    final_cash = e.get_account_balance("india")
    gross_pnl = (115.0 - 100.0) * 10     # 150.0
    expected = initial + gross_pnl
    assert abs(final_cash - expected) < 0.01, (
        f"Expected cash ≈ {expected:.2f} after profit, got {final_cash:.2f}"
    )


def test_cash_reduced_on_loss_close(zero_fee_engine):
    """After stop-loss close, cash = initial - loss."""
    e = zero_fee_engine
    initial = e.get_account_balance("india")

    e.execute_signal(_signal(entry=100.0, sl=95.0, qty=10, strategy="swing"))
    snaps = {"TEST": _snap("TEST", price=95.0)}
    e.evaluate_open_positions("india", snaps)

    final_cash = e.get_account_balance("india")
    loss = (95.0 - 100.0) * 10     # -50.0
    expected = initial + loss
    assert abs(final_cash - expected) < 0.01


# ── 3. Reserved margin ─────────────────────────────────────────────────────────

def test_reserved_margin_blocked_on_open(zero_fee_engine):
    """reserved_margin increases by margin_blocked when position is opened."""
    e = zero_fee_engine
    assert e.get_reserved_margin("india") == 0.0

    e.execute_signal(_signal(entry=100.0, qty=10, strategy="swing"))

    rm = e.get_reserved_margin("india")
    assert abs(rm - 1000.0) < 0.01, f"Expected reserved_margin=1000, got {rm}"


def test_reserved_margin_released_on_close(zero_fee_engine):
    """reserved_margin returns to 0 after position is fully closed."""
    e = zero_fee_engine
    e.execute_signal(_signal(entry=100.0, qty=10, strategy="swing",
                              sl=95.0, tp=115.0))

    snaps = {"TEST": _snap("TEST", price=115.0)}   # exactly at target → triggers close
    e.evaluate_open_positions("india", snaps)

    assert e.get_reserved_margin("india") == 0.0


def test_reserved_margin_intraday_3x(zero_fee_engine):
    """For intraday (3x leverage), margin_blocked = notional / 3."""
    e = zero_fee_engine
    e.execute_signal(_signal(entry=300.0, qty=10, strategy="intraday",
                              sl=295.0, tp=310.0))
    rm = e.get_reserved_margin("india")
    expected = (300.0 * 10) / 3.0    # 1000.0
    assert abs(rm - expected) < 0.01


# ── 4. Buying power ────────────────────────────────────────────────────────────

def test_buying_power_before_any_trade(engine):
    """Buying power = 10,000 cash before any trade."""
    bp = engine.get_buying_power("india")
    assert bp["cash"] == 10_000.0
    assert bp["buying_power_equity"] == 10_000.0
    assert bp["buying_power_intraday"] == 10_000.0 * 3.0


def test_buying_power_decreases_after_buy(zero_fee_engine):
    """After a swing buy costing 1,000, equity BP decreases by 1,000."""
    e = zero_fee_engine
    e.execute_signal(_signal(entry=100.0, qty=10, strategy="swing"))
    bp = e.get_buying_power("india")
    assert abs(bp["cash"] - 9_000.0) < 0.01


# ── 5. Realized P&L (net of fees) ─────────────────────────────────────────────

def test_realized_pnl_net_of_fees():
    """Realized PnL stored in DB = gross_pnl minus exit fees."""
    fees = FeeSchedule(paper_per_side_pct=0.001, slippage_pct=0.0)  # 0.1% fees, no slippage
    e = PaperTradingEngine(fee_schedule=fees)
    e.full_reset()

    # tp=110 so price=110 triggers target hit
    e.execute_signal(_signal(entry=100.0, qty=10, strategy="swing", sl=90.0, tp=110.0))

    # Close at target (110)
    snaps = {"TEST": _snap("TEST", price=110.0)}
    e.evaluate_open_positions("india", snaps)

    daily = e.get_daily_realized_pnl("india")
    gross = (110.0 - 100.0) * 10   # 100.0
    exit_fee = 110.0 * 10 * 0.001  # 1.10
    expected_net = gross - exit_fee
    assert abs(daily - expected_net) < 0.05, (
        f"Expected realized PnL ≈ {expected_net:.2f}, got {daily:.2f}"
    )


def test_realized_pnl_negative_on_loss(zero_fee_engine):
    """Realized PnL is negative after a loss (price hits stop loss)."""
    e = zero_fee_engine
    e.execute_signal(_signal(entry=100.0, sl=95.0, tp=115.0, qty=10, strategy="swing"))
    # curr_p <= sl (95.0) triggers stop loss close
    snaps = {"TEST": _snap("TEST", price=94.0)}
    e.evaluate_open_positions("india", snaps)

    pnl = e.get_daily_realized_pnl("india")
    assert pnl < 0.0


# ── 6. Unrealized P&L ─────────────────────────────────────────────────────────

def test_unrealized_pnl_in_stats(zero_fee_engine):
    """Unrealized PnL reflects current price vs avg_cost for open positions."""
    e = zero_fee_engine
    e.execute_signal(_signal(entry=100.0, qty=10, strategy="swing"))

    # Update price mark
    with _conn() as conn:
        conn.execute("UPDATE positions SET current_price = 108.0 WHERE ticker = 'TEST'")
        conn.commit()

    stats = e.get_daily_stats("india")
    assert abs(stats["unrealized_pnl"] - 80.0) < 0.01   # (108-100)*10


def test_unrealized_pnl_in_nav(zero_fee_engine):
    """NAV increases by unrealized_pnl when position appreciates."""
    e = zero_fee_engine
    initial_nav = e.get_portfolio_nav("india")["nav"]

    e.execute_signal(_signal(entry=100.0, qty=10, strategy="swing"))

    # Price goes up 10 points
    with _conn() as conn:
        conn.execute("UPDATE positions SET current_price = 110.0 WHERE ticker = 'TEST'")
        conn.commit()

    nav_after = e.get_portfolio_nav("india")["nav"]
    assert abs(nav_after - initial_nav - 100.0) < 0.01   # unrealized gain = 100


# ── 7. Portfolio NAV — the critical bug fix ────────────────────────────────────

def test_nav_not_inflated_by_leverage(zero_fee_engine):
    """
    CRITICAL: Portfolio NAV must NOT increase on trade entry.

    With 3x leverage: entry at 300×10 = 3000 notional, margin = 1000.
    At entry (price unchanged), NAV should equal initial capital.
    Old bug: NAV = cash + full_notional = 9000 + 3000 = 12000 (WRONG).
    Correct:  NAV = cash + reserved_margin + 0 unrealized = 9000 + 1000 + 0 = 10000.
    """
    e = zero_fee_engine
    initial = e.get_account_balance("india")  # 10_000.0

    e.execute_signal(_signal(entry=300.0, qty=10, strategy="intraday",
                              sl=295.0, tp=315.0))

    nav = e.get_portfolio_nav("india")
    assert abs(nav["nav"] - initial) < 0.01, (
        f"NAV should equal initial capital at entry, got {nav['nav']:.2f} vs {initial:.2f}"
    )


def test_nav_equals_initial_after_swing_entry(zero_fee_engine):
    """Same test for swing (1x leverage): NAV = initial at entry (no price move)."""
    e = zero_fee_engine
    initial = e.get_account_balance("india")

    e.execute_signal(_signal(entry=100.0, qty=10, strategy="swing"))

    nav = e.get_portfolio_nav("india")
    assert abs(nav["nav"] - initial) < 0.01


def test_nav_components_sum_correctly(zero_fee_engine):
    """NAV = cash + reserved_margin + unrealized_pnl."""
    e = zero_fee_engine
    e.execute_signal(_signal(entry=100.0, qty=10, strategy="swing"))

    with _conn() as conn:
        conn.execute("UPDATE positions SET current_price = 105.0 WHERE ticker = 'TEST'")
        conn.commit()

    nav_data = e.get_portfolio_nav("india")
    expected = nav_data["cash"] + nav_data["reserved_margin"] + nav_data["unrealized_pnl"]
    assert abs(nav_data["nav"] - expected) < 0.01


def test_nav_increases_only_on_realized_pnl(zero_fee_engine):
    """NAV only increases by realized PnL after a profitable close."""
    e = zero_fee_engine
    initial_nav = e.get_portfolio_nav("india")["nav"]

    e.execute_signal(_signal(entry=100.0, qty=10, strategy="swing"))
    snaps = {"TEST": _snap("TEST", price=120.0)}
    e.evaluate_open_positions("india", snaps)

    final_nav = e.get_portfolio_nav("india")["nav"]
    realized = (120.0 - 100.0) * 10   # 200
    assert abs(final_nav - initial_nav - realized) < 0.01


# ── 8. Position average price ──────────────────────────────────────────────────

def test_avg_cost_stored_correctly(zero_fee_engine):
    """avg_cost in DB equals the entry fill price."""
    e = zero_fee_engine
    e.execute_signal(_signal(entry=100.0, qty=10, strategy="swing"))

    with _conn() as conn:
        row = conn.execute("SELECT avg_cost FROM positions WHERE ticker='TEST' ORDER BY id DESC LIMIT 1").fetchone()
    assert row is not None
    assert row["avg_cost"] == 100.0


def test_avg_cost_with_slippage():
    """avg_cost stored = entry_price × (1 + slippage) on a BUY."""
    fees = FeeSchedule(paper_per_side_pct=0.0, slippage_pct=0.001)
    e = PaperTradingEngine(fee_schedule=fees)
    e.full_reset()
    e.execute_signal(_signal(entry=1000.0, qty=1, strategy="swing",
                              sl=950.0, tp=1100.0))

    with _conn() as conn:
        row = conn.execute("SELECT avg_cost FROM positions WHERE ticker='TEST' ORDER BY id DESC LIMIT 1").fetchone()
    expected_fill = 1000.0 * 1.001   # slippage applied
    assert abs(row["avg_cost"] - expected_fill) < 0.01



# ── 9. Fees ────────────────────────────────────────────────────────────────────

def test_fees_deducted_from_cash_on_entry():
    """Entry fees are deducted from cash in addition to margin."""
    fees = FeeSchedule(paper_per_side_pct=0.001, slippage_pct=0.0)
    e = PaperTradingEngine(fee_schedule=fees)
    e.full_reset()

    before = e.get_account_balance("india")
    e.execute_signal(_signal(entry=100.0, qty=10, strategy="swing"))
    after = e.get_account_balance("india")

    margin = 100.0 * 10
    entry_fee = 100.0 * 10 * 0.001
    expected_deduction = margin + entry_fee
    assert abs(before - after - expected_deduction) < 0.01


def test_fees_reduce_realized_pnl():
    """Net realized PnL = gross PnL minus exit fees."""
    fees = FeeSchedule(paper_per_side_pct=0.001, slippage_pct=0.0)
    e = PaperTradingEngine(fee_schedule=fees)
    e.full_reset()

    # entry=100, qty=50 -> notional=5000, entry fee=5.00
    e.execute_signal(_signal(entry=100.0, qty=50, strategy="swing",
                              sl=90.0, tp=110.0))

    snaps = {"TEST": _snap("TEST", price=110.0)}
    e.evaluate_open_positions("india", snaps)

    # Gross PnL = (110-100)*50 = 500
    # Exit fee  = 110*50*0.001 = 5.50
    # Net PnL   = 494.50
    pnl = e.get_daily_realized_pnl("india")
    assert abs(pnl - 494.50) < 0.5


def test_fee_schedule_compute_buy_side():
    """FeeSchedule.compute_fees: 0.05% of notional on BUY."""
    fs = FeeSchedule(paper_per_side_pct=0.0005, slippage_pct=0.0)
    fee = fs.compute_fees("india", "BUY", filled_price=200.0, quantity=50, is_paper=True)
    assert abs(fee - 200.0 * 50 * 0.0005) < 0.001


def test_fee_schedule_compute_sell_side():
    """FeeSchedule.compute_fees: 0.05% of notional on SELL."""
    fs = FeeSchedule(paper_per_side_pct=0.0005, slippage_pct=0.0)
    fee = fs.compute_fees("india", "SELL", filled_price=210.0, quantity=50, is_paper=True)
    assert abs(fee - 210.0 * 50 * 0.0005) < 0.001


# ── 10. Slippage ──────────────────────────────────────────────────────────────

def test_slippage_applied_on_buy():
    """BUY fill price = entry_price × (1 + slippage_pct)."""
    fees = FeeSchedule(paper_per_side_pct=0.0, slippage_pct=0.001)
    e = PaperTradingEngine(fee_schedule=fees)
    e.full_reset()
    e.execute_signal(_signal(entry=1000.0, qty=1, strategy="swing",
                              sl=950.0, tp=1100.0))

    with _conn() as conn:
        row = conn.execute("SELECT avg_cost FROM positions WHERE ticker='TEST' ORDER BY id DESC LIMIT 1").fetchone()
    assert abs(row["avg_cost"] - 1001.0) < 0.01   # 1000 × 1.001


def test_slippage_applied_on_sell_reduces_proceeds():
    """
    Exit via evaluate_open_positions should apply sell-side slippage.
    Net PnL is lower when slippage > 0.
    """
    fees_no_slip = FeeSchedule(paper_per_side_pct=0.0, slippage_pct=0.0)
    fees_with_slip = FeeSchedule(paper_per_side_pct=0.0, slippage_pct=0.001)

    def run(fs):
        with _conn() as conn:
            conn.execute("DELETE FROM positions")
            conn.commit()
        eng = PaperTradingEngine(fee_schedule=fs)
        eng.reset_account_balances()
        eng.execute_signal(_signal(entry=100.0, qty=10, strategy="swing", sl=90.0, tp=110.0))
        snaps = {"TEST": _snap("TEST", price=110.0)}
        eng.evaluate_open_positions("india", snaps)
        return eng.get_daily_realized_pnl("india")

    pnl_no_slip = run(fees_no_slip)
    pnl_with_slip = run(fees_with_slip)

    assert pnl_no_slip > pnl_with_slip, (
        "Slippage should reduce exit proceeds → lower net PnL"
    )



# ── 11. Leverage accounting ────────────────────────────────────────────────────

def test_3x_leverage_uses_less_cash(zero_fee_engine):
    """Intraday (3x) blocks only 1/3 of notional as margin."""
    e = zero_fee_engine
    before = e.get_account_balance("india")

    e.execute_signal(_signal(entry=300.0, qty=10, strategy="intraday",
                              sl=295.0, tp=315.0))
    after = e.get_account_balance("india")

    margin_blocked = before - after
    notional = 300.0 * 10
    assert abs(margin_blocked - notional / 3.0) < 0.01, (
        f"3x leverage: expected margin={notional/3:.2f}, got {margin_blocked:.2f}"
    )


def test_leverage_margin_returned_on_close(zero_fee_engine):
    """After closing leveraged position, cash returns to initial ± PnL."""
    e = zero_fee_engine
    initial = e.get_account_balance("india")

    e.execute_signal(_signal(entry=300.0, qty=10, strategy="intraday",
                              sl=290.0, tp=315.0))
    snaps = {"TEST": _snap("TEST", price=315.0)}
    e.evaluate_open_positions("india", snaps)

    final = e.get_account_balance("india")
    gain = (315.0 - 300.0) * 10   # 150.0
    assert abs(final - initial - gain) < 0.01


def test_nav_not_inflated_by_3x_leverage_multiple_positions(zero_fee_engine):
    """Multiple leveraged open positions must not inflate NAV beyond initial capital."""
    e = zero_fee_engine
    initial_nav = e.get_portfolio_nav("india")["nav"]

    # Open 3 separate positions; combined notional > initial capital
    e.execute_signal(_signal("AAAA", entry=200.0, qty=5, strategy="intraday",
                              sl=195.0, tp=210.0))
    e.execute_signal(_signal("BBBB", entry=150.0, qty=5, strategy="intraday",
                              sl=145.0, tp=160.0))
    e.execute_signal(_signal("CCCC", entry=100.0, qty=5, strategy="scalping",
                              sl=99.0, tp=102.0))

    nav = e.get_portfolio_nav("india")
    # At entry with no price move: NAV must equal initial (no value created)
    assert abs(nav["nav"] - initial_nav) < 1.0, (
        f"NAV should stay at initial capital at entry, got {nav['nav']:.2f}"
    )


# ── 12. Transaction ledger ─────────────────────────────────────────────────────

def test_ledger_entries_on_buy(zero_fee_engine):
    """BUY creates at minimum a MARGIN_BLOCK ledger entry."""
    e = zero_fee_engine
    e.execute_signal(_signal(entry=100.0, qty=10, strategy="swing"))

    ledger = e.get_transaction_ledger("india")
    types = [row["entry_type"] for row in ledger]
    assert "MARGIN_BLOCK" in types


def test_ledger_entries_on_close(zero_fee_engine):
    """Close creates MARGIN_RELEASE + REALIZED_PNL ledger entries."""
    e = zero_fee_engine
    e.execute_signal(_signal(entry=100.0, qty=10, strategy="swing", sl=90.0, tp=110.0))
    snaps = {"TEST": _snap("TEST", price=110.0)}
    e.evaluate_open_positions("india", snaps)

    ledger = e.get_transaction_ledger("india")
    types = {row["entry_type"] for row in ledger}
    assert "MARGIN_RELEASE" in types
    assert "REALIZED_PNL" in types


def test_ledger_fee_entries_present():
    """FEE entries appear in ledger when fees > 0."""
    fees = FeeSchedule(paper_per_side_pct=0.001, slippage_pct=0.0)
    e = PaperTradingEngine(fee_schedule=fees)
    e.full_reset()

    e.execute_signal(_signal(entry=100.0, qty=10, strategy="swing", sl=90.0, tp=110.0))
    snaps = {"TEST": _snap("TEST", price=110.0)}
    e.evaluate_open_positions("india", snaps)

    ledger = e.get_transaction_ledger("india")
    fee_entries = [row for row in ledger if row["entry_type"] == "FEE"]
    assert len(fee_entries) >= 2, "Expected at least entry + exit FEE ledger rows"



def test_ledger_balance_after_is_consistent(zero_fee_engine):
    """
    Ledger entries' balance_after should be monotonically trackable.
    After a buy then profitable sell, final balance_after in ledger
    should match actual account cash.
    """
    e = zero_fee_engine
    e.execute_signal(_signal(entry=100.0, qty=10, strategy="swing"))
    snaps = {"TEST": _snap("TEST", price=120.0)}
    e.evaluate_open_positions("india", snaps)

    ledger = e.get_transaction_ledger("india", limit=1)
    assert len(ledger) >= 1
    latest_balance = ledger[0]["balance_after"]
    actual_cash = e.get_account_balance("india")
    assert abs(latest_balance - actual_cash) < 0.01


def test_ledger_reset_entry():
    """full_reset writes RESET entries to ledger."""
    fees = FeeSchedule(paper_per_side_pct=0.0, slippage_pct=0.0)
    e = PaperTradingEngine(fee_schedule=fees)
    e.full_reset()

    ledger = e.get_transaction_ledger("india")
    types = [row["entry_type"] for row in ledger]
    assert "RESET" in types


# ── Profit / Loss lifecycle ────────────────────────────────────────────────────

def test_profit_exit_full_lifecycle(zero_fee_engine):
    """Complete buy → profit exit lifecycle: all accounting checks."""
    e = zero_fee_engine
    initial_cash = e.get_account_balance("india")
    initial_nav = e.get_portfolio_nav("india")["nav"]

    # 1. BUY 10 shares @ 100
    order = e.execute_signal(_signal(entry=100.0, qty=10, strategy="swing"))
    assert order is not None
    assert e.get_account_balance("india") == initial_cash - 1000.0
    assert e.get_reserved_margin("india") == 1000.0
    assert abs(e.get_portfolio_nav("india")["nav"] - initial_nav) < 0.01

    # 2. Update price mark to 120
    with _conn() as conn:
        conn.execute("UPDATE positions SET current_price=120.0 WHERE ticker='TEST'")
        conn.commit()
    nav_with_unrealized = e.get_portfolio_nav("india")["nav"]
    assert abs(nav_with_unrealized - initial_nav - 200.0) < 0.01   # +200 unrealized

    # 3. Close at target (price hits 115)
    snaps = {"TEST": _snap("TEST", price=115.0)}
    reports = e.evaluate_open_positions("india", snaps)
    assert len(reports) > 0

    # 4. After close: no open positions, reserved_margin = 0
    assert e.get_reserved_margin("india") == 0.0
    nav = e.get_portfolio_nav("india")
    assert nav["open_positions_count"] == 0

    # 5. Final cash = initial + gain
    gain = (115.0 - 100.0) * 10   # 150
    assert abs(e.get_account_balance("india") - initial_cash - gain) < 0.01

    # 6. Final NAV = initial + gain
    assert abs(nav["nav"] - initial_nav - gain) < 0.01


def test_loss_exit_full_lifecycle(zero_fee_engine):
    """Buy → stop loss hit: cash reduced by exact loss amount."""
    e = zero_fee_engine
    initial = e.get_account_balance("india")

    e.execute_signal(_signal(entry=100.0, sl=95.0, qty=20, strategy="swing",
                              tp=115.0))
    snaps = {"TEST": _snap("TEST", price=95.0)}
    e.evaluate_open_positions("india", snaps)

    expected = initial + (95.0 - 100.0) * 20   # -100
    assert abs(e.get_account_balance("india") - expected) < 0.01


# ── Partial sell ───────────────────────────────────────────────────────────────

def test_partial_sell_reduces_quantity(zero_fee_engine):
    """sell_partial closes part of a position, leaving remainder open."""
    e = zero_fee_engine
    e.execute_signal(_signal(entry=100.0, qty=20, strategy="swing",
                              sl=90.0, tp=120.0))

    report = e.sell_partial("TEST", "india", exit_price_raw=110.0, sell_quantity=10)
    assert report is not None

    with _conn() as conn:
        row = conn.execute(
            "SELECT quantity, status FROM positions WHERE ticker='TEST' AND status='OPEN'"
        ).fetchone()
    assert row is not None
    assert row["quantity"] == 10


def test_partial_sell_returns_proportional_cash(zero_fee_engine):
    """Partial sell returns margin for only the closed shares."""
    e = zero_fee_engine
    e.execute_signal(_signal(entry=100.0, qty=20, strategy="swing",
                              sl=90.0, tp=120.0))

    before_partial = e.get_account_balance("india")
    e.sell_partial("TEST", "india", exit_price_raw=110.0, sell_quantity=10)
    after_partial = e.get_account_balance("india")

    # margin released = (100*10)/1 = 1000, gain = (110-100)*10 = 100
    gain = (110.0 - 100.0) * 10
    returned = 100.0 * 10 + gain
    assert abs(after_partial - before_partial - returned) < 0.01


def test_partial_sell_then_full_close(zero_fee_engine):
    """Partial sell then full close: total PnL and cash add up correctly."""
    e = zero_fee_engine
    initial = e.get_account_balance("india")

    e.execute_signal(_signal(entry=100.0, qty=20, strategy="swing",
                              sl=90.0, tp=125.0))   # tp=125 so remaining 10 close at price=125


    # Sell half at 110 (gain=100)
    e.sell_partial("TEST", "india", exit_price_raw=110.0, sell_quantity=10)

    # Close remaining 10 at target 120 (gain=200)
    snaps = {"TEST": _snap("TEST", price=125.0)}
    e.evaluate_open_positions("india", snaps)

    final_cash = e.get_account_balance("india")
    gain1 = (110.0 - 100.0) * 10
    gain2 = (125.0 - 100.0) * 10
    expected = initial + gain1 + gain2
    assert abs(final_cash - expected) < 0.01


# ── Multiple positions ─────────────────────────────────────────────────────────

def test_multiple_positions_nav_is_additive(zero_fee_engine):
    """NAV with multiple open positions = initial + sum(unrealized_pnl)."""
    e = zero_fee_engine
    initial_nav = e.get_portfolio_nav("india")["nav"]

    e.execute_signal(_signal("AAAA", entry=100.0, qty=10, strategy="swing",
                              sl=90.0, tp=120.0))
    e.execute_signal(_signal("BBBB", entry=200.0, qty=5, strategy="swing",
                              sl=190.0, tp=220.0))

    with _conn() as conn:
        conn.execute("UPDATE positions SET current_price=108.0 WHERE ticker='AAAA'")
        conn.execute("UPDATE positions SET current_price=210.0 WHERE ticker='BBBB'")
        conn.commit()

    nav = e.get_portfolio_nav("india")
    upnl_a = (108.0 - 100.0) * 10   # 80
    upnl_b = (210.0 - 200.0) * 5    # 50
    expected_nav = initial_nav + upnl_a + upnl_b
    assert abs(nav["nav"] - expected_nav) < 0.01


def test_multiple_positions_independent_close(zero_fee_engine):
    """Closing one position does not affect another's accounting."""
    e = zero_fee_engine
    initial = e.get_account_balance("india")

    e.execute_signal(_signal("AAAA", entry=100.0, qty=10, strategy="swing",
                              sl=90.0, tp=115.0))
    e.execute_signal(_signal("BBBB", entry=50.0, qty=10, strategy="swing",
                              sl=45.0, tp=60.0))

    # Close only AAAA at profit
    snaps = {
        "AAAA": _snap("AAAA", price=115.0),
        "BBBB": _snap("BBBB", price=48.0),  # below target, above SL — stays open
    }
    e.evaluate_open_positions("india", snaps)

    # BBBB should still be open — use ORDER BY id DESC to get the most recent row
    with _conn() as conn:
        bbbb = conn.execute(
            "SELECT status FROM positions WHERE ticker='BBBB' ORDER BY id DESC LIMIT 1"
        ).fetchone()
    assert bbbb["status"] == "OPEN"


    # AAAA should be closed with correct PnL
    pnl_a = (115.0 - 100.0) * 10   # 150
    # Cash = initial - 500 (BBBB margin) + 1000 (AAAA margin back) + 150 (gain)
    expected_cash = initial - 1000.0 - 500.0 + 1000.0 + pnl_a
    assert abs(e.get_account_balance("india") - expected_cash) < 0.01


# ── Intraday square-off ────────────────────────────────────────────────────────

def test_intraday_squareoff_closes_both_strategies(zero_fee_engine):
    """square_off_intraday closes both 'intraday' and 'scalping' positions."""
    e = zero_fee_engine
    e.execute_signal(_signal("INTRA", entry=100.0, qty=5, strategy="intraday",
                              sl=95.0, tp=108.0))
    e.execute_signal(_signal("SCALP", entry=50.0, qty=10, strategy="scalping",
                              sl=49.5, tp=51.0))
    e.execute_signal(_signal("SWING", entry=200.0, qty=2, strategy="swing",
                              sl=190.0, tp=220.0))

    snaps = {
        "INTRA": _snap("INTRA", price=103.0),
        "SCALP": _snap("SCALP", price=50.5),
        "SWING": _snap("SWING", price=205.0),
    }
    closed = e.square_off_intraday("india", snaps)
    assert len(closed) == 2

    # SWING position must remain open
    with _conn() as conn:
        swing = conn.execute(
            "SELECT status FROM positions WHERE ticker='SWING' ORDER BY id DESC LIMIT 1"
        ).fetchone()
    assert swing["status"] == "OPEN"



def test_intraday_squareoff_cash_restored(zero_fee_engine):
    """After squaring off an intraday position, cash reflects correct P&L."""
    e = zero_fee_engine
    initial = e.get_account_balance("india")

    e.execute_signal(_signal("INTRA", entry=200.0, qty=5, strategy="intraday",
                              sl=195.0, tp=215.0))
    snaps = {"INTRA": _snap("INTRA", price=204.0)}
    e.square_off_intraday("india", snaps)

    gain = (204.0 - 200.0) * 5   # 20
    assert abs(e.get_account_balance("india") - initial - gain) < 0.01


# ── Rejection cases ────────────────────────────────────────────────────────────

def test_insufficient_cash_rejects_signal(zero_fee_engine):
    """Signal is rejected when margin_required > available cash."""
    e = zero_fee_engine
    # Try to buy 1000 shares @ 1000 each with swing (1x) → margin = 1,000,000
    # Cash is only 10,000
    sig = _signal(entry=1000.0, qty=1000, strategy="swing", sl=950.0, tp=1100.0)
    order = e.execute_signal(sig)
    assert order is None
    # Cash should be unchanged
    assert e.get_account_balance("india") == 10_000.0


# ── US market accounting ───────────────────────────────────────────────────────

def test_us_account_independent_from_india(zero_fee_engine):
    """INR and USD accounts are completely separate."""
    e = zero_fee_engine
    assert e.get_account_balance("us") == 1_000.0

    sig = _signal("AAPL", market="us", entry=150.0, qty=2, strategy="swing",
                  sl=140.0, tp=170.0)
    order = e.execute_signal(sig)
    assert order is not None

    # INR account should be untouched
    assert e.get_account_balance("india") == 10_000.0
    # USD account should have margin deducted
    assert e.get_account_balance("us") < 1_000.0
