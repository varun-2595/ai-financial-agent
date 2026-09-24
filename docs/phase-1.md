# Phase 1: Correct Portfolio Accounting & Execution Engine

## 1. Overview & Objectives

Phase 1 establishes mathematically sound, production-grade portfolio and transaction accounting for the Aegis AI trading system across both India (NSE/BSE) and US (NYSE/NASDAQ) paper trading environments.

The primary objective was fixing accounting bugs—most notably the **artificial inflation of portfolio NAV by leverage**—while introducing explicit reserved margin tracking, buying power calculation, fee and slippage models, partial-sell support, and a complete double-entry transaction ledger.

---

## 2. Core Accounting Principles & Formulas

### 2.1. Portfolio Net Asset Value (NAV / Equity)
**The Golden Rule:** Portfolio equity must never be artificially increased or created by opening leveraged positions.

$$\text{NAV} = \text{Cash} + \text{Reserved Margin} + \sum \text{Unrealized P\&L}_{\text{open positions}}$$

* **Cash:** Free, liquid cash available for withdrawal or new positions.
* **Reserved Margin:** Capital locked as collateral for active open positions ($\frac{\text{Filled Price} \times \text{Quantity}}{\text{Leverage}}$).
* **Unrealized P&L:** Current mark-to-market gain/loss across open positions.
* **Leverage Isolation:** The borrowed portion of leveraged positions is strictly excluded from equity. At trade entry (with 0 slippage/fees and unchanged market price), $\Delta \text{NAV} = 0$.

### 2.2. Buying Power
* **Equity Buying Power (Swing / Positional):**
  $$\text{Buying Power}_{\text{equity}} = \text{Cash}$$
* **Intraday Buying Power (Scalping / Intraday):**
  $$\text{Buying Power}_{\text{intraday}} = \text{Cash} \times \text{Leverage Multiplier} \quad (\text{default } 3\times)$$

### 2.3. Trade Execution Lifecycle & Accounting Entries

#### On BUY (Entry):
1. **Slippage Applied:** $\text{Filled Price} = \text{Signal Entry Price} \times (1 + \text{Slippage Pct})$
2. **Margin Blocked:** $\text{Margin Blocked} = \frac{\text{Filled Price} \times \text{Quantity}}{\text{Leverage}}$
3. **Entry Fees:** $\text{Entry Fees} = \text{FeeSchedule.compute\_fees}(\text{market}, \text{"BUY"}, \text{Filled Price}, \text{Quantity})$
4. **Cash Deduction:**
   $$\text{Cash}_{\text{new}} = \text{Cash}_{\text{old}} - (\text{Margin Blocked} + \text{Entry Fees})$$
   $$\text{Reserved Margin}_{\text{new}} = \text{Reserved Margin}_{\text{old}} + \text{Margin Blocked}$$
5. **Ledger Entries:** `MARGIN_BLOCK` (debit) and `FEE` (debit).

#### On SELL (Exit / Target / Stop Loss / Square-Off):
1. **Slippage Applied:** $\text{Exit Price} = \text{Market Price} \times (1 - \text{Slippage Pct})$
2. **Gross P&L (for LONG):** $\text{Gross P\&L} = (\text{Exit Price} - \text{Avg Cost}) \times \text{Quantity}$
3. **Exit Fees:** $\text{Exit Fees} = \text{FeeSchedule.compute\_fees}(\text{market}, \text{"SELL"}, \text{Exit Price}, \text{Quantity})$
4. **Net Realized P&L:** $\text{Net P\&L} = \text{Gross P\&L} - \text{Exit Fees}$
5. **Cash Returned:** $\text{Cash Returned} = \text{Margin Blocked} + \text{Net P\&L}$
6. **Cash & Margin Update:**
   $$\text{Cash}_{\text{new}} = \text{Cash}_{\text{old}} + \text{Cash Returned}$$
   $$\text{Reserved Margin}_{\text{new}} = \text{Reserved Margin}_{\text{old}} - \text{Margin Blocked}$$
7. **Ledger Entries:** `MARGIN_RELEASE` (credit), `FEE` (debit), and `REALIZED_PNL` (credit/debit).

---

## 3. Database Schema Updates (`trading.db`)

### `accounts` Table
Added `reserved_margin` column:
```sql
CREATE TABLE IF NOT EXISTS accounts (
    account_id      TEXT PRIMARY KEY,  -- 'paper_inr' or 'paper_usd'
    currency        TEXT NOT NULL,     -- 'INR' or 'USD'
    cash            REAL NOT NULL,     -- free liquid cash
    initial_cash    REAL NOT NULL,     -- starting principal
    reserved_margin REAL NOT NULL DEFAULT 0.0, -- collateral locked in active trades
    updated_at      TEXT NOT NULL
);
```

### `positions` Table
Added `margin_blocked` and `fees_paid` columns:
```sql
CREATE TABLE IF NOT EXISTS positions (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    ticker          TEXT NOT NULL,
    market          TEXT NOT NULL,
    strategy        TEXT NOT NULL,
    direction       TEXT NOT NULL,
    quantity        INTEGER NOT NULL,
    avg_cost        REAL NOT NULL,
    current_price   REAL,
    stop_loss       REAL,
    target_price    REAL,
    margin_blocked  REAL NOT NULL DEFAULT 0.0, -- exact cash collateral deducted
    fees_paid       REAL NOT NULL DEFAULT 0.0, -- total cumulative fees for position
    status          TEXT NOT NULL DEFAULT 'OPEN', -- 'OPEN', 'CLOSED', 'CANCELLED'
    is_paper        INTEGER NOT NULL DEFAULT 1,
    opened_at       TEXT NOT NULL,
    closed_at       TEXT,
    realized_pnl    REAL DEFAULT 0.0          -- net PnL after all exit fees
);
```

### `ledger` Table (New Double-Entry Audit Trail)
```sql
CREATE TABLE IF NOT EXISTS ledger (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    account_id      TEXT NOT NULL,
    entry_type      TEXT NOT NULL,  -- 'MARGIN_BLOCK', 'MARGIN_RELEASE', 'FEE', 'REALIZED_PNL', 'RESET'
    amount          REAL NOT NULL,  -- positive = credit, negative = debit
    balance_after   REAL NOT NULL,  -- snapshot of cash balance after transaction
    ref_order_id    TEXT,
    ref_position_id INTEGER,
    description     TEXT,
    created_at      TEXT NOT NULL
);
```

---

## 4. Key Architectural Changes & Bug Fixes

1. **Leverage NAV Bug Fixed:** Eliminated double counting where full notional value was added to cash in NAV calculation. Implemented `get_portfolio_nav()` using cash + reserved margin + unrealized P&L.
2. **Fee Schedule Module (`src/trading/fees.py`):** Configurable `FeeSchedule` providing realistic models for paper trading (0.05% flat per-side default), NSE (STT, Exchange turnover, SEBI, GST, Stamp duty), and US markets (SEC fee, FINRA TAF).
3. **Symmetric Adverse Slippage:** Fill price adjustments applied on both entry (BUY at $+0.05\%$) and exits (SELL at $-0.05\%$).
4. **Partial Sell Support (`sell_partial`):** Enables partial lot liquidations with exact proportional margin release, FIFO cost accounting, and dedicated ledger entries while keeping the remaining position active.
5. **Config Root Path Fix:** Fixed `ROOT` path resolution in `src/utils/config.py` from `src/config/` to project root `config/`, ensuring `settings.yaml` settings are consistently loaded.
6. **Scan Loop Margin Deduction:** Fixed `src/scheduler/jobs.py` where in-loop signal sizing deducted full notional from the temporary `cash` variable instead of the actual margin reserved.

---

## 5. Test Suite Verification

The complete test suite runs 61 automated tests verifying every accounting requirement:

* `tests/test_core.py` (7 tests): Market hours, technical indicators, indicators math, advisory classification, position sizer.
* `tests/test_phase1_accounting.py` (45 tests):
  * Cash deduction on BUY / credit on profitable & loss SELL
  * Reserved margin locking & release
  * Buying power calculations (equity & $3\times$ intraday)
  * Realized P&L net of fees
  * Unrealized P&L mark-to-market
  * Portfolio NAV invariance under leverage
  * Average cost tracking with slippage
  * Fee computations (BUY & SELL legs)
  * Adverse slippage impact on trade proceeds
  * Multi-position NAV additivity & independent exits
  * Partial sell accounting & remainder handling
  * Intraday square-off lifecycle
  * Double-entry ledger generation & balance auditing
  * US vs India account isolation
* `tests/test_scalping_and_learning.py` (5 tests): Capital sizing, scalping signal generation, learning engine autopsies.
* `tests/test_telegram_commands.py` (4 tests): Trading state management, dynamic watchlists, authorization, emergency close.

```bash
$ pytest tests/ -v
======================== 61 passed, 1 warning in 1.09s =========================
```
