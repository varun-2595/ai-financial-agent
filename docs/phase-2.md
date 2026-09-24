# Phase 2: Deterministic Historical Backtesting Engine

## 1. Overview & Objectives

Phase 2 establishes a high-performance, deterministic historical backtesting engine for the Aegis AI trading system. 

The backtester executes the **exact same strategy, risk management, and portfolio accounting logic** as paper and live trading, enabling verifiable historical validation across India (NSE/BSE) and US (NYSE/NASDAQ) markets.

---

## 2. System Architecture

```
Historical Data (yfinance / Cache)
       │
       ▼
Point-in-Time Data Provider (Strict No-Lookahead Slicing)
       │
       ▼
Autonomous Screener / Universe Filter
       │
       ▼
Signal Generator (Technical Heuristics & Key Price Levels)
       │
       ▼
Risk Engine (Capital Caps, R:R >= 1.3, Sector Exposure)
       │
       ▼
Execution Simulator (Slippage, Fees, Intrabar SL/TP Hits)
       │
       ▼
Portfolio Accounting (Cash, Reserved Margin, NAV Invariance)
       │
       ▼
Quantitative Metrics & Multi-Format Reporting (CLI, Markdown, CSV)
```

---

## 3. Core Modules (`src/backtesting/`)

### 3.1. Data Provider (`src/backtesting/data_provider.py`)
* **Point-in-Time Slicing:** Given simulated timestamp $T$, the provider strictly filters $\text{timestamp} \le T$, ensuring zero future data leakage.
* **Local Parquet Caching:** Caches downloaded ticker histories under `data/backtest_cache/` to guarantee fast, repeatable offline runs.
* **Deterministic Fallback:** Features built-in synthetic OHLCV generation when operating offline without live network access.
* **Data Model Bridge:** Converts raw price series into canonical `Quote` and `StockSnapshot` structures used across technicals, risk, and screening modules.

### 3.2. Execution Simulator (`src/backtesting/execution_simulator.py`)
* **Realistic Slippage:** Applies symmetric adverse slippage on both entry ($+0.05\%$) and exit ($-0.05\%$) orders.
* **Transaction Costs:** Uses Phase 1 `FeeSchedule` to simulate brokerage, exchange turnover, STT/SEC, FINRA, and stamp charges.
* **Intrabar SL/TP Simulation:** Evaluates bar `High` and `Low` to check if Stop Loss ($\text{Low} \le \text{SL}$) or Target Price ($\text{High} \ge \text{TP}$) was triggered.
* **Conservative Conflict Resolution:** When a wide-range bar satisfies both SL and TP conditions ($\text{Low} \le \text{SL}$ and $\text{High} \ge \text{TP}$), the simulator conservatively resolves the trade as a **Stop Loss**.
* **Intraday Square-Off:** Automatically squares off `scalping` and `intraday` positions at bar close.

### 3.3. Backtest Engine (`src/backtesting/engine.py`)
* **State Machine:** Steps through trading days chronologically, managing cash balances, reserved margin, active positions, closed trades, and double-entry transaction ledger logs.
* **NAV Invariance:** Uses the Phase 1 NAV formula:
  $$\text{NAV} = \text{Cash} + \text{Reserved Margin} + \sum \text{Unrealized P\&L}_{\text{open positions}}$$
* **Position Limits:** Enforces concurrent open position caps (`max_positions`, default 5) and portfolio risk ceilings.

### 3.4. Performance Metrics (`src/backtesting/metrics.py`)
Computes institutional-grade quantitative trading metrics:
* **Returns:** Total Net Profit, Total Return (%), CAGR (%)
* **Benchmark Comparison:** Alpha vs. Benchmark (Nifty 50 for India, S&P 500 for US)
* **Risk Ratios:** Annualized Volatility, Sharpe Ratio ($R_f=5\%$), Sortino Ratio (downside risk), Calmar Ratio
* **Drawdown Analysis:** Max Drawdown (%), Max Drawdown Amount, Max Drawdown Duration (days)
* **Trade Analytics:** Win Rate (%), Profit Factor ($\frac{\sum \text{Gains}}{\sum |\text{Losses}|}$), Expectancy ($/₹ per trade), Win/Loss Ratio, Average Holding Time (days)
* **Cost Impact:** Cumulative Brokerage & Regulatory Fees, Cumulative Slippage Drag
* **Market Exposure:** % of calendar/trading days capital was actively deployed in positions

### 3.5. Reporting & CLI (`src/backtesting/reporting.py`, `src/backtesting/__main__.py`)
* **Terminal Summary:** Rich ASCII dashboard.
* **Markdown Export:** Generates full `.md` report tables.
* **CSV Export:** Exports `trades.csv` and `nav.csv`.

---

## 4. CLI Usage & Examples

Run backtests directly from the terminal using `python -m src.backtesting`:

### India (NSE) Backtest
```bash
python3 -m src.backtesting \
  --market india \
  --start 2025-01-01 \
  --end 2025-06-30 \
  --strategy swing \
  --capital 10000 \
  --report reports/sample_backtest_india.md \
  --csv-dir reports/csv_india
```

### US (NYSE/NASDAQ) Backtest
```bash
python3 -m src.backtesting \
  --market us \
  --start 2025-01-01 \
  --end 2025-06-30 \
  --tickers AAPL,MSFT,NVDA,TSLA,AVGO \
  --strategy swing \
  --capital 1000 \
  --report reports/sample_backtest_us.md \
  --csv-dir reports/csv_us
```

### CLI Options Reference
| Flag | Type | Default | Description |
|---|---|---|---|
| `--market` | `india \| us` | `india` | Target market |
| `--start` | `YYYY-MM-DD` | `2025-01-01` | Backtest start date |
| `--end` | `YYYY-MM-DD` | `2025-12-31` | Backtest end date |
| `--strategy` | `scalping \| intraday \| swing \| positional` | `swing` | Trading strategy |
| `--capital` | `float` | Configured paper capital | Starting cash balance |
| `--tickers` | `comma-separated list` | Default market pool | Specific universe of symbols |
| `--leverage` | `float` | `3.0` (intra) / `1.0` (swing) | Margin leverage multiplier |
| `--max-positions` | `int` | `5` | Maximum concurrent positions |
| `--report` | `path` | `reports/backtest_...md` | Markdown report export path |
| `--csv-dir` | `path` | `None` | CSV directory for trades & NAV curves |

---

## 5. Sample Backtest Results

### 5.1. US Tech Universe (2025-01-01 to 2025-06-30)
```
══════════════════════════════════════════════════════════════════════
📊 AEGIS BACKTEST REPORT — US (SWING)
📅 Period: 2025-01-01 to 2025-06-30 (129 trading days)
══════════════════════════════════════════════════════════════════════

💰 PORTFOLIO PERFORMANCE
──────────────────────────────────────────────────────────────────────
  Initial Capital:       $    1,000.00
  Ending NAV:            $    1,028.76
  Peak NAV:              $    1,045.44
  Total Net Profit:      $       28.76 ( +2.88%)
  CAGR:                                +5.92%
  Benchmark Return:                    +1.66%
  Alpha vs Benchmark:                  +1.22%

🛡️ RISK & DRAWDOWN
──────────────────────────────────────────────────────────────────────
  Annualized Volatility:                5.47%
  Sharpe Ratio (Rf=5%):                 0.13
  Sortino Ratio:                        0.18
  Calmar Ratio:                         1.61
  Max Drawdown:                      -  3.67% ($37.18)
  Max Drawdown Duration:              72 days
  Market Exposure:                     94.57%

🎯 TRADE ANALYTICS
──────────────────────────────────────────────────────────────────────
  Total Closed Trades:                    10
  Win / Loss / BE:                    5 / 5 / 0
  Win Rate:                            50.00%
  Profit Factor:                        1.30
  Expectancy:            $        1.62 per trade
  Average Trade P&L:     $        1.62
  Average Win / Loss:    $   14.07 / $   10.82 (Ratio: 1.30)
  Largest Win / Loss:    $   22.53 / $  -17.04
  Average Hold Duration:                40.8 days
══════════════════════════════════════════════════════════════════════
```

---

## 6. Test Suite & Verification

The test suite contains 73 automated unit and integration tests passing with zero errors:

```bash
$ pytest tests/ -v
======================== 73 passed, 1 warning in 1.61s =========================
```

* `tests/test_phase2_backtesting.py` (12 tests): Point-in-time slicing, lookahead prevention, execution simulator, intrabar SL/TP conflict resolution, metrics math, multi-market runs, position constraints, CSV/Markdown export.
* `tests/test_phase1_accounting.py` (45 tests): Portfolio accounting, NAV invariance, double-entry ledger, partial exits, fees, slippage.
* `tests/test_core.py` (7 tests): Core technical indicators, support/resistance, market calendars.
* `tests/test_scalping_and_learning.py` (5 tests): Sizing, signal generation, learning autopsies.
* `tests/test_telegram_commands.py` (4 tests): Trading state management, dynamic watchlists, bot controls.
