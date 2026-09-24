# Phase 3: Rigorous Performance Evaluation & Benchmark Analytics

## 1. Overview & Objectives

Phase 3 introduces institutional-grade performance evaluation and benchmark comparative analytics for the Aegis AI trading system. 

It provides rigorous mathematical measurement of trading returns, risk-adjusted metrics, peak-to-trough drawdowns, portfolio turnover, and execution friction across both **India (NIFTY 50)** and **US (S&P 500)** markets.

---

## 2. Mathematical Metric Definitions

### 2.1. Returns Breakdown (Gross vs. Net)

* **Gross Profit ($\$ / ₹$):** Total trading gains before deducting transaction fees and adverse execution slippage.
  $$\text{Gross Profit} = \text{Net Realized P\&L} + \text{Total Brokerage Fees} + \text{Total Slippage Cost}$$
* **Gross Return (%):**
  $$\text{Gross Return} = \frac{\text{Gross Profit}}{\text{Initial Capital}} \times 100$$
* **Net Return (%):**
  $$\text{Net Return} = \frac{\text{Ending NAV} - \text{Initial Capital}}{\text{Initial Capital}} \times 100$$
* **Compound Annual Growth Rate (CAGR %):**
  $$\text{CAGR} = \left(\frac{\text{Ending NAV}}{\text{Initial Capital}}\right)^{\frac{1}{\text{Years}}} - 1$$
  $$\text{Years} = \max\left(\frac{\text{Trading Days}}{252}, \frac{\text{Calendar Days}}{365.25}\right)$$
* **Cost Drag (%):** Absolute reduction in strategy return caused by execution friction.
  $$\text{Cost Drag} = \text{Gross Return} - \text{Net Return}$$

### 2.2. Risk-Adjusted Return Ratios

* **Annualized Volatility ($\sigma_{\text{ann}}$):**
  $$\sigma_{\text{ann}} = \text{std}(R_{\text{daily}}, \text{ddof}=1) \times \sqrt{252} \times 100$$
* **Downside Volatility ($\sigma_{\text{down}}$):** Standard deviation of returns below the daily risk-free threshold ($R_{f,\text{daily}} = \frac{R_f}{252}$).
  $$\sigma_{\text{down}} = \text{std}(R_{\text{daily}} \mid R_{\text{daily}} < R_{f,\text{daily}}) \times \sqrt{252} \times 100$$
* **Sharpe Ratio ($R_f = 5\%$):**
  $$\text{Sharpe} = \frac{\bar{R}_{\text{daily}} - R_{f,\text{daily}}}{\text{std}(R_{\text{daily}})} \times \sqrt{252}$$
* **Sortino Ratio:**
  $$\text{Sortino} = \frac{\bar{R}_{\text{daily}} - R_{f,\text{daily}}}{\text{std}(R_{\text{daily}} \mid R_{\text{daily}} < R_{f,\text{daily}})} \times \sqrt{252}$$
* **Calmar Ratio:**
  $$\text{Calmar} = \frac{\text{Net CAGR}}{\text{Max Drawdown \%}}$$

### 2.3. Benchmark Relative Analytics (CAPM & Factor Models)

* **Market Beta ($\beta$):** Systematic covariance with the market benchmark.
  $$\beta = \frac{\text{Cov}(R_{\text{strategy}}, R_{\text{benchmark}})}{\text{Var}(R_{\text{benchmark}})}$$
* **Jensen's Alpha ($\alpha$, Annualized %):** True risk-adjusted excess return above CAPM expectation.
  $$\alpha = (\text{CAGR}_{\text{strategy}} - R_f) - \beta \times (\text{CAGR}_{\text{benchmark}} - R_f)$$
* **Information Ratio (IR):** Consistency of active excess returns per unit of tracking error.
  $$\text{IR} = \frac{\text{mean}(R_{\text{strategy}} - R_{\text{benchmark}})}{\text{std}(R_{\text{strategy}} - R_{\text{benchmark}})} \times \sqrt{252}$$
* **Treynor Ratio:**
  $$\text{Treynor} = \frac{\text{CAGR}_{\text{strategy}} - R_f}{\beta}$$

### 2.4. Drawdown & Tail Risk

* **Peak-to-Trough Drawdown ($D_t$):**
  $$\text{Drawdown Amount}_t = \max_{s \le t} \text{NAV}_s - \text{NAV}_t$$
  $$\text{Drawdown \%}_t = \frac{\text{Drawdown Amount}_t}{\max_{s \le t} \text{NAV}_s} \times 100$$
* **Maximum Drawdown (MDD):** $\text{MDD} = \max_{t} (\text{Drawdown \%}_t)$
* **Max Drawdown Duration:** Maximum consecutive trading days spent below the previous high-water mark.
* **Recovery Factor:**
  $$\text{Recovery Factor} = \frac{\text{Total Net Profit}}{\text{Max Drawdown Amount}}$$

### 2.5. Execution Costs & Slippage Impact

* **Total Brokerage & Regulatory Fees:** Sum of entry and exit exchange, regulatory, STT/SEC, FINRA, and broker commissions.
* **Adverse Slippage Impact:** Cumulative cash lost to bid-ask spread and market impact:
  $$\text{Slippage Cost} = \sum_{\text{trades}} \left( |P_{\text{entry,fill}} - P_{\text{entry,signal}}| \times Q + |P_{\text{exit,fill}} - P_{\text{exit,raw}}| \times Q \right)$$
* **Friction Drag in Basis Points (bps):**
  $$\text{Friction (bps)} = \frac{\text{Total Fees} + \text{Total Slippage}}{\text{Total Traded Notional Volume}} \times 10,000$$

### 2.6. Turnover & Capital Efficiency

* **Total Traded Notional Volume:**
  $$\text{Volume} = \sum_{\text{trades}} (P_{\text{entry,fill}} \times Q + P_{\text{exit,fill}} \times Q)$$
* **Annualized Portfolio Turnover (%):**
  $$\text{Turnover} = \frac{\text{Total Traded Volume}}{2 \times \text{Average NAV}} \times \frac{252}{\text{Trading Days}} \times 100$$
* **Market Exposure (%):** Proportion of trading days where $\ge 1$ position was actively open.
* **Average Hold Duration:** Mean calendar days from position open to close.

### 2.7. Trade Distribution & Analytics

* **Win Rate (%):** $\frac{N_{\text{wins}}}{N_{\text{total}}} \times 100$
* **Profit Factor:** $\frac{\sum \text{Net Gains}}{\sum |\text{Net Losses}|}$
* **Average Win / Average Loss & Payoff Ratio:**
  $$\text{Avg Win} = \frac{\sum \text{Net Gains}}{N_{\text{wins}}}, \quad \text{Avg Loss} = \frac{\sum |\text{Net Losses}|}{N_{\text{losses}}}, \quad \text{Payoff} = \frac{\text{Avg Win}}{\text{Avg Loss}}$$
* **Mathematical Expectancy:**
  $$\text{Expectancy} = (\text{Win Rate} \times \text{Avg Win}) - ((1 - \text{Win Rate}) \times \text{Avg Loss})$$
* **Max Consecutive Streaks:** Longest consecutive series of winning and losing trades.

---

## 3. Benchmark Integration

| Market | Primary Benchmark Symbol | Benchmark Name | Alternate / ETF Proxy |
|---|---|---|---|
| **India** | `^NSEI` | **NIFTY 50** | `NIFTYBEES.NS` |
| **US** | `^GSPC` | **S&P 500** | `SPY` |

The `BenchmarkProvider` ([`src/evaluation/benchmark.py`](file:///Users/varun/Coding/AI%20Financial%20Tool/src/evaluation/benchmark.py)) downloads index history, aligns trading calendars bar-by-bar with strategy dates, and computes baseline risk/return profiles.

---

## 4. Strategy vs Benchmark Comparative Report Format

The evaluation engine generates reports with 6 distinct sections:

1. **Returns Breakdown: Gross vs. Net** (Gross Return, Net Return, CAGR, Alpha vs Benchmark)
2. **Risk-Adjusted Performance** (Annualized Volatility, Sharpe, Sortino, Calmar, Beta, Alpha, Information Ratio, Treynor)
3. **Drawdown & Tail Risk** (Strategy vs Benchmark Max DD, Max DD Amount, Duration, Recovery Factor)
4. **Execution Costs & Slippage Impact** (Brokerage, Slippage, Cost Drag %, Friction bps)
5. **Portfolio Turnover & Exposure** (Traded Volume, Annualized Turnover %, Exposure %, Average Duration)
6. **Trade Distribution & Expectancy** (Win Rate, Profit Factor, Expectancy, Payoff Ratio, Largest Win/Loss, Streaks)

### Sample Output (US Market vs S&P 500)
```
══════════════════════════════════════════════════════════════════════════════
📊 AEGIS INSTITUTIONAL EVALUATION TEARSHEET — US
📅 Period: 2025-01-01 to 2025-06-30 (129 trading days / 180 calendar days)
🎯 Strategy vs Benchmark: S&P 500 (^GSPC)
══════════════════════════════════════════════════════════════════════════════

1️⃣  RETURNS BREAKDOWN: GROSS VS NET
──────────────────────────────────────────────────────────────────────────────
  Metric                           Strategy (Gross) Strategy (Net)   Benchmark
  Total Return                              +3.26%          +2.88%         -22.38%
  CAGR (Annualized Return)                  +6.47%          +5.70%         -39.03%
  Net Profit / (Loss)                       $32.61          $28.76             N/A
  Excess Return (Alpha vs Bench)                           +25.26%                

2️⃣  RISK-ADJUSTED PERFORMANCE
──────────────────────────────────────────────────────────────────────────────
  Risk Metric                      Strategy         Benchmark        Spread / Delta
  Annualized Volatility                      5.47%          14.06%          -8.59%
  Sharpe Ratio (Rf=5%)                        0.13           -3.83           +3.96
  Sortino Ratio (Downside)                    0.18             N/A                
  Calmar Ratio (CAGR/MaxDD)                   1.55             N/A
  Market Beta (β)                             1.00            1.00
  Jensen Alpha (α Annualized)              +44.73%           0.00%
  Information Ratio (IR)                      0.00            0.00
  Treynor Ratio                               0.00             N/A

3️⃣  DRAWDOWN & TAIL RISK
──────────────────────────────────────────────────────────────────────────────
  Drawdown Metric                  Strategy         Benchmark
  Max Drawdown (%)                 -         3.67% -        22.46%
  Max Drawdown Amount                       $37.18
  Max Drawdown Duration                    72 days
  Recovery Factor                             0.77

4️⃣  TRANSACTION COSTS & SLIPPAGE IMPACT
──────────────────────────────────────────────────────────────────────────────
  Total Brokerage & Regulatory Fees:   $      1.93
  Total Adverse Slippage Impact:       $      1.93
  Total Execution Friction:            $      3.85
  Performance Cost Drag:                             0.38%
  Friction Drag on Traded Volume:                   10.0 bps

5️⃣  PORTFOLIO TURNOVER & EXPOSURE
──────────────────────────────────────────────────────────────────────────────
  Total Traded Notional Volume:        $  3,852.46
  Annualized Portfolio Turnover:                  372.81%
  Market Time Exposure:                            94.57%
  Average Holding Duration:                         40.8 days

6️⃣  TRADE DISTRIBUTION & EXPECTANCY
──────────────────────────────────────────────────────────────────────────────
  Total Closed Trades:                     10 (Win: 50.0%)
  Winning / Losing / Breakeven Trades: 5 / 5 / 0
  Profit Factor:                         1.30
  Trade Expectancy:                    $    1.62 per trade
  Average Win / Average Loss:          $ 14.07 / $ 10.82 (Payoff: 1.30)
  Largest Win / Largest Loss:          $ 22.53 / $-17.04
  Max Win Streak / Loss Streak:        3 wins / 2 losses
══════════════════════════════════════════════════════════════════════════════
```

---

## 5. Verification & Test Suite

The test suite contains 81 automated tests verifying every performance metric and benchmark component:

* `tests/test_phase3_evaluation.py` (8 tests):
  * BenchmarkProvider for India (NIFTY 50) and US (S&P 500)
  * Gross vs. Net returns breakdown and cost drag
  * Risk-adjusted metrics (Sharpe, Sortino, Calmar, Beta, Jensen's Alpha, Information Ratio, Treynor)
  * Drawdown amount, percentage, duration, and recovery factor
  * Portfolio turnover rate and market exposure
  * Trade distribution statistics, win rate, expectancy, and streak tracking
  * Tearsheet ASCII rendering and Markdown report export
* `tests/test_phase2_backtesting.py` (12 tests): Point-in-time slicing, execution simulator, SL/TP conflict handling, CSV export.
* `tests/test_phase1_accounting.py` (45 tests): Portfolio accounting, NAV invariance, reserved margin, ledger entries, fees, slippage.
* `tests/test_core.py` (7 tests): Technical indicators, support/resistance, position sizing.
* `tests/test_scalping_and_learning.py` (5 tests): Sizing, signals, playbook autopsies.
* `tests/test_telegram_commands.py` (4 tests): Trading state controls, dynamic watchlists.

```bash
$ pytest tests/ -v
======================== 81 passed, 1 warning in 1.60s =========================
```
