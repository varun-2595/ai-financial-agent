# Phase 8: Walk-Forward Validation & Out-of-Sample Robustness Engine

## 1. Overview & Objective

Traditional backtesting is highly prone to **in-sample overfitting**, **selection bias**, and **look-ahead leakage**. A strategy backtested over a single fixed historical window often fails when deployed in live market conditions.

Phase 8 implements institutional-grade **Walk-Forward Validation (WFV)** for the Aegis AI Trading Agent. By employing consecutive rolling windows of In-Sample (IS) training and Out-of-Sample (OOS) testing, Aegis validates whether alpha survives across unseen market regimes without optimizing or curve-fitting on test periods.

```
Rolling Walk-Forward Schedule:
2020–2021 [IS Train: 2y] ──> 2022 [OOS Test: 1y] (Fold 1)
2021–2022 [IS Train: 2y] ──> 2023 [OOS Test: 1y] (Fold 2)
2022–2023 [IS Train: 2y] ──> 2024 [OOS Test: 1y] (Fold 3)
2023–2024 [IS Train: 2y] ──> 2025 [OOS Test: 1y] (Fold 4)
2024–2025 [IS Train: 2y] ──> 2026 [OOS Test: 1y] (Fold 5)
```

---

## 2. Core Principles & Strict Invariants

1. **Zero Optimization on Test Periods**:
   - Hyperparameters, indicator periods, risk constraints, and agent synthesis prompts are strictly frozen prior to evaluating the out-of-sample period.
   - Test periods serve strictly as unpolluted validation benchmarks.
2. **Point-in-Time Realism**:
   - Data slicing enforces strict chronological bar progression. No forward bars or future data leakage are permitted.
3. **Execution Friction Across All Windows**:
   - Full exchange brokerage, regulatory taxes (STT / stamp duty / SEBI turnover charges), and dynamic slippage models are computed on both IS and OOS runs.

---

## 3. Metrics Computed Per Period

For every rolling period (both In-Sample and Out-of-Sample), the following 7 quantitative metrics are computed:

| Metric | Symbol / Formula | Description |
|:---|:---|:---|
| **CAGR** | Compound Annual Growth Rate | Annualized geometric growth rate of portfolio NAV. |
| **Sharpe Ratio** | $\frac{R_p - R_f}{\sigma_p}$ | Risk-adjusted excess return over annualized risk-free rate ($5\%$). |
| **Sortino Ratio** | $\frac{R_p - R_f}{\sigma_{downside}}$ | Risk-adjusted return penalizing only downside volatility. |
| **Max Drawdown** | $\max\left(\frac{Peak - NAV}{Peak}\right)$ | Worst peak-to-trough decline experienced. |
| **Profit Factor** | $\frac{\sum Gross\ Profits}{\sum Gross\ Losses}$ | Absolute profitability ratio of winning trades vs losing trades. |
| **Win Rate** | $\frac{Winning\ Trades}{Total\ Trades} \times 100$ | Percentage of closed trades with positive net P&L. |
| **Turnover** | Annualized Traded Volume / NAV | Portfolio churn and execution friction intensity. |

### Walk-Forward Efficiency (WFE)

Walk-Forward Efficiency measures the performance retention from In-Sample to Out-of-Sample:

$$\text{WFE}_{\text{Sharpe}} = \frac{\text{Sharpe}_{\text{OOS}}}{\text{Sharpe}_{\text{IS}}}$$

$$\text{WFE}_{\text{CAGR}} = \frac{\text{CAGR}_{\text{OOS}}}{\text{CAGR}_{\text{IS}}}$$

- **WFE $\ge 0.70$**: Highly robust strategy exhibiting strong generalization.
- **$0.50 \le$ WFE $< 0.70$**: Moderate degradation; acceptable for production.
- **WFE $< 0.50$**: Overfitted strategy failing out-of-sample.

---

## 4. Automated Anomaly & Defect Detection

The validation engine runs 4 specialized anomaly detectors across the rolling folds:

### 1. Overfitting Detection (`OVERFITTING_WFE_BELOW_50PCT`)
- **Trigger**: Mean WFE drops below $0.50$, or a strategy with positive IS CAGR ($>10\%$) yields negative OOS CAGR ($<0\%$), or $\ge 50\%$ of periods suffer total Sharpe collapse (IS $\ge 1.0 \rightarrow$ OOS $< 0.0$).
- **Remediation**: Reduce parameter complexity, increase training sample size, or tighten signal confidence thresholds.

### 2. Look-Ahead Bias Detection (`SUSPECTED_LOOKAHEAD_BIAS`)
- **Trigger**: Suspiciously perfect metrics such as OOS Win Rate $>80\%$, Max Drawdown $<0.5\%$, or Profit Factor $>5.0$ with high trade counts.
- **Remediation**: Inspect indicator lag, verify price bar alignment, ensure decisions use bar close without forward price leakage.

### 3. Unstable Performance Detection (`UNSTABLE_OOS_PERFORMANCE`)
- **Trigger**: High cross-period variance across OOS folds ($\sigma_{\text{Sharpe}} > 1.2$ or $\sigma_{\text{CAGR}} > 20\%$ with $>40\%$ peak-to-trough swing between folds).
- **Remediation**: Implement volatility-adaptive position sizing and sector diversification constraints.

### 4. Regime Dependence Detection (`REGIME_DEPENDENCE_BEAR_VULNERABILITY`)
- **Trigger**: Performance failures concentrated exclusively in known bear market/high-volatility test windows (e.g. 2022 rate hike drawdown or 2020 crash).
- **Remediation**: Introduce 200 EMA macro trend filters or ADX regime switches to disarm aggressive buying during market downturns.

---

## 5. Architecture & Code Structure

```
src/
├── evaluation/
│   ├── __init__.py           # Exports WalkForwardValidator, WalkForwardConfig, WalkForwardReport
│   ├── walk_forward.py       # Walk-Forward engine, rolling window generator, defect detectors
│   ├── evaluator.py          # Comprehensive institutional quantitative metrics
│   ├── benchmark.py          # NIFTY 50 & S&P 500 benchmark providers
│   └── tearsheet.py          # Markdown tearsheet & summary formatter
├── backtesting/
│   ├── __main__.py           # CLI entry point supporting --walk-forward
│   ├── engine.py             # Point-in-time backtest execution engine
│   └── data_provider.py      # Historical OHLCV data provider with disk caching
tests/
└── test_phase8_walkforward.py # 11 unit & integration tests
```

---

## 6. CLI Usage & Examples

### Run Rolling Walk-Forward Validation (India / NSE)
```bash
python -m src.backtesting --walk-forward --market india --wf-start-year 2020 --wf-end-year 2026 --strategy swing
```

### Run Rolling Walk-Forward Validation (US / S&P 500)
```bash
python -m src.backtesting --walk-forward --market us --strategy swing --wf-train-years 2 --wf-test-years 1 --report reports/wf_us_swing.md --csv-dir reports/
```

### Run Single Point-in-Time Backtest
```bash
python -m src.backtesting --market india --start 2024-01-01 --end 2025-12-31 --strategy swing
```

---

## 7. Sample Walk-Forward Tearsheet

```markdown
# Aegis Walk-Forward Validation & Robustness Report

**Status**: 🟢 ROBUST | **Robustness Score**: `88.5/100`
- **Market**: `INDIA` | **Strategy**: `Swing`
- **Rolling Configuration**: Train `2y` ──> Test `1y` (Step `1y`)
- **Rolling Windows Evaluated**: `5`

## 1. Executive Summary & Aggregate Robustness

| Metric | Mean In-Sample (IS) | Mean Out-of-Sample (OOS) | Degradation / Efficiency |
|:---|---:|---:|---:|
| **CAGR** | `+21.40%` | `+17.80%` | **CAGR WFE**: `0.83` |
| **Sharpe Ratio** | `1.65` | `1.38` | **Sharpe WFE**: `0.84` |
| **Sortino Ratio** | `2.20` | `1.85` | `84.1% retention` |
| **Max Drawdown** | `8.50%` | `11.20%` | `+2.70% Δ` |
| **Profit Factor** | `1.82` | `1.58` | `-0.24 Δ` |
| **Win Rate** | `61.5%` | `57.2%` | `-4.3% Δ` |
| **Turnover** | `135.0%` | `128.0%` | `-7.0% Δ` |

## 2. Automated Defect & Anomaly Diagnostics
- **Overfitting Detected**: `✅ NO`
- **Look-Ahead Bias Detected**: `✅ NO`
- **Unstable Performance**: `✅ NO`
- **Regime Dependence**: `✅ NO`
```
