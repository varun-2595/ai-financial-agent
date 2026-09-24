# Phase 6: Portfolio-Level Risk Management

## 1. Executive Summary

Phase 6 upgrades Aegis from trade-level sizing to **portfolio-level risk management**. Every proposed trade must pass through an isolated, deterministic Python **Portfolio Risk Engine** before execution.

```
┌─────────────────────────────────────────────────────────────┐
│                    MULTI-AGENT NETWORK                      │
│ (Technical, Fundamental, News, Macro, Risk, Portfolio Mgr) │
└──────────────────────────────┬──────────────────────────────┘
                               │ (Trade Proposal & Conviction)
                               ▼
┌─────────────────────────────────────────────────────────────┐
│                   PORTFOLIO RISK MANAGER                    │
│             Strict Deterministic Python Controls            │
├─────────────────────────────────────────────────────────────┤
│  1. Available Cash & Buying Power Check                     │
│  2. Maximum Position Size Cap                               │
│  3. Maximum Sector Exposure Cap                             │
│  4. Maximum Market Gross Exposure Cap                       │
│  5. Maximum Portfolio Leverage Multiplier                   │
│  6. Daily Loss Circuit Breaker                              │
│  7. Maximum Portfolio Drawdown Circuit Breaker              │
│  8. Portfolio Beta Limit                                    │
│  9. Correlation & Sector Concentration Limit                │
│ 10. Liquidity & 30-Day ADV Volume Constraint                │
│ 11. Risk-to-Reward Ratio (Min 1:1.0 / 1:1.2 / 1:1.3)        │
└──────────────────────────────┬──────────────────────────────┘
                               │
       ┌───────────────────────┼───────────────────────┐
       ▼                       ▼                       ▼
┌─────────────┐         ┌─────────────┐         ┌─────────────┐
│   APPROVE   │         │   REDUCE    │         │   REJECT    │
│  (Full Qty) │         │(Scaled Qty) │         │   (0 Qty)   │
└──────┬──────┘         └──────┬──────┘         └──────┬──────┘
       │                       │                       │
       └───────────────────────┼───────────────────────┘
                               ▼
┌─────────────────────────────────────────────────────────────┐
│                       RISK AUDIT LOG                        │
│     (Immutable Record Persisted to SQLite `risk_audit_log`) │
└──────────────────────────────┬──────────────────────────────┘
                               │ (Approved / Reduced Orders Only)
                               ▼
┌─────────────────────────────────────────────────────────────┐
│                    PAPER / LIVE EXECUTION                   │
└─────────────────────────────────────────────────────────────┘
```

---

## 2. Core Guardrails & Isolation

1. **LLM Isolation**: The LLM / Multi-Agent Network is strictly decoupled from risk execution. **An LLM can NEVER override risk limits.**
2. **Three-State Decision Engine**:
   - `APPROVE`: Trade satisfies all 10 portfolio risk rules; executed at 100% requested size.
   - `REDUCE`: Trade is valid, but requested size breaches position cap, risk budget, sector room, or liquidity; automatically scaled down to maximum allowed shares.
   - `REJECT`: Trade breaches a hard circuit breaker (Daily Loss, Drawdown, Insufficient Cash, Sector Max Count, Illiquidity, or Invalid Bounds); blocked immediately.
3. **Immutable Regulatory Audit Trail**: Every trade proposal (whether approved, reduced, or rejected) is recorded in SQLite with full state snapshot values.

---

## 3. The 10 Deterministic Risk Controls

| # | Risk Control | Limit / Formula | Enforcement Action |
| :--- | :--- | :--- | :--- |
| **1** | **Available Cash** | $\text{Req Margin} \le \text{Available Unallocated Cash}$ | Rejects if cash insufficient for order |
| **2** | **Max Position Size** | $\text{Position Value} \le \text{NAV} \times \text{Max Position \%}$ (up to $50\%$) | Scales down to max position value |
| **3** | **Max Sector Exposure** | $\text{Sector Exposure} \le \text{NAV} \times 50\%$ | Scales down to remaining sector room |
| **4** | **Max Market Exposure** | $\text{Gross Exposure} \le \text{NAV} \times 3.0\text{x}$ | Restricts total market allocation |
| **5** | **Max Portfolio Leverage**| $\text{Gross Exposure} / \text{NAV} \le 3.0\text{x}$ (MIS) / $1.0\text{x}$ (Swing) | Rejects/Reduces margin borrowing |
| **6** | **Daily Loss Circuit Breaker** | $\text{Daily Loss} \ge \$50\text{ USD} / ₹500\text{ INR}$ | Immediate hard block on all new buys |
| **7** | **Portfolio Drawdown** | $(\text{Peak NAV} - \text{NAV}) / \text{Peak NAV} \ge 15\%$ | Immediate hard circuit breaker halt |
| **8** | **Portfolio Beta** | $\text{Projected Portfolio Beta} \le 1.50$ | Restricts high-beta asset weighting |
| **9** | **Sector Concentration** | $\text{Positions in same sector} \le 3$ | Rejects 4th asset in same sector |
| **10** | **Liquidity (ADV)** | $\text{Order Qty} \le 2\% \text{ of 30d ADV}$; $\text{ADV} \ge 10,000$ | Rejects illiquid; scales large orders |

---

## 4. Risk Decision Audit Schema

Every risk evaluation is recorded in `risk_audit_log`:

```sql
CREATE TABLE risk_audit_log (
    audit_id            TEXT PRIMARY KEY,
    timestamp           TEXT NOT NULL,
    ticker              TEXT NOT NULL,
    market              TEXT NOT NULL,
    strategy            TEXT NOT NULL,
    direction           TEXT NOT NULL,
    requested_quantity  INTEGER NOT NULL,
    approved_quantity   INTEGER NOT NULL,
    decision            TEXT NOT NULL, -- APPROVE / REDUCE / REJECT
    entry_price         REAL NOT NULL,
    approved_margin     REAL NOT NULL,
    nav_at_decision     REAL NOT NULL,
    cash_at_decision    REAL NOT NULL,
    drawdown_at_decision REAL NOT NULL,
    leverage_at_decision REAL NOT NULL,
    violations_json     TEXT NOT NULL,
    checks_json         TEXT NOT NULL,
    reason              TEXT NOT NULL
);
```

### Audit Record Sample:
```json
{
  "audit_id": "AUD-9F408B12EC",
  "ticker": "AAPL",
  "decision": "REDUCE",
  "requested_quantity": 8,
  "approved_quantity": 5,
  "entry_price": 100.0,
  "approved_margin": 500.0,
  "nav_at_decision": 1000.0,
  "cash_at_decision": 1000.0,
  "drawdown_at_decision": 0.0,
  "leverage_at_decision": 0.0,
  "reason": "Reduced from 8 to 5 shares due to position cap (5 shs), risk budget (5 shs)"
}
```

---

## 5. Verification & Test Suite

The test suite in [`tests/test_phase6_portfolio_risk.py`](file:///Users/varun/Coding/AI%20Financial%20Tool/tests/test_phase6_portfolio_risk.py) verifies:
- `APPROVE` on clean signals
- `REJECT` on daily loss circuit breaker
- `REJECT` on portfolio drawdown limit ($\ge 15\%$)
- `REJECT` on insufficient available cash
- `REJECT` on max positions per sector concentration ($> 3$)
- `REJECT` on illiquid stock ($\text{ADV} < 10,000$)
- `REJECT` on unfavorable risk-reward ratio
- `REJECT` on inverted stop loss / target boundaries
- `REDUCE` on risk budget limit
- `REDUCE` on single position cap
- `REDUCE` on sector room exhaustion
- `REDUCE` on 2% ADV liquidity constraint
- Audit record SQLite persistence and query retrieval
- Full pipeline integration through `PaperTradingEngine`

**Total Test Suite**: **116 / 116 tests passing** (`pytest tests/ -v`).
