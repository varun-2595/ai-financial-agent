# Phase 4: Multi-Agent Intelligence Network & ModelGateway

## 1. Overview & Objectives

Phase 4 establishes Aegis's multi-agent analytical architecture, decoupling financial intelligence generation into specialized domain agents while unifying local and cloud LLM execution through a transparent **ModelGateway** and **ModelRouter**.

---

## 2. Multi-Agent Network Architecture

```
                               ┌────────────────────────┐
                               │     StockSnapshot      │
                               └───────────┬────────────┘
                                           │
         ┌──────────────────┬──────────────┼──────────────────┬──────────────────┐
         ▼                  ▼              ▼                  ▼                  ▼
┌─────────────────┐┌─────────────────┐┌─────────┐    ┌─────────────────┐┌─────────────────┐
│ TechnicalAgent  ││FundamentalAgent ││NewsAgent│    │   MacroAgent    ││   RiskAgent     │
│ (Price & Levels)││(Ratios & ROE)   ││(Catalyst│    │ (Regime & VIX)  ││ (Invalidation)  │
└────────┬────────┘└────────┬────────┘└───┬─────┘    └────────┬────────┘└────────┬────────┘
         │                  │             │                   │                  │
         └──────────────────┼─────────────┴───────────────────┴──────────────────┘
                            │
                            ▼
                ┌────────────────────────┐
                │  MultiAgentConsensus   │
                └───────────┬────────────┘
                            │
                            ▼
              ┌───────────────────────────┐
              │   PortfolioManagerAgent   │
              │  (Coordination/Conviction)│
              └─────────────┬─────────────┘
                            │
                            ▼
              ┌───────────────────────────┐
              │    AgentSignalOutput      │
              └─────────────┬─────────────┘
                            │
             (NO DIRECT TRADE EXECUTION)
                            │
                            ▼
              ┌───────────────────────────┐
              │ RiskEngine & Paper Engine │
              │  (Deterministic Python)   │
              └───────────────────────────┘
```

### Specialized Agents Overview

| Agent | Domain Mandate | Input Signals | Complexity Route |
|---|---|---|---|
| **TechnicalAgent** | Chart structure, momentum, support/resistance | OHLCV history, EMA20/50/200, RSI, MACD, ATR, Pivots | `HIGH_VOLUME` |
| **FundamentalAgent** | Earnings quality, valuation, debt health | P/E, P/B, ROE, Profit Margin, Debt-to-Equity, Market Cap | `COMPLEX` |
| **NewsAgent** | Catalysts, corporate disclosures, sentiment | Recent news headlines, publisher sources, catalyst keywords | `HIGH_VOLUME` |
| **MacroAgent** | Economic regime, monetary policy, market posture | Interest rates, inflation regime, VIX, USD/INR dynamics | `COMPLEX` |
| **RiskAgent** | Downside protection, structural volatility | ATR %, support distance, invalidation price levels | `COMPLEX` |
| **PortfolioManagerAgent** | Synthesis of all 5 agent signals into final conviction | Combined consensus vector from all domain agents | `PORTFOLIO_DECISION` |

---

## 3. Standardized Agent Output Contract

Every agent strictly outputs a validated `AgentSignalOutput` object:

```json
{
  "agent": "TechnicalAgent",
  "signal": "BUY",
  "confidence": 0.85,
  "reasons": [
    "Short-term trend is bullish with EMA alignment (EMA20 > EMA50)",
    "RSI 14 at 58.2 shows healthy momentum without overbought exhaustion"
  ],
  "risks": [
    "Breach of primary support at 2480.00"
  ],
  "evidence": [
    "RSI=58.2",
    "Trend=BULLISH",
    "S1=2480.00",
    "R1=2560.00"
  ]
}
```

---

## 4. ModelGateway & Intelligent ModelRouter

### 4.1. Provider Abstraction

```
                   ModelGateway (BaseModelProvider)
                                 │
                 ┌───────────────┴───────────────┐
                 ▼                               ▼
      LocalModelProvider                  GeminiProvider
   (M5 Mac Local Inference)          (Google GenAI Cloud API)
```

The rest of Aegis interacts **only** with the `ModelRouter`. Agents are agnostic to whether inference occurs on the local Apple Silicon M5 Mac or remote Google Gemini servers.

### 4.2. Routing Policy Matrix

* **Simple Tasks:** Routed to **Local Model** when available.
* **High-Volume Tasks (Screening triage, Technicals):** Routed to **Local Model** when available.
* **Complex Financial Reasoning (Fundamentals, Macro, Risk):** Routed to **Gemini**.
* **Final Portfolio Decision (PortfolioManagerAgent):** Routed to **Gemini** initially.

### 4.3. Automated Fallback & Fault Tolerance

1. **M5 Offline / Unavailable:** The router automatically delegates to **Gemini**.
2. **Gemini Offline / Rate-Limited:** The router falls back to the **Local Model**.
3. **All Providers Offline / Malformed JSON:** Agents safely engage **deterministic rule-based heuristic fallbacks**.
4. **Resilience Principle:** The local M5 Mac is an **optional acceleration node**, not a required server. Aegis never crashes due to offline hardware.

---

## 5. Strict Separation of Concerns & Deterministic Guardrails

### Immutable Golden Rules:
1. **Agents never execute trades directly.** They produce intelligence and structured recommendations.
2. **Never use an LLM for:**
   * Portfolio accounting
   * Margin calculations
   * Position sizing
   * Financial arithmetic
3. Position sizing, stop-loss calculations, margin blocking, fee deduction, and slippage application remain strictly executed in deterministic Python (`RiskEngine`, `PaperTradingEngine`, `ExecutionSimulator`).

---

## 6. Health Checking Diagnostics

```python
from src.gateway import get_gateway_health

health = get_gateway_health()
# Returns:
# {
#   "LOCAL_MODEL_AVAILABLE": False,
#   "GEMINI_AVAILABLE": True,
#   "primary_inference_route": "gemini",
#   "m5_node_status": "OFFLINE (Optional)",
#   "gemini_status": "ONLINE"
# }
```

---

## 7. Verification & Test Suite

The test suite contains 93 automated tests verifying all phases and multi-agent components:

```bash
$ pytest tests/ -v
======================== 93 passed, 1 warning in 1.60s =========================
```

* `tests/test_phase4_multiagent.py` (12 tests):
  * ModelRouter routing policies (simple, high-volume, complex, portfolio decision)
  * Local model availability & fallback to Gemini
  * Graceful handling when all providers are offline
  * Malformed JSON / schema validation handling
  * Schema compliance for `TechnicalAgent`, `FundamentalAgent`, `NewsAgent`, `MacroAgent`, `RiskAgent`, and `PortfolioManagerAgent`
  * Multi-agent consensus synthesis
  * Health check diagnostics
* `tests/test_phase3_evaluation.py` (8 tests): Benchmarks (NIFTY 50, S&P 500), risk-adjusted metrics, tearsheet reports.
* `tests/test_phase2_backtesting.py` (12 tests): Point-in-time data slicing, execution simulator, intrabar SL/TP.
* `tests/test_phase1_accounting.py` (45 tests): Portfolio accounting, NAV invariance, reserved margin, transaction ledger.
* `tests/test_core.py` (7 tests): Indicators, support/resistance, market hours.
* `tests/test_scalping_and_learning.py` (5 tests): Playbook autopsy and sizing.
* `tests/test_telegram_commands.py` (4 tests): Telegram controls.
