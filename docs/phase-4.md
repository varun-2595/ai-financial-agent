# Phase 4: Multi-Agent Architecture & Local 10B Model Gateway

## 1. Executive Summary

Phase 4 transforms Aegis into a **multi-agent reasoning system** powered by a hybrid **Model Gateway**. The architecture pairs a local **10B Financial Inference Model** (`aegis-10b-financial` running directly on Apple Silicon / M5 Mac) with remote **Gemini API** capabilities.

```
                    ┌─────────────────────────┐
                    │   Domain Market Data    │
                    └────────────┬────────────┘
                                 │
     ┌───────────────┬───────────┼───────────┬───────────────┐
     ▼               ▼           ▼           ▼               ▼
┌───────────┐ ┌─────────────┐ ┌─────┐ ┌───────────┐ ┌─────────────┐
│ Technical │ │ Fundamental │ │News │ │   Macro   │ │    Risk     │
│   Agent   │ │    Agent    │ │Agent│ │   Agent   │ │    Agent    │
└─────┬─────┘ └──────┬──────┘ └──┬──┘ └─────┬─────┘ └──────┬──────┘
      │              │           │          │               │
      └──────────────┴───────────┼──────────┴───────────────┘
                                 ▼
                    ┌─────────────────────────┐
                    │ Portfolio Manager Agent │
                    └────────────┬────────────┘
                                 ▼
                     ┌───────────────────────┐
                     │ Multi-Agent Consensus │
                     │   Structured Output   │
                     └───────────┬───────────┘
                                 │ (Signal & Conviction Only)
                                 ▼
                     ┌───────────────────────┐
                     │   RiskEngine / Python │
                     │  Deterministic Sizing │
                     └───────────────────────┘
```

---

## 2. Core Architectural Principles & Guardrails

1. **No Direct Trade Execution by Agents**: Domain agents and the Portfolio Manager Agent provide intelligence, signals (`BUY` / `SELL` / `HOLD`), confidence scores (0.0 to 1.0), and reasoning. They **never execute trades directly**.
2. **Deterministic Risk & Accounting**: Position sizing, buying power, margin reservation, fees, slippage, and NAV calculations remain strictly Python arithmetic in `RiskEngine` and `PaperEngine`.
3. **Structured JSON Output**: Every agent conforms strictly to the `AgentSignalOutput` schema:
   ```json
   {
     "agent": "FundamentalAgent",
     "signal": "BUY",
     "confidence": 0.80,
     "reasons": ["High ROE (147.0%) and disciplined capital structure", "Attractive valuation multiple of 28.4x P/E"],
     "risks": ["Earnings deceleration or multiple compression risk"],
     "evidence": ["PE=28.4", "ROE=147.0%", "DE=1.20", "Engine=Aegis-10B-Local"]
   }
   ```
4. **M5 Mac / Apple Silicon Inference Node**: The local Mac acts as an accelerated inference node. If the local endpoint is offline or unreachable, Aegis seamlessly and automatically falls back to Gemini without failing.

---

## 3. Specialized Domain Agents

| Agent | Purpose | Primary Inputs |
| :--- | :--- | :--- |
| **`TechnicalAgent`** | Trend structure, EMA ribbons, RSI momentum, breakout validity | OHLCV prices, RSI, MACD, Moving Averages |
| **`FundamentalAgent`** | Balance sheet health, ROE, valuation multiples (P/E, P/B), leverage (D/E) | Financial statements, margins, growth metrics |
| **`NewsAgent`** | Catalysts, earnings disclosures, sentiment extraction | Live headlines, press releases, SEC filings |
| **`MacroAgent`** | Regime shifts, interest rates, inflation trends, VIX volatility | Macro regime indicators, VIX index, FX rates |
| **`RiskAgent`** | Downside floor, invalidation triggers, structural stop levels | ATR volatility, Support/Resistance levels |
| **`PortfolioManagerAgent`** | Synthesis of all 5 domain agents into high-conviction decision | Domain agent outputs + market snapshot |

---

## 4. Model Gateway & Hybrid Routing

The `ModelGateway` abstracts all LLM calls behind an OpenAI-compatible layer:

- **`LocalModelProvider`**: Connects to `http://127.0.0.1:11434/v1` (or in-process `FinancialReasoning10BEngine`) running `aegis-10b-financial` / `gemma-2-9b` / `qwen2.5-14b`.
- **`GeminiProvider`**: Remote Google GenAI client (`gemini-3.8-flash`, `gemini-3.6-flash`).
- **`ModelRouter` Policy**:
  - `TaskComplexity.SIMPLE` / `HIGH_VOLUME`: Routed to Local 10B Model when available.
  - `TaskComplexity.COMPLEX`: Routed to Local 10B Model / Gemini.
  - `TaskComplexity.PORTFOLIO_DECISION`: Hybrid routing with automatic fallback.
  - **Graceful Fallback**: If local node times out or errors, the router automatically fails over to Gemini; if all LLMs are unreachable, deterministic heuristic algorithms return safe signals.

---

## 5. Configuration & Verification

### Configuration (`config/settings.yaml`):
```yaml
llm:
  provider: "hybrid"            # uses local M5 node (10B model) + Gemini fallback
  model: "gemini-3.8-flash"
  fallback_model: "gemini-3.6-flash"
  local_enabled: true
  local_endpoint: "http://127.0.0.1:11434/v1"
  local_model: "aegis-10b-financial"
  local_timeout_seconds: 5.0
```

### Health Diagnostics:
```python
from src.gateway.health import get_gateway_health
# Returns:
# {'LOCAL_MODEL_AVAILABLE': True, 'GEMINI_AVAILABLE': False, 'primary_inference_route': 'local', 'm5_node_status': 'ONLINE', 'gemini_status': 'OFFLINE'}
```
