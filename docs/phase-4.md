# Phase 4: Multi-Agent Architecture & Model Gateway

## 1. Executive Summary

Phase 4 transforms Aegis into a **multi-agent reasoning system** powered by an intelligent **Model Gateway**. The architecture pairs an optional local inference node (running on Apple Silicon / M5 Mac via Ollama, MLX, LM Studio, or vLLM) with remote **Google Gemini API** models.

> [!IMPORTANT]
> **Optional M5 Inference Node**: The local Apple Silicon (M5 Mac) node is strictly an **optional inference acceleration node** designed for zero-cost, high-volume processing. It is **NOT required** for Aegis operation. If the local endpoint is offline, unreachable, or unconfigured, Aegis operates with full analytical depth via Gemini remote models or safe deterministic fallbacks.

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
                     │ Deterministic Risk    │
                     │  Gatekeeper (Python)  │
                     └───────────┬───────────┘
                                 │ (APPROVE / REDUCE / REJECT)
                                 ▼
                     ┌───────────────────────┐
                     │   Execution Engine    │
                     └───────────────────────┘
```

---

## 2. Core Architectural Principles & Guardrails

1. **No Direct Trade Execution by Agents**: Domain agents and the Portfolio Manager Agent provide intelligence, signals (`BUY` / `SELL` / `HOLD`), confidence scores ($0.0$ to $1.0$), and reasoning. They **never execute trades directly**.
2. **Deterministic Risk & Accounting**: Position sizing, buying power, margin reservation, fees, slippage, and NAV calculations remain strictly Python arithmetic in `PortfolioRiskManager` and `PaperTradingEngine`. The absence or presence of the local M5 node has **zero impact** on accounting integrity.
3. **Structured JSON Output**: Every agent conforms strictly to the `AgentSignalOutput` schema:
   ```json
   {
     "agent": "FundamentalAgent",
     "signal": "BUY",
     "confidence": 0.80,
     "reasons": ["High ROE (18.2%) and disciplined capital structure", "Attractive valuation multiple of 20.4x Forward P/E"],
     "risks": ["Earnings deceleration or multiple compression risk"],
     "evidence": ["PE=24.5", "ROE=18.2%", "DE=0.65", "[10-K, Item 7, 2024-10-31]"]
   }
   ```
4. **Genuine OpenAI Client (No Fake Engines)**: `LocalModelProvider` connects strictly as an OpenAI-compatible client to a genuine external inference runtime. All fake in-process reasoning engines have been removed from the production path.
5. **Rigorous Endpoint & Model Discovery**: `LocalModelProvider.is_available()` performs a live HTTP probe against `/v1/models` and only reports available if the endpoint is reachable (HTTP 200) **and** the configured model is explicitly present in the returned model registry.
6. **Graceful Failover Hierarchy**:
   - **Local M5 Available** $\rightarrow$ Routes high-volume / simple tasks to local model to minimize latency and API costs.
   - **Local M5 Unavailable** $\rightarrow$ Automatically routes all requests to remote Gemini API without interruption.
   - **Both Unavailable** $\rightarrow$ Enters safe deterministic fallback mode generating a safe `HOLD` / no-trade state.

---

## 3. Specialized Domain Agents

| Agent | Purpose | Primary Inputs |
| :--- | :--- | :--- |
| **`TechnicalAgent`** | Trend structure, EMA ribbons, RSI momentum, breakout validity | OHLCV prices, RSI, MACD, Moving Averages |
| **`FundamentalAgent`** | Balance sheet health, ROE, valuation multiples (P/E, P/B), leverage (D/E) | Financial statements, SEC / NSE filings, margins |
| **`NewsAgent`** | Catalysts, earnings disclosures, headline sentiment | Live headlines, press releases, corporate disclosures |
| **`MacroAgent`** | Regime shifts, interest rates, inflation trends, VIX volatility | Macro regime indicators, VIX index, FX rates |
| **`RiskAgent`** | Downside floor, invalidation triggers, structural stop levels | ATR volatility, Support/Resistance levels |
| **`PortfolioManagerAgent`** | Synthesis of all 5 domain agents into high-conviction decision | Domain agent outputs + market snapshot |

---

## 4. Model Gateway & Hybrid Routing

The `ModelGateway` abstracts all LLM calls behind a unified interface:

- **`LocalModelProvider`**: Connects to an external OpenAI-compatible runtime (`http://127.0.0.1:11434/v1` via Ollama, MLX, LM Studio, or vLLM).
- **`GeminiProvider`**: Remote Google GenAI client (`gemini-2.5-flash`, `gemini-1.5-flash`).
- **`ModelRouter` Policy**:
  - `TaskComplexity.SIMPLE` / `HIGH_VOLUME`: Routed to Local M5 Model when available; falls back to Gemini.
  - `TaskComplexity.COMPLEX` / `PORTFOLIO_DECISION`: Routed to Gemini for institutional reasoning; falls back to Local M5 if Gemini is unconfigured.
  - **Offline Resilience**: If both providers are offline, the router returns a safe offline state that triggers deterministic heuristic fallbacks in domain agents.

---

## 5. Configuration & Diagnostics

### Configuration (`config/settings.yaml`):
```yaml
llm:
  provider: "hybrid"            # uses local M5 node when available + Gemini fallback
  model: "gemini-2.5-flash"
  fallback_model: "gemini-1.5-flash"
  local_enabled: true
  local_endpoint: "http://127.0.0.1:11434/v1"
  local_model: "qwen2.5:14b-instruct"
  local_timeout_seconds: 5.0
```

### Health Diagnostics API:
```python
from src.gateway.health import get_gateway_health

health = get_gateway_health()
print(health)
# Output:
# {
#   "LOCAL_MODEL_AVAILABLE": False,
#   "GEMINI_AVAILABLE": True,
#   "status_summary": "HEALTHY_GEMINI_ONLY",
#   "primary_inference_route": "gemini (all tasks; M5 offline/optional)",
#   "m5_node_status": "OFFLINE (Optional)",
#   "gemini_status": "ONLINE"
# }
```

---

## 6. Provider Health Test Matrix

The test suite (`tests/test_phase4_multiagent.py`) verifies all 6 provider states:

| Test Case | M5 Endpoint | Model Loaded | Gemini API Key | Router Behavior | Downstream Output |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **State 1: Dual Online** | HTTP 200 | Yes | Valid | Local (Simple) + Gemini (Complex) | Multi-Agent LLM Output |
| **State 2: M5 Offline** | Unreachable | N/A | Valid | Full Gemini Delegation | Multi-Agent LLM Output |
| **State 3: Model Missing** | HTTP 200 | No | Valid | Full Gemini Delegation | Multi-Agent LLM Output |
| **State 4: Gemini Only** | Unconfigured | N/A | Valid | Full Gemini Delegation | Multi-Agent LLM Output |
| **State 5: Gemini Offline** | HTTP 200 | Yes | None / Error | Full Local M5 Delegation | Multi-Agent LLM Output |
| **State 6: Both Offline** | Unreachable | N/A | None / Error | Heuristic Fallback State | Safe `HOLD` / No-Trade |
