# 🏛️ Aegis — Autonomous Multi-Agent AI Financial & Trading System

Aegis is an institutional-grade, evidence-grounded **autonomous multi-agent financial trading and investment analysis system** designed for dual-market execution across **India (NSE/BSE)** and the **United States (NYSE/NASDAQ)**.

Aegis combines deep fundamental and macroeconomic reasoning with deterministic, zero-trust risk controls, evidence retrieval from regulatory filings, double-entry portfolio accounting, and an immutable post-trade decision journal.

---

## ⚡ Core Pillars & Architecture

```mermaid
flowchart TD
    subgraph MarketData["1. Market Data & Universe Screening"]
        NSE["NSE 500 (India)"]
        SP500["S&P 500 (US)"]
        Filings["SEC (10-K/10-Q) & NSE/BSE Filings"]
        NSE --> Screener["5-Factor Opportunity Screener"]
        SP500 --> Screener
        Filings --> RAG["Evidence-Grounded RAG Engine"]
    end

    subgraph MultiAgent["2. Multi-Agent Consensus (Local M5 + Gemini)"]
        RAG --> FundAgent["Fundamental Agent"]
        RAG --> NewsAgent["News & Sentiment Agent"]
        Screener --> TechAgent["Technical Agent"]
        MacroData["Macro Indicators"] --> MacroAgent["Macro Agent"]
        
        TechAgent --> PM["Portfolio Manager Agent (Synthesis)"]
        FundAgent --> PM
        NewsAgent --> PM
        MacroAgent --> PM
        RiskAgent["Risk Agent"] --> PM
        
        Gateway["Model Gateway & Router\n(Local 10B M5 Mac ↔ Remote Gemini Fallback)"] -.-> MultiAgent
    end

    subgraph DeterministicRisk["3. Deterministic Portfolio Risk Engine (Hard Gatekeeper)"]
        PM --> RiskGate["10 Portfolio-Level Risk Controls\n(Cash, Sector Caps, Drawdown, ADV, Beta)"]
        RiskGate --> Decision{"APPROVE / REDUCE / REJECT"}
    end

    subgraph Execution["4. Execution & Accounting"]
        Decision -- "Approved / Reduced" --> Exec["Paper / Live Execution Engine\n(3x Leverage, Slippage, Fees)"]
        Exec --> Ledger["Double-Entry Transaction Ledger"]
        Exec --> NAV["True Portfolio NAV Tracking"]
    end

    subgraph Observability["5. Post-Trade Decision Journal & Attribution"]
        Decision --> Journal["Immutable Decision Journal"]
        Exec --> ClosePos["Position Closed"]
        ClosePos --> PostMortem["Post-Trade Evaluator"]
        PostMortem --> Scorecards["Agent Scorecards & Brier Calibration"]
    end
```

---

## 🏗️ Phase-by-Phase System Architecture

Aegis is engineered in 7 modular, production-ready phases:

| Phase | Subsystem | Description | Documentation |
| :--- | :--- | :--- | :--- |
| **Phase 1** | **Portfolio Accounting & NAV** | Double-entry transaction ledger, reserved margin, 3x intraday leverage, slippage modeling, and true equity NAV calculation. | [`docs/phase-1.md`](docs/phase-1.md) |
| **Phase 2** | **Historical Backtesting Engine** | Deterministic, point-in-time historical simulation engine for India & US with zero look-ahead bias and order conflict resolution. | [`docs/phase-2.md`](docs/phase-2.md) |
| **Phase 3** | **Institutional Performance Evaluation** | Comprehensive analytics: CAGR, Sharpe, Sortino, Calmar, Max Drawdown, Turnover, Benchmark Tearsheets vs NIFTY 50 & S&P 500. | [`docs/phase-3.md`](docs/phase-3.md) |
| **Phase 4** | **Multi-Agent Architecture & Local Gateway** | Standardized 6-agent system (`TechnicalAgent`, `FundamentalAgent`, `NewsAgent`, `MacroAgent`, `RiskAgent`, `PortfolioManagerAgent`) powered by a Local M5 10B Gateway with automatic Gemini fallback. | [`docs/phase-4.md`](docs/phase-4.md) |
| **Phase 5** | **Evidence-Grounded Financial RAG** | Document parsers for SEC 10-K/10-Q and Indian exchange filings, metadata breadcrumbs, local embeddings cache, and strict no-fabrication citations. | [`docs/phase-5.md`](docs/phase-5.md) |
| **Phase 6** | **Portfolio-Level Risk Management** | 10 hard deterministic limits (Position Caps, Sector Concentration, Portfolio Beta, ADV Liquidity, Daily Loss & Drawdown Circuit Breakers). | [`docs/phase-6.md`](docs/phase-6.md) |
| **Phase 7** | **Immutable Decision Journal & Attribution** | Point-in-time decision snapshots, post-trade autopsies, directional accuracy, thesis validation, root-cause failure taxonomy, and per-agent scorecards. | [`docs/phase-7.md`](docs/phase-7.md) |

---

## 🚀 Getting Started

### 1. Prerequisites & Installation

```bash
# Clone the repository
git clone https://github.com/varun-2595/ai-financial-agent.git
cd ai-financial-agent

# Install dependencies
pip install -r requirements.txt
```

### 2. Configure Environment

Copy the example environment file:
```bash
cp .env.example .env
```

Configure the required variables in `.env`:
```ini
# Google Gemini API Key
GEMINI_API_KEY=your_gemini_api_key_here

# Local Model Gateway (Optional M5 Mac Inference)
LOCAL_MODEL_BASE_URL=http://localhost:11434/v1
LOCAL_MODEL_NAME=qwen2.5:14b-instruct

# Telegram Bot (Optional for Real-Time Alerts)
TELEGRAM_BOT_TOKEN=your_telegram_bot_token
TELEGRAM_CHAT_ID=your_telegram_chat_id
```

---

## 💻 CLI & Operational Commands

### 1. Run Historical Backtests (Phase 2 & 3)
```bash
# Backtest India strategy over custom date range
python -m src.backtesting --market india --start 2024-01-01 --end 2024-12-31

# Backtest US strategy with benchmark comparison against S&P 500
python -m src.backtesting --market us --start 2024-01-01 --end 2024-12-31 --report-dir reports/
```

### 2. Launch Local Model Gateway (Phase 4)
```bash
# Start the Local M5 10B/14B OpenAI-compatible Gateway server
python run_local_model.py --port 11434 --model qwen2.5:14b-instruct
```

### 3. Financial Research & Analysis CLI
```bash
# Deep multi-agent analysis on an Indian stock with filing citations
python run_analysis.py --ticker RELIANCE.NS

# Deep analysis on a US stock
python run_analysis.py --ticker NVDA

# Check live trading hours and exchange status
python run_analysis.py --market-status
```

### 4. Financial Advisory Mode
```bash
# Monthly high-conviction stock & ETF recommendation scan
python run_advisory.py --monthly-scan

# Portfolio review and allocation breakdown
python run_advisory.py --portfolio

# Weekly profit-taking and stop-loss scan
python run_advisory.py --sell-scan
```

### 5. Autonomous Daemon
```bash
# Test execution cycle in dry-run mode
python main.py --dry-run

# Run continuous 24/7 dual-market trading daemon
python main.py
```

---

## 🧪 Comprehensive Test Suite

Aegis includes **128 unit and integration tests** verifying all accounting rules, backtesting mechanics, agent routing, RAG retrieval, risk controls, and post-trade attribution:

```bash
pytest tests/ -v
```

```text
======================== 128 passed, 1 warning in 2.04s ========================
```

---

## 🛡️ Risk Management & Non-Interference Principles

1. **Deterministic Override**: AI models never have permission to execute trades directly. Every trade proposal must pass through the deterministic `PortfolioRiskManager` (`APPROVE`, `REDUCE`, or `REJECT`).
2. **Double-Entry Accounting**: Buying power and reserved margin are updated atomically. Portfolio equity (NAV) is never artificially inflated by leverage.
3. **Evidence Grounding**: Fundamental claims must be backed by verifiable document citations. If no filing evidence exists, the agent reports `"Insufficient evidence"`.
4. **Non-Mutating Observability**: Post-trade evaluation and agent attribution scorecards strictly measure performance and calibrate predictions without mutating agent weights or model configurations.

---

## 📜 License & Disclaimer

*Aegis is an open-source quantitative and financial research tool. Live financial trading carries risk. Past performance does not guarantee future results. Ensure adequate testing in paper trading mode before committing capital.*
