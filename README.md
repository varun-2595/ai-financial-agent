# 🏛️ Aegis — Autonomous AI Financial & Trading Agent

Aegis is an autonomous AI financial analyst and algorithmic trading agent designed for dual-market monitoring across **India (NSE/BSE)** and the **US (NYSE/NASDAQ)**.

It operates in two modes:
1. **Trading Mode**: Autonomous screening, technical analysis, position sizing, risk guardrails, simulated paper trading execution, and automated intraday square-off.
2. **Advisory Mode**: Monthly quality stock/ETF recommendations with short-term (1–6 months) and long-term (1–5 years) horizon allocation, ongoing portfolio tracking, and weekly exit/trim signal scans.

---

## ⚡ Key Highlights
- **Autonomous Opportunity Discovery**: Screens the full **NSE 500** and **S&P 500** daily on a 5-factor scoring model (0–100) — no manual ticker configuration required.
- **Downside-First Research**: Powered by **Google Gemini API** (`gemini-2.5-flash` / `gemini-1.5-flash`), enforcing institutional structured analysis.
- **Risk Management**: Enforces max 2% capital risk per trade, 10% maximum single position size, 30% sector concentration limit, and min 1:1.3 Risk-to-Reward ratio.
- **Dual-Market Daily Cycle**: Timezone-aware scheduling across IST and EST/EDT with automatic Daylight Saving Time handling.
- **Real-Time Delivery**: Push alerts via **Telegram** (signals, fills, alerts) and daily statement digests via **Gmail**.
- **Zerodha Kite Live Gateway**: Safety-gated integration for real order routing when you're ready to graduate from paper trading.

---

## 📅 Dual-Market Daily Agenda (IST)

| Time (IST) | Event | Description |
|---|---|---|
| **05:30 AM** | 🔍 Universe Screen | Screens NSE 500 & S&P 500, scores opportunities, updates watchlist |
| **08:00 AM** | 📊 US Overnight Report | Delivered to Telegram & Gmail when you wake up |
| **09:00 AM** | 🇮🇳 India Pre-Market | Scans for morning gap-ups/downs and intraday setups |
| **09:15 AM** | 🟢 NSE OPENS | Evaluates setups every 15 mins, executes paper orders |
| **03:20 PM** | 🔔 NSE Square-Off | Force-closes all open intraday positions |
| **03:30 PM** | 🔴 NSE CLOSES | Session wrap-up |
| **03:45 PM** | 📊 India EOD Report | P&L summary and open positions delivered |
| **06:30 PM** | 🇺🇸 US Pre-Market | 9:00 AM EST pre-market volume and news scan |
| **07:00 PM** | 🟢 NYSE OPENS | 9:30 AM EDT/EST US trading session starts |
| **01:25 AM** | 🔔 US Square-Off | 3:55 PM EST intraday square-off |
| **01:30 AM** | 🔴 NYSE CLOSES | 4:00 PM EST session close |

---

## 🚀 Getting Started

### 1. Installation
```bash
# Clone the repository
cd "AI Financial Tool"

# Install dependencies
pip install -r requirements.txt
```

### 2. Configure Credentials
Copy the `.env.example` template:
```bash
cp .env.example .env
```
Open `.env` and fill in your keys:
- `GEMINI_API_KEY`: Get a free key at [Google AI Studio](https://aistudio.google.com/app/apikey)
- `TELEGRAM_BOT_TOKEN` & `TELEGRAM_CHAT_ID`: From `@BotFather`
- `GMAIL_SENDER` & `GMAIL_APP_PASSWORD`: From your Google Account App Passwords

---

## 💻 CLI Commands

### Instant Market Analysis
```bash
# Analyze an Indian stock
python run_analysis.py --ticker RELIANCE.NS

# Analyze a US stock
python run_analysis.py --ticker NVDA

# Check live exchange open/close status
python run_analysis.py --market-status
```

### Financial Advisory Mode
```bash
# Run monthly stock & ETF recommendation screener
python run_advisory.py --monthly-scan

# View current advisory portfolio and returns
python run_advisory.py --portfolio

# Run weekly sell & profit trim scan
python run_advisory.py --sell-scan
```

### Continuous Agent Daemon
```bash
# Run one immediate test cycle
python main.py --dry-run

# Start the continuous 24/7 autonomous trading & advisory daemon
python main.py
```

### Run Tests
```bash
pytest tests/ -v
```

---

## 🐳 Docker Deployment
Run Aegis as a background service:
```bash
docker-compose up -d --build
```
Check logs:
```bash
docker-compose logs -f
```

---

## 🛡️ Risk & Safety Disclaimer
*Aegis is an algorithmic tool designed for educational and paper-trading research. Live trading involves financial risk. Never trade with capital you cannot afford to lose.*
