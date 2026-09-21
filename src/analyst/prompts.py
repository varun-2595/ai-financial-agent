"""
Expert analyst prompt templates for Google Gemini LLM.
Enforces rigorous, data-backed 5-lens equity analysis.
"""
from __future__ import annotations

from src.data.models import StockSnapshot
from src.technicals.indicators import TechnicalSummary
from src.technicals.levels import PriceLevels

SYSTEM_PROMPT = """You are a senior equity research analyst and institutional portfolio manager with 15+ years of experience across fundamental analysis, technical analysis, macroeconomics, and risk management.
You advise retail and institutional clients on buy/sell/hold decisions across Indian (NSE/BSE) and US (NYSE/NASDAQ) markets.

Your hallmarks:
1. Rigorous, quantitative, and data-driven — zero hype, zero speculation.
2. DOWN-SIDE FIRST: Always evaluate downside risks, invalidation triggers, and max loss potential before upside potential.
3. Objective classification:
   - Risk Level: Low, Medium, High
   - Action Category: Must Buy, Good Buy, Hold-Watch, Must Avoid
4. Clear trade & investment suitability: Identify exact strategy fit (intraday, swing, positional) and horizon fit (short_term, long_term).
5. Always state concrete, falsifiable invalidation price triggers (e.g. "Break below 200 EMA at 1240").

SUPPLY-CHAIN & AFFILIATE INVESTMENT THESIS (critical lens):
You understand that mega-cap product companies (Apple, Samsung, Tesla, Google, etc.) create entire ecosystems of investable suppliers, component makers, and contract manufacturers.
Examples of this framework (Apple ecosystem):
- Chip fab: TSMC (TSM) — A/M-series chip fabrication (+385% historical outperformance)
- RF/Wireless: Broadcom (AVGO), Qorvo (QRVO), Skyworks (SWKS)
- Memory: Micron (MU), Western Digital (WDC)
- Optics/Sensing: Lumentum (LITE), Coherent (COHR)
- Equipment: Applied Materials (AMAT), Amphenol (APH)
- Contract Mfg: Jabil (JBL)
- Glass/Display: Corning (GLW)
- Audio/Power ICs: Cirrus Logic (CRUS), Analog Devices (ADI)

When analyzing any stock, look for:
a) Is this company a component supplier, chip designer, sensor maker, or contract manufacturer for a dominant OEM?
b) Does the OEM's product cycle (iPhone launch, EV refresh, AI hardware) directly impact revenue visibility?
c) Are there similar "picks-and-shovels" plays in India? (e.g. Dixon Technologies, Tata Elxsi, Kaynes Technology for electronics manufacturing)
d) Suggest 1-2 RELATED affiliate/supply-chain tickers worth monitoring when relevant.
"""


def build_analysis_prompt(
    snapshot: StockSnapshot,
    technicals: TechnicalSummary,
    levels: PriceLevels,
) -> str:
    f = snapshot.fundamentals
    currency = "₹" if snapshot.currency == "INR" else "$"

    news_text = "No recent headlines available."
    if snapshot.recent_news:
        headlines = [f"- {n.title} ({n.publisher or 'Source'})" for n in snapshot.recent_news[:5]]
        news_text = "\n".join(headlines)

    return f"""Analyze the following asset and provide an institutional-grade structured research report.

ASSET OVERVIEW:
- Ticker: {snapshot.ticker} ({snapshot.name or 'N/A'})
- Market: {snapshot.market.upper()} | Sector: {snapshot.sector or 'N/A'} | Asset Type: {'ETF' if snapshot.is_etf else 'Equity'}
- Current Price: {currency}{snapshot.current_price:,.2f}
- Price Change: 1D: {snapshot.price_change_pct_1d or 0:+.2f}%, 1W: {snapshot.price_change_pct_1w or 0:+.2f}%, 1M: {snapshot.price_change_pct_1m or 0:+.2f}%, 3M: {snapshot.price_change_pct_3m or 0:+.2f}%

FUNDAMENTALS:
- Market Cap: {f"{currency}{f.market_cap / 1e9:,.2f}B" if f.market_cap else 'N/A'}
- P/E Ratio: {f.pe_ratio or 'N/A'} | P/B Ratio: {f.pb_ratio or 'N/A'}
- EPS: {currency}{f.eps or 'N/A'} | Debt-to-Equity: {f.debt_to_equity or 'N/A'}
- Return on Equity (ROE): {f'{f.return_on_equity*100:.1f}%' if f.return_on_equity else 'N/A'}
- Profit Margin: {f'{f.profit_margin*100:.1f}%' if f.profit_margin else 'N/A'}
- 52-Week High/Low: {currency}{f.week_52_high or 'N/A'} / {currency}{f.week_52_low or 'N/A'}
- Average 30D Volume: {f.avg_volume_30d or 'N/A'}

TECHNICAL PROFILE:
- RSI (14): {technicals.rsi_14 or 'N/A'} ({technicals.rsi_condition})
- MACD Line: {technicals.macd or 'N/A'} | Signal: {technicals.macd_signal or 'N/A'} | Histogram: {technicals.macd_hist or 'N/A'}
- EMA 20: {technicals.ema_20 or 'N/A'} | EMA 50: {technicals.ema_50 or 'N/A'} | EMA 200: {technicals.ema_200 or 'N/A'}
- Short-Term Trend: {technicals.trend_short} | Long-Term Trend: {technicals.trend_long}
- ATR (14): {technicals.atr_14 or 'N/A'} ({f'{technicals.atr_pct:.2f}% of price' if technicals.atr_pct else 'N/A'})
- Bollinger Bands: Upper: {technicals.bb_upper or 'N/A'}, Mid: {technicals.bb_middle or 'N/A'}, Lower: {technicals.bb_lower or 'N/A'}

KEY PRICE LEVELS:
- Pivot: {currency}{levels.pivot_point}
- Resistance 1: {currency}{levels.resistance_1} | Resistance 2: {currency}{levels.resistance_2}
- Support 1: {currency}{levels.support_1} | Support 2: {currency}{levels.support_2}
- Fibonacci 61.8%: {currency}{levels.fib_618} | 50.0%: {currency}{levels.fib_500}

RECENT HEADLINES:
{news_text}

TASK:
Evaluate this security across fundamental valuation, technical momentum, and risk/reward symmetry.
Return your evaluation adhering strictly to the structured schema.
"""
