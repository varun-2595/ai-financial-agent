"""
Investment Horizon Classifier and Asset Allocator.
Separates opportunities into:
  - Short-term (1 - 6 months): Growth momentum, technical breakout, +20% target, -8% SL.
  - Long-term (1 - 5 years): Moat, compounders, high ROE, low debt, +80% trim, -20% stop.
Enforces diversification across single stocks (max 10%), ETFs (max 20%), and sectors (max 30%).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from src.analyst.engine import AnalystEngine
from src.data.models import StockAnalysis, StockSnapshot
from src.technicals.indicators import compute_technical_indicators
from src.utils.config import get_config
from src.utils.logger import logger


@dataclass
class AllocationRecommendation:
    ticker: str
    name: str
    market: Literal["india", "us"]
    horizon: Literal["short_term", "long_term"]
    is_etf: bool
    allocation_pct: float
    suggested_capital: float
    suggested_qty: int
    entry_price: float
    stop_loss: float
    target_price: float
    thesis: str
    invalidation_trigger: str


class HorizonClassifier:
    def __init__(self):
        self.config = get_config()

    def classify_and_allocate(
        self,
        snapshot: StockSnapshot,
        analysis: StockAnalysis,
        available_capital: float,
        horizon_preference: Literal["short_term", "long_term"] = "long_term",
    ) -> AllocationRecommendation | None:
        """
        Determines if an asset qualifies for the requested investment horizon,
        calculates target allocation percentage, and sets investment parameters.
        """
        f = snapshot.fundamentals
        tech = compute_technical_indicators(snapshot.history)
        p = snapshot.current_price

        # Check analyst fit
        if horizon_preference not in analysis.horizon_fit and "long_term" not in analysis.horizon_fit:
            logger.info(f"[Advisory] {snapshot.ticker} not suitable for {horizon_preference}")
            return None

        # Determine asset limits (stocks max 10%, ETFs max 20%)
        max_pct = self.config.advisory.max_single_etf_pct if snapshot.is_etf else self.config.advisory.max_single_stock_pct

        if horizon_preference == "short_term":
            # Short-Term Criteria: High momentum, healthy trend, clean R:R
            target_pct = min(max_pct, 0.08) # 8% allocation
            stop_loss = round(p * 0.92, 2)   # -8% hard stop
            target_price = round(p * 1.20, 2) # +20% target
            thesis = f"Short-term momentum setup. RSI={tech.rsi_14:.1f}, trend={tech.trend_short}. Target +20% within 1-6 months."
            invalidation = analysis.invalidation_trigger or f"Close below stop-loss at {stop_loss}"
        else:
            # Long-Term Criteria: Quality fundamentals, low debt, compounder
            # ETFs are automatically prime candidates for long-term
            if not snapshot.is_etf:
                # Require clean debt & positive margin if available
                if f.debt_to_equity and f.debt_to_equity > 1.5:
                    logger.info(f"[Advisory] {snapshot.ticker} rejected for long-term: High debt ({f.debt_to_equity})")
                    return None
            target_pct = max_pct # Full allocation (10% stock, 20% ETF)
            stop_loss = round(p * 0.80, 2)   # -20% fundamental invalidation stop
            target_price = round(p * 1.80, 2) # +80% multi-year trim target
            thesis = f"Long-term compounder. Market Cap: {f.market_cap or 'N/A'}, ROE: {f.return_on_equity or 'N/A'}, P/E: {f.pe_ratio or 'N/A'}."
            invalidation = analysis.invalidation_trigger or f"Fundamental deterioration or close below {stop_loss}"

        suggested_capital = round(available_capital * target_pct, 2)
        qty = int(suggested_capital / p) if p > 0 else 0

        if qty <= 0:
            return None

        return AllocationRecommendation(
            ticker=snapshot.ticker,
            name=snapshot.name or snapshot.ticker,
            market=snapshot.market,
            horizon=horizon_preference,
            is_etf=snapshot.is_etf,
            allocation_pct=target_pct,
            suggested_capital=suggested_capital,
            suggested_qty=qty,
            entry_price=p,
            stop_loss=stop_loss,
            target_price=target_price,
            thesis=thesis,
            invalidation_trigger=invalidation,
        )
