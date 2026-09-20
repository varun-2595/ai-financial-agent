"""
Monthly Financial Advisor Recommendation Pipeline.
Scans the advisory universe (stocks + ETFs across India & US),
runs deep LLM analysis, selects top picks for short and long-term horizons,
and records recommendations into the Advisory Portfolio.
"""
from __future__ import annotations

from pathlib import Path
from typing import Literal

import yaml

from src.advisory.horizon_classifier import HorizonClassifier
from src.analyst.engine import AnalystEngine
from src.data.fetcher_india import fetch_india_batch
from src.data.fetcher_us import fetch_us_batch
from src.data.models import StockSnapshot
from src.db.advisory_store import init_advisory_db, save_advisory_recommendation
from src.utils.config import get_config
from src.utils.logger import logger

UNIVERSE_CONFIG = Path(__file__).parent.parent.parent / "config" / "advisory_universe.yaml"


class AdvisoryRecommender:
    def __init__(self):
        self.config = get_config()
        self.analyst = AnalystEngine()
        self.classifier = HorizonClassifier()
        init_advisory_db()

    def run_monthly_advisory(self) -> dict:
        """
        Executes the monthly advisory selection process:
        - Reads universe of quality stocks & ETFs
        - Fetches full market data
        - Runs analyst research
        - Classifies into short-term & long-term buckets
        - Saves top recommendations
        """
        logger.info("=" * 60)
        logger.info("[Advisory Recommender] 🌟 Starting monthly advisory screening...")
        logger.info("=" * 60)

        with open(UNIVERSE_CONFIG) as f:
            univ = yaml.safe_load(f).get("advisory_universe", {})

        in_tickers = [item["ticker"] for item in univ.get("india", {}).get("large_cap", [])]
        in_etfs = [item["ticker"] for item in univ.get("india", {}).get("etfs", [])]
        us_tickers = [item["ticker"] for item in univ.get("us", {}).get("large_cap", [])]
        us_etfs = [item["ticker"] for item in univ.get("us", {}).get("etfs", [])]

        all_india = in_tickers + in_etfs
        all_us = us_tickers + us_etfs

        # Fetch data
        in_snaps = fetch_india_batch(all_india, delay_seconds=0.2)
        us_snaps = fetch_us_batch(all_us, delay_seconds=0.2)

        recommendations = {
            "india_short_term": [],
            "india_long_term": [],
            "us_short_term": [],
            "us_long_term": [],
        }

        cap_inr = self.config.advisory.virtual_capital_inr
        cap_usd = self.config.advisory.virtual_capital_usd

        # Process India
        for ticker, snap in in_snaps.items():
            analysis = self.analyst.analyze_stock(snap)
            if analysis.action in ("Must Buy", "Good Buy"):
                # Try long-term first, then short-term
                rec_lt = self.classifier.classify_and_allocate(snap, analysis, cap_inr * 0.75, "long_term")
                if rec_lt:
                    save_advisory_recommendation(
                        ticker=rec_lt.ticker, name=rec_lt.name, market="india",
                        horizon="long_term", entry_price=rec_lt.entry_price,
                        suggested_alloc_pct=rec_lt.allocation_pct, quantity_suggested=rec_lt.suggested_qty,
                        stop_loss=rec_lt.stop_loss, target_price=rec_lt.target_price,
                        thesis=rec_lt.thesis, invalidation_trigger=rec_lt.invalidation_trigger
                    )
                    recommendations["india_long_term"].append(rec_lt)

                rec_st = self.classifier.classify_and_allocate(snap, analysis, cap_inr * 0.25, "short_term")
                if rec_st:
                    save_advisory_recommendation(
                        ticker=rec_st.ticker, name=rec_st.name, market="india",
                        horizon="short_term", entry_price=rec_st.entry_price,
                        suggested_alloc_pct=rec_st.allocation_pct, quantity_suggested=rec_st.suggested_qty,
                        stop_loss=rec_st.stop_loss, target_price=rec_st.target_price,
                        thesis=rec_st.thesis, invalidation_trigger=rec_st.invalidation_trigger
                    )
                    recommendations["india_short_term"].append(rec_st)

        # Process US
        for ticker, snap in us_snaps.items():
            analysis = self.analyst.analyze_stock(snap)
            if analysis.action in ("Must Buy", "Good Buy"):
                rec_lt = self.classifier.classify_and_allocate(snap, analysis, cap_usd * 0.75, "long_term")
                if rec_lt:
                    save_advisory_recommendation(
                        ticker=rec_lt.ticker, name=rec_lt.name, market="us",
                        horizon="long_term", entry_price=rec_lt.entry_price,
                        suggested_alloc_pct=rec_lt.allocation_pct, quantity_suggested=rec_lt.suggested_qty,
                        stop_loss=rec_lt.stop_loss, target_price=rec_lt.target_price,
                        thesis=rec_lt.thesis, invalidation_trigger=rec_lt.invalidation_trigger
                    )
                    recommendations["us_long_term"].append(rec_lt)

        logger.info(
            f"[Advisory Recommender] Finished: "
            f"India LT: {len(recommendations['india_long_term'])}, ST: {len(recommendations['india_short_term'])}, "
            f"US LT: {len(recommendations['us_long_term'])}, ST: {len(recommendations['us_short_term'])}"
        )
        return recommendations
