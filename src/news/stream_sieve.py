"""
High-Velocity Zero-Cost News Stream Sieve.

Leverages Apple Silicon M5 local inference to continuously filter and classify hundreds
of raw news headlines per minute at zero API cost.
Categorizes into:
  - BREAKING_ALPHA: Massive contracts, earnings beats, FDA approvals, strategic M&A.
  - HEADLINE_RISK: Regulatory probes, executive resignations, debt defaults, fraud.
  - NOISE: Discarded immediately to preserve cognitive bandwidth.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Literal, Optional

from src.data.models import NewsItem
from src.gateway.gateway import ModelRequest, TaskComplexity
from src.gateway.router import ModelRouter
from src.utils.logger import logger


@dataclass
class SieveResult:
    title: str
    category: Literal["BREAKING_ALPHA", "HEADLINE_RISK", "NOISE"]
    ticker: Optional[str] = None
    market: Literal["india", "us"] = "india"
    impact_score: float = 0.0  # -1.0 (extreme negative) to +1.0 (extreme positive)
    key_takeaway: str = ""


class NewsStreamSieve:
    """Zero-cost local M5 news classification and alpha extraction sieve."""

    def __init__(self, router: Optional[ModelRouter] = None):
        self.router = router or ModelRouter()

    def filter_batch(
        self,
        news_items: list[NewsItem],
        market: Literal["india", "us"] = "india",
        force_heuristic: bool = False,
    ) -> list[SieveResult]:
        """
        Classifies a batch of news headlines using local M5 fast inference.
        """
        if not news_items:
            return []

        if force_heuristic:
            return [self._heuristic_classify(item, market) for item in news_items]

        headlines = [f"{i+1}. {item.title}" for i, item in enumerate(news_items[:10])]
        headlines_text = "\n".join(headlines)

        prompt = f"""You are a High-Frequency Financial Intelligence Filter.
Classify each headline into one of: 'BREAKING_ALPHA', 'HEADLINE_RISK', or 'NOISE'.
Market: {market.upper()}

Headlines:
{headlines_text}

Respond ONLY with a JSON array in this exact format:
[
  {{"index": 1, "category": "BREAKING_ALPHA", "impact_score": 0.8, "key_takeaway": "Large contract win"}},
  {{"index": 2, "category": "NOISE", "impact_score": 0.0, "key_takeaway": "Routine commentary"}}
]
"""

        try:
            req = ModelRequest(
                prompt=prompt,
                task_complexity=TaskComplexity.HIGH_VOLUME,
                temperature=0.1,
                max_tokens=600,
            )
            resp = self.router.route(req)
            if resp.content:
                cleaned = re.sub(r"```json\s*", "", resp.content)
                cleaned = re.sub(r"```\s*", "", cleaned).strip()
                match = re.search(r"\[.*\]", cleaned, re.DOTALL)
                if match:
                    items_data = json.loads(match.group(0))
                    results: list[SieveResult] = []
                    for entry in items_data:
                        idx = int(entry.get("index", 1)) - 1
                        if 0 <= idx < len(news_items):
                            cat = entry.get("category", "NOISE").upper()
                            if cat not in ("BREAKING_ALPHA", "HEADLINE_RISK", "NOISE"):
                                cat = "NOISE"
                            results.append(SieveResult(
                                title=news_items[idx].title,
                                category=cat,  # type: ignore
                                market=market,
                                impact_score=float(entry.get("impact_score", 0.0)),
                                key_takeaway=entry.get("key_takeaway", ""),
                            ))
                    if results:
                        return results
        except Exception as e:
            logger.debug(f"[News Sieve] Fast LLM parse error: {e}; using heuristic fallback.")

        return [self._heuristic_classify(item, market) for item in news_items]

    def _heuristic_classify(self, item: NewsItem, market: Literal["india", "us"]) -> SieveResult:
        """Deterministic keyword-based classification fallback."""
        title_lower = item.title.lower()
        
        # Risk keywords
        risk_words = [
            "fraud", "cbi", "sebi", "sec probe", "lawsuit", "investigation", "probe",
            "default", "resigns", "scam", "downgrade", "pledge", "penalty", "subpoena", "raids"
        ]
        if any(w in title_lower for w in risk_words):
            return SieveResult(
                title=item.title,
                category="HEADLINE_RISK",
                market=market,
                impact_score=-0.8,
                key_takeaway="Regulatory or governance risk flag detected",
            )

        # Alpha keywords
        alpha_words = [
            "record profit", "bags order", "wins", "contract", "fda approval", "acquisition",
            "beats estimates", "upgrades target", "bonus shares", "dividend", "order win"
        ]
        if any(w in title_lower for w in alpha_words):
            return SieveResult(
                title=item.title,
                category="BREAKING_ALPHA",
                market=market,
                impact_score=0.8,
                key_takeaway="High-impact positive operational catalyst",
            )

        return SieveResult(
            title=item.title,
            category="NOISE",
            market=market,
            impact_score=0.0,
            key_takeaway="Routine financial commentary",
        )
