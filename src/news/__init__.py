"""
Real-Time Financial News & Catalyst Intelligence Package for Aegis.
"""
from __future__ import annotations

from src.news.fetcher import RealTimeNewsFetcher
from src.news.catalyst_engine import CatalystReport, NewsCatalystEngine

__all__ = [
    "CatalystReport",
    "NewsCatalystEngine",
    "RealTimeNewsFetcher",
]
