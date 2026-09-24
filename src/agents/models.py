"""
Standardized Data Contracts for Aegis Multi-Agent Architecture.

Every agent MUST return AgentSignalOutput:
{
  "agent": "...",
  "signal": "BUY|SELL|HOLD",
  "confidence": 0.0,
  "reasons": [],
  "risks": [],
  "evidence": []
}
"""
from __future__ import annotations

from typing import Any, Literal, Optional
from pydantic import BaseModel, Field


class AgentSignalOutput(BaseModel):
    """
    Standardized institutional signal output required from all Aegis agents.
    """
    agent: str = Field(description="Name of the reporting agent")
    signal: Literal["BUY", "SELL", "HOLD"] = Field(description="Action recommendation")
    confidence: float = Field(ge=0.0, le=1.0, description="Confidence score from 0.0 to 1.0")
    reasons: list[str] = Field(default_factory=list, description="Core supporting rationale")
    risks: list[str] = Field(default_factory=list, description="Primary downside or invalidation risks")
    evidence: list[str] = Field(default_factory=list, description="Quantitative data points and key facts")


class MacroContext(BaseModel):
    """Macroeconomic and market regime inputs."""
    interest_rate_trend: str = "NEUTRAL"    # "RISING", "FALLING", "NEUTRAL"
    inflation_regime: str = "MODERATE"      # "HIGH", "MODERATE", "LOW"
    market_regime: str = "BULLISH_TREND"    # "BULLISH_TREND", "BEARISH_TREND", "SIDEWAYS", "HIGH_VOLATILITY"
    vix_level: float = 16.0
    usd_inr_trend: str = "STABLE"
    notes: Optional[str] = None


class MultiAgentConsensus(BaseModel):
    """Aggregated bundle of individual agent signals."""
    ticker: str
    market: str
    technical: Optional[AgentSignalOutput] = None
    fundamental: Optional[AgentSignalOutput] = None
    news: Optional[AgentSignalOutput] = None
    macro: Optional[AgentSignalOutput] = None
    risk: Optional[AgentSignalOutput] = None
    portfolio_manager: Optional[AgentSignalOutput] = None
