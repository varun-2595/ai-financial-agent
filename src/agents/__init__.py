"""
Aegis Multi-Agent Intelligence Network Package.

Specialized Domain Agents:
  - TechnicalAgent (Chart momentum, volume, support/resistance)
  - FundamentalAgent (Financial quality, multiples, ROE, debt)
  - NewsAgent (Sentiment, corporate disclosures, catalysts)
  - MacroAgent (Rates, inflation, market regime, VIX)
  - RiskAgent (Drawdown risk, invalidation triggers, volatility)
  - PortfolioManagerAgent (Multi-agent coordination & synthesis)
"""
from __future__ import annotations

from src.agents.base import BaseAgent
from src.agents.fundamental_agent import FundamentalAgent
from src.agents.macro_agent import MacroAgent
from src.agents.models import AgentSignalOutput, MacroContext, MultiAgentConsensus
from src.agents.news_agent import NewsAgent
from src.agents.portfolio_manager_agent import PortfolioManagerAgent
from src.agents.risk_agent import RiskAgent
from src.agents.technical_agent import TechnicalAgent

__all__ = [
    "AgentSignalOutput",
    "BaseAgent",
    "FundamentalAgent",
    "MacroAgent",
    "MacroContext",
    "MultiAgentConsensus",
    "NewsAgent",
    "PortfolioManagerAgent",
    "RiskAgent",
    "TechnicalAgent",
]
