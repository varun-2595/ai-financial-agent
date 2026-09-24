"""
Risk Models for Portfolio-Level Risk Management and Audit Logging.
"""
from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
from typing import Any, Literal, Optional
from pydantic import BaseModel, Field


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


class RiskDecision(str, Enum):
    APPROVE = "APPROVE"
    REDUCE = "REDUCE"
    REJECT = "REJECT"


class RiskCheckName(str, Enum):
    AVAILABLE_CASH = "AVAILABLE_CASH"
    MAX_POSITION_SIZE = "MAX_POSITION_SIZE"
    MAX_SECTOR_EXPOSURE = "MAX_SECTOR_EXPOSURE"
    MAX_MARKET_EXPOSURE = "MAX_MARKET_EXPOSURE"
    MAX_LEVERAGE = "MAX_LEVERAGE"
    DAILY_LOSS_CIRCUIT_BREAKER = "DAILY_LOSS_CIRCUIT_BREAKER"
    PORTFOLIO_DRAWDOWN = "PORTFOLIO_DRAWDOWN"
    PORTFOLIO_BETA = "PORTFOLIO_BETA"
    CORRELATION_CONCENTRATION = "CORRELATION_CONCENTRATION"
    LIQUIDITY_ADV = "LIQUIDITY_ADV"
    RISK_REWARD_RATIO = "RISK_REWARD_RATIO"


class RiskCheckResult(BaseModel):
    """Result of a single deterministic risk rule verification."""
    check_name: RiskCheckName
    passed: bool
    limit_value: float
    current_value: float
    projected_value: float
    message: str


class PortfolioRiskState(BaseModel):
    """Live risk snapshot of the trading portfolio."""
    market: Literal["india", "us"]
    currency: Literal["INR", "USD"]
    nav: float
    peak_nav: float
    cash: float
    reserved_margin: float
    gross_exposure: float
    daily_realized_pnl: float = 0.0
    daily_unrealized_pnl: float = 0.0
    current_drawdown_pct: float = 0.0
    current_leverage: float = 0.0
    portfolio_beta: float = 1.0
    sector_exposures: dict[str, float] = Field(default_factory=dict)
    sector_position_counts: dict[str, int] = Field(default_factory=dict)
    open_positions: list[dict[str, Any]] = Field(default_factory=list)


class RiskEvaluationResult(BaseModel):
    """Overall outcome of portfolio-level risk evaluation."""
    decision: RiskDecision
    ticker: str
    market: Literal["india", "us"]
    strategy: str
    direction: Literal["BUY", "SELL"]
    requested_quantity: int
    approved_quantity: int
    entry_price: float
    stop_loss: float
    target_price: float
    approved_margin: float
    capital_allocated: float
    risk_amount: float
    checks: list[RiskCheckResult] = Field(default_factory=list)
    violations: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    reason: str
    timestamp: datetime = Field(default_factory=_utcnow)


class RiskAuditRecord(BaseModel):
    """Immutable audit record stored in SQLite."""
    audit_id: str
    timestamp: str
    ticker: str
    market: str
    strategy: str
    direction: str
    requested_quantity: int
    approved_quantity: int
    decision: str
    entry_price: float
    approved_margin: float
    nav_at_decision: float
    cash_at_decision: float
    drawdown_at_decision: float
    leverage_at_decision: float
    violations_json: str
    checks_json: str
    reason: str
