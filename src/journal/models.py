"""
Data contracts and models for the Immutable Decision Journal & Agent Attribution System.

Phase 7: Measurement, Observability & Post-Trade Evaluation.
Strict Constraint: All data records are immutable. Agent weights are NOT automatically adjusted.
"""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Literal, Optional
from pydantic import BaseModel, Field


class DecisionJournalEntry(BaseModel):
    """
    Immutable snapshot of every trade decision before and during execution.
    Preserves market context, individual agent outputs, evidence citations,
    risk engine decisions, and execution levels.
    """
    journal_id: str = Field(description="Unique decision ID (e.g. DEC-xxxx)")
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc), description="Decision timestamp")
    symbol: str = Field(description="Ticker symbol, e.g., RELIANCE.NS, NVDA")
    market: Literal["india", "us"] = Field(description="Market identifier")
    strategy: str = Field(default="multi_agent_consensus", description="Strategy identifier")
    direction: Literal["BUY", "SELL", "HOLD"] = Field(default="BUY", description="Decision direction")
    
    # 1. Market Snapshot
    market_snapshot: dict[str, Any] = Field(
        default_factory=dict,
        description="Point-in-time market data (price, RSI, MACD, ATR, 50/200 SMA, sector, volume, VIX)"
    )
    
    # 2. Agent Outputs
    agent_outputs: dict[str, Any] = Field(
        default_factory=dict,
        description="Individual outputs from Technical, Fundamental, News, Macro, Risk, and PM agents"
    )
    
    # 3. Decision Consensus & Evidence
    confidence: float = Field(ge=0.0, le=1.0, description="Synthesized confidence score")
    evidence: list[str] = Field(default_factory=list, description="Grounding evidence and document citations")
    final_thesis: str = Field(description="Portfolio Manager's synthesis and core investment thesis")
    
    # 4. Risk Engine Decision
    risk_decision: str = Field(description="APPROVE, REDUCE, or REJECT")
    risk_reasons: list[str] = Field(default_factory=list, description="Audit trail of risk checks")
    
    # 5. Position Sizing & Order Pricing
    requested_quantity: int = Field(gt=0, description="Quantity proposed by AI")
    approved_quantity: int = Field(ge=0, description="Quantity permitted by deterministic risk engine")
    entry_price: float = Field(gt=0.0, description="Proposed/executed entry price")
    stop_loss: float = Field(gt=0.0, description="Calculated stop loss price")
    target_price: float = Field(gt=0.0, description="Calculated profit target price")
    
    # 6. Execution Lifecycle
    order_id: Optional[str] = Field(default=None, description="Associated broker/paper order ID")
    position_id: Optional[int] = Field(default=None, description="Associated open position ID")
    status: Literal["PROPOSED", "EXECUTED", "REJECTED", "CLOSED"] = Field(default="PROPOSED")
    
    # 7. Post-Close Realized Result (Filled once position closes)
    exit_price: Optional[float] = Field(default=None, description="Actual realized exit price")
    exit_timestamp: Optional[datetime] = Field(default=None, description="Timestamp of trade closure")
    exit_reason: Optional[str] = Field(default=None, description="Reason for exit (e.g., STOP_LOSS, TAKE_PROFIT, MANUAL)")
    realized_pnl: Optional[float] = Field(default=None, description="Net realized P&L after all transaction costs")
    return_pct: Optional[float] = Field(default=None, description="Net return percentage on trade")
    holding_period_seconds: Optional[float] = Field(default=None, description="Duration trade remained open")


class AgentAttributionScore(BaseModel):
    """Post-trade attribution score for a single agent on a single trade."""
    agent_name: str
    signal: str
    confidence: float
    directional_accuracy: float = Field(
        description="1.0 if agent direction was correct relative to market movement, 0.0 if incorrect, 0.5 if neutral"
    )
    brier_score_loss: float = Field(
        description="Squared error calibration metric: (confidence - binary_outcome)^2"
    )
    notes: str = Field(default="")


class TradeEvaluation(BaseModel):
    """
    Immutable post-trade post-mortem evaluation record created when a position closes.
    """
    evaluation_id: str = Field(description="Unique evaluation identifier (e.g. EVAL-xxxx)")
    journal_id: str = Field(description="Reference to DecisionJournalEntry")
    symbol: str
    market: str
    direction: str
    
    # Outcome Metrics
    realized_pnl: float
    return_pct: float
    is_winner: bool
    holding_period_seconds: float
    
    # Core Evaluation Criteria
    directional_accuracy: float = Field(
        description="1.0 if trade thesis direction was correct (price moved favorably), 0.0 if wrong"
    )
    thesis_accuracy: float = Field(
        description="Score (0.0 to 1.0) evaluating if the fundamental/technical thesis held"
    )
    thesis_notes: str = Field(description="Detailed explanation of whether thesis proved valid or was invalidated")
    
    # Agent Attribution
    agent_accuracy: dict[str, AgentAttributionScore] = Field(
        default_factory=dict,
        description="Attribution and calibration breakdown per agent"
    )
    
    # Risk Decision Evaluation
    risk_decision_evaluation: str = Field(
        description="Assessment of whether risk reduction/approval protected capital or capped gain"
    )
    
    # Failure Analysis
    major_failure_reason: Literal[
        "STOP_LOSS_HIT",
        "EARNINGS_DISAPPOINTMENT",
        "MACRO_REVERSAL",
        "TECHNICAL_BREAKDOWN",
        "NEWS_HEADLINE_RISK",
        "CHOPPY_SIDEWAYS_TIMEOUT",
        "SLIPPAGE_OR_SPREAD",
        "NONE_WINNING_TRADE",
        "UNSPECIFIED"
    ] = Field(default="NONE_WINNING_TRADE")
    failure_details: str = Field(default="No failure detected; trade closed profitably.")
    
    evaluated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class AgentScorecard(BaseModel):
    """
    Cumulative performance attribution metrics for a specific agent.
    Strictly used for observability and tracking over time.
    """
    agent_name: str
    total_evaluations: int = 0
    bullish_calls: int = 0
    bearish_calls: int = 0
    neutral_calls: int = 0
    correct_calls: int = 0
    incorrect_calls: int = 0
    win_rate: float = 0.0
    accuracy_rate: float = 0.0
    avg_confidence: float = 0.0
    brier_calibration_score: float = 0.0  # Lower is better (0.0 = perfect calibration)
    total_pnl_attributed: float = 0.0
    avg_return_when_bullish: float = 0.0
    avg_return_when_bearish: float = 0.0
