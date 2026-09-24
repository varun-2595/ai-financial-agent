"""
Post-Trade Evaluation Engine — Phase 7.

Deterministic evaluation of closed trades:
- Directional accuracy
- Thesis accuracy
- Per-agent accuracy attribution & Brier score calibration
- Risk decision effectiveness
- Major failure reason classification
"""
from __future__ import annotations

import uuid
from datetime import datetime, timezone
from typing import Optional

from src.journal.models import (
    AgentAttributionScore,
    DecisionJournalEntry,
    TradeEvaluation,
)
from src.utils.logger import logger


class PostTradeEvaluator:
    """
    Evaluates completed trade outcomes against pre-trade theses, agent recommendations,
    and risk limits.
    """

    @staticmethod
    def evaluate(
        entry: DecisionJournalEntry,
        exit_price: float,
        exit_timestamp: Optional[datetime] = None,
        exit_reason: Optional[str] = None,
        realized_pnl: Optional[float] = None,
        return_pct: Optional[float] = None,
    ) -> TradeEvaluation:
        """
        Produce a deterministic post-trade evaluation.
        """
        now = exit_timestamp or datetime.now(timezone.utc)
        eval_id = f"EVAL-{uuid.uuid4().hex[:8].upper()}"
        
        # 1. Basic Outcome Calculations
        entry_price = entry.entry_price
        direction = entry.direction.upper()
        
        # Duration
        entry_time = entry.timestamp
        if entry_time.tzinfo is None:
            entry_time = entry_time.replace(tzinfo=timezone.utc)
        if now.tzinfo is None:
            now = now.replace(tzinfo=timezone.utc)
        holding_seconds = max(0.0, (now - entry_time).total_seconds())

        # Returns & PnL
        if realized_pnl is None:
            if direction == "BUY":
                gross = (exit_price - entry_price) * entry.approved_quantity
            else:
                gross = (entry_price - exit_price) * entry.approved_quantity
            realized_pnl = round(gross, 4)

        if return_pct is None and entry_price > 0:
            if direction == "BUY":
                return_pct = round((exit_price - entry_price) / entry_price * 100, 2)
            else:
                return_pct = round((entry_price - exit_price) / entry_price * 100, 2)
        elif return_pct is None:
            return_pct = 0.0

        is_winner = realized_pnl > 0.0

        # 2. Directional Accuracy (Price movement relative to position direction)
        if direction == "BUY":
            price_delta = exit_price - entry_price
        else:
            price_delta = entry_price - exit_price

        if price_delta > 0:
            directional_accuracy = 1.0
        elif price_delta == 0:
            directional_accuracy = 0.5
        else:
            directional_accuracy = 0.0

        # 3. Thesis Accuracy & Notes
        thesis_accuracy, thesis_notes = PostTradeEvaluator._evaluate_thesis(
            entry=entry,
            exit_price=exit_price,
            exit_reason=exit_reason,
            is_winner=is_winner,
            directional_accuracy=directional_accuracy,
        )

        # 4. Agent Attribution Scores & Calibration (Brier Score Component)
        market_move = exit_price - entry_price
        agent_accuracy = PostTradeEvaluator._evaluate_agent_attribution(
            entry=entry,
            directional_accuracy=directional_accuracy,
            is_winner=is_winner,
            market_move=market_move,
        )

        # 5. Risk Decision Evaluation
        risk_eval = PostTradeEvaluator._evaluate_risk_decision(
            entry=entry,
            realized_pnl=realized_pnl,
            is_winner=is_winner,
        )

        # 6. Failure Reason Taxonomy
        failure_reason, failure_details = PostTradeEvaluator._classify_failure(
            entry=entry,
            exit_price=exit_price,
            exit_reason=exit_reason,
            realized_pnl=realized_pnl,
            holding_seconds=holding_seconds,
        )

        evaluation = TradeEvaluation(
            evaluation_id=eval_id,
            journal_id=entry.journal_id,
            symbol=entry.symbol,
            market=entry.market,
            direction=entry.direction,
            realized_pnl=realized_pnl,
            return_pct=return_pct,
            is_winner=is_winner,
            holding_period_seconds=round(holding_seconds, 1),
            directional_accuracy=directional_accuracy,
            thesis_accuracy=thesis_accuracy,
            thesis_notes=thesis_notes,
            agent_accuracy=agent_accuracy,
            risk_decision_evaluation=risk_eval,
            major_failure_reason=failure_reason,
            failure_details=failure_details,
            evaluated_at=datetime.now(timezone.utc),
        )

        logger.info(
            f"[PostTradeEvaluator] Evaluated {entry.symbol} ({entry.journal_id}): "
            f"Winner={is_winner} (PnL={realized_pnl:+.2f}, Return={return_pct:+.2f}%) | "
            f"DirectionAcc={directional_accuracy} | ThesisAcc={thesis_accuracy} | "
            f"FailureReason={failure_reason}"
        )
        return evaluation

    @staticmethod
    def _evaluate_thesis(
        entry: DecisionJournalEntry,
        exit_price: float,
        exit_reason: Optional[str],
        is_winner: bool,
        directional_accuracy: float,
    ) -> tuple[float, str]:
        """Assess how well the core thesis played out."""
        reason_upper = (exit_reason or "").upper()
        
        # Long position expectations
        if entry.direction == "BUY":
            target_diff = max(0.001, entry.target_price - entry.entry_price)
            actual_move = exit_price - entry.entry_price

            if "TARGET" in reason_upper or "PROFIT" in reason_upper or exit_price >= entry.target_price:
                return 1.0, f"Target price ₹/{entry.target_price:.2f} reached. Core thesis fully validated."
            
            if "STOP" in reason_upper or exit_price <= entry.stop_loss:
                return 0.0, f"Stop loss ₹/{entry.stop_loss:.2f} hit. Thesis invalidated by adverse price action."

            if is_winner:
                pct_target = min(1.0, max(0.0, actual_move / target_diff))
                return round(pct_target, 2), f"Trade closed profitably (+{round(pct_target*100)}% of target distance). Thesis partially captured."
            else:
                return 0.0, "Trade closed with loss before target was attained."

        # Short position expectations
        target_diff = max(0.001, entry.entry_price - entry.target_price)
        actual_move = entry.entry_price - exit_price

        if "TARGET" in reason_upper or "PROFIT" in reason_upper or exit_price <= entry.target_price:
            return 1.0, f"Target profit price {entry.target_price:.2f} reached. Short thesis validated."
        if "STOP" in reason_upper or exit_price >= entry.stop_loss:
            return 0.0, f"Stop loss {entry.stop_loss:.2f} hit. Short thesis invalidated."
        
        if is_winner:
            pct_target = min(1.0, max(0.0, actual_move / target_diff))
            return round(pct_target, 2), f"Short trade closed profitably (+{round(pct_target*100)}% of target). Thesis partially captured."
        return 0.0, "Short trade closed with loss."

    @staticmethod
    def _evaluate_agent_attribution(
        entry: DecisionJournalEntry,
        directional_accuracy: float,
        is_winner: bool,
        market_move: float,
    ) -> dict[str, AgentAttributionScore]:
        """Compute accuracy and calibration (Brier score component) for each participating agent."""
        scores: dict[str, AgentAttributionScore] = {}

        for agent_name, out in entry.agent_outputs.items():
            if isinstance(out, dict):
                sig = out.get("signal", "HOLD").upper()
                conf = float(out.get("confidence", 0.5))
            else:
                sig = getattr(out, "signal", "HOLD").upper()
                conf = float(getattr(out, "confidence", 0.5))

            # Directional correctness for this agent's signal
            if sig == "BUY":
                if market_move > 0:
                    agent_dir_acc = 1.0
                    target_outcome = 1.0
                    notes = f"{agent_name} correctly recommended BUY on upward move (+{market_move:.2f})."
                elif market_move == 0:
                    agent_dir_acc = 0.5
                    target_outcome = 0.5
                    notes = f"{agent_name} recommended BUY on flat price action."
                else:
                    agent_dir_acc = 0.0
                    target_outcome = 0.0
                    notes = f"{agent_name} incorrectly recommended BUY on downward move ({market_move:.2f})."
            elif sig == "SELL":
                if market_move < 0:
                    agent_dir_acc = 1.0
                    target_outcome = 1.0
                    notes = f"{agent_name} correctly recommended SELL on downward move ({market_move:.2f})."
                elif market_move == 0:
                    agent_dir_acc = 0.5
                    target_outcome = 0.5
                    notes = f"{agent_name} recommended SELL on flat price action."
                else:
                    agent_dir_acc = 0.0
                    target_outcome = 0.0
                    notes = f"{agent_name} incorrectly recommended SELL on upward move (+{market_move:.2f})."
            else:  # HOLD / NEUTRAL
                agent_dir_acc = 0.5
                target_outcome = 0.5
                if not is_winner:
                    notes = f"{agent_name} recommended HOLD, which appropriately signaled caution on non-winning trade."
                else:
                    notes = f"{agent_name} recommended HOLD, sitting out a winning trade."

            # Brier score component: (confidence - target_outcome)^2
            brier_loss = round((conf - target_outcome) ** 2, 4)

            scores[agent_name] = AgentAttributionScore(
                agent_name=agent_name,
                signal=sig,
                confidence=round(conf, 4),
                directional_accuracy=agent_dir_acc,
                brier_score_loss=brier_loss,
                notes=notes,
            )

        return scores

    @staticmethod
    def _evaluate_risk_decision(
        entry: DecisionJournalEntry,
        realized_pnl: float,
        is_winner: bool,
    ) -> str:
        """Evaluate deterministic risk management efficacy."""
        r_decision = (entry.risk_decision or "APPROVE").upper()
        req = entry.requested_quantity
        app = entry.approved_quantity

        if r_decision == "REDUCE":
            if not is_winner:
                saved_qty = req - app
                per_share_loss = abs(realized_pnl) / max(1, app)
                est_saved = saved_qty * per_share_loss
                return (
                    f"PROTECTIVE_REDUCTION: Risk engine reduced quantity from {req} to {app}. "
                    f"Curbed downside loss by estimated ₹/${est_saved:,.2f}."
                )
            else:
                return (
                    f"CONSERVATIVE_CAP: Risk engine reduced quantity from {req} to {app}. "
                    f"Secured winning trade under enforced portfolio risk budget."
                )
        elif r_decision == "APPROVE":
            if is_winner:
                return (
                    f"OPTIMAL_SIZING: Risk engine approved full {app} shares. "
                    f"Maximized upside capture without violating portfolio limits."
                )
            else:
                return (
                    f"CONTROLLED_LOSS: Risk engine approved {app} shares within configured single-trade loss limit."
                )
        elif r_decision == "REJECT":
            return "REJECTED_TRADE: Risk engine prevented trade execution."
        
        return f"STANDARD_EXECUTION: Decision={r_decision} (Approved {app}/{req} shares)."

    @staticmethod
    def _classify_failure(
        entry: DecisionJournalEntry,
        exit_price: float,
        exit_reason: Optional[str],
        realized_pnl: float,
        holding_seconds: float,
    ) -> tuple[str, str]:
        """Classify root cause of trade failure according to standardized taxonomy."""
        if realized_pnl > 0:
            return "NONE_WINNING_TRADE", "Trade achieved net positive profitability."

        reason_str = (exit_reason or "").upper()
        entry_price = entry.entry_price
        stop_loss = entry.stop_loss

        # Check macro warnings from agents
        macro_out = entry.agent_outputs.get("MacroAgent")
        if macro_out:
            m_risks = macro_out.get("risks", []) if isinstance(macro_out, dict) else getattr(macro_out, "risks", [])
            m_sig = macro_out.get("signal", "") if isinstance(macro_out, dict) else getattr(macro_out, "signal", "")
            if m_sig == "SELL" or any("macro" in r.lower() or "interest" in r.lower() or "volatility" in r.lower() for r in m_risks):
                return (
                    "MACRO_REVERSAL",
                    f"Adverse macroeconomic headwind materialized: {m_risks[:2] if m_risks else 'Macro signal BEARISH'}"
                )

        # Check news warnings
        news_out = entry.agent_outputs.get("NewsAgent")
        if news_out:
            n_risks = news_out.get("risks", []) if isinstance(news_out, dict) else getattr(news_out, "risks", [])
            if any("headline" in r.lower() or "regulatory" in r.lower() or "probe" in r.lower() for r in n_risks):
                return (
                    "NEWS_HEADLINE_RISK",
                    f"Negative headline or regulatory news materialized: {n_risks[:2]}"
                )

        # Check fundamental earnings warnings
        fund_out = entry.agent_outputs.get("FundamentalAgent")
        if fund_out:
            f_risks = fund_out.get("risks", []) if isinstance(fund_out, dict) else getattr(fund_out, "risks", [])
            if any("earnings" in r.lower() or "revenue" in r.lower() or "margin" in r.lower() for r in f_risks):
                return (
                    "EARNINGS_DISAPPOINTMENT",
                    f"Fundamental weakness or earnings pressure materialized: {f_risks[:2]}"
                )

        # Stop loss check
        if "STOP" in reason_str or (entry.direction == "BUY" and exit_price <= stop_loss):
            return (
                "STOP_LOSS_HIT",
                f"Stop loss triggered at ₹/{exit_price:.2f} (boundary ₹/{stop_loss:.2f}). Risk strictly cut."
            )

        # Long duration sideways timeout
        if holding_seconds > 86400 * 5 and abs(exit_price - entry_price) / max(0.001, entry_price) < 0.01:
            return (
                "CHOPPY_SIDEWAYS_TIMEOUT",
                f"Position stagnated over {holding_seconds / 86400:.1f} days without momentum."
            )

        # Technical breakdown default
        return (
            "TECHNICAL_BREAKDOWN",
            f"Price momentum reversed below support levels (Entry: {entry_price:.2f}, Exit: {exit_price:.2f})."
        )
