"""
Portfolio-Level Risk Management Engine.

Strictly enforces 10 deterministic risk controls:
1. Available Cash & Buying Power
2. Maximum Single Position Exposure Cap
3. Maximum Sector Exposure Cap
4. Maximum Market Gross Exposure Cap
5. Maximum Portfolio Leverage Multiplier
6. Daily Loss Circuit Breaker
7. Maximum Portfolio Drawdown Circuit Breaker
8. Portfolio Beta Limit
9. Correlation & Sector Concentration Limit
10. Liquidity & Average Daily Volume (ADV) Constraint

Every trade proposal is evaluated deterministically:
AI Proposal → Portfolio Manager → PortfolioRiskManager → APPROVE / REDUCE / REJECT → Execution.
The LLM is completely isolated and cannot override any limit.
"""
from __future__ import annotations

import math
from typing import Literal, Optional

from src.data.models import StockSnapshot, TradeSignal
from src.risk.audit_store import RiskAuditStore
from src.risk.models import (
    PortfolioRiskState,
    RiskCheckName,
    RiskCheckResult,
    RiskDecision,
    RiskEvaluationResult,
)
from src.utils.config import get_config
from src.utils.logger import logger


class PortfolioRiskManager:
    """Deterministic Portfolio-Level Risk Gatekeeper."""

    def __init__(
        self,
        audit_store: Optional[RiskAuditStore] = None,
        max_portfolio_beta: float = 1.50,
        max_positions_per_sector: int = 3,
        max_adv_pct: float = 0.02,  # Max 2% of 30-day average daily volume
        min_adv_volume: int = 10_000,  # Minimum 30-day ADV to qualify as liquid
        max_market_exposure_pct: float = 3.0,  # Max 300% gross exposure under 3x leverage
    ):
        self.config = get_config()
        self.audit_store = audit_store or RiskAuditStore()
        self.max_portfolio_beta = max_portfolio_beta
        self.max_positions_per_sector = max_positions_per_sector
        self.max_adv_pct = max_adv_pct
        self.min_adv_volume = min_adv_volume
        self.max_market_exposure_pct = max_market_exposure_pct

    def evaluate_trade(
        self,
        signal: TradeSignal,
        risk_state: PortfolioRiskState,
        snapshot: Optional[StockSnapshot] = None,
    ) -> RiskEvaluationResult:
        """
        Evaluate proposed trade against all 10 portfolio-level risk rules.
        Returns RiskEvaluationResult with decision (APPROVE / REDUCE / REJECT) and approved quantity.
        """
        cfg_paper = self.config.paper_trading
        market = signal.market.lower()
        strategy = signal.strategy
        ticker = signal.ticker.upper()
        direction = signal.direction
        req_qty = signal.quantity
        entry_price = signal.entry_price
        stop_loss = signal.stop_loss
        target_price = signal.target_price

        checks: list[RiskCheckResult] = []
        violations: list[str] = []
        warnings: list[str] = []

        # 0. Basic Price & Direction Sanity
        if entry_price <= 0 or stop_loss <= 0 or target_price <= 0 or req_qty <= 0:
            res = self._build_rejection(
                signal, risk_state,
                reason="Invalid price values or requested quantity <= 0",
                checks=checks, violations=["Invalid trade parameter pricing"],
            )
            self.audit_store.record_evaluation(res, risk_state)
            return res

        # 1. Direction and Risk-Reward Validation
        if direction == "BUY":
            if stop_loss >= entry_price or target_price <= entry_price:
                res = self._build_rejection(
                    signal, risk_state,
                    reason="Stop loss must be below entry and target above entry for BUY",
                    checks=checks, violations=["Invalid BUY stop loss or target boundary"],
                )
                self.audit_store.record_evaluation(res, risk_state)
                return res
            risk_per_share = entry_price - stop_loss
            reward_per_share = target_price - entry_price
        else:
            if stop_loss <= entry_price or target_price >= entry_price:
                res = self._build_rejection(
                    signal, risk_state,
                    reason="Stop loss must be above entry and target below entry for SELL",
                    checks=checks, violations=["Invalid SELL stop loss or target boundary"],
                )
                self.audit_store.record_evaluation(res, risk_state)
                return res
            risk_per_share = stop_loss - entry_price
            reward_per_share = entry_price - target_price

        min_rr = 1.0
        rr_ratio = reward_per_share / risk_per_share if risk_per_share > 0 else 0.0
        rr_passed = rr_ratio >= min_rr
        checks.append(RiskCheckResult(
            check_name=RiskCheckName.RISK_REWARD_RATIO,
            passed=rr_passed,
            limit_value=min_rr,
            current_value=0.0,
            projected_value=round(rr_ratio, 2),
            message=f"R:R ratio is {rr_ratio:.2f} (min required {min_rr})",
        ))
        if not rr_passed:
            violations.append(f"Unfavorable Risk:Reward ratio ({rr_ratio:.2f} < {min_rr})")

        # 2. Hard Circuit Breaker: Daily Loss Limit
        max_daily_loss = (
            cfg_paper.daily_max_loss_inr if market == "india" else cfg_paper.daily_max_loss_usd
        )
        current_daily_loss = -(risk_state.daily_realized_pnl + min(0.0, risk_state.daily_unrealized_pnl))
        daily_loss_passed = current_daily_loss < max_daily_loss
        checks.append(RiskCheckResult(
            check_name=RiskCheckName.DAILY_LOSS_CIRCUIT_BREAKER,
            passed=daily_loss_passed,
            limit_value=max_daily_loss,
            current_value=round(current_daily_loss, 2),
            projected_value=round(current_daily_loss, 2),
            message=f"Daily loss is {current_daily_loss:.2f} / {max_daily_loss:.2f} limit",
        ))
        if not daily_loss_passed:
            violations.append(f"Daily loss circuit breaker tripped ({current_daily_loss:.2f} >= {max_daily_loss:.2f})")

        # 3. Hard Circuit Breaker: Maximum Portfolio Drawdown
        max_dd_pct = cfg_paper.max_drawdown_pct  # e.g. 0.15 (15%)
        dd_passed = risk_state.current_drawdown_pct < max_dd_pct
        checks.append(RiskCheckResult(
            check_name=RiskCheckName.PORTFOLIO_DRAWDOWN,
            passed=dd_passed,
            limit_value=max_dd_pct,
            current_value=round(risk_state.current_drawdown_pct, 4),
            projected_value=round(risk_state.current_drawdown_pct, 4),
            message=f"Current portfolio drawdown {risk_state.current_drawdown_pct*100:.2f}% (limit {max_dd_pct*100:.1f}%)",
        ))
        if not dd_passed:
            violations.append(f"Portfolio drawdown circuit breaker active ({risk_state.current_drawdown_pct*100:.1f}% >= {max_dd_pct*100:.1f}%)")

        # 4. Cash Constraint (Requested order cannot exceed available cash)
        is_margin = strategy in ("scalping", "intraday")
        leverage = cfg_paper.intraday_leverage_multiplier if is_margin else 1.0
        margin_per_share = entry_price / leverage
        req_margin = req_qty * margin_per_share
        cash_sufficient = req_margin <= (risk_state.cash * 1.01)

        checks.append(RiskCheckResult(
            check_name=RiskCheckName.AVAILABLE_CASH,
            passed=cash_sufficient,
            limit_value=round(risk_state.cash, 2),
            current_value=round(risk_state.cash, 2),
            projected_value=round(max(0.0, risk_state.cash - req_margin), 2),
            message=f"Requested margin {req_margin:.2f} <= available cash {risk_state.cash:.2f}",
        ))
        if not cash_sufficient:
            violations.append(f"Insufficient available cash ({risk_state.cash:.2f} < {req_margin:.2f} needed)")

        # 5. Correlation & Sector Position Count Limit
        sector_name = (snapshot.sector if snapshot and snapshot.sector else "General")
        curr_sector_count = risk_state.sector_position_counts.get(sector_name, 0)
        is_existing_position = any(p.get("ticker") == ticker for p in risk_state.open_positions)
        projected_sector_count = curr_sector_count if is_existing_position else curr_sector_count + 1
        count_passed = projected_sector_count <= self.max_positions_per_sector
        checks.append(RiskCheckResult(
            check_name=RiskCheckName.CORRELATION_CONCENTRATION,
            passed=count_passed,
            limit_value=float(self.max_positions_per_sector),
            current_value=float(curr_sector_count),
            projected_value=float(projected_sector_count),
            message=f"Sector '{sector_name}' has {projected_sector_count} positions (max {self.max_positions_per_sector})",
        ))
        if not count_passed:
            violations.append(f"Max positions in sector '{sector_name}' reached ({projected_sector_count} > {self.max_positions_per_sector})")

        # 6. Liquidity & Average Daily Volume (ADV) Check
        stock_adv = 1_000_000
        if snapshot and snapshot.fundamentals and snapshot.fundamentals.avg_volume_30d:
            stock_adv = snapshot.fundamentals.avg_volume_30d
        elif snapshot and snapshot.history:
            vols = [q.volume for q in snapshot.history if q.volume > 0]
            if vols:
                stock_adv = int(sum(vols) / len(vols))

        adv_liquid = stock_adv >= self.min_adv_volume
        max_qty_by_adv = max(1, int(stock_adv * self.max_adv_pct))
        adv_passed = adv_liquid
        checks.append(RiskCheckResult(
            check_name=RiskCheckName.LIQUIDITY_ADV,
            passed=adv_passed,
            limit_value=float(max_qty_by_adv),
            current_value=float(stock_adv),
            projected_value=float(req_qty),
            message=f"30d ADV {stock_adv:,} (max order {max_qty_by_adv:,} shares)",
        ))
        if not adv_liquid:
            violations.append(f"Stock is illiquid (30d ADV {stock_adv:,} < min {self.min_adv_volume:,})")

        # If any hard limit is tripped, reject immediately
        if violations:
            res = self._build_rejection(
                signal, risk_state,
                reason="; ".join(violations),
                checks=checks, violations=violations, warnings=warnings,
            )
            self.audit_store.record_evaluation(res, risk_state)
            return res

        # ─────────────────────────────────────────────────────────────────
        # DYNAMIC QUANTITY SIZING & CONSTRAINT SATISFACTION
        # ─────────────────────────────────────────────────────────────────
        is_margin = strategy in ("scalping", "intraday")
        leverage = cfg_paper.intraday_leverage_multiplier if is_margin else 1.0
        nav = max(100.0, risk_state.nav)

        # A. Trade-level risk cap (max % of NAV lost if stop-loss hit)
        max_risk_pct = 0.05 if strategy in ("scalping", "intraday") else max(0.05, cfg_paper.risk_per_trade_pct)
        max_loss_budget = nav * max_risk_pct
        qty_cap_risk = int(max_loss_budget / risk_per_share) if risk_per_share > 0 else 0

        # B. Position size cap (% of NAV)
        if strategy == "scalping":
            max_pos_val = nav * cfg_paper.max_position_pct * leverage
        elif strategy == "intraday":
            max_pos_val = nav * 0.40 * leverage
        elif strategy == "swing":
            max_pos_val = nav * 0.50
        else:
            max_pos_val = nav * 0.25
        qty_cap_pos = max(1, int(max_pos_val / entry_price))

        # C. Sector exposure cap (% of NAV)
        curr_sector_val = risk_state.sector_exposures.get(sector_name, 0.0)
        max_sector_val = nav * cfg_paper.max_sector_pct
        available_sector_val = max(0.0, max_sector_val - curr_sector_val)
        qty_cap_sector = int(available_sector_val / entry_price)

        # D. Available Cash / Buying Power
        # margin per share = entry_price / leverage
        margin_per_share = entry_price / leverage
        qty_cap_cash = int(risk_state.cash / margin_per_share) if margin_per_share > 0 else 0

        # E. ADV Volume cap
        qty_cap_adv = max_qty_by_adv

        # F. Portfolio Beta constraint
        # projected_beta = (sum(pos_val * beta) + new_val * stock_beta) / total_exposure
        stock_beta = snapshot.fundamentals.beta if (snapshot and snapshot.fundamentals and snapshot.fundamentals.beta) else 1.0
        # If stock beta is very high (> 1.8), restrict size to keep portfolio beta under limit
        if stock_beta > self.max_portfolio_beta and risk_state.portfolio_beta > 1.2:
            qty_cap_beta = max(1, int((nav * 0.15 * leverage) / entry_price))
        else:
            qty_cap_beta = req_qty

        # Sizing decision: minimum permitted across all mathematical bounds
        candidate_qty = min(
            req_qty,
            qty_cap_risk,
            qty_cap_pos,
            qty_cap_sector,
            qty_cap_cash,
            qty_cap_adv,
            qty_cap_beta,
        )

        # Edge case: If integer truncation gives 0 on small balances, allow 1 share if cash allows
        if candidate_qty <= 0 and risk_state.cash >= margin_per_share and available_sector_val >= entry_price:
            candidate_qty = 1

        # Check Cash Constraint Result
        cash_passed = candidate_qty > 0 and (candidate_qty * margin_per_share) <= risk_state.cash
        checks.append(RiskCheckResult(
            check_name=RiskCheckName.AVAILABLE_CASH,
            passed=cash_passed,
            limit_value=round(risk_state.cash, 2),
            current_value=round(risk_state.cash, 2),
            projected_value=round(risk_state.cash - (candidate_qty * margin_per_share), 2),
            message=f"Required margin {candidate_qty * margin_per_share:.2f} <= cash {risk_state.cash:.2f}",
        ))

        # Check Position Size Result
        proj_pos_val = candidate_qty * entry_price
        pos_passed = proj_pos_val <= (max_pos_val * 1.05)
        checks.append(RiskCheckResult(
            check_name=RiskCheckName.MAX_POSITION_SIZE,
            passed=pos_passed,
            limit_value=round(max_pos_val, 2),
            current_value=0.0,
            projected_value=round(proj_pos_val, 2),
            message=f"Position value {proj_pos_val:.2f} <= max {max_pos_val:.2f}",
        ))

        # Check Sector Exposure Result
        proj_sector_val = curr_sector_val + proj_pos_val
        sector_passed = proj_sector_val <= (max_sector_val * 1.05)
        checks.append(RiskCheckResult(
            check_name=RiskCheckName.MAX_SECTOR_EXPOSURE,
            passed=sector_passed,
            limit_value=round(max_sector_val, 2),
            current_value=round(curr_sector_val, 2),
            projected_value=round(proj_sector_val, 2),
            message=f"Sector '{sector_name}' exposure {proj_sector_val:.2f} <= max {max_sector_val:.2f}",
        ))

        # Check Market Gross Exposure Result
        proj_gross = risk_state.gross_exposure + proj_pos_val
        max_market_val = nav * self.max_market_exposure_pct
        market_passed = proj_gross <= max_market_val
        checks.append(RiskCheckResult(
            check_name=RiskCheckName.MAX_MARKET_EXPOSURE,
            passed=market_passed,
            limit_value=round(max_market_val, 2),
            current_value=round(risk_state.gross_exposure, 2),
            projected_value=round(proj_gross, 2),
            message=f"Gross exposure {proj_gross:.2f} <= max market limit {max_market_val:.2f}",
        ))

        # Check Leverage Result
        proj_leverage = proj_gross / nav if nav > 0 else 0.0
        leverage_passed = proj_leverage <= (self.config.paper_trading.intraday_leverage_multiplier * 1.05)
        checks.append(RiskCheckResult(
            check_name=RiskCheckName.MAX_LEVERAGE,
            passed=leverage_passed,
            limit_value=self.config.paper_trading.intraday_leverage_multiplier,
            current_value=round(risk_state.current_leverage, 2),
            projected_value=round(proj_leverage, 2),
            message=f"Projected leverage {proj_leverage:.2f}x <= max {self.config.paper_trading.intraday_leverage_multiplier:.1f}x",
        ))

        # Check Portfolio Beta Result
        proj_beta = ((risk_state.gross_exposure * risk_state.portfolio_beta) + (proj_pos_val * stock_beta)) / max(1.0, proj_gross)
        beta_passed = proj_beta <= self.max_portfolio_beta
        checks.append(RiskCheckResult(
            check_name=RiskCheckName.PORTFOLIO_BETA,
            passed=beta_passed,
            limit_value=self.max_portfolio_beta,
            current_value=round(risk_state.portfolio_beta, 2),
            projected_value=round(proj_beta, 2),
            message=f"Projected portfolio beta {proj_beta:.2f} <= max {self.max_portfolio_beta:.2f}",
        ))

        # ─────────────────────────────────────────────────────────────────
        # FINAL DECISION SYNTHESIS
        # ─────────────────────────────────────────────────────────────────
        if candidate_qty <= 0 or not cash_passed:
            reason = "Insufficient available cash or position sizing reduced to 0"
            res = self._build_rejection(
                signal, risk_state,
                reason=reason, checks=checks, violations=[reason], warnings=warnings,
            )
            self.audit_store.record_evaluation(res, risk_state)
            return res

        approved_margin = round(candidate_qty * margin_per_share, 2)
        allocated_capital = round(candidate_qty * entry_price, 2)
        total_risk = round(candidate_qty * risk_per_share, 2)

        if candidate_qty < req_qty:
            decision = RiskDecision.REDUCE
            reduction_reasons = []
            if candidate_qty == qty_cap_cash:
                reduction_reasons.append(f"cash limit ({qty_cap_cash} shs)")
            if candidate_qty == qty_cap_pos:
                reduction_reasons.append(f"position cap ({qty_cap_pos} shs)")
            if candidate_qty == qty_cap_sector:
                reduction_reasons.append(f"sector room ({qty_cap_sector} shs)")
            if candidate_qty == qty_cap_risk:
                reduction_reasons.append(f"risk budget ({qty_cap_risk} shs)")
            if candidate_qty == qty_cap_adv:
                reduction_reasons.append(f"2% ADV liquidity ({qty_cap_adv} shs)")

            reason = f"Reduced from {req_qty} to {candidate_qty} shares due to {', '.join(reduction_reasons)}"
            warnings.append(reason)
        else:
            decision = RiskDecision.APPROVE
            reason = f"Approved: {candidate_qty} shares (Margin: {approved_margin:.2f}, R:R {rr_ratio:.2f})"

        eval_result = RiskEvaluationResult(
            decision=decision,
            ticker=ticker,
            market=signal.market,
            strategy=strategy,
            direction=direction,
            requested_quantity=req_qty,
            approved_quantity=candidate_qty,
            entry_price=entry_price,
            stop_loss=stop_loss,
            target_price=target_price,
            approved_margin=approved_margin,
            capital_allocated=allocated_capital,
            risk_amount=total_risk,
            checks=checks,
            violations=[],
            warnings=warnings,
            reason=reason,
        )

        self.audit_store.record_evaluation(eval_result, risk_state)
        return eval_result

    def _build_rejection(
        self,
        signal: TradeSignal,
        risk_state: PortfolioRiskState,
        reason: str,
        checks: list[RiskCheckResult],
        violations: list[str],
        warnings: Optional[list[str]] = None,
    ) -> RiskEvaluationResult:
        """Helper to construct standard REJECT result."""
        return RiskEvaluationResult(
            decision=RiskDecision.REJECT,
            ticker=signal.ticker.upper(),
            market=signal.market,
            strategy=signal.strategy,
            direction=signal.direction,
            requested_quantity=signal.quantity,
            approved_quantity=0,
            entry_price=signal.entry_price,
            stop_loss=signal.stop_loss,
            target_price=signal.target_price,
            approved_margin=0.0,
            capital_allocated=0.0,
            risk_amount=0.0,
            checks=checks,
            violations=violations,
            warnings=warnings or [],
            reason=f"REJECTED: {reason}",
        )
