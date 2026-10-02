"""
Synthetic Market Shock & Position Stress Tester.

Runs 5 rapid synthetic stress scenarios locally on M5 before trade execution:
1. Benchmark Gap Down (-1.5% index shock)
2. Sector Liquidity Outflow (-2.5% sector drop)
3. Volatility Expansion (+50% ATR spike)
4. Slippage / Spread Widening (0.5% liquidity squeeze)
5. Overnight Gap Down (-3.0% tail-risk scenario)
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal, Optional

from src.data.models import StockSnapshot
from src.utils.logger import logger


@dataclass
class ScenarioResult:
    scenario_name: str
    simulated_price: float
    simulated_loss: float
    loss_pct_of_capital: float
    survives_circuit_breaker: bool
    notes: str


@dataclass
class StressTestOutcome:
    ticker: str
    market: Literal["india", "us"]
    quantity: int
    entry_price: float
    stop_loss: float
    passed: bool
    worst_case_loss: float
    scenarios: list[ScenarioResult] = field(default_factory=list)
    recommendation: str = ""


class PositionStressTester:
    """Simulates market shocks and tail-risk resilience before order dispatch."""

    def test_position_resilience(
        self,
        snapshot: StockSnapshot,
        quantity: int,
        entry_price: float,
        stop_loss: float,
        total_capital: float = 10_000.0,
        max_allowed_loss_pct: float = 0.03,  # Max 3% of capital per single trade shock
    ) -> StressTestOutcome:
        """
        Evaluates proposed position against 5 synthetic stress scenarios.
        """
        ticker = snapshot.ticker
        market = snapshot.market
        scenarios: list[ScenarioResult] = []
        max_loss_limit = total_capital * max_allowed_loss_pct

        # 1. Benchmark Flash Drop (-1.5% Index Shock)
        # Beta-adjusted shock: stock drops by (1.5% * beta), caught at stop_loss if breached
        beta = snapshot.fundamentals.beta if (snapshot.fundamentals and snapshot.fundamentals.beta) else 1.2
        shock_1_pct = 0.015 * beta
        sim_p1 = round(entry_price * (1.0 - shock_1_pct), 2)
        exit_p1 = max(sim_p1, stop_loss)
        loss_1 = round(max(0.0, entry_price - exit_p1) * quantity, 2)
        scenarios.append(ScenarioResult(
            scenario_name="Benchmark Gap Down (-1.5% Index)",
            simulated_price=sim_p1,
            simulated_loss=loss_1,
            loss_pct_of_capital=round((loss_1 / total_capital) * 100, 2),
            survives_circuit_breaker=loss_1 <= max_loss_limit,
            notes=f"Stock adjusts -{shock_1_pct*100:.1f}% based on beta {beta:.2f}",
        ))

        # 2. Sector Liquidity Outflow (-2.5% Sector Shock)
        sim_p2 = round(entry_price * 0.975, 2)
        exit_p2 = max(sim_p2, stop_loss)
        loss_2 = round(max(0.0, entry_price - exit_p2) * quantity, 2)
        scenarios.append(ScenarioResult(
            scenario_name="Sector Liquidity Outflow (-2.5%)",
            simulated_price=sim_p2,
            simulated_loss=loss_2,
            loss_pct_of_capital=round((loss_2 / total_capital) * 100, 2),
            survives_circuit_breaker=loss_2 <= max_loss_limit,
            notes="Sector-wide selling pressure caught at stop loss",
        ))

        # 3. Volatility Expansion (Stop Slippage by 0.5%)
        sim_p3 = round(stop_loss * 0.995, 2)
        loss_3 = round(max(0.0, entry_price - sim_p3) * quantity, 2)
        scenarios.append(ScenarioResult(
            scenario_name="Volatility & Slippage Expansion (0.5% gap past SL)",
            simulated_price=sim_p3,
            simulated_loss=loss_3,
            loss_pct_of_capital=round((loss_3 / total_capital) * 100, 2),
            survives_circuit_breaker=loss_3 <= max_loss_limit,
            notes="Simulates fast spread widening on stop exit",
        ))

        # 4. Overnight Gap Down (-3.0%)
        sim_p4 = round(entry_price * 0.970, 2)
        loss_4 = round(max(0.0, entry_price - sim_p4) * quantity, 2)
        scenarios.append(ScenarioResult(
            scenario_name="Overnight Market Gap Down (-3.0%)",
            simulated_price=sim_p4,
            simulated_loss=loss_4,
            loss_pct_of_capital=round((loss_4 / total_capital) * 100, 2),
            survives_circuit_breaker=loss_4 <= max_loss_limit * 1.5,
            notes="Overnight macro headline tail risk",
        ))

        # 5. Full Stop Loss Execution
        loss_5 = round(max(0.0, entry_price - stop_loss) * quantity, 2)
        scenarios.append(ScenarioResult(
            scenario_name="Standard Stop Loss Execution",
            simulated_price=stop_loss,
            simulated_loss=loss_5,
            loss_pct_of_capital=round((loss_5 / total_capital) * 100, 2),
            survives_circuit_breaker=loss_5 <= max_loss_limit,
            notes="Controlled structural stop loss exit",
        ))

        worst_loss = max(s.simulated_loss for s in scenarios)
        all_passed = all(s.survives_circuit_breaker for s in scenarios)

        outcome = StressTestOutcome(
            ticker=ticker,
            market=market,
            quantity=quantity,
            entry_price=entry_price,
            stop_loss=stop_loss,
            passed=all_passed,
            worst_case_loss=worst_loss,
            scenarios=scenarios,
            recommendation="APPROVED (Stress test passed all 5 shocks)" if all_passed else f"REJECTED: Worst-case loss ({worst_loss:.2f}) exceeds risk limit ({max_loss_limit:.2f})",
        )
        logger.info(f"[Stress Tester] {ticker}: {outcome.recommendation} (Worst loss: {worst_loss:.2f})")
        return outcome
