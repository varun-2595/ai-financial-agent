"""
Trade Signal Generator combining Analyst evaluation, technical indicators,
and key price levels for Intraday, Swing, and Positional trading strategies.
"""
from __future__ import annotations

from typing import Literal, Optional

from src.analyst.debate_engine import RedTeamDebateEngine
from src.analyst.engine import AnalystEngine
from src.analyst.playbook_memory import PlaybookMemoryRetriever
from src.data.models import StockSnapshot, Strategy, TradeSignal
from src.news.catalyst_engine import NewsCatalystEngine
from src.risk.position_sizer import RiskEngine
from src.risk.stress_tester import PositionStressTester
from src.technicals.indicators import compute_technical_indicators
from src.technicals.levels import compute_support_resistance
from src.utils.logger import logger


class SignalGenerator:
    def __init__(
        self,
        analyst: Optional[AnalystEngine] = None,
        risk: Optional[RiskEngine] = None,
        catalyst_engine: Optional[NewsCatalystEngine] = None,
        memory_retriever: Optional[PlaybookMemoryRetriever] = None,
        debate_engine: Optional[RedTeamDebateEngine] = None,
        stress_tester: Optional[PositionStressTester] = None,
    ):
        self.analyst = analyst or AnalystEngine()
        self.risk = risk or RiskEngine()
        self.catalyst_engine = catalyst_engine or NewsCatalystEngine()
        self.memory_retriever = memory_retriever or PlaybookMemoryRetriever()
        self.debate_engine = debate_engine or RedTeamDebateEngine()
        self.stress_tester = stress_tester or PositionStressTester()

    def generate_signal(
        self,
        snapshot: StockSnapshot,
        strategy: Strategy = "scalping",
        current_cash: float = 10_000.0,
        portfolio_val: float = 10_000.0,
        sector_exposure_pct: float = 0.0,
        skip_db_check: bool = False,
        force_heuristic: bool = False,
    ) -> Optional[TradeSignal]:
        """
        Synthesizes technical conditions, historical memory, Red-Team debate,
        stress testing, and LLM analysis to produce high-conviction TradeSignal.
        """
        # Prevent duplicate entries if a position in this ticker is already open or closed recently at a loss
        if not skip_db_check:
            try:
                from datetime import datetime, timezone, timedelta
                from src.db.trading_store import _conn, get_open_positions
                open_pos = get_open_positions(market=snapshot.market)
                if any(p["ticker"] == snapshot.ticker for p in open_pos):
                    logger.debug(f"[Signal] Position already active for {snapshot.ticker}; skipping duplicate.")
                    return None

                # 4-hour cooldown after a stopped-out / losing trade to prevent churn
                cutoff_loss = (datetime.now(timezone.utc) - timedelta(hours=4)).isoformat()
                with _conn() as conn:
                    last_closed = conn.execute("""
                        SELECT realized_pnl, closed_at FROM positions
                        WHERE ticker = ? AND status = 'CLOSED'
                        ORDER BY closed_at DESC LIMIT 1
                    """, (snapshot.ticker,)).fetchone()
                    if last_closed:
                        pnl = float(last_closed["realized_pnl"] or 0.0)
                        closed_at_str = last_closed["closed_at"]
                        if pnl < 0 and closed_at_str and closed_at_str >= cutoff_loss:
                            logger.info(
                                f"[Signal] ❄️ Ticker {snapshot.ticker} in loss cooldown until {closed_at_str} + 4h (PnL: {pnl:+.2f}). Skipping entry."
                            )
                            return None
            except Exception:
                pass

        use_heuristic = force_heuristic or getattr(self.analyst, "force_heuristic", False)

        # 1. Historical Playbook Memory Check
        memory_report = self.memory_retriever.get_memory_warning(
            ticker=snapshot.ticker,
            market=snapshot.market,
            strategy=strategy,
        )

        # 2. Headline Risk / Catalyst Gatekeeper
        cat_report = None
        if snapshot.recent_news:
            cat_report = self.catalyst_engine.evaluate_catalysts(
                ticker=snapshot.ticker,
                market=snapshot.market,
                news_items=snapshot.recent_news,
                force_heuristic=use_heuristic,
            )
            if cat_report.has_headline_risk:
                logger.warning(
                    f"[Signal] 🚫 Trade aborted for {snapshot.ticker} due to headline risk: {cat_report.catalyst_summary}"
                )
                return None

        # 3. Technical calculations
        tech = compute_technical_indicators(snapshot.history)
        levels = compute_support_resistance(snapshot.history)

        # 4. Analyst evaluation
        analysis = self.analyst.analyze_stock(snapshot)

        # Only trigger buy on high conviction
        if analysis.action not in ("Must Buy", "Good Buy"):
            logger.info(f"[Signal] No buy signal for {snapshot.ticker} (action: {analysis.action})")
            return None

        current_p = snapshot.current_price
        direction: Literal["BUY", "SELL", "HOLD"] = "BUY"
        atr = tech.atr_14 if (tech.atr_14 and tech.atr_14 > 0) else (current_p * 0.02)

        # 5. Pre-Trade "Red Team" Adversarial Debate (M5 Local Node)
        debate = self.debate_engine.conduct_debate(
            snapshot=snapshot,
            direction=direction,
            strategy=strategy,
            memory_report=memory_report,
            force_heuristic=use_heuristic,
        )
        if debate.verdict == "REJECT" or debate.fatal_flaw_detected:
            logger.warning(
                f"[Signal] 🚫 Trade for {snapshot.ticker} REJECTED by Red Team: {debate.verdict_reasoning}"
            )
            return None

        # Calculate entry, SL, and target based on strategy & technical levels
        if strategy == "scalping":
            entry = current_p
            # Volatility-aware scalp: 1.5% or 1.2*ATR stop loss, 2.5% or 2.2*ATR profit target
            sl_dist = max(current_p * 0.015, atr * 1.2)
            tgt_dist = max(current_p * 0.025, atr * 2.2)
            stop_loss = round(current_p - sl_dist, 2)
            target = round(current_p + tgt_dist, 2)
        elif strategy == "intraday":
            entry = current_p
            # Intraday momentum breakout
            sl_dist = max(current_p * 0.020, atr * 1.5)
            tgt_dist = max(current_p * 0.038, atr * 2.8)
            stop_loss = round(levels.support_1 if (levels.support_1 < current_p and (current_p - levels.support_1) <= sl_dist * 1.2) else (current_p - sl_dist), 2)
            target = round(levels.resistance_1 if (levels.resistance_1 > current_p and (levels.resistance_1 - current_p) >= tgt_dist * 0.8) else (current_p + tgt_dist), 2)
        elif strategy == "swing":
            entry = current_p
            # Swing setup: support/ATR based
            sl_dist = max(current_p * 0.035, atr * 2.0)
            tgt_dist = max(current_p * 0.075, atr * 4.0)
            stop_loss = round(min(levels.support_1, current_p - sl_dist) if levels.support_1 < current_p else (current_p - sl_dist), 2)
            target = round(levels.resistance_2 if levels.resistance_2 > current_p else (current_p + tgt_dist), 2)
        else: # positional
            entry = current_p
            stop_loss = round(levels.support_2 if levels.support_2 < current_p else current_p * 0.90, 2)
            target = round(current_p * 1.15, 2)

        # 6. Risk Sizing & Approval
        size_res = self.risk.calculate_position_size(
            snapshot=snapshot,
            direction=direction,
            entry_price=entry,
            stop_loss=stop_loss,
            target_price=target,
            current_cash=current_cash,
            total_portfolio_value=portfolio_val,
            sector_exposure_pct=sector_exposure_pct,
            strategy=strategy,
        )

        if not size_res.allowed:
            logger.warning(f"[Signal] Signal for {snapshot.ticker} rejected by RiskEngine: {size_res.reason}")
            return None

        final_quantity = size_res.quantity
        if debate.verdict == "REDUCE":
            final_quantity = max(1, final_quantity // 2)

        # 7. Synthetic Stress Testing
        stress = self.stress_tester.test_position_resilience(
            snapshot=snapshot,
            quantity=final_quantity,
            entry_price=entry,
            stop_loss=stop_loss,
            total_capital=portfolio_val,
        )
        if not stress.passed:
            logger.warning(f"[Signal] 🛑 Signal for {snapshot.ticker} failed stress test: {stress.recommendation}")
            return None

        sig = TradeSignal(
            ticker=snapshot.ticker,
            market=snapshot.market,
            strategy=strategy,
            direction=direction,
            entry_price=entry,
            stop_loss=stop_loss,
            target_price=target,
            quantity=final_quantity,
            confidence=round(min(analysis.confidence_score, debate.confidence_score), 2),
            reasoning=f"Action: {analysis.action} | Red-Team: {debate.verdict} ({debate.verdict_reasoning}) | Stress: Passed",
        )
        logger.success(
            f"[Signal] 🔥 GENERATED {sig.direction} {sig.quantity}x {sig.ticker} @ {sig.entry_price} "
            f"(SL: {sig.stop_loss}, TGT: {sig.target_price}, Conf: {sig.confidence:.2f})"
        )
        return sig
