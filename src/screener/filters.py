"""
Quantitative pre-filters applied to the raw universe before LLM analysis.
"""
from __future__ import annotations

from dataclasses import dataclass

from src.data.models import StockSnapshot
from src.utils.logger import logger


@dataclass
class FilterConfig:
    min_avg_volume_india: int = 200_000      # NSE: min 200k shares/day
    min_avg_volume_us: int = 500_000         # NYSE: min 500K shares/day
    min_price_inr: float = 50.0
    min_price_usd: float = 5.0
    min_candles: int = 60
    max_drawdown_from_1y_high_pct: float = 60.0
    min_atr_pct_intraday: float = 1.0
    max_atr_pct_positional: float = 8.0


DEFAULT_FILTERS = FilterConfig()


def _compute_atr_pct(history: list, lookback: int = 14) -> float | None:
    if len(history) < lookback + 1:
        return None
    recent = history[-lookback - 1:]
    true_ranges = []
    for i in range(1, len(recent)):
        high = recent[i].high
        low  = recent[i].low
        prev_close = recent[i - 1].close
        tr = max(high - low, abs(high - prev_close), abs(low - prev_close))
        true_ranges.append(tr)
    if not true_ranges or recent[-1].close == 0:
        return None
    atr = sum(true_ranges) / len(true_ranges)
    return (atr / recent[-1].close) * 100


def _compute_drawdown_from_high(history: list) -> float | None:
    if not history:
        return None
    peak = max(q.high for q in history)
    current = history[-1].close
    if peak == 0:
        return None
    return ((peak - current) / peak) * 100


def passes_filters(
    snapshot: StockSnapshot,
    strategy: str = "swing",
    cfg: FilterConfig = DEFAULT_FILTERS,
) -> tuple[bool, str]:
    h = snapshot.history
    f = snapshot.fundamentals
    is_india = snapshot.market == "india"

    # 1. Data quality
    if len(h) < cfg.min_candles:
        return False, f"Insufficient history ({len(h)} candles, need {cfg.min_candles})"

    # 2. Price
    min_price = cfg.min_price_inr if is_india else cfg.min_price_usd
    if snapshot.current_price < min_price:
        sym = "₹" if is_india else "$"
        return False, f"Price too low ({sym}{snapshot.current_price:.2f} < {sym}{min_price})"

    # 3. Liquidity — fail explicitly if volume is unknown or too low
    avg_vol = f.avg_volume_30d
    min_vol = cfg.min_avg_volume_india if is_india else cfg.min_avg_volume_us
    if avg_vol is None:
        return False, "Unknown average volume (missing liquidity data)"
    if avg_vol < min_vol:
        return False, f"Insufficient liquidity (avg vol {avg_vol:,} < {min_vol:,})"

    # 4. Trend
    dd = _compute_drawdown_from_high(h)
    if dd is not None and dd > cfg.max_drawdown_from_1y_high_pct:
        return False, f"Severe downtrend (down {dd:.1f}% from 1yr high)"

    # 5. Volatility
    atr_pct = _compute_atr_pct(h)
    if atr_pct is not None:
        if strategy == "intraday" and atr_pct < cfg.min_atr_pct_intraday:
            return False, f"Too low volatility for intraday (ATR {atr_pct:.2f}% < {cfg.min_atr_pct_intraday}%)"
        if strategy == "positional" and atr_pct > cfg.max_atr_pct_positional:
            return False, f"Too volatile for positional (ATR {atr_pct:.2f}% > {cfg.max_atr_pct_positional}%)"

    return True, "OK"


def filter_batch(
    snapshots: dict[str, StockSnapshot],
    strategy: str = "swing",
    cfg: FilterConfig = DEFAULT_FILTERS,
) -> dict[str, StockSnapshot]:
    passed: dict[str, StockSnapshot] = {}
    for ticker, snap in snapshots.items():
        ok, reason = passes_filters(snap, strategy=strategy, cfg=cfg)
        if ok:
            passed[ticker] = snap
        else:
            logger.debug(f"[Filter] ✗ {ticker}: {reason}")

    logger.info(
        f"[Filter] {len(passed)}/{len(snapshots)} tickers passed "
        f"({strategy} strategy filters)"
    )
    return passed
