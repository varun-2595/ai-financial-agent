"""
Support, Resistance, Fibonacci Retracement, and Key Price Levels detection.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
import pandas as pd

from src.data.models import Quote


@dataclass
class PriceLevels:
    current_price: float
    support_1: float
    support_2: float
    resistance_1: float
    resistance_2: float
    fib_382: float
    fib_500: float
    fib_618: float
    pivot_point: float


def compute_support_resistance(quotes: Sequence[Quote], window: int = 20) -> PriceLevels:
    """
    Compute pivot points and dynamic support/resistance levels.
    """
    if not quotes:
        return PriceLevels(0, 0, 0, 0, 0, 0, 0, 0, 0)

    recent = quotes[-window:]
    high = max(q.high for q in recent)
    low = min(q.low for q in recent)
    close = quotes[-1].close

    # Standard Floor Pivot Points
    pivot = (high + low + close) / 3.0
    r1 = (2.0 * pivot) - low
    s1 = (2.0 * pivot) - high
    r2 = pivot + (high - low)
    s2 = pivot - (high - low)

    # Fibonacci Retracements from swing high/low
    price_range = high - low
    fib_382 = high - (0.382 * price_range)
    fib_500 = high - (0.500 * price_range)
    fib_618 = high - (0.618 * price_range)

    return PriceLevels(
        current_price=close,
        support_1=round(s1, 2),
        support_2=round(s2, 2),
        resistance_1=round(r1, 2),
        resistance_2=round(r2, 2),
        fib_382=round(fib_382, 2),
        fib_500=round(fib_500, 2),
        fib_618=round(fib_618, 2),
        pivot_point=round(pivot, 2),
    )
