"""
Tests for Statutory Fee Engine, ATR Volatility Parity, and Half-Kelly Sizing.
"""
from __future__ import annotations

from datetime import datetime, timezone
import pytest
from src.trading.fees import FeeSchedule, DEFAULT_FEE_SCHEDULE
from src.risk.position_sizer import RiskEngine
from src.data.models import StockSnapshot, Quote


def test_statutory_fees_india_delivery_vs_intraday():
    fees = FeeSchedule()
    
    # 1. India Delivery Buy (is_paper=False)
    # Buy ₹25,000 notional
    buy_delivery_fee = fees.compute_fees(
        market="india",
        direction="BUY",
        filled_price=2500.0,
        quantity=10,
        is_paper=False,
        strategy="swing",
    )
    # 2. India Delivery Sell (is_paper=False)
    sell_delivery_fee = fees.compute_fees(
        market="india",
        direction="SELL",
        filled_price=2500.0,
        quantity=10,
        is_paper=False,
        strategy="swing",
    )
    assert buy_delivery_fee > 0
    assert sell_delivery_fee > 0
    
    # 3. India Intraday Sell vs Buy
    buy_intraday_fee = fees.compute_fees(
        market="india",
        direction="BUY",
        filled_price=2500.0,
        quantity=10,
        is_paper=False,
        strategy="intraday",
    )
    sell_intraday_fee = fees.compute_fees(
        market="india",
        direction="SELL",
        filled_price=2500.0,
        quantity=10,
        is_paper=False,
        strategy="intraday",
    )
    # Intraday STT is only on SELL side (0.025%), BUY side STT is 0%
    assert sell_intraday_fee > buy_intraday_fee


def test_statutory_fees_us():
    fees = FeeSchedule()
    
    # US Buy (is_paper=False)
    us_buy_fee = fees.compute_fees(
        market="us",
        direction="BUY",
        filled_price=150.0,
        quantity=100,
        is_paper=False,
    )
    # US Sell (is_paper=False, includes SEC fee + FINRA TAF + clearing)
    us_sell_fee = fees.compute_fees(
        market="us",
        direction="SELL",
        filled_price=150.0,
        quantity=100,
        is_paper=False,
    )
    assert us_buy_fee > 0
    assert us_sell_fee > us_buy_fee  # SEC fee and FINRA TAF applied to sell


def test_atr_volatility_parity_sizing():
    engine = RiskEngine()
    
    # Create fake history with 20 Quote bars
    history = [
        Quote(
            timestamp=datetime.now(timezone.utc),
            open=100.0,
            high=105.0,
            low=95.0,
            close=100.0 + i,
            volume=100000,
        )
        for i in range(20)
    ]
    snapshot = StockSnapshot(
        ticker="TEST.NS",
        market="india",
        current_price=110.0,
        currency="INR",
        history=history,
    )
    
    res = engine.calculate_position_size(
        snapshot=snapshot,
        direction="BUY",
        entry_price=110.0,
        stop_loss=105.0,
        target_price=125.0,
        current_cash=50000.0,
        total_portfolio_value=50000.0,
        strategy="swing",
    )
    assert res.allowed is True
    assert res.quantity > 0
    assert res.capital_allocated <= 50000.0
    assert res.risk_amount > 0
