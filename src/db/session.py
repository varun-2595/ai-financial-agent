"""
Unified Database Session and SQLAlchemy 2.0 Async Engine.
Supports PostgreSQL (asyncpg) with persistent connection pooling and graceful SQLite fallback.
"""
from __future__ import annotations

import os
from contextlib import asynccontextmanager, contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, AsyncGenerator, Generator, Optional

from sqlalchemy import (
    BigInteger,
    Column,
    DateTime,
    Float,
    Index,
    Integer,
    MetaData,
    Numeric,
    SmallInteger,
    String,
    Text,
    create_engine,
    text,
)
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.ext.asyncio import (
    AsyncEngine,
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)
from sqlalchemy.orm import DeclarativeBase, declarative_base, sessionmaker

from src.utils.logger import logger

DEFAULT_SQLITE_PATH = Path(__file__).parent.parent.parent / "data" / "trading.db"


def _get_database_url() -> str:
    env_url = os.getenv("DATABASE_URL", "").strip()
    if env_url:
        # Normalise postgres:// -> postgresql+asyncpg:// if necessary
        if env_url.startswith("postgres://"):
            env_url = env_url.replace("postgres://", "postgresql+asyncpg://", 1)
        elif env_url.startswith("postgresql://") and not env_url.startswith("postgresql+asyncpg://"):
            env_url = env_url.replace("postgresql://", "postgresql+asyncpg://", 1)
        return env_url
    DEFAULT_SQLITE_PATH.parent.mkdir(parents=True, exist_ok=True)
    return f"sqlite+aiosqlite:///{DEFAULT_SQLITE_PATH}"


class Base(DeclarativeBase):
    pass


# ── SQLAlchemy Models ─────────────────────────────────────────────────────────

class AccountModel(Base):
    __tablename__ = "accounts"
    account_id = Column(String(32), primary_key=True)
    currency = Column(String(8), nullable=False)
    cash = Column(Float, nullable=False)
    initial_cash = Column(Float, nullable=False)
    reserved_margin = Column(Float, nullable=False, default=0.0)
    peak_nav = Column(Float, nullable=False, default=0.0)
    updated_at = Column(DateTime(timezone=True), nullable=False, default=lambda: datetime.now(timezone.utc))


class OrderModel(Base):
    __tablename__ = "orders"
    order_id = Column(String(64), primary_key=True)
    broker_order_ref = Column(String(64), nullable=True)
    ticker = Column(String(32), nullable=False, index=True)
    market = Column(String(16), nullable=False)
    exchange = Column(String(16), nullable=False, default="NSE")
    strategy = Column(String(32), nullable=False)
    order_type = Column(String(16), nullable=False)
    direction = Column(String(8), nullable=False)
    quantity = Column(Integer, nullable=False)
    limit_price = Column(Float, nullable=True)
    trigger_price = Column(Float, nullable=True)
    requested_price = Column(Float, nullable=True)
    filled_price = Column(Float, nullable=True)
    executed_price = Column(Float, nullable=True)
    fees = Column(Float, nullable=False, default=0.0)
    statutory_fees = Column(Float, nullable=False, default=0.0)
    slippage = Column(Float, nullable=False, default=0.0)
    slippage_pct = Column(Float, nullable=False, default=0.0005)
    status = Column(String(24), nullable=False, index=True)
    is_paper = Column(SmallInteger, nullable=False, default=1)
    created_at = Column(DateTime(timezone=True), nullable=False, default=lambda: datetime.now(timezone.utc))
    filled_at = Column(DateTime(timezone=True), nullable=True)


class PositionModel(Base):
    __tablename__ = "positions"
    id = Column(BigInteger, primary_key=True, autoincrement=True)
    position_id = Column(String(64), unique=True, nullable=True)
    ticker = Column(String(32), nullable=False, index=True)
    market = Column(String(16), nullable=False)
    strategy = Column(String(32), nullable=False)
    direction = Column(String(8), nullable=False)
    quantity = Column(Integer, nullable=False)
    avg_cost = Column(Float, nullable=False)
    current_price = Column(Float, nullable=True)
    stop_loss = Column(Float, nullable=True)
    target_price = Column(Float, nullable=True)
    margin_blocked = Column(Float, nullable=False, default=0.0)
    fees_paid = Column(Float, nullable=False, default=0.0)
    status = Column(String(16), nullable=False, default="OPEN", index=True)
    is_paper = Column(SmallInteger, nullable=False, default=1)
    opened_at = Column(DateTime(timezone=True), nullable=False, default=lambda: datetime.now(timezone.utc))
    closed_at = Column(DateTime(timezone=True), nullable=True)
    realized_pnl = Column(Float, default=0.0)
    return_pct = Column(Float, default=0.0)


class JournalEntryModel(Base):
    __tablename__ = "journal_entries"
    journal_id = Column(String(64), primary_key=True)
    trade_id = Column(String(64), nullable=True)
    symbol = Column(String(32), nullable=False, index=True)
    market = Column(String(16), nullable=False)
    strategy = Column(String(32), nullable=False)
    direction = Column(String(8), nullable=False)
    entry_timestamp = Column(DateTime(timezone=True), nullable=False)
    exit_timestamp = Column(DateTime(timezone=True), nullable=True)
    entry_price = Column(Float, nullable=False)
    exit_price = Column(Float, nullable=True)
    requested_quantity = Column(Integer, nullable=False)
    approved_quantity = Column(Integer, nullable=False)
    stop_loss = Column(Float, nullable=False)
    target_price = Column(Float, nullable=False)
    status = Column(String(24), nullable=False, default="PROPOSED", index=True)
    realized_pnl = Column(Float, nullable=True)
    return_pct = Column(Float, nullable=True)
    holding_period_seconds = Column(Float, nullable=True)
    agent_deliberation_logs = Column(Text, nullable=False, default="{}")
    market_snapshot = Column(Text, nullable=False, default="{}")
    confidence = Column(Float, nullable=False)
    evidence = Column(Text, nullable=True)
    final_thesis = Column(Text, nullable=False)
    risk_decision = Column(String(32), nullable=False)
    risk_reasons = Column(Text, nullable=True)
    qualitative_tags = Column(Text, nullable=False, default="[]")
    created_at = Column(DateTime(timezone=True), nullable=False, default=lambda: datetime.now(timezone.utc))
    updated_at = Column(DateTime(timezone=True), nullable=False, default=lambda: datetime.now(timezone.utc))


class TradeEvaluationModel(Base):
    __tablename__ = "trade_evaluations"
    evaluation_id = Column(String(64), primary_key=True)
    journal_id = Column(String(64), nullable=False, index=True)
    symbol = Column(String(32), nullable=False)
    market = Column(String(16), nullable=False)
    direction = Column(String(8), nullable=False)
    realized_pnl = Column(Float, nullable=False)
    return_pct = Column(Float, nullable=False)
    is_winner = Column(SmallInteger, nullable=False)
    holding_period_seconds = Column(Float, nullable=False)
    directional_accuracy = Column(Float, nullable=False)
    thesis_accuracy = Column(Float, nullable=False)
    thesis_notes = Column(Text, nullable=False)
    agent_accuracy = Column(Text, nullable=False, default="{}")
    risk_decision_evaluation = Column(Text, nullable=False)
    major_failure_reason = Column(String(64), nullable=False)
    failure_details = Column(Text, nullable=False)
    evaluated_at = Column(DateTime(timezone=True), nullable=False, default=lambda: datetime.now(timezone.utc))


class AccountSnapshotModel(Base):
    __tablename__ = "account_snapshots"
    id = Column(BigInteger, primary_key=True, autoincrement=True)
    timestamp = Column(DateTime(timezone=True), nullable=False, default=lambda: datetime.now(timezone.utc), index=True)
    market = Column(String(16), nullable=False, index=True)
    cash_balance = Column(Float, nullable=False)
    margin_utilized = Column(Float, nullable=False)
    portfolio_nav = Column(Float, nullable=False)
    peak_nav = Column(Float, nullable=False)
    drawdown_pct = Column(Float, nullable=False)
    created_at = Column(DateTime(timezone=True), nullable=False, default=lambda: datetime.now(timezone.utc))


class LedgerModel(Base):
    __tablename__ = "ledger"
    id = Column(BigInteger, primary_key=True, autoincrement=True)
    account_id = Column(String(32), nullable=False)
    entry_type = Column(String(32), nullable=False)
    amount = Column(Float, nullable=False)
    balance_after = Column(Float, nullable=False)
    ref_order_id = Column(String(64), nullable=True)
    ref_position_id = Column(BigInteger, nullable=True)
    description = Column(Text, nullable=True)
    created_at = Column(DateTime(timezone=True), nullable=False, default=lambda: datetime.now(timezone.utc))


class SignalModel(Base):
    __tablename__ = "signals"
    id = Column(BigInteger, primary_key=True, autoincrement=True)
    ticker = Column(String(32), nullable=False)
    market = Column(String(16), nullable=False)
    strategy = Column(String(32), nullable=False)
    direction = Column(String(8), nullable=False)
    entry_price = Column(Float, nullable=False)
    stop_loss = Column(Float, nullable=False)
    target_price = Column(Float, nullable=False)
    quantity = Column(Integer, nullable=False)
    confidence = Column(Float, nullable=False)
    reasoning = Column(Text, nullable=True)
    generated_at = Column(DateTime(timezone=True), nullable=False, default=lambda: datetime.now(timezone.utc))


class WatchlistModel(Base):
    __tablename__ = "watchlist"
    ticker = Column(String(32), primary_key=True)
    market = Column(String(16), primary_key=True)
    strategies = Column(Text, nullable=False)
    pinned = Column(SmallInteger, nullable=False, default=0)
    score = Column(Float, nullable=False, default=50.0)
    notes = Column(Text, nullable=True)
    source = Column(String(16), nullable=False, default="auto")
    added_at = Column(DateTime(timezone=True), nullable=False, default=lambda: datetime.now(timezone.utc))
    updated_at = Column(DateTime(timezone=True), nullable=False, default=lambda: datetime.now(timezone.utc))


class AdvisoryRecommendationModel(Base):
    __tablename__ = "advisory_recommendations"
    id = Column(BigInteger, primary_key=True, autoincrement=True)
    horizon = Column(String(16), nullable=False)
    market = Column(String(16), nullable=False)
    ticker = Column(String(32), nullable=False)
    action = Column(String(16), nullable=False)
    allocation_pct = Column(Float, nullable=False)
    conviction_score = Column(Float, nullable=False)
    thesis = Column(Text, nullable=False)
    risks = Column(Text, nullable=False)
    run_date = Column(String(16), nullable=False)
    created_at = Column(DateTime(timezone=True), nullable=False, default=lambda: datetime.now(timezone.utc))


class RiskAuditLogModel(Base):
    __tablename__ = "risk_audit_log"
    audit_id = Column(String(64), primary_key=True)
    timestamp = Column(DateTime(timezone=True), nullable=False, default=lambda: datetime.now(timezone.utc))
    ticker = Column(String(32), nullable=False)
    market = Column(String(16), nullable=False)
    strategy = Column(String(32), nullable=False)
    direction = Column(String(8), nullable=False)
    requested_qty = Column(Integer, nullable=False)
    approved_qty = Column(Integer, nullable=False)
    decision = Column(String(16), nullable=False)
    risk_score = Column(Float, nullable=False)
    max_loss = Column(Float, nullable=False)
    capital_required = Column(Float, nullable=False)
    checks = Column(Text, nullable=False, default="[]")
    violations = Column(Text, nullable=False, default="[]")
    warnings = Column(Text, nullable=False, default="[]")
    portfolio_state = Column(Text, nullable=False, default="{}")


class SystemStateModel(Base):
    __tablename__ = "system_state"
    key = Column(String(64), primary_key=True)
    value = Column(Text, nullable=False)
    updated_at = Column(DateTime(timezone=True), nullable=False, default=lambda: datetime.now(timezone.utc))


# ── Engine & Session Management ───────────────────────────────────────────────

_ASYNC_ENGINE: Optional[AsyncEngine] = None
_ASYNC_SESSION_MAKER: Optional[async_sessionmaker[AsyncSession]] = None


def get_async_engine() -> AsyncEngine:
    global _ASYNC_ENGINE
    if _ASYNC_ENGINE is None:
        db_url = _get_database_url()
        is_sqlite = db_url.startswith("sqlite")
        connect_args = {"check_same_thread": False} if is_sqlite else {}
        _ASYNC_ENGINE = create_async_engine(
            db_url,
            echo=False,
            future=True,
            connect_args=connect_args,
            pool_pre_ping=True,
            **({} if is_sqlite else {"pool_size": 10, "max_overflow": 20}),
        )
    return _ASYNC_ENGINE


def get_async_session_maker() -> async_sessionmaker[AsyncSession]:
    global _ASYNC_SESSION_MAKER
    if _ASYNC_SESSION_MAKER is None:
        engine = get_async_engine()
        _ASYNC_SESSION_MAKER = async_sessionmaker(
            engine,
            expire_on_commit=False,
            class_=AsyncSession,
        )
    return _ASYNC_SESSION_MAKER


@asynccontextmanager
async def get_async_session() -> AsyncGenerator[AsyncSession, None]:
    maker = get_async_session_maker()
    async with maker() as session:
        try:
            yield session
            await session.commit()
        except Exception:
            await session.rollback()
            raise


async def init_db() -> None:
    """Initialize all tables asynchronously."""
    engine = get_async_engine()
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    logger.info(f"[DB] Initialized database schema on {_get_database_url().split('@')[-1] if '@' in _get_database_url() else 'local store'}")
