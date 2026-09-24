-- =============================================================================
-- AEGIS TRADING AGENT: Unified PostgreSQL Relational Schema
-- =============================================================================

CREATE TABLE IF NOT EXISTS accounts (
    account_id      VARCHAR(32) PRIMARY KEY,
    currency        VARCHAR(8) NOT NULL,
    cash            NUMERIC(18, 4) NOT NULL,
    initial_cash    NUMERIC(18, 4) NOT NULL,
    reserved_margin NUMERIC(18, 4) NOT NULL DEFAULT 0.0,
    peak_nav        NUMERIC(18, 4) NOT NULL DEFAULT 0.0,
    updated_at      TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS orders (
    order_id         VARCHAR(64) PRIMARY KEY,
    broker_order_ref VARCHAR(64),
    ticker           VARCHAR(32) NOT NULL,
    market           VARCHAR(16) NOT NULL,
    exchange         VARCHAR(16) NOT NULL DEFAULT 'NSE',
    strategy         VARCHAR(32) NOT NULL,
    order_type       VARCHAR(16) NOT NULL,
    direction        VARCHAR(8) NOT NULL,
    quantity         INTEGER NOT NULL,
    limit_price      NUMERIC(18, 4),
    trigger_price    NUMERIC(18, 4),
    requested_price  NUMERIC(18, 4),
    filled_price     NUMERIC(18, 4),
    executed_price   NUMERIC(18, 4),
    fees             NUMERIC(18, 4) NOT NULL DEFAULT 0.0,
    statutory_fees   NUMERIC(18, 4) NOT NULL DEFAULT 0.0,
    slippage         NUMERIC(18, 4) NOT NULL DEFAULT 0.0,
    slippage_pct     NUMERIC(10, 6) NOT NULL DEFAULT 0.0005,
    status           VARCHAR(24) NOT NULL,
    is_paper         SMALLINT NOT NULL DEFAULT 1,
    created_at       TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
    filled_at        TIMESTAMPTZ
);

CREATE TABLE IF NOT EXISTS positions (
    id              BIGSERIAL PRIMARY KEY,
    position_id     VARCHAR(64) UNIQUE,
    ticker          VARCHAR(32) NOT NULL,
    market          VARCHAR(16) NOT NULL,
    strategy        VARCHAR(32) NOT NULL,
    direction       VARCHAR(8) NOT NULL,
    quantity        INTEGER NOT NULL,
    avg_cost        NUMERIC(18, 4) NOT NULL,
    current_price   NUMERIC(18, 4),
    stop_loss       NUMERIC(18, 4),
    target_price    NUMERIC(18, 4),
    margin_blocked  NUMERIC(18, 4) NOT NULL DEFAULT 0.0,
    fees_paid       NUMERIC(18, 4) NOT NULL DEFAULT 0.0,
    status          VARCHAR(16) NOT NULL DEFAULT 'OPEN',
    is_paper        SMALLINT NOT NULL DEFAULT 1,
    opened_at       TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
    closed_at       TIMESTAMPTZ,
    realized_pnl    NUMERIC(18, 4) DEFAULT 0.0,
    return_pct      NUMERIC(10, 4) DEFAULT 0.0
);

CREATE TABLE IF NOT EXISTS journal_entries (
    journal_id             VARCHAR(64) PRIMARY KEY,
    trade_id               VARCHAR(64),
    symbol                 VARCHAR(32) NOT NULL,
    market                 VARCHAR(16) NOT NULL,
    strategy               VARCHAR(32) NOT NULL,
    direction              VARCHAR(8) NOT NULL,
    entry_timestamp        TIMESTAMPTZ NOT NULL,
    exit_timestamp         TIMESTAMPTZ,
    entry_price            NUMERIC(18, 4) NOT NULL,
    exit_price             NUMERIC(18, 4),
    requested_quantity     INTEGER NOT NULL,
    approved_quantity      INTEGER NOT NULL,
    stop_loss              NUMERIC(18, 4) NOT NULL,
    target_price           NUMERIC(18, 4) NOT NULL,
    status                 VARCHAR(24) NOT NULL DEFAULT 'PROPOSED',
    realized_pnl           NUMERIC(18, 4),
    return_pct             NUMERIC(10, 4),
    holding_period_seconds NUMERIC(14, 2),
    agent_deliberation_logs JSONB NOT NULL DEFAULT '{}'::jsonb,
    market_snapshot        JSONB NOT NULL DEFAULT '{}'::jsonb,
    confidence             NUMERIC(6, 4) NOT NULL,
    evidence               TEXT,
    final_thesis           TEXT NOT NULL,
    risk_decision          VARCHAR(32) NOT NULL,
    risk_reasons           TEXT,
    qualitative_tags       JSONB NOT NULL DEFAULT '[]'::jsonb,
    created_at             TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at             TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS trade_evaluations (
    evaluation_id            VARCHAR(64) PRIMARY KEY,
    journal_id               VARCHAR(64) NOT NULL REFERENCES journal_entries(journal_id),
    symbol                   VARCHAR(32) NOT NULL,
    market                   VARCHAR(16) NOT NULL,
    direction                VARCHAR(8) NOT NULL,
    realized_pnl             NUMERIC(18, 4) NOT NULL,
    return_pct               NUMERIC(10, 4) NOT NULL,
    is_winner                SMALLINT NOT NULL,
    holding_period_seconds   NUMERIC(14, 2) NOT NULL,
    directional_accuracy     NUMERIC(6, 4) NOT NULL,
    thesis_accuracy          NUMERIC(6, 4) NOT NULL,
    thesis_notes             TEXT NOT NULL,
    agent_accuracy           JSONB NOT NULL DEFAULT '{}'::jsonb,
    risk_decision_evaluation TEXT NOT NULL,
    major_failure_reason     VARCHAR(64) NOT NULL,
    failure_details          TEXT NOT NULL,
    evaluated_at             TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS account_snapshots (
    id              BIGSERIAL PRIMARY KEY,
    timestamp       TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
    market          VARCHAR(16) NOT NULL,
    cash_balance    NUMERIC(18, 4) NOT NULL,
    margin_utilized NUMERIC(18, 4) NOT NULL,
    portfolio_nav   NUMERIC(18, 4) NOT NULL,
    peak_nav        NUMERIC(18, 4) NOT NULL,
    drawdown_pct    NUMERIC(10, 4) NOT NULL,
    created_at      TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS ledger (
    id              BIGSERIAL PRIMARY KEY,
    account_id      VARCHAR(32) NOT NULL,
    entry_type      VARCHAR(32) NOT NULL,
    amount          NUMERIC(18, 4) NOT NULL,
    balance_after   NUMERIC(18, 4) NOT NULL,
    ref_order_id    VARCHAR(64),
    ref_position_id BIGINT,
    description     TEXT,
    created_at      TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS signals (
    id           BIGSERIAL PRIMARY KEY,
    ticker       VARCHAR(32) NOT NULL,
    market       VARCHAR(16) NOT NULL,
    strategy     VARCHAR(32) NOT NULL,
    direction    VARCHAR(8) NOT NULL,
    entry_price  NUMERIC(18, 4) NOT NULL,
    stop_loss    NUMERIC(18, 4) NOT NULL,
    target_price NUMERIC(18, 4) NOT NULL,
    quantity     INTEGER NOT NULL,
    confidence   NUMERIC(6, 4) NOT NULL,
    reasoning    TEXT,
    generated_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS watchlist (
    ticker     VARCHAR(32) NOT NULL,
    market     VARCHAR(16) NOT NULL,
    strategies TEXT NOT NULL,
    pinned     SMALLINT NOT NULL DEFAULT 0,
    score      NUMERIC(8, 2) NOT NULL DEFAULT 50.0,
    notes      TEXT,
    source     VARCHAR(16) NOT NULL DEFAULT 'auto',
    added_at   TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (ticker, market)
);

CREATE TABLE IF NOT EXISTS advisory_recommendations (
    id               BIGSERIAL PRIMARY KEY,
    horizon          VARCHAR(16) NOT NULL,
    market           VARCHAR(16) NOT NULL,
    ticker           VARCHAR(32) NOT NULL,
    action           VARCHAR(16) NOT NULL,
    allocation_pct   NUMERIC(6, 2) NOT NULL,
    conviction_score NUMERIC(6, 2) NOT NULL,
    thesis           TEXT NOT NULL,
    risks            TEXT NOT NULL,
    run_date         VARCHAR(16) NOT NULL,
    created_at       TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS risk_audit_log (
    audit_id         VARCHAR(64) PRIMARY KEY,
    timestamp        TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
    ticker           VARCHAR(32) NOT NULL,
    market           VARCHAR(16) NOT NULL,
    strategy         VARCHAR(32) NOT NULL,
    direction        VARCHAR(8) NOT NULL,
    requested_qty    INTEGER NOT NULL,
    approved_qty     INTEGER NOT NULL,
    decision         VARCHAR(16) NOT NULL,
    risk_score       NUMERIC(6, 2) NOT NULL,
    max_loss         NUMERIC(18, 4) NOT NULL,
    capital_required NUMERIC(18, 4) NOT NULL,
    checks           JSONB NOT NULL DEFAULT '[]'::jsonb,
    violations       JSONB NOT NULL DEFAULT '[]'::jsonb,
    warnings         JSONB NOT NULL DEFAULT '[]'::jsonb,
    portfolio_state  JSONB NOT NULL DEFAULT '{}'::jsonb
);

CREATE TABLE IF NOT EXISTS system_state (
    key        VARCHAR(64) PRIMARY KEY,
    value      TEXT NOT NULL,
    updated_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP
);

-- Indices for performance
CREATE INDEX IF NOT EXISTS idx_orders_ticker ON orders(ticker);
CREATE INDEX IF NOT EXISTS idx_orders_status ON orders(status);
CREATE INDEX IF NOT EXISTS idx_positions_status ON positions(status);
CREATE INDEX IF NOT EXISTS idx_positions_ticker ON positions(ticker);
CREATE INDEX IF NOT EXISTS idx_journal_symbol ON journal_entries(symbol);
CREATE INDEX IF NOT EXISTS idx_journal_status ON journal_entries(status);
CREATE INDEX IF NOT EXISTS idx_evaluations_journal ON trade_evaluations(journal_id);
CREATE INDEX IF NOT EXISTS idx_account_snapshots_market ON account_snapshots(market, timestamp);
