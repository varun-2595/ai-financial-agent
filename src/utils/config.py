"""
Config loader for settings.yaml, watchlist.yaml, and advisory_universe.yaml.
Provides strongly typed access to global configuration across all modules.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, Field

ROOT = Path(__file__).parent.parent.parent  # src/utils/ -> src/ -> project_root
CONFIG_DIR = ROOT / "config"


class GeneralSettings(BaseModel):
    agent_name: str = "Aegis"
    version: str = "1.0.0"
    log_level: str = "INFO"


class LLMSettings(BaseModel):
    provider: str = "gemini"
    model: str = "gemini-3.8-flash"
    fallback_model: str = "gemini-3.6-flash"
    temperature: float = 0.2
    top_p: float = 0.8
    max_output_tokens: int = 4096


class PaperTradingSettings(BaseModel):
    enabled: bool = True
    virtual_capital_inr: float = 10_000.0
    virtual_capital_usd: float = 1_000.0
    daily_profit_target_inr: float = 1_000.0
    daily_profit_target_usd: float = 150.0
    daily_max_loss_inr: float = 500.0
    daily_max_loss_usd: float = 50.0
    intraday_leverage_multiplier: float = 3.0
    risk_per_trade_pct: float = 0.03
    max_position_pct: float = 0.40
    max_sector_pct: float = 0.50
    max_drawdown_pct: float = 0.15


class AdvisorySettings(BaseModel):
    enabled: bool = True
    virtual_capital_inr: float = 500_000.0
    virtual_capital_usd: float = 5_000.0
    short_term_pct: float = 0.25
    long_term_pct: float = 0.75
    max_single_stock_pct: float = 0.10
    max_single_etf_pct: float = 0.20
    max_sector_pct: float = 0.30
    min_instruments: int = 6


class AppConfig(BaseModel):
    general: GeneralSettings = Field(default_factory=GeneralSettings)
    llm: LLMSettings = Field(default_factory=LLMSettings)
    paper_trading: PaperTradingSettings = Field(default_factory=PaperTradingSettings)
    advisory: AdvisorySettings = Field(default_factory=AdvisorySettings)
    raw_settings: dict[str, Any] = Field(default_factory=dict)


_LOADED_CONFIG: AppConfig | None = None


def get_config(reload: bool = False) -> AppConfig:
    global _LOADED_CONFIG
    if _LOADED_CONFIG is not None and not reload:
        return _LOADED_CONFIG

    settings_path = CONFIG_DIR / "settings.yaml"
    raw = {}
    if settings_path.exists():
        with open(settings_path) as f:
            raw = yaml.safe_load(f) or {}

    # settings.yaml uses 'agent:' key; 'general:' kept for backward compat
    general_raw = raw.get("general", raw.get("agent", {}))

    llm_raw = raw.get("llm", {})
    paper_raw = raw.get("paper_trading", {})
    adv_raw = raw.get("advisory", {})
    adv_alloc = adv_raw.get("horizon_allocation", {})
    adv_div = adv_raw.get("diversification", {})

    config = AppConfig(
        general=GeneralSettings(**general_raw),
        llm=LLMSettings(**llm_raw),
        paper_trading=PaperTradingSettings(**paper_raw),
        advisory=AdvisorySettings(
            enabled=adv_raw.get("enabled", True),
            virtual_capital_inr=adv_raw.get("virtual_capital_inr", 500000.0),
            virtual_capital_usd=adv_raw.get("virtual_capital_usd", 5000.0),
            short_term_pct=adv_alloc.get("short_term_pct", 0.25),
            long_term_pct=adv_alloc.get("long_term_pct", 0.75),
            max_single_stock_pct=adv_div.get("max_single_stock_pct", 0.10),
            max_single_etf_pct=adv_div.get("max_single_etf_pct", 0.20),
            max_sector_pct=adv_div.get("max_sector_pct", 0.30),
            min_instruments=adv_div.get("min_instruments", 6),
        ),
        raw_settings=raw,
    )
    _LOADED_CONFIG = config
    return _LOADED_CONFIG
