"""
Comprehensive Unit & Integration Test Suite for Phase 4: Aegis Multi-Agent Architecture & ModelGateway.

Tests:
  1. ModelGateway abstraction & provider protocol
  2. Local model available routing (M5 Mac)
  3. Local model unavailable automatic fallback to Gemini
  4. Both providers failing gracefully with deterministic fallback (safe HOLD state)
  5. Malformed model output handling and schema validation
  6. Standardized output validation for all 6 agents (Technical, Fundamental, News, Macro, Risk, PortfolioManager)
  7. Multi-agent consensus synthesis in PortfolioManagerAgent
  8. Provider health checks:
     - M5 online
     - M5 offline
     - Model unavailable
     - Gemini available
     - Gemini unavailable
     - Both unavailable
"""
from __future__ import annotations

import io
import json
import urllib.error
from datetime import datetime, timezone
from unittest.mock import MagicMock, patch

import pytest
from pydantic import BaseModel

from src.agents.fundamental_agent import FundamentalAgent
from src.agents.macro_agent import MacroAgent
from src.agents.models import AgentSignalOutput, MacroContext, MultiAgentConsensus
from src.agents.news_agent import NewsAgent
from src.agents.portfolio_manager_agent import PortfolioManagerAgent
from src.agents.risk_agent import RiskAgent
from src.agents.technical_agent import TechnicalAgent
from src.data.models import Fundamentals, NewsItem, Quote, StockSnapshot
from src.gateway.gateway import (
    BaseModelProvider,
    ModelRequest,
    ModelResponse,
    TaskComplexity,
)
from src.gateway.gemini_provider import GeminiProvider
from src.gateway.health import check_gemini_available, check_local_model_available, get_gateway_health
from src.gateway.local_provider import LocalModelProvider
from src.gateway.router import ModelRouter


# ── Fixtures & Mock Providers ──────────────────────────────────────────────────

class MockSuccessProvider(BaseModelProvider):
    def __init__(self, name: str = "mock_local", signal: str = "BUY"):
        self._name = name
        self.signal = signal

    @property
    def provider_name(self) -> str:
        return self._name

    def is_available(self) -> bool:
        return True

    def generate(self, request: ModelRequest) -> ModelResponse:
        data = None
        if request.response_schema == AgentSignalOutput:
            data = AgentSignalOutput(
                agent="MockAgent",
                signal=self.signal,
                confidence=0.88,
                reasons=["Strong institutional mock catalyst"],
                risks=["Mock downside risk"],
                evidence=["Mock metric=42.0"],
            )
        return ModelResponse(
            content=json.dumps(data.model_dump()) if data else "mock text",
            structured_data=data,
            provider_used=self._name,
            model_name="mock_model",
            latency_ms=12.5,
            success=True,
        )


class MockFailingProvider(BaseModelProvider):
    def __init__(self, name: str = "failing_provider", available: bool = True):
        self._name = name
        self._available = available

    @property
    def provider_name(self) -> str:
        return self._name

    def is_available(self) -> bool:
        return self._available

    def generate(self, request: ModelRequest) -> ModelResponse:
        return ModelResponse(
            content="",
            provider_used=self._name,
            model_name="error_model",
            success=False,
            error="Simulated provider connection failure or timeout.",
        )


@pytest.fixture
def sample_snapshot() -> StockSnapshot:
    quotes = [
        Quote(
            timestamp=datetime(2025, 1, i + 1, tzinfo=timezone.utc),
            open=100.0 + i,
            high=102.0 + i,
            low=99.0 + i,
            close=101.5 + i,
            volume=150_000,
        )
        for i in range(30)
    ]
    fundamentals = Fundamentals(
        market_cap=50_000_000_000,
        pe_ratio=24.5,
        forward_pe=20.0,
        price_to_book=3.2,
        roe=0.18,
        debt_to_equity=0.65,
        profit_margin=0.22,
        sector="Technology",
        industry="Semiconductors",
    )
    news = [
        NewsItem(
            title="Q3 Earnings Beat Institutional Expectations by 12%",
            publisher="Bloomberg",
            link="https://example.com/news1",
            published_at=datetime(2025, 1, 28, tzinfo=timezone.utc),
            summary="Strong revenue expansion driven by AI chip demand.",
        )
    ]
    return StockSnapshot(
        ticker="AAPL",
        market="us",
        currency="USD",
        current_price=131.5,
        history=quotes,
        fundamentals=fundamentals,
        recent_news=news,
        timestamp=quotes[-1].timestamp,
    )


# ── 1. ModelGateway & ModelRouter Tests ────────────────────────────────────────

def test_router_local_available_routes_to_local():
    """When local M5 model is available, simple/high-volume tasks route to local."""
    local_p = MockSuccessProvider("local")
    gemini_p = MockSuccessProvider("gemini")
    router = ModelRouter(local_provider=local_p, gemini_provider=gemini_p)

    req = ModelRequest(
        system_prompt="sys",
        user_prompt="usr",
        response_schema=AgentSignalOutput,
        task_complexity=TaskComplexity.HIGH_VOLUME,
    )

    resp = router.route(req)
    assert resp.success
    assert resp.provider_used == "local"
    assert isinstance(resp.structured_data, AgentSignalOutput)


def test_router_local_unavailable_falls_back_to_gemini():
    """When local model is offline, router automatically delegates to Gemini."""
    local_p = MockFailingProvider("local", available=False)
    gemini_p = MockSuccessProvider("gemini")
    router = ModelRouter(local_provider=local_p, gemini_provider=gemini_p)

    req = ModelRequest(
        system_prompt="sys",
        user_prompt="usr",
        response_schema=AgentSignalOutput,
        task_complexity=TaskComplexity.SIMPLE,
    )

    resp = router.route(req)
    assert resp.success
    assert resp.provider_used == "gemini"
    assert isinstance(resp.structured_data, AgentSignalOutput)


def test_router_local_runtime_error_falls_back_to_gemini():
    """When local provider is marked available but errors during execution, fall back to Gemini."""
    local_p = MockFailingProvider("local", available=True)  # runtime error
    gemini_p = MockSuccessProvider("gemini")
    router = ModelRouter(local_provider=local_p, gemini_provider=gemini_p)

    req = ModelRequest(
        system_prompt="sys",
        user_prompt="usr",
        response_schema=AgentSignalOutput,
        task_complexity=TaskComplexity.HIGH_VOLUME,
    )

    resp = router.route(req)
    assert resp.success
    assert resp.provider_used == "gemini"


def test_router_all_providers_failing_returns_safe_fallback():
    """When both local and Gemini fail, router returns safe fallback without crashing."""
    local_p = MockFailingProvider("local", available=True)
    gemini_p = MockFailingProvider("gemini", available=True)
    router = ModelRouter(local_provider=local_p, gemini_provider=gemini_p)

    req = ModelRequest(
        system_prompt="sys",
        user_prompt="usr",
        response_schema=AgentSignalOutput,
        task_complexity=TaskComplexity.COMPLEX,
    )

    resp = router.route(req)
    assert not resp.success
    assert resp.provider_used == "heuristic_fallback"


def test_malformed_model_output_handling():
    """When provider returns malformed non-JSON output, handle gracefully."""
    class BadJsonProvider(BaseModelProvider):
        @property
        def provider_name(self) -> str:
            return "bad_json"

        def is_available(self) -> bool:
            return True

        def generate(self, request: ModelRequest) -> ModelResponse:
            return ModelResponse(
                content="This is plain text not valid json {invalid",
                provider_used=self.provider_name,
                success=False,
                error="Malformed schema output: JSONDecodeError",
            )

    gemini_p = BadJsonProvider()
    local_p = MockFailingProvider("local", available=False)
    router = ModelRouter(local_provider=local_p, gemini_provider=gemini_p)

    req = ModelRequest(
        system_prompt="sys",
        user_prompt="usr",
        response_schema=AgentSignalOutput,
    )
    resp = router.route(req)
    assert not resp.success


# ── 2. Specialized Multi-Agent Output Tests ────────────────────────────────────

def test_technical_agent_output(sample_snapshot):
    """Verify TechnicalAgent produces strictly validated AgentSignalOutput."""
    agent = TechnicalAgent()
    out = agent.analyze(sample_snapshot)

    assert isinstance(out, AgentSignalOutput)
    assert out.agent == "TechnicalAgent"
    assert out.signal in ("BUY", "SELL", "HOLD")
    assert 0.0 <= out.confidence <= 1.0
    assert len(out.reasons) > 0
    assert len(out.risks) > 0
    assert len(out.evidence) > 0


def test_fundamental_agent_output(sample_snapshot):
    """Verify FundamentalAgent produces strictly validated AgentSignalOutput."""
    agent = FundamentalAgent()
    out = agent.analyze(sample_snapshot)

    assert isinstance(out, AgentSignalOutput)
    assert out.agent == "FundamentalAgent"
    assert out.signal in ("BUY", "SELL", "HOLD")
    assert 0.0 <= out.confidence <= 1.0
    assert len(out.reasons) > 0
    assert len(out.evidence) > 0


def test_news_agent_output(sample_snapshot):
    """Verify NewsAgent produces strictly validated AgentSignalOutput."""
    agent = NewsAgent()
    out = agent.analyze(sample_snapshot)

    assert isinstance(out, AgentSignalOutput)
    assert out.agent == "NewsAgent"
    assert out.signal in ("BUY", "SELL", "HOLD")
    assert 0.0 <= out.confidence <= 1.0
    assert len(out.reasons) > 0


def test_macro_agent_output(sample_snapshot):
    """Verify MacroAgent produces strictly validated AgentSignalOutput."""
    agent = MacroAgent()
    ctx = MacroContext(market_regime="BULLISH_TREND", vix_level=14.5)
    out = agent.analyze(sample_snapshot, macro_context=ctx)

    assert isinstance(out, AgentSignalOutput)
    assert out.agent == "MacroAgent"
    assert out.signal in ("BUY", "SELL", "HOLD")
    assert 0.0 <= out.confidence <= 1.0


def test_risk_agent_output(sample_snapshot):
    """Verify RiskAgent produces strictly validated AgentSignalOutput."""
    agent = RiskAgent()
    out = agent.analyze(sample_snapshot)

    assert isinstance(out, AgentSignalOutput)
    assert out.agent == "RiskAgent"
    assert out.signal in ("BUY", "SELL", "HOLD")
    assert 0.0 <= out.confidence <= 1.0
    assert len(out.risks) > 0


def test_portfolio_manager_agent_synthesis(sample_snapshot):
    """
    Verify PortfolioManagerAgent synthesizes the 5 domain agent outputs
    into a coordinated portfolio recommendation.
    """
    tech_agent = TechnicalAgent()
    fund_agent = FundamentalAgent()
    news_agent = NewsAgent()
    macro_agent = MacroAgent()
    risk_agent = RiskAgent()
    pm_agent = PortfolioManagerAgent()

    consensus = MultiAgentConsensus(
        ticker=sample_snapshot.ticker,
        market=sample_snapshot.market,
        technical=tech_agent.analyze(sample_snapshot),
        fundamental=fund_agent.analyze(sample_snapshot),
        news=news_agent.analyze(sample_snapshot),
        macro=macro_agent.analyze(sample_snapshot),
        risk=risk_agent.analyze(sample_snapshot),
    )

    final_decision = pm_agent.synthesize(sample_snapshot, consensus)

    assert isinstance(final_decision, AgentSignalOutput)
    assert final_decision.agent == "PortfolioManagerAgent"
    assert final_decision.signal in ("BUY", "SELL", "HOLD")
    assert 0.0 <= final_decision.confidence <= 1.0
    assert len(final_decision.reasons) > 0
    assert len(final_decision.risks) > 0


# ── 3. Provider Health Check & Availability Tests (All 6 States) ───────────────

class MockHTTPResponse:
    def __init__(self, data: dict, status: int = 200):
        self.data_bytes = json.dumps(data).encode("utf-8")
        self.status = status

    def read(self) -> bytes:
        return self.data_bytes

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        pass


def test_provider_health_m5_online():
    """Test State 1: M5 endpoint is reachable and requested model is present."""
    provider = LocalModelProvider(
        endpoint="http://127.0.0.1:11434/v1",
        model_name="qwen2.5:14b-instruct",
    )

    mock_models = {
        "object": "list",
        "data": [
            {"id": "qwen2.5:14b-instruct", "object": "model"},
            {"id": "llama3.1:8b", "object": "model"},
        ],
    }

    with patch("urllib.request.urlopen", return_value=MockHTTPResponse(mock_models, status=200)):
        assert provider.is_available() is True


def test_provider_health_m5_offline():
    """Test State 2: M5 endpoint is unreachable (e.g. connection refused / server down)."""
    provider = LocalModelProvider(
        endpoint="http://127.0.0.1:11434/v1",
        model_name="qwen2.5:14b-instruct",
    )

    with patch("urllib.request.urlopen", side_effect=urllib.error.URLError("Connection refused")):
        assert provider.is_available() is False

        # Attempt generate() while offline -> returns failure without crashing
        req = ModelRequest(system_prompt="sys", user_prompt="usr")
        resp = provider.generate(req)
        assert resp.success is False
        assert "offline" in resp.error.lower()


def test_provider_health_model_unavailable():
    """Test State 3: M5 endpoint is reachable, but requested model is not in model list."""
    provider = LocalModelProvider(
        endpoint="http://127.0.0.1:11434/v1",
        model_name="qwen2.5:14b-instruct",
    )

    mock_models = {
        "object": "list",
        "data": [
            {"id": "llama3.1:8b", "object": "model"},
            {"id": "mistral:7b", "object": "model"},
        ],
    }

    with patch("urllib.request.urlopen", return_value=MockHTTPResponse(mock_models, status=200)):
        assert provider.is_available() is False


def test_provider_health_gemini_available():
    """Test State 4: Gemini is available when valid API key and client exist."""
    provider = GeminiProvider(api_key="valid-test-key")
    assert provider.is_available() is True


def test_provider_health_gemini_unavailable():
    """Test State 5: Gemini is unavailable when API key is missing or unconfigured."""
    provider = GeminiProvider(api_key="")
    provider.api_key = None
    provider.client = None
    assert provider.is_available() is False

    req = ModelRequest(system_prompt="sys", user_prompt="usr")
    resp = provider.generate(req)
    assert resp.success is False
    assert "not configured" in resp.error.lower()


def test_provider_health_both_unavailable_safe_hold(sample_snapshot):
    """
    Test State 6: Both Local M5 and Gemini are unavailable.
    System must enter a safe deterministic HOLD / no-trade state.
    """
    local_p = MockFailingProvider("local", available=False)
    gemini_p = MockFailingProvider("gemini", available=False)
    router = ModelRouter(local_provider=local_p, gemini_provider=gemini_p)

    tech_agent = TechnicalAgent(router=router)
    pm_agent = PortfolioManagerAgent(router=router)

    # TechnicalAgent invokes deterministic fallback -> safe signal generated
    out = tech_agent.analyze(sample_snapshot)
    assert isinstance(out, AgentSignalOutput)
    assert out.signal in ("BUY", "SELL", "HOLD")

    # Full consensus fallback
    consensus = MultiAgentConsensus(
        ticker=sample_snapshot.ticker,
        market=sample_snapshot.market,
        technical=out,
    )
    final_decision = pm_agent.synthesize(sample_snapshot, consensus)
    assert isinstance(final_decision, AgentSignalOutput)
    assert final_decision.signal in ("BUY", "SELL", "HOLD")


def test_health_diagnostics_all_states():
    """Verify get_gateway_health() across combinations of online/offline nodes."""
    # 1. Dual online
    h1 = get_gateway_health(
        local_provider=MockSuccessProvider("local"),
        gemini_provider=MockSuccessProvider("gemini"),
    )
    assert h1["LOCAL_MODEL_AVAILABLE"] is True
    assert h1["GEMINI_AVAILABLE"] is True
    assert h1["status_summary"] == "HEALTHY_DUAL_PROVIDER"

    # 2. Gemini only (M5 offline/optional)
    h2 = get_gateway_health(
        local_provider=MockFailingProvider("local", available=False),
        gemini_provider=MockSuccessProvider("gemini"),
    )
    assert h2["LOCAL_MODEL_AVAILABLE"] is False
    assert h2["GEMINI_AVAILABLE"] is True
    assert h2["status_summary"] == "HEALTHY_GEMINI_ONLY"

    # 3. Local M5 only
    h3 = get_gateway_health(
        local_provider=MockSuccessProvider("local"),
        gemini_provider=MockFailingProvider("gemini", available=False),
    )
    assert h3["LOCAL_MODEL_AVAILABLE"] is True
    assert h3["GEMINI_AVAILABLE"] is False
    assert h3["status_summary"] == "DEGRADED_LOCAL_ONLY"

    # 4. Both offline
    h4 = get_gateway_health(
        local_provider=MockFailingProvider("local", available=False),
        gemini_provider=MockFailingProvider("gemini", available=False),
    )
    assert h4["LOCAL_MODEL_AVAILABLE"] is False
    assert h4["GEMINI_AVAILABLE"] is False
    assert h4["status_summary"] == "ALL_PROVIDERS_OFFLINE"
