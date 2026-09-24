"""
Intelligent Model Router for Aegis Multi-Agent Architecture.

Enforces configurable routing policies across M5 Local Node and Gemini Remote Provider:
  - simple tasks -> local model when available
  - high-volume tasks -> local model when available
  - complex financial reasoning -> Gemini (fallback to local if Gemini unavailable)
  - final portfolio decision -> Gemini initially (fallback to local if Gemini unavailable)

Ensures Aegis NEVER fails due to offline local hardware or remote rate limits.
"""
from __future__ import annotations

from typing import Optional

from src.gateway.gateway import (
    BaseModelProvider,
    ModelRequest,
    ModelResponse,
    TaskComplexity,
)
from src.gateway.gemini_provider import GeminiProvider
from src.gateway.local_provider import LocalModelProvider
from src.utils.logger import logger


class ModelRouter:
    """
    Directs model requests to the optimal inference provider based on task complexity
    and node availability.
    """

    def __init__(
        self,
        local_provider: Optional[BaseModelProvider] = None,
        gemini_provider: Optional[BaseModelProvider] = None,
    ):
        self.local = local_provider or LocalModelProvider()
        self.gemini = gemini_provider or GeminiProvider()

    def route(self, request: ModelRequest) -> ModelResponse:
        """
        Execute request according to policy with automated fallback.
        """
        complexity = request.task_complexity
        local_available = self.local.is_available()
        gemini_available = self.gemini.is_available()

        # ── 1. Policy Determination: Select Primary & Secondary ────────────
        if complexity in (TaskComplexity.SIMPLE, TaskComplexity.HIGH_VOLUME):
            # Prefer local M5 node for speed & zero-cost high volume
            if local_available:
                primary = self.local
                secondary = self.gemini if gemini_available else None
            else:
                primary = self.gemini if gemini_available else None
                secondary = None
        else:
            # COMPLEX or PORTFOLIO_DECISION: Prefer Gemini for deep reasoning
            if gemini_available:
                primary = self.gemini
                secondary = self.local if local_available else None
            else:
                primary = self.local if local_available else None
                secondary = None

        # ── 2. Primary Execution ───────────────────────────────────────────
        if primary is not None:
            logger.info(f"[ModelRouter] Routing {complexity.value} task to {primary.provider_name.upper()}...")
            resp = primary.generate(request)
            if resp.success:
                return resp

            logger.warning(
                f"[ModelRouter] Primary provider ({primary.provider_name}) failed: {resp.error}. "
                f"Attempting fallback..."
            )

        # ── 3. Secondary Fallback Execution ────────────────────────────────
        if secondary is not None:
            logger.info(f"[ModelRouter] Fallback routing to {secondary.provider_name.upper()}...")
            resp = secondary.generate(request)
            if resp.success:
                return resp

            logger.warning(f"[ModelRouter] Secondary provider ({secondary.provider_name}) failed: {resp.error}.")

        # ── 4. Graceful Offline / Fallback Response ────────────────────────
        logger.warning("[ModelRouter] All LLM providers unavailable or failed. Returning heuristic fallback state.")
        return ModelResponse(
            content="",
            structured_data=None,
            provider_used="heuristic_fallback",
            model_name="deterministic_heuristic",
            success=False,
            error="All LLM providers unavailable (M5 local offline & Gemini unreachable).",
        )


_ROUTER_SINGLETON: Optional[ModelRouter] = None


def get_router() -> ModelRouter:
    """Return global singleton ModelRouter instance."""
    global _ROUTER_SINGLETON
    if _ROUTER_SINGLETON is None:
        _ROUTER_SINGLETON = ModelRouter()
    return _ROUTER_SINGLETON
