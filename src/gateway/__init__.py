"""
Aegis Model Gateway & Multi-Model Routing Package.

Provides a unified LLM abstraction layer supporting:
  - LocalModelProvider (M5 Mac local inference node)
  - GeminiProvider (Google GenAI remote inference)
  - Intelligent ModelRouter with policy-based routing and automatic fallback
"""
from __future__ import annotations

from src.gateway.gateway import (
    BaseModelProvider,
    ModelRequest,
    ModelResponse,
    TaskComplexity,
)
from src.gateway.gemini_provider import GeminiProvider
from src.gateway.health import check_gemini_available, check_local_model_available
from src.gateway.local_provider import LocalModelProvider
from src.gateway.router import ModelRouter, get_router

__all__ = [
    "BaseModelProvider",
    "GeminiProvider",
    "LocalModelProvider",
    "ModelRequest",
    "ModelResponse",
    "ModelRouter",
    "TaskComplexity",
    "check_gemini_available",
    "check_local_model_available",
    "get_router",
]
