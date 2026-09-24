"""
Model Gateway Health Check Diagnostics.

Monitors availability of local (M5 Mac) and remote (Gemini) inference nodes.
"""
from __future__ import annotations

import os
from typing import Any

from src.gateway.gemini_provider import GeminiProvider
from src.gateway.local_provider import LocalModelProvider


def check_local_model_available(endpoint: str | None = None) -> bool:
    """Return True if local M5 inference endpoint is reachable."""
    return LocalModelProvider(endpoint=endpoint).is_available()


def check_gemini_available(api_key: str | None = None) -> bool:
    """Return True if Gemini API credentials and client are operational."""
    return GeminiProvider(api_key=api_key).is_available()


def get_gateway_health() -> dict[str, Any]:
    """Inspect full gateway health state."""
    local_ok = check_local_model_available()
    gemini_ok = check_gemini_available()
    return {
        "LOCAL_MODEL_AVAILABLE": local_ok,
        "GEMINI_AVAILABLE": gemini_ok,
        "primary_inference_route": "local" if local_ok else ("gemini" if gemini_ok else "heuristic_fallback"),
        "m5_node_status": "ONLINE" if local_ok else "OFFLINE (Optional)",
        "gemini_status": "ONLINE" if gemini_ok else "OFFLINE",
    }
