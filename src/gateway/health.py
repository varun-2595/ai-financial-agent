"""
Model Gateway Health Check Diagnostics.

Monitors availability of local (M5 Mac) and remote (Gemini) inference nodes.
The local M5 node is strictly treated as an OPTIONAL acceleration node.
"""
from __future__ import annotations

import os
from typing import Any, Optional

from src.gateway.gemini_provider import GeminiProvider
from src.gateway.local_provider import LocalModelProvider


def check_local_model_available(
    endpoint: Optional[str] = None,
    model_name: Optional[str] = None,
) -> bool:
    """
    Return True ONLY if local M5 inference endpoint is actually reachable
    AND the configured model is available in the models list.
    """
    return LocalModelProvider(endpoint=endpoint, model_name=model_name).is_available()


def check_gemini_available(api_key: Optional[str] = None) -> bool:
    """Return True if Gemini API credentials and client are operational."""
    return GeminiProvider(api_key=api_key).is_available()


def get_gateway_health(
    local_provider: Optional[LocalModelProvider] = None,
    gemini_provider: Optional[GeminiProvider] = None,
) -> dict[str, Any]:
    """
    Inspect full gateway health state across local and remote inference nodes.
    """
    local_p = local_provider or LocalModelProvider()
    gemini_p = gemini_provider or GeminiProvider()

    local_ok = local_p.is_available()
    gemini_ok = gemini_p.is_available()

    if local_ok and gemini_ok:
        status_summary = "HEALTHY_DUAL_PROVIDER"
        primary_route = "local (simple/high-vol) & gemini (complex/PM)"
    elif gemini_ok and not local_ok:
        status_summary = "HEALTHY_GEMINI_ONLY"
        primary_route = "gemini (all tasks; M5 offline/optional)"
    elif local_ok and not gemini_ok:
        status_summary = "DEGRADED_LOCAL_ONLY"
        primary_route = "local M5 (all tasks; Gemini unconfigured)"
    else:
        status_summary = "ALL_PROVIDERS_OFFLINE"
        primary_route = "deterministic_fallback (HOLD / no-trade)"

    return {
        "LOCAL_MODEL_AVAILABLE": local_ok,
        "GEMINI_AVAILABLE": gemini_ok,
        "local_endpoint": getattr(local_p, "endpoint", "N/A"),
        "local_model": getattr(local_p, "model", "N/A"),
        "status_summary": status_summary,
        "primary_inference_route": primary_route,
        "m5_node_status": "ONLINE" if local_ok else "OFFLINE (Optional)",
        "gemini_status": "ONLINE" if gemini_ok else "OFFLINE",
    }
