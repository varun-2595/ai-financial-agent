"""
Local Model Provider for Apple Silicon (M5 Mac) Inference.

Connects to local inference runtimes (e.g. Ollama, LMStudio, llama.cpp, MLX)
exposing an OpenAI-compatible /v1 endpoint.
"""
from __future__ import annotations

import json
import time
from typing import Optional
import urllib.error
import urllib.request

from src.gateway.gateway import BaseModelProvider, ModelRequest, ModelResponse
from src.utils.config import get_config
from src.utils.logger import logger


class LocalModelProvider(BaseModelProvider):
    """
    Client for local LLM running on the user's M5 Mac.
    Treated as an optional inference acceleration node.
    """

    def __init__(
        self,
        endpoint: Optional[str] = None,
        model_name: Optional[str] = None,
        timeout_seconds: float = 3.0,
    ):
        cfg = get_config()
        self.endpoint = (endpoint or getattr(cfg.llm, "local_endpoint", "http://127.0.0.1:11434/v1")).rstrip("/")
        self.model = model_name or getattr(cfg.llm, "local_model", "aegis-10b-financial")
        self.timeout_seconds = timeout_seconds

    @property
    def provider_name(self) -> str:
        return "local"

    def is_available(self) -> bool:
        """
        Fast health-check probe to determine if the M5 local endpoint is active.
        If local_enabled is true, returns True (supports HTTP and embedded engine).
        """
        cfg = get_config()
        if not getattr(cfg.llm, "local_enabled", True):
            return False

        # If pointing to localhost / 127.0.0.1, embedded 10B engine is always available on Mac
        if "127.0.0.1" in self.endpoint or "localhost" in self.endpoint:
            return True

        url = f"{self.endpoint}/models"
        req = urllib.request.Request(url, headers={"User-Agent": "AegisGateway/1.0"})
        try:
            with urllib.request.urlopen(req, timeout=0.5) as response:
                return response.status == 200
        except Exception:
            return False

    def generate(self, request: ModelRequest) -> ModelResponse:
        """
        Send completion request to the local M5 model (via HTTP endpoint or embedded 10B engine).
        """
        t0 = time.perf_counter()
        url = f"{self.endpoint}/chat/completions"

        # Prepare system & user prompt
        system_content = request.system_prompt
        if request.response_schema is not None:
            schema_json = json.dumps(request.response_schema.model_json_schema())
            system_content += f"\nYou MUST respond strictly in valid JSON conforming to this JSON Schema:\n{schema_json}"

        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system_content},
                {"role": "user", "content": request.user_prompt},
            ],
            "temperature": request.temperature,
            "max_tokens": request.max_tokens,
            "response_format": {"type": "json_object"} if request.response_schema else None,
        }

        # 1. Try HTTP local server
        try:
            req_data = json.dumps(payload).encode("utf-8")
            req = urllib.request.Request(
                url,
                data=req_data,
                headers={"Content-Type": "application/json", "User-Agent": "AegisGateway/1.0"},
            )
            timeout = min(request.timeout_seconds, self.timeout_seconds)
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                raw_body = resp.read().decode("utf-8")

            latency = (time.perf_counter() - t0) * 1000.0
            data = json.loads(raw_body)
            content = data.get("choices", [{}])[0].get("message", {}).get("content", "")

        except Exception as http_exc:
            # 2. Seamless fallback to in-process 10B Financial Engine on Mac
            try:
                from src.gateway.local_server import ENGINE
                dict_output = ENGINE.generate_response(payload["messages"], self.model)
                content = json.dumps(dict_output)
                latency = (time.perf_counter() - t0) * 1000.0
            except Exception as eng_exc:
                latency = (time.perf_counter() - t0) * 1000.0
                return ModelResponse(
                    content="",
                    provider_used=self.provider_name,
                    model_name=self.model,
                    latency_ms=latency,
                    success=False,
                    error=f"Local M5 model error: {http_exc} / {eng_exc}",
                )

        # Parse structured output if schema requested
        structured_data = None
        if request.response_schema is not None and content:
            try:
                clean_content = content.strip()
                if clean_content.startswith("```"):
                    clean_content = clean_content.split("\n", 1)[-1].rsplit("```", 1)[0].strip()
                structured_data = request.response_schema.model_validate_json(clean_content)
            except Exception as parse_exc:
                logger.warning(f"[LocalProvider] Schema parsing failed: {parse_exc}")
                return ModelResponse(
                    content=content,
                    provider_used=self.provider_name,
                    model_name=self.model,
                    latency_ms=latency,
                    success=False,
                    error=f"Malformed schema output: {parse_exc}",
                )

        return ModelResponse(
            content=content,
            structured_data=structured_data,
            provider_used=self.provider_name,
            model_name=self.model,
            latency_ms=latency,
            success=True,
        )
