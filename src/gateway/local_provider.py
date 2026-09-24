"""
Local Model Provider for Apple Silicon (M5 Mac) / External Inference Server.

Genuine OpenAI-compatible client for external local inference servers (e.g. Ollama, LM Studio, MLX, vLLM).
Treated strictly as an OPTIONAL inference acceleration node.

Rules:
1. Never assumes localhost or 127.0.0.1 is available.
2. Only reports available if the endpoint responds to HTTP GET /models AND the configured model exists in the returned list.
3. Strictly no fake reasoning engines or in-memory mock fallbacks.
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
    Genuine OpenAI-compatible client for an external local inference runtime
    running on the user's M5 Mac or local network (Ollama, MLX, LM Studio, vLLM).
    
    Strictly performs real HTTP probes against /v1/models.
    Never assumes localhost is online and never uses fake in-memory engines.
    """

    def __init__(
        self,
        endpoint: Optional[str] = None,
        model_name: Optional[str] = None,
        api_key: Optional[str] = None,
        timeout_seconds: float = 30.0,
        health_timeout_seconds: float = 1.0,
    ):
        cfg = get_config()
        raw_endpoint = endpoint or getattr(cfg.llm, "local_endpoint", "http://127.0.0.1:11434/v1")
        self.endpoint = raw_endpoint.rstrip("/")
        self.model = model_name or getattr(cfg.llm, "local_model", "qwen2.5:14b-instruct")
        self.api_key = api_key or getattr(cfg.llm, "local_api_key", None)
        self.timeout_seconds = timeout_seconds
        self.health_timeout_seconds = health_timeout_seconds

    @property
    def provider_name(self) -> str:
        return "local"

    def is_available(self) -> bool:
        """
        Probe the local inference server endpoint to verify:
        1. The HTTP endpoint is reachable.
        2. The requested model is actually loaded / available in the models list.
        
        Returns False if unreachable, timed out, or model is absent.
        """
        cfg = get_config()
        if not getattr(cfg.llm, "local_enabled", True):
            return False

        if not self.endpoint:
            return False

        url = f"{self.endpoint}/models"
        req = urllib.request.Request(url, headers={"User-Agent": "AegisGateway/1.0"})
        if self.api_key:
            req.add_header("Authorization", f"Bearer {self.api_key}")

        try:
            with urllib.request.urlopen(req, timeout=self.health_timeout_seconds) as response:
                if response.status != 200:
                    return False
                raw = response.read().decode("utf-8")
                data = json.loads(raw)

                # Extract model identifiers from OpenAI or Ollama schema
                models_list: list[str] = []
                if isinstance(data, dict):
                    if "data" in data and isinstance(data["data"], list):
                        for m in data["data"]:
                            if isinstance(m, dict):
                                m_id = m.get("id") or m.get("name")
                                if m_id:
                                    models_list.append(str(m_id).lower())
                    elif "models" in data and isinstance(data["models"], list):
                        for m in data["models"]:
                            if isinstance(m, dict):
                                m_id = m.get("name") or m.get("model") or m.get("id")
                                if m_id:
                                    models_list.append(str(m_id).lower())

                if not models_list:
                    return False

                # Exact or prefix match against requested model
                target = self.model.lower().strip()
                for m_id in models_list:
                    if target == m_id or target in m_id or m_id in target:
                        return True

                logger.debug(
                    f"[LocalModelProvider] Endpoint online at {self.endpoint}, but model '{self.model}' not found in {models_list}"
                )
                return False

        except Exception as exc:
            logger.debug(f"[LocalModelProvider] Health probe failed for {url}: {exc}")
            return False

    def generate(self, request: ModelRequest) -> ModelResponse:
        """
        Send completion request to genuine OpenAI-compatible /v1/chat/completions endpoint.
        No fake in-process fallback engines.
        """
        if not self.is_available():
            return ModelResponse(
                content="",
                provider_used=self.provider_name,
                model_name=self.model,
                success=False,
                error=f"Local M5 endpoint ({self.endpoint}) is offline or model '{self.model}' is unavailable.",
            )

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
        }
        if request.response_schema is not None:
            payload["response_format"] = {"type": "json_object"}

        try:
            req_data = json.dumps(payload).encode("utf-8")
            req = urllib.request.Request(
                url,
                data=req_data,
                headers={"Content-Type": "application/json", "User-Agent": "AegisGateway/1.0"},
            )
            if self.api_key:
                req.add_header("Authorization", f"Bearer {self.api_key}")

            timeout = min(request.timeout_seconds, self.timeout_seconds)
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                raw_body = resp.read().decode("utf-8")

            latency = (time.perf_counter() - t0) * 1000.0
            data = json.loads(raw_body)
            content = data.get("choices", [{}])[0].get("message", {}).get("content", "")

        except Exception as http_exc:
            latency = (time.perf_counter() - t0) * 1000.0
            logger.warning(f"[LocalModelProvider] Request failed to {url}: {http_exc}")
            return ModelResponse(
                content="",
                provider_used=self.provider_name,
                model_name=self.model,
                latency_ms=latency,
                success=False,
                error=f"Local M5 model error: {http_exc}",
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
                logger.warning(f"[LocalModelProvider] Schema parsing failed: {parse_exc}")
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
