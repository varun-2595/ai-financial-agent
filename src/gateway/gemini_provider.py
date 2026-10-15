"""
Gemini Provider for Google GenAI Remote Inference.

Provides structured response generation, automated retries, and schema validation.
"""
from __future__ import annotations

import os
import time
from typing import Optional

from google import genai
from google.genai import types

from src.gateway.gateway import BaseModelProvider, ModelRequest, ModelResponse
from src.utils.config import get_config
from src.utils.logger import logger


class GeminiProvider(BaseModelProvider):
    """
    Remote inference provider backed by Google Gemini models.
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        model_name: Optional[str] = None,
    ):
        raw_key = api_key or os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
        self.api_key = raw_key.strip("'\" \t\r\n") if raw_key else None
        cfg = get_config()
        self.model = model_name or cfg.llm.model
        self.fallback_model = cfg.llm.fallback_model
        self.client = genai.Client(api_key=self.api_key) if self.api_key else None

    @property
    def provider_name(self) -> str:
        return "gemini"

    def is_available(self) -> bool:
        """Gemini is available if an API key is present."""
        return bool(self.api_key and self.client is not None)

    def validate_connection(self) -> tuple[bool, str]:
        """
        Performs a lightweight 1-token health check probe to verify API key validity.
        """
        if not self.api_key:
            return False, "GEMINI_API_KEY environment variable is not configured."
        try:
            client = genai.Client(api_key=self.api_key)
            resp = client.models.generate_content(
                model=self.model,
                contents="ping",
                config=types.GenerateContentConfig(max_output_tokens=2),
            )
            return True, "Gemini API connection healthy."
        except Exception as e:
            err_str = str(e)
            if "API_KEY_INVALID" in err_str or "API key not valid" in err_str:
                return False, "API_KEY_INVALID: Configured Gemini API key is rejected by Google API."
            return False, f"Gemini probe error: {err_str}"

    def generate(self, request: ModelRequest) -> ModelResponse:
        """
        Execute completion with Gemini using structured schema or text output.
        """
        if not self.is_available():
            return ModelResponse(
                content="",
                provider_used=self.provider_name,
                model_name=self.model,
                success=False,
                error="Gemini API key is not configured.",
            )

        t0 = time.perf_counter()
        config_args = {
            "system_instruction": request.system_prompt,
            "temperature": request.temperature,
        }

        if request.response_schema is not None:
            config_args["response_mime_type"] = "application/json"
            config_args["response_schema"] = request.response_schema

        gen_config = types.GenerateContentConfig(**config_args)

        try:
            response = self.client.models.generate_content(
                model=self.model,
                contents=request.user_prompt,
                config=gen_config,
            )
            latency = (time.perf_counter() - t0) * 1000.0
            content = response.text or ""

            structured_data = None
            if request.response_schema is not None and content:
                try:
                    structured_data = request.response_schema.model_validate_json(content)
                except Exception as parse_exc:
                    logger.warning(f"[GeminiProvider] Schema parse error: {parse_exc}")
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

        except Exception as exc:
            # Attempt fallback model if configured
            if self.fallback_model and self.fallback_model != self.model:
                try:
                    logger.info(f"[GeminiProvider] Primary model failed, trying fallback {self.fallback_model}...")
                    response = self.client.models.generate_content(
                        model=self.fallback_model,
                        contents=request.user_prompt,
                        config=gen_config,
                    )
                    latency = (time.perf_counter() - t0) * 1000.0
                    content = response.text or ""
                    structured_data = request.response_schema.model_validate_json(content) if (request.response_schema and content) else None
                    return ModelResponse(
                        content=content,
                        structured_data=structured_data,
                        provider_used=self.provider_name,
                        model_name=self.fallback_model,
                        latency_ms=latency,
                        success=True,
                    )
                except Exception as fb_exc:
                    latency = (time.perf_counter() - t0) * 1000.0
                    return ModelResponse(
                        content="",
                        provider_used=self.provider_name,
                        model_name=self.model,
                        latency_ms=latency,
                        success=False,
                        error=f"Gemini error (primary: {exc}, fallback: {fb_exc})",
                    )

            latency = (time.perf_counter() - t0) * 1000.0
            return ModelResponse(
                content="",
                provider_used=self.provider_name,
                model_name=self.model,
                latency_ms=latency,
                success=False,
                error=f"Gemini API error: {exc}",
            )
