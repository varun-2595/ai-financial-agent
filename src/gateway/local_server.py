"""
Aegis Mock OpenAI Inference Server for Testing & Integration.

Provides a lightweight, genuine OpenAI-compatible HTTP server exposing:
- GET /v1/models (returns available models list)
- POST /v1/chat/completions (returns valid OpenAI chat response format)

Used for unit/integration testing of LocalModelProvider without external dependencies.
"""
from __future__ import annotations

import argparse
import json
import threading
import time
from http.server import BaseHTTPRequestHandler, HTTPServer
from typing import Any, Optional

from src.utils.logger import logger

DEFAULT_PORT = 11434
DEFAULT_HOST = "127.0.0.1"
DEFAULT_MODELS = ["qwen2.5:14b-instruct", "qwen2.5:14b", "llama3.1:8b", "mistral:7b"]


class MockOpenAIHTTPHandler(BaseHTTPRequestHandler):
    """OpenAI-compatible HTTP request handler for testing."""

    available_models = DEFAULT_MODELS

    def log_message(self, format: str, *args: Any) -> None:
        pass  # Suppress default stdio access logging

    def do_GET(self) -> None:
        if self.path in ("/v1/models", "/models"):
            models_data = [
                {
                    "id": m,
                    "object": "model",
                    "created": 1700000000,
                    "owned_by": "local-test-runner",
                }
                for m in self.available_models
            ]
            resp = {"object": "list", "data": models_data}
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps(resp).encode("utf-8"))
        else:
            self.send_response(404)
            self.end_headers()

    def do_POST(self) -> None:
        if self.path in ("/v1/chat/completions", "/chat/completions"):
            content_len = int(self.headers.get("Content-Length", 0))
            body = self.rfile.read(content_len).decode("utf-8")

            try:
                data = json.loads(body)
                model = data.get("model", "qwen2.5:14b-instruct")

                # Default structured mock signal response
                output_dict = {
                    "agent": "TechnicalAgent",
                    "signal": "BUY",
                    "confidence": 0.85,
                    "reasons": ["Test neural inference catalyst", "Bullish structure confirmed"],
                    "risks": ["Test volatility boundary"],
                    "evidence": ["Volume +18%"],
                }
                content_str = json.dumps(output_dict)

                resp = {
                    "id": f"chatcmpl-mock-{int(time.time()*1000)}",
                    "object": "chat.completion",
                    "created": int(time.time()),
                    "model": model,
                    "choices": [
                        {
                            "index": 0,
                            "message": {"role": "assistant", "content": content_str},
                            "finish_reason": "stop",
                        }
                    ],
                    "usage": {
                        "prompt_tokens": 50,
                        "completion_tokens": 30,
                        "total_tokens": 80,
                    },
                }

                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                self.wfile.write(json.dumps(resp).encode("utf-8"))

            except Exception as e:
                self.send_response(500)
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                self.wfile.write(json.dumps({"error": str(e)}).encode("utf-8"))
        else:
            self.send_response(404)
            self.end_headers()


_SERVER_INSTANCE: Optional[HTTPServer] = None
_SERVER_THREAD: Optional[threading.Thread] = None


def start_mock_local_server(
    host: str = DEFAULT_HOST,
    port: int = DEFAULT_PORT,
    models: Optional[list[str]] = None,
) -> HTTPServer:
    """Start mock local OpenAI server in a background daemon thread."""
    global _SERVER_INSTANCE, _SERVER_THREAD
    if _SERVER_INSTANCE is not None:
        return _SERVER_INSTANCE

    handler = MockOpenAIHTTPHandler
    if models is not None:
        handler.available_models = models

    server = HTTPServer((host, port), handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()

    _SERVER_INSTANCE = server
    _SERVER_THREAD = thread
    logger.info(f"[MockLocalServer] Running at http://{host}:{port}/v1")
    return server


def stop_mock_local_server() -> None:
    """Stop the background mock server."""
    global _SERVER_INSTANCE, _SERVER_THREAD
    if _SERVER_INSTANCE is not None:
        _SERVER_INSTANCE.shutdown()
        _SERVER_INSTANCE.server_close()
        _SERVER_INSTANCE = None
        _SERVER_THREAD = None
        logger.info("[MockLocalServer] Stopped.")
