"""
Model Gateway Base Abstractions & Protocols.

Hides provider-specific details (M5 local vs Gemini remote) from the rest of Aegis.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Optional, Type, TypeVar
from pydantic import BaseModel

T = TypeVar("T", bound=BaseModel)


class TaskComplexity(str, Enum):
    """Routing classification for model tasks."""
    SIMPLE = "simple"                     # Fast formatting, quick classification
    HIGH_VOLUME = "high_volume"           # Bulk screening, batch tagging
    COMPLEX = "complex"                   # Multi-factor financial reasoning, macro analysis
    PORTFOLIO_DECISION = "portfolio_decision"  # Final allocation & multi-agent synthesis


@dataclass
class ModelRequest:
    """Standardized request envelope sent through the ModelGateway."""
    system_prompt: str
    user_prompt: str
    response_schema: Optional[Type[BaseModel]] = None
    task_complexity: TaskComplexity = TaskComplexity.COMPLEX
    temperature: float = 0.2
    max_tokens: int = 2048
    timeout_seconds: float = 10.0


@dataclass
class ModelResponse:
    """Standardized response returned by any ModelProvider."""
    content: str
    structured_data: Optional[Any] = None
    provider_used: str = "unknown"  # "local", "gemini", "heuristic_fallback"
    model_name: str = ""
    latency_ms: float = 0.0
    tokens_used: int = 0
    success: bool = True
    error: Optional[str] = None


class BaseModelProvider(ABC):
    """Abstract interface for LLM providers."""

    @property
    @abstractmethod
    def provider_name(self) -> str:
        """Name of the provider (e.g. 'local', 'gemini')."""
        pass

    @abstractmethod
    def is_available(self) -> bool:
        """Check if the provider is currently reachable and operational."""
        pass

    @abstractmethod
    def generate(self, request: ModelRequest) -> ModelResponse:
        """Execute inference and return validated ModelResponse."""
        pass
