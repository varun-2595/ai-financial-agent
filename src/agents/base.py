"""
Base Agent Abstract Class for Aegis Multi-Agent Network.

Standardizes model gateway communication, structured response parsing,
and deterministic rule-based fallback handling.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Optional

from src.agents.models import AgentSignalOutput
from src.data.models import StockSnapshot
from src.gateway.gateway import ModelRequest, TaskComplexity
from src.gateway.router import ModelRouter, get_router
from src.utils.logger import logger


class BaseAgent(ABC):
    """
    Abstract base class for all Aegis specialized analytical agents.
    """

    def __init__(
        self,
        name: str,
        system_prompt: str,
        task_complexity: TaskComplexity = TaskComplexity.COMPLEX,
        router: Optional[ModelRouter] = None,
    ):
        self.name = name
        self.system_prompt = system_prompt
        self.task_complexity = task_complexity
        self.router = router or get_router()

    def _query_gateway(
        self,
        user_prompt: str,
        fallback_handler: Any,
        *fallback_args: Any,
        **fallback_kwargs: Any,
    ) -> AgentSignalOutput:
        """
        Execute request via the ModelGateway/ModelRouter.
        If LLM is offline or output fails schema validation, delegates to deterministic fallback.
        """
        req = ModelRequest(
            system_prompt=self.system_prompt,
            user_prompt=user_prompt,
            response_schema=AgentSignalOutput,
            task_complexity=self.task_complexity,
            temperature=0.2,
        )

        resp = self.router.route(req)

        if resp.success and isinstance(resp.structured_data, AgentSignalOutput):
            # Ensure agent name is accurately stamped
            resp.structured_data.agent = self.name
            logger.info(
                f"[{self.name}] ✓ {resp.structured_data.signal} "
                f"(Conf: {resp.structured_data.confidence:.2f}, Provider: {resp.provider_used})"
            )
            return resp.structured_data

        logger.warning(
            f"[{self.name}] LLM unavailable or malformed ({resp.error}). "
            "Engaging deterministic rule-based fallback."
        )
        return fallback_handler(*fallback_args, **fallback_kwargs)

    @abstractmethod
    def _heuristic_fallback(self, *args: Any, **kwargs: Any) -> AgentSignalOutput:
        """Deterministic rule-based fallback if all LLMs are offline."""
        pass
