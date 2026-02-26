"""
core/base_agent.py
==================
Base class for all research agents.

Replaces the previous ``google.adk.Agent`` dependency with a
lightweight standalone base that integrates with LangGraph
and the ``LLMProvider`` brain.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from core.llm_provider import LLMProvider


class ResearchAgent:
    """
    Base class for all research agents in the system.

    Every agent:
    - Has a ``name`` and ``description``
    - Can be ``initialize``-d with shared context (project metadata, etc.)
    - Validates required input fields before processing
    - Optionally receives an ``LLMProvider`` to do reasoning

    Subclasses override ``process()`` and ``get_required_fields()``.
    """

    def __init__(
        self,
        name: str,
        description: str,
        llm: Optional[LLMProvider] = None,
    ):
        self.name = name
        self.description = description
        self.llm = llm
        self.logger = logging.getLogger(f"research_agent.{name}")
        self.context: Optional[Dict[str, Any]] = None

    async def initialize(self, context: Dict[str, Any]) -> None:
        """Initialize the agent with context from the orchestrator or graph."""
        self.context = context

        # Auto-create LLM from context if not already set
        if self.llm is None and "llm" in context:
            self.llm = context["llm"]

        self.logger.info(f"Initialized {self.name}")

    async def validate_input(self, input_data: Dict[str, Any]) -> bool:
        """Validate input data before processing."""
        required_fields = self.get_required_fields()
        missing_fields = [field for field in required_fields if field not in input_data]
        if missing_fields:
            self.logger.error(f"Missing required fields: {missing_fields}")
            return False
        return True

    def get_required_fields(self) -> List[str]:
        """Get list of required input fields for the agent."""
        return []

    async def process(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process input data and return results.

        Subclasses must call ``super().process(input_data)`` first —
        it runs validation and raises on invalid input.
        """
        if not await self.validate_input(input_data):
            raise ValueError(
                f"Invalid input data for {self.name}: "
                f"missing {[f for f in self.get_required_fields() if f not in input_data]}"
            )
        return {}

    async def cleanup(self) -> None:
        """Clean up any resources used by the agent."""
        self.context = None
        self.logger.info(f"Cleaned up {self.name}")
