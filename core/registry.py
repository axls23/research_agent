"""Agent registry that maps agent type names to live agent instances.

The registry is the glue between the orchestrator's workflow templates
(which reference agents by string name) and the concrete
:class:`ResearchAgent` subclasses that live in the ``agents/`` package.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, Optional, Type

from core.base_agent import ResearchAgent

logger = logging.getLogger(__name__)


class AgentRegistry:
    """Maintains a mapping from agent-type name → agent instance.

    Usage::

        registry = AgentRegistry()
        registry.register("LiteratureReviewAgent", LiteratureReviewAgent())
        result = await registry.call_agent(
            "LiteratureReviewAgent",
            {"action": "formulate_search_query", "topic": "AI"},
        )
    """

    def __init__(self) -> None:
        self._agents: Dict[str, ResearchAgent] = {}

    # ------------------------------------------------------------------
    # Registration
    # ------------------------------------------------------------------

    def register(self, name: str, agent: ResearchAgent) -> None:
        """Register an agent instance under *name*."""
        self._agents[name] = agent
        logger.info("Registered agent %s (%s)", name, type(agent).__name__)

    def register_class(self, name: str, cls: Type[ResearchAgent]) -> None:
        """Instantiate *cls* and register the result under *name*."""
        self.register(name, cls())

    # ------------------------------------------------------------------
    # Lookup
    # ------------------------------------------------------------------

    def get(self, name: str) -> Optional[ResearchAgent]:
        """Return the agent registered under *name*, or ``None``."""
        return self._agents.get(name)

    def list_agents(self) -> list[str]:
        """Return the names of all registered agents."""
        return list(self._agents.keys())

    # ------------------------------------------------------------------
    # Execution
    # ------------------------------------------------------------------

    async def call_agent(
        self, name: str, input_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Look up the agent registered as *name* and call its ``process``
        method with *input_data*.

        Raises :class:`KeyError` if *name* is not registered.
        """
        agent = self._agents.get(name)
        if agent is None:
            raise KeyError(f"Agent not registered: {name}")
        return await agent.process(input_data)
