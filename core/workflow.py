"""Lightweight workflow and task definitions for the multi-agent
research orchestrator.

These replace the unavailable ``google.adk.agents.Workflow`` and
``google.adk.agents.Task`` classes with simple, self-contained data
structures that the orchestrator can use to define and execute
research pipelines.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class Task:
    """A single unit of work inside a research workflow.

    Parameters
    ----------
    name:
        Human-readable identifier for the task (e.g. ``"query_formulation"``).
    agent_type:
        The agent class name that should execute this task
        (e.g. ``"LiteratureReviewAgent"``).
    parameters:
        Key/value pairs forwarded to the agent's ``process`` method.
    dependencies:
        Names of other tasks that must complete before this one starts.
    """

    name: str
    agent_type: str
    parameters: Dict[str, Any] = field(default_factory=dict)
    dependencies: List[str] = field(default_factory=list)


@dataclass
class Workflow:
    """An ordered collection of :class:`Task` objects that together form
    a research pipeline (e.g. *literature review*, *data analysis*).

    Parameters
    ----------
    name:
        Short slug used as a lookup key (e.g. ``"literature_review"``).
    description:
        Human-readable description shown in UIs / logs.
    tasks:
        The tasks that make up this workflow, in dependency order.
    """

    name: str
    description: str
    tasks: List[Task] = field(default_factory=list)
