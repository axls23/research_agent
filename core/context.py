"""Execution context passed through a research workflow.

The ``Context`` object carries project-level metadata and researcher
preferences so that every agent in a pipeline has access to shared
state without tight coupling.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Optional


@dataclass
class Context:
    """Shared context threaded through every task in a workflow run.

    Parameters
    ----------
    project_id:
        Unique identifier of the research project.
    researcher_preferences:
        User-specified preferences (citation style, statistical methods, …).
    knowledge_graph_id:
        Identifier of the knowledge graph associated with the project.
    extra:
        Arbitrary additional parameters supplied at workflow start.
    """

    project_id: str = ""
    researcher_preferences: Dict[str, Any] = field(default_factory=dict)
    knowledge_graph_id: Optional[str] = None
    extra: Dict[str, Any] = field(default_factory=dict)
