"""
core/capabilities.py
====================
Mosaic capability registry for NEXUS subagents.

Each subagent is a self-contained *tile*: a declarative ``AgentCapability``
that names its prompt, tools, model tier, and dispatch policy. The
orchestrator composes whatever tiles are registered — it never hard-codes
agent internals. Growing the system means registering a new tile, not
editing the supervisor.

Every tile execution returns the same structured ``AgentResult`` contract,
so the ReAct supervisor always reasons over real, uniform outcomes.
"""

from __future__ import annotations

import os
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Literal, Optional, Tuple

from typing_extensions import TypedDict

DispatchMode = Literal["inline", "queue"]

#: Global default when a tile does not pin its own dispatch mode.
#: ``inline``  — execute the subagent synchronously; the supervisor sees results.
#: ``queue``   — enqueue for an external worker (``python nexus.py worker``).
DEFAULT_DISPATCH: DispatchMode = "inline"

_DISPATCH_ENV_VAR = "NEXUS_AGENT_DISPATCH"


# ---------------------------------------------------------------------------
# Contracts
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class AgentCapability:
    """One mosaic tile: the full declarative contract of a subagent.

    Attributes:
        name: Stable kebab-case identity (e.g. ``"rosetta-core"``).
        description: Supervisor-facing summary used for tool selection.
        system_prompt: The tile's own system prompt.
        tool_names: Names of callables resolved from ``core.agent_tools``.
        model_tier: ``"fast"`` (retrieve/ground) or ``"deep"`` (reason/generate);
            maps onto the tier routing in ``config/config.yaml``.
        dispatch: Per-tile override of the dispatch mode; ``None`` defers to
            ``NEXUS_AGENT_DISPATCH`` env var, then ``DEFAULT_DISPATCH``.
        catalog_note: Extra guidance appended in the orchestrator prompt
            catalog (e.g. air-gap warnings). Not part of the tool description.
        version: Bump when the tile's contract changes; echoed in results.
    """

    name: str
    description: str
    system_prompt: str
    tool_names: Tuple[str, ...] = ()
    model_tier: Literal["fast", "deep"] = "fast"
    dispatch: Optional[DispatchMode] = None
    catalog_note: str = ""
    version: str = "1.0.0"


class AgentResult(TypedDict):
    """Uniform result envelope returned by every tile execution."""

    agent: str
    status: Literal["ok", "error", "queued"]
    summary: str  # Final subagent message (or dispatch note for "queued")
    error: Optional[str]
    job_id: Optional[int]  # Ledger row in the job queue, when available
    duration_ms: int
    capability_version: str


def make_result(
    agent: str,
    status: Literal["ok", "error", "queued"],
    summary: str = "",
    error: Optional[str] = None,
    job_id: Optional[int] = None,
    duration_ms: int = 0,
    capability_version: str = "",
) -> AgentResult:
    """Build a fully-populated AgentResult (all keys always present)."""
    return AgentResult(
        agent=agent,
        status=status,
        summary=summary,
        error=error,
        job_id=job_id,
        duration_ms=int(duration_ms),
        capability_version=capability_version,
    )


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

_REGISTRY: "OrderedDict[str, AgentCapability]" = OrderedDict()


def register_capability(cap: AgentCapability, replace: bool = False) -> None:
    """Add a tile to the mosaic. Raises on duplicates unless ``replace``."""
    if cap.name in _REGISTRY and not replace:
        raise ValueError(
            f"Capability '{cap.name}' is already registered; "
            "pass replace=True to override it."
        )
    _REGISTRY[cap.name] = cap


def unregister_capability(name: str) -> Optional[AgentCapability]:
    """Remove a tile (used by tests and plugins). Returns it, or None."""
    return _REGISTRY.pop(name, None)


def get_capability(name: str) -> AgentCapability:
    try:
        return _REGISTRY[name]
    except KeyError:
        raise KeyError(
            f"Unknown capability '{name}'. Registered: {sorted(_REGISTRY)}"
        ) from None


def all_capabilities() -> List[AgentCapability]:
    """All tiles in registration order (order defines catalog order)."""
    return list(_REGISTRY.values())


def resolve_tools(cap: AgentCapability) -> List[Callable[..., Any]]:
    """Resolve a tile's tool names against ``core.agent_tools``."""
    from core import agent_tools

    tools: List[Callable[..., Any]] = []
    for tool_name in cap.tool_names:
        tool_fn = getattr(agent_tools, tool_name, None)
        if tool_fn is None:
            raise AttributeError(
                f"Capability '{cap.name}' declares tool '{tool_name}' "
                "but core.agent_tools does not define it."
            )
        tools.append(tool_fn)
    return tools


def resolve_dispatch(cap: Optional[AgentCapability]) -> DispatchMode:
    """Effective dispatch mode: tile override → env var → default."""
    if cap is not None and cap.dispatch in ("inline", "queue"):
        return cap.dispatch
    env_value = (os.getenv(_DISPATCH_ENV_VAR) or "").strip().lower()
    if env_value in ("inline", "queue"):
        return env_value  # type: ignore[return-value]
    return DEFAULT_DISPATCH


def render_catalog() -> str:
    """Markdown bullet list of tiles for the orchestrator system prompt."""
    lines = []
    for cap in all_capabilities():
        note = f" {cap.catalog_note}" if cap.catalog_note else ""
        lines.append(f"- **{cap.name}**: {cap.description}{note}")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Default tiles
# ---------------------------------------------------------------------------

_DEFAULT_CAPABILITIES: Tuple[AgentCapability, ...] = (
    AgentCapability(
        name="deep-reasoner",
        description=(
            "Call this agent for complex reasoning, math, or planning and "
            "final validation consistency checks."
        ),
        system_prompt=(
            "You are a deep-reasoning agent with the ability to run Python code to solve "
            "complex mathematical, logical, or data-heavy problems. Use your Python sandbox "
            "to verify hypotheses or perform complex calculations when requested."
        ),
        tool_names=("validate_quality",),
        model_tier="deep",
    ),
    AgentCapability(
        name="literature-search",
        description=(
            "Search external academic databases (arXiv, Semantic Scholar) "
            "based on extracted research topics."
        ),
        system_prompt=(
            "You are an academic literature search specialist. Extract research topics "
            "from the orchestrator and query external databases (e.g., arXiv, Semantic "
            "Scholar) to retrieve papers."
        ),
        tool_names=("literature_search", "validate_quality"),
        catalog_note=(
            "OPTIONAL enrichment. Only use when the user explicitly asks for "
            "external/public literature — external calls violate the air-gap "
            "policy otherwise."
        ),
    ),
    AgentCapability(
        name="dark-data-ingestion",
        description="Ingest enterprise dark data from local files and staged directories.",
        system_prompt=(
            "You are an enterprise dark-data ingestion specialist. Load local artifacts "
            "from configured connectors and filesystem sources while preserving provenance."
        ),
        tool_names=("ingest_dark_data", "validate_quality"),
        catalog_note="This is the PRIMARY data source.",
    ),
    AgentCapability(
        name="data-processing",
        description="Process staged PDFs or text documents into chunks.",
        system_prompt=(
            "You are a document processing specialist. Your job is to extract text from "
            "enterprise documents and split them into chunks suitable for embedding and "
            "knowledge extraction."
        ),
        tool_names=("process_documents",),
    ),
    AgentCapability(
        name="rosetta-core",
        description=(
            "Translate domain-specific terminology/jargon into shared core "
            "engineering principles."
        ),
        system_prompt=(
            "You are the Rosetta Core specialist. Translate silo-specific jargon into "
            "shared principles that can be mapped across departments and domains."
        ),
        tool_names=("rosetta_translate",),
        model_tier="deep",
    ),
    AgentCapability(
        name="knowledge-graph",
        description="Extract entities and construct structured Neo4j reasoning graphs.",
        system_prompt=(
            "You are a knowledge graph specialist. You build structured knowledge "
            "graphs aligned with NEXUS principles. Extract entities (Paper, Objective, "
            "Methodology, Result, Limitation, Implication) and relationships."
        ),
        tool_names=("extract_prisma_knowledge", "neo4j_vector_search", "neo4j_query"),
        model_tier="deep",
        catalog_note="Stores embeddings using native Neo4j Vectors.",
    ),
    AgentCapability(
        name="analysis",
        description="Run evidence synthesis and GraphRAG context retrieval.",
        system_prompt=(
            "You are a systematic review analyst. Analyze extracted PRISMA entities to "
            "identify patterns, contradictions, and gaps. Use GraphRAG retrieval "
            "(neo4j_vector_search -> neo4j_query) to build rich context before synthesis."
        ),
        tool_names=("analyze_evidence", "neo4j_vector_search", "neo4j_query"),
        model_tier="deep",
    ),
    AgentCapability(
        name="writing",
        description="Draft academic/enterprise review sections.",
        system_prompt=(
            "You are an enterprise intelligence writer. Draft evidence-based findings "
            "that surface cross-silo isomorphic mappings: which teams are solving "
            "structurally identical problems, what redundant work can be averted, and "
            "which translated principles transfer between domains. Report opportunity "
            "gaps with concrete, actionable recommendations. After drafting, check for "
            "evidence gaps."
        ),
        tool_names=("draft_section", "neo4j_vector_search", "validate_quality"),
        model_tier="deep",
    ),
)

for _cap in _DEFAULT_CAPABILITIES:
    register_capability(_cap)
