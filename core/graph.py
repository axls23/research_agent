"""
core/graph.py
=============
LangGraph StateGraph builder — compiles the full research pipeline
with validation gates, human-in-the-loop, and conditional routing.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, Literal, Optional

from langgraph.graph import END, StateGraph

from core.state import ResearchState, make_initial_state
from core.llm_provider import (
    LLMProvider,
    create_llm_from_config,
    create_tiered_providers,
)
from core.nodes.literature_review_node import literature_review_node
from core.nodes.data_processing_node import data_processing_node
from core.nodes.knowledge_graph_node import knowledge_graph_node
from core.nodes.analysis_node import analysis_node
from core.nodes.writing_node import writing_node
from core.nodes.quality_validator_node import quality_validator_node
from core.nodes.human_intervention_node import human_intervention_node
from core.nodes.audit_formatter_node import audit_formatter_node

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Routing functions (conditional edges)
# ---------------------------------------------------------------------------


def _route_after_validation(state: ResearchState) -> str:
    """
    After quality_validator_node runs, decide next step:
    - If validation passed → continue to next node
    - If validation failed → go to human_intervention
    """
    if state.get("last_validation_passed", True):
        return "continue"
    return "human_intervention"


def _route_after_human(state: ResearchState) -> str:
    """
    After human_intervention_node runs, decide next step:
    - If abort → go to audit_formatter (exit cleanly)
    - If override → continue to next node
    - If retry → the node retried will be set by returning the
      previous node name; for simplicity we re-enter the same stage
    """
    if state.get("abort", False):
        return "abort"
    # override or retry both continue (retry would need more complex logic)
    return "continue"


def _should_validate(state: ResearchState) -> str:
    """
    Check if we need validation gates (rigor != exploratory).
    """
    if state.get("rigor_level", "exploratory") == "exploratory":
        return "skip"
    return "validate"


def _route_after_writing(state: ResearchState) -> str:
    """
    After writing_node runs, decide next step:
    - If needs_more_papers AND backtrack budget remains → loop back
    - Otherwise → proceed to audit_formatter
    """
    if state.get("needs_more_papers", False):
        return "backtrack"
    return "finish"


# ---------------------------------------------------------------------------
# Graph Builder
# ---------------------------------------------------------------------------


def build_research_graph(
    rigor_level: str = "exploratory",
) -> Any:
    """
    Build and compile the LangGraph research pipeline.

    The graph structure:

    literature_review → [validator → human?] → data_processing →
    [validator → human?] → knowledge_graph → analysis →
    [validator → human?] → writing → audit_formatter → END

    For "exploratory" rigor, validation gates are skipped.
    """
    graph = StateGraph(ResearchState)

    # ---- Add all nodes ----
    graph.add_node("literature_review", literature_review_node)
    graph.add_node("data_processing", data_processing_node)
    graph.add_node("knowledge_graph", knowledge_graph_node)
    graph.add_node("analysis", analysis_node)
    graph.add_node("writing", writing_node)
    graph.add_node("audit_formatter", audit_formatter_node)

    # Validation + human intervention nodes
    # We use separate named instances for each gate to allow
    # different routing from each point
    graph.add_node("validator_post_lit", quality_validator_node)
    graph.add_node("validator_post_data", quality_validator_node)
    graph.add_node("validator_post_analysis", quality_validator_node)
    graph.add_node("human_post_lit", human_intervention_node)
    graph.add_node("human_post_data", human_intervention_node)
    graph.add_node("human_post_analysis", human_intervention_node)

    # ---- Entry point ----
    graph.set_entry_point("literature_review")

    # ---- Edges: Literature Review → Validation Gate 1 ----
    if rigor_level == "exploratory":
        # Skip all validation gates
        graph.add_edge("literature_review", "data_processing")
        graph.add_edge("data_processing", "knowledge_graph")
        graph.add_edge("knowledge_graph", "analysis")
        graph.add_edge("analysis", "writing")
        graph.add_conditional_edges(
            "writing",
            _route_after_writing,
            {
                "backtrack": "literature_review",
                "finish": "audit_formatter",
            },
        )
    else:
        # === Gate 1: After literature review ===
        graph.add_edge("literature_review", "validator_post_lit")
        graph.add_conditional_edges(
            "validator_post_lit",
            _route_after_validation,
            {
                "continue": "data_processing",
                "human_intervention": "human_post_lit",
            },
        )
        graph.add_conditional_edges(
            "human_post_lit",
            _route_after_human,
            {
                "continue": "data_processing",
                "abort": "audit_formatter",
            },
        )

        # === Gate 2: After data processing ===
        graph.add_edge("data_processing", "validator_post_data")
        graph.add_conditional_edges(
            "validator_post_data",
            _route_after_validation,
            {
                "continue": "knowledge_graph",
                "human_intervention": "human_post_data",
            },
        )
        graph.add_conditional_edges(
            "human_post_data",
            _route_after_human,
            {
                "continue": "knowledge_graph",
                "abort": "audit_formatter",
            },
        )

        # knowledge_graph → analysis (no gate here)
        graph.add_edge("knowledge_graph", "analysis")

        # === Gate 3: After analysis ===
        graph.add_edge("analysis", "validator_post_analysis")
        graph.add_conditional_edges(
            "validator_post_analysis",
            _route_after_validation,
            {
                "continue": "writing",
                "human_intervention": "human_post_analysis",
            },
        )
        graph.add_conditional_edges(
            "human_post_analysis",
            _route_after_human,
            {
                "continue": "writing",
                "abort": "audit_formatter",
            },
        )

        # writing → audit_formatter OR backtrack to literature_review
        graph.add_conditional_edges(
            "writing",
            _route_after_writing,
            {
                "backtrack": "literature_review",
                "finish": "audit_formatter",
            },
        )

    # ---- Terminal edge ----
    graph.add_edge("audit_formatter", END)

    # ---- Compile ----
    compiled = graph.compile()
    logger.info(
        f"Compiled research graph with rigor_level={rigor_level!r}, "
        f"validation_gates={'enabled' if rigor_level != 'exploratory' else 'disabled'}"
    )
    return compiled


# ---------------------------------------------------------------------------
# Convenience runner
# ---------------------------------------------------------------------------


async def run_research_pipeline(
    project_name: str,
    research_topic: str,
    research_goals: list[str],
    rigor_level: Literal["exploratory", "prisma", "cochrane"] = "exploratory",
    llm: Optional[LLMProvider] = None,
    interactive: bool = True,
    config_path: str = "config/config.yaml",
    mode: Literal["deterministic", "agentic"] = "deterministic",
) -> ResearchState:
    """
    High-level function to run the full research pipeline.

    Args:
        mode: Pipeline execution mode.
            - "deterministic": Fixed StateGraph pipeline (default, backward-compatible)
            - "agentic": ReAct orchestrator with Deep Agents (agent decides flow)

    Usage::

        result = await run_research_pipeline(
            project_name="My Review",
            research_topic="Machine Learning in Healthcare",
            research_goals=["accuracy", "interpretability"],
            rigor_level="prisma",
            mode="agentic",  # <-- ReAct agent loop
        )
    """
    # ---- Agentic Mode: ReAct Orchestrator ----
    if mode == "agentic":
        from core.orchestrator import run_agentic_pipeline

        logger.info(f"Starting AGENTIC pipeline: {project_name} ({rigor_level})")
        result = await run_agentic_pipeline(
            project_name=project_name,
            research_topic=research_topic,
            research_goals=research_goals,
        )
        logger.info("Agentic pipeline complete!")
        return result

    # ---- Deterministic Mode: StateGraph (default) ----
    import uuid

    # Create LLM providers (tiered routing)
    if llm is None:
        try:
            tiers = create_tiered_providers(config_path)
            llm_fast = tiers.get("fast")
            llm_deep = tiers.get("deep")
            agent_tiers = tiers.get("agent_tiers", {})
            # Use deep as the default / backward-compatible "llm"
            llm = llm_deep
            logger.info(
                f"Tiered routing active: "
                f"fast={getattr(llm_fast, 'model', '?')}, "
                f"deep={getattr(llm_deep, 'model', '?')}"
            )
        except Exception as e:
            logger.warning(f"Failed to create tiered providers: {e}")
            llm_fast = None
            llm_deep = None
            agent_tiers = {}
    else:
        # Custom LLM passed — use it for everything
        llm_fast = llm
        llm_deep = llm
        agent_tiers = {}

    # Build initial state
    initial_state = make_initial_state(
        project_id=str(uuid.uuid4())[:8],
        project_name=project_name,
        research_topic=research_topic,
        research_goals=research_goals,
        rigor_level=rigor_level,
    )

    # Build and compile graph
    graph = build_research_graph(rigor_level=rigor_level)

    # Run — pass tiered LLMs through config so each node picks the right one
    config = {
        "configurable": {
            "llm": llm,  # backward-compatible default
            "llm_fast": llm_fast,  # 8B — screening, queries, validation
            "llm_deep": llm_deep,  # 70B — synthesis, analysis, writing
            "agent_tiers": agent_tiers,
            "interactive": interactive,
            "output_dir": "outputs",
        }
    }

    logger.info(f"Starting DETERMINISTIC pipeline: {project_name} ({rigor_level})")
    result = await graph.ainvoke(initial_state, config=config)
    logger.info("Pipeline complete!")

    return result
