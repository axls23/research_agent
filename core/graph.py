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
from core.llm_provider import LLMProvider, create_llm_from_config
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
    - If retry → re-run the previous stage via its validation gate
    """
    if state.get("abort", False):
        return "abort"
    decision = ""
    decisions = state.get("human_decisions", [])
    if decisions:
        decision = decisions[-1].get("decision", "")
    if decision == "retry":
        return "retry"
    return "continue"


def _should_validate(state: ResearchState) -> str:
    """
    Check if we need validation gates (rigor != exploratory).
    """
    if state.get("rigor_level", "exploratory") == "exploratory":
        return "skip"
    return "validate"


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
        graph.add_edge("writing", "audit_formatter")
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
                "retry": "literature_review",
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
                "retry": "data_processing",
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
                "retry": "analysis",
                "abort": "audit_formatter",
            },
        )

        # writing → audit_formatter
        graph.add_edge("writing", "audit_formatter")

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
) -> ResearchState:
    """
    High-level function to run the full research pipeline.

    Usage::

        result = await run_research_pipeline(
            project_name="My Review",
            research_topic="Machine Learning in Healthcare",
            research_goals=["accuracy", "interpretability"],
            rigor_level="prisma",
        )
    """
    import uuid

    # Create LLM if not provided
    if llm is None:
        try:
            llm = create_llm_from_config(config_path)
        except Exception as e:
            logger.warning(f"Failed to create LLM from config: {e}")
            llm = None

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

    # Run
    config = {
        "configurable": {
            "llm": llm,
            "interactive": interactive,
            "output_dir": "outputs",
        }
    }

    logger.info(f"Starting pipeline: {project_name} ({rigor_level})")
    result = await graph.ainvoke(initial_state, config=config)
    logger.info("Pipeline complete!")

    return result
