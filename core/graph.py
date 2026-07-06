"""
core/graph.py - Eager Edition
=============
LangGraph StateGraph builder — compiles the full research pipeline
with validation gates, human-in-the-loop, and conditional routing.
"""

from __future__ import annotations

import logging
import uuid
from dataclasses import dataclass
from typing import Any, Dict, Literal, Optional

from langgraph.graph import END, StateGraph

from core.state import ResearchState, make_initial_state
from core.llm_provider import (
    LLMProvider,
    create_llm_from_config,
    create_tiered_providers,
)
from core.nodes.dark_data_ingestion_node import dark_data_ingestion_node
from core.nodes.data_processing_node import data_processing_node
from core.nodes.literature_review_node import literature_review_node
from core.nodes.rosetta_core_node import rosetta_core_node
from core.nodes.knowledge_graph_node import knowledge_graph_node
from core.nodes.hypergraph_reasoning_node import hypergraph_reasoning_node
from core.nodes.analysis_node import analysis_node
from core.nodes.writing_node import writing_node
from core.nodes.quality_validator_node import quality_validator_node
from core.nodes.human_intervention_node import human_intervention_node
from core.nodes.audit_formatter_node import audit_formatter_node

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class _GraphEdge:
    source: str
    target: str


class _GraphSnapshot:
    def __init__(self, nodes: list[str], edges: list[_GraphEdge]):
        self.nodes = nodes
        self.edges = edges


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
    decision = state.get("human_decision")
    if decision == "abort" or state.get("abort", False):
        return "abort"
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
# Eager Graph Runner (Immediate Execution)
# ---------------------------------------------------------------------------


class EagerGraphRunner:
    """
    Simulates a LangGraph CompiledGraph but executes nodes immediately (eagerly).
    This bypasses LangGraph's black-box executor for better debugging and
    more flexible state management.
    """

    def __init__(
        self,
        nodes: Dict[str, Any],
        entry_point: str,
        rigor_level: str,
        enable_external_search: bool = False,
    ):
        self.nodes = nodes
        self.entry_point = entry_point
        self.rigor_level = rigor_level
        # NEXUS air-gap policy: external literature APIs are an opt-in
        # enrichment stage, never a required pipeline dependency.
        self.enable_external_search = enable_external_search

    def _next_after_ingest(self) -> str:
        return "literature_review" if self.enable_external_search else "data_processing"

    def get_graph(self) -> _GraphSnapshot:
        """Return a lightweight graph snapshot for tests and introspection."""
        nodes = list(self.nodes.keys())
        edges: list[_GraphEdge] = [_GraphEdge("__start__", self.entry_point)]
        after_ingest = self._next_after_ingest()

        if self.rigor_level == "exploratory":
            edge_map = {
                "dark_data_ingestion": after_ingest,
                "data_processing": "rosetta_core",
                "rosetta_core": "knowledge_graph",
                "knowledge_graph": "hypergraph_reasoning",
                "hypergraph_reasoning": "analysis",
                "analysis": "writing",
                "writing": "audit_formatter",
                "audit_formatter": "__end__",
            }
            if self.enable_external_search:
                edge_map["literature_review"] = "data_processing"
        else:
            edge_map = {
                "dark_data_ingestion": "validator_post_ingest",
                "validator_post_ingest": after_ingest,
                "human_post_ingest": after_ingest,
                "data_processing": "validator_post_data",
                "validator_post_data": "rosetta_core",
                "human_post_data": "rosetta_core",
                "rosetta_core": "knowledge_graph",
                "knowledge_graph": "hypergraph_reasoning",
                "hypergraph_reasoning": "analysis",
                "analysis": "validator_post_analysis",
                "validator_post_analysis": "writing",
                "human_post_analysis": "writing",
                "writing": "audit_formatter",
                "audit_formatter": "__end__",
            }
            if self.enable_external_search:
                edge_map["literature_review"] = "validator_post_lit"
                edge_map["validator_post_lit"] = "data_processing"
                edge_map["human_post_lit"] = "data_processing"

        gate_nodes = {
            "writing", "audit_formatter",
            "validator_post_ingest", "human_post_ingest",
            "validator_post_lit", "human_post_lit",
            "validator_post_data", "human_post_data",
            "validator_post_analysis", "human_post_analysis",
        }
        for source, target in edge_map.items():
            if source in self.nodes or source in gate_nodes:
                edges.append(_GraphEdge(source, target))

        return _GraphSnapshot(nodes=nodes, edges=edges)

    async def ainvoke(self, state: ResearchState, config: Dict[str, Any]) -> ResearchState:
        """Asynchronously run the graph eagerly."""
        logger.info(f"Starting EAGER execution loop from '{self.entry_point}'")
        config = config or {}
        cfgr = config.get("configurable", {})
        max_iterations = int(cfgr.get("max_iterations", 50))

        current_node = self.entry_point
        iteration_count = 0

        after_ingest = self._next_after_ingest()

        gate_name_by_validator = {
            "validator_post_ingest": "post_dark_data_ingestion",
            "validator_post_lit": "post_literature_review",
            "validator_post_data": "post_data_processing",
            "validator_post_analysis": "post_analysis",
        }

        retry_default_target = {
            "human_post_ingest": "dark_data_ingestion",
            "human_post_lit": "literature_review",
            "human_post_data": "data_processing",
            "human_post_analysis": "analysis",
        }
        continue_target = {
            "human_post_ingest": after_ingest,
            "human_post_lit": "data_processing",
            "human_post_data": "rosetta_core",
            "human_post_analysis": "writing",
        }
        
        while current_node != "END":
            iteration_count += 1
            if iteration_count > max_iterations:
                logger.error(
                    "Eager graph exceeded max_iterations=%s; forcing graceful stop.",
                    max_iterations,
                )
                state["abort"] = True
                state["max_iterations_reached"] = True
                current_node = "audit_formatter"
                # Run formatter once to preserve audit export before ending.
                if iteration_count > (max_iterations + 1):
                    break

            logger.info(f"Eagerly Executing [ {current_node} ]")

            if current_node in gate_name_by_validator:
                state["current_gate_name"] = gate_name_by_validator[current_node]
            
            # Execute node function (must be an async node)
            node_func = self.nodes.get(current_node)
            if not node_func:
                logger.error(f"Node '{current_node}' not found in graph.")
                break
                
            # Node execution
            result = await node_func(state, config=config)
            
            # Merge result into state (emulating LangGraph's merge logic)
            if isinstance(result, dict):
                state.update(result) # TypedDict update
            
            # -----------------------------------------------------------------------
            # ROUTING LOGIC (Replicating the edges from the original StateGraph)
            # -----------------------------------------------------------------------
            next_node = "END"
            
            if self.rigor_level == "exploratory":
                # Linear exploratory path
                mapping = {
                    "dark_data_ingestion": after_ingest,
                    "literature_review": "data_processing",
                    "data_processing": "rosetta_core",
                    "rosetta_core": "knowledge_graph",
                    "knowledge_graph": "hypergraph_reasoning",
                    "hypergraph_reasoning": "analysis",
                    "analysis": "writing",
                    "audit_formatter": "END"
                }
                if current_node == "writing":
                    route = _route_after_writing(state)
                    next_node = "dark_data_ingestion" if route == "backtrack" else "audit_formatter"
                else:
                    next_node = mapping.get(current_node, "END")
            else:
                # Full Rigor Path with Validation Gates
                if current_node == "dark_data_ingestion":
                    next_node = "validator_post_ingest"
                elif current_node == "validator_post_ingest":
                    route = _route_after_validation(state)
                    next_node = after_ingest if route == "continue" else "human_post_ingest"
                elif current_node == "human_post_ingest":
                    route = _route_after_human(state)
                    if route == "retry":
                        next_node = state.get("retry_target") or retry_default_target[current_node]
                    elif route == "abort":
                        next_node = "audit_formatter"
                    else:
                        next_node = continue_target[current_node]

                elif current_node == "literature_review":
                    next_node = "validator_post_lit"
                elif current_node == "validator_post_lit":
                    route = _route_after_validation(state)
                    next_node = "data_processing" if route == "continue" else "human_post_lit"
                elif current_node == "human_post_lit":
                    route = _route_after_human(state)
                    if route == "retry":
                        next_node = state.get("retry_target") or retry_default_target[current_node]
                    elif route == "abort":
                        next_node = "audit_formatter"
                    else:
                        next_node = continue_target[current_node]

                elif current_node == "data_processing":
                    next_node = "validator_post_data"
                elif current_node == "validator_post_data":
                    route = _route_after_validation(state)
                    next_node = "rosetta_core" if route == "continue" else "human_post_data"
                elif current_node == "human_post_data":
                    route = _route_after_human(state)
                    if route == "retry":
                        next_node = state.get("retry_target") or retry_default_target[current_node]
                    elif route == "abort":
                        next_node = "audit_formatter"
                    else:
                        next_node = continue_target[current_node]

                elif current_node == "rosetta_core":
                    next_node = "knowledge_graph"

                elif current_node == "knowledge_graph":
                    next_node = "hypergraph_reasoning"

                elif current_node == "hypergraph_reasoning":
                    next_node = "analysis"

                elif current_node == "analysis":
                    next_node = "validator_post_analysis"
                elif current_node == "validator_post_analysis":
                    route = _route_after_validation(state)
                    next_node = "writing" if route == "continue" else "human_post_analysis"
                elif current_node == "human_post_analysis":
                    route = _route_after_human(state)
                    if route == "retry":
                        next_node = state.get("retry_target") or retry_default_target[current_node]
                    elif route == "abort":
                        next_node = "audit_formatter"
                    else:
                        next_node = continue_target[current_node]

                elif current_node == "writing":
                    route = _route_after_writing(state)
                    next_node = "dark_data_ingestion" if route == "backtrack" else "audit_formatter"
                
                elif current_node == "audit_formatter":
                    next_node = "END"

            current_node = next_node

        logger.info("Eager execution loop completed.")
        return state


# ---------------------------------------------------------------------------
# Graph Builder (Eager version)
# ---------------------------------------------------------------------------


def build_research_graph(
    rigor_level: str = "exploratory",
    enable_external_search: bool = False,
) -> EagerGraphRunner:
    """
    Builds an eager graph runner that executes the NEXUS dark-data pipeline:

        dark_data_ingestion → [literature_review?] → data_processing →
        rosetta_core → knowledge_graph → hypergraph_reasoning →
        analysis → writing → audit_formatter

    Local dark-data ingestion is always the entry point. External literature
    search (ArXiv/S2/Crossref) is an opt-in enrichment stage — disabled by
    default so on-premise deployments stay air-gapped.
    """

    # Define node mapping
    nodes = {
        "dark_data_ingestion": dark_data_ingestion_node,
        "literature_review": literature_review_node,
        "data_processing": data_processing_node,
        "rosetta_core": rosetta_core_node,
        "knowledge_graph": knowledge_graph_node,
        "hypergraph_reasoning": hypergraph_reasoning_node,
        "analysis": analysis_node,
        "writing": writing_node,
        "audit_formatter": audit_formatter_node,
        # Shared instances for gates
        "validator_post_ingest": quality_validator_node,
        "validator_post_lit": quality_validator_node,
        "validator_post_data": quality_validator_node,
        "validator_post_analysis": quality_validator_node,
        "human_post_ingest": human_intervention_node,
        "human_post_lit": human_intervention_node,
        "human_post_data": human_intervention_node,
        "human_post_analysis": human_intervention_node,
    }

    logger.info(
        "Initialized EAGER research graph with rigor_level=%r "
        "(external_search=%s)",
        rigor_level,
        enable_external_search,
    )

    return EagerGraphRunner(
        nodes=nodes,
        entry_point="dark_data_ingestion",
        rigor_level=rigor_level,
        enable_external_search=enable_external_search,
    )


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
    allow_auto_override: bool = False,
    config_path: str = "config/config.yaml",
    mode: str = "agentic",
    agentic_model: Optional[str] = None,
    enable_external_search: bool = False,
) -> ResearchState:
    """
    High-level function to run the full research pipeline.

    Args:
        mode: Pipeline execution mode.
            - "agentic": ReAct supervisor orchestrating subagents dynamically
            - "default"/"langgraph"/"deterministic": fixed eager StateGraph with
              validation gates and human-in-the-loop
        enable_external_search: opt into the external literature enrichment
            stage (off by default — air-gap policy).

    Usage::

        result = await run_research_pipeline(
            project_name="My Review",
            research_topic="Machine Learning in Healthcare",
            research_goals=["accuracy", "interpretability"],
            rigor_level="prisma",
        )
    """
    requested_mode = (mode or "agentic").lower()

    if requested_mode == "agentic":
        from core.orchestrator import run_agentic_pipeline

        logger.info(
            f"Starting AGENTIC pipeline for rigor {rigor_level.upper()}: {project_name}"
        )
        result = await run_agentic_pipeline(
            project_name=project_name,
            research_topic=research_topic,
            research_goals=research_goals,
            model=agentic_model,
            rigor_level=rigor_level,
        )
        logger.info("Agentic pipeline complete!")
        return result

    # ---- Deterministic mode: fixed StateGraph with validation gates ----
    logger.info(
        f"Starting DETERMINISTIC pipeline for rigor {rigor_level.upper()}: {project_name}"
    )

    runner = build_research_graph(
        rigor_level=rigor_level,
        enable_external_search=enable_external_search,
    )
    state = make_initial_state(
        project_id=str(uuid.uuid4()),
        project_name=project_name,
        research_topic=research_topic,
        research_goals=research_goals,
        rigor_level=rigor_level,
    )

    tiers = create_tiered_providers(config_path)
    deep_llm = llm or tiers.get("deep")
    configurable: Dict[str, Any] = {
        "llm": deep_llm,
        "llm_fast": tiers.get("fast") or deep_llm,
        "llm_deep": deep_llm,
        "interactive": interactive,
        "allow_auto_override": allow_auto_override,
    }

    result = await runner.ainvoke(state, {"configurable": configurable})
    logger.info("Deterministic pipeline complete!")
    return result
