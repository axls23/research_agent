"""
core/graph.py - Eager Edition
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

    def __init__(self, nodes: Dict[str, Any], entry_point: str, rigor_level: str):
        self.nodes = nodes
        self.entry_point = entry_point
        self.rigor_level = rigor_level

    async def ainvoke(self, state: ResearchState, config: Dict[str, Any]) -> ResearchState:
        """Asynchronously run the graph eagerly."""
        logger.info(f"Starting EAGER execution loop from '{self.entry_point}'")
        config = config or {}
        cfgr = config.get("configurable", {})
        max_iterations = int(cfgr.get("max_iterations", 50))

        current_node = self.entry_point
        iteration_count = 0

        gate_name_by_validator = {
            "validator_post_lit": "post_literature_review",
            "validator_post_data": "post_data_processing",
            "validator_post_analysis": "post_analysis",
        }

        retry_default_target = {
            "human_post_lit": "literature_review",
            "human_post_data": "data_processing",
            "human_post_analysis": "analysis",
        }
        continue_target = {
            "human_post_lit": "data_processing",
            "human_post_data": "knowledge_graph",
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
                    "literature_review": "data_processing",
                    "data_processing": "knowledge_graph",
                    "knowledge_graph": "analysis",
                    "analysis": "writing",
                    "audit_formatter": "END"
                }
                if current_node == "writing":
                    route = _route_after_writing(state)
                    next_node = "literature_review" if route == "backtrack" else "audit_formatter"
                else:
                    next_node = mapping.get(current_node, "END")
            else:
                # Full Rigor Path with Validation Gates
                if current_node == "literature_review":
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
                    next_node = "knowledge_graph" if route == "continue" else "human_post_data"
                elif current_node == "human_post_data":
                    route = _route_after_human(state)
                    if route == "retry":
                        next_node = state.get("retry_target") or retry_default_target[current_node]
                    elif route == "abort":
                        next_node = "audit_formatter"
                    else:
                        next_node = continue_target[current_node]

                elif current_node == "knowledge_graph":
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
                    next_node = "literature_review" if route == "backtrack" else "audit_formatter"
                
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
) -> EagerGraphRunner:
    """
    Builds an eager graph runner that executes the PRISMA research pipeline.
    
    This replaces the compiled LangGraph approach to give the user immediate
    execution feedback and easier manual control.
    """
    
    # Define node mapping
    nodes = {
        "literature_review": literature_review_node,
        "data_processing": data_processing_node,
        "knowledge_graph": knowledge_graph_node,
        "analysis": analysis_node,
        "writing": writing_node,
        "audit_formatter": audit_formatter_node,
        # Shared instances for gates
        "validator_post_lit": quality_validator_node,
        "validator_post_data": quality_validator_node,
        "validator_post_analysis": quality_validator_node,
        "human_post_lit": human_intervention_node,
        "human_post_data": human_intervention_node,
        "human_post_analysis": human_intervention_node,
    }

    logger.info(
        f"Initialized EAGER research graph with rigor_level={rigor_level!r}"
    )
    
    return EagerGraphRunner(nodes=nodes, entry_point="literature_review", rigor_level=rigor_level)


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
    mode: str = "agentic",
    agentic_model: Optional[str] = None,
) -> ResearchState:
    """
    High-level function to run the full research pipeline.

    Args:
        mode: Pipeline execution mode.
            - "agentic": ReAct orchestrator with Deep Agents (enforced for exploratory)
            - "default"/"langgraph": eagerly executed for prisma/cochrane

    Usage::

        result = await run_research_pipeline(
            project_name="My Review",
            research_topic="Machine Learning in Healthcare",
            research_goals=["accuracy", "interpretability"],
            rigor_level="prisma",
        )
    """
    requested_mode = (mode or "agentic").lower()

    if rigor_level == "exploratory":
        if requested_mode != "agentic":
            logger.warning(
                "Mode '%s' requested, but exploratory mode routes to agentic execution. Routing to agentic mode.",
                requested_mode,
            )
        from core.orchestrator import run_agentic_pipeline

        logger.info(f"Starting AGENTIC pipeline for EXPLORATORY: {project_name}")
        result = await run_agentic_pipeline(
            project_name=project_name,
            research_topic=research_topic,
            research_goals=research_goals,
            model=agentic_model,
            rigor_level=rigor_level,
        )
        logger.info("Agentic pipeline complete!")
        return result
    else:
        logger.info(f"Starting DETERMINISTIC Eager pipeline for STRICT rigor ({rigor_level}): {project_name}")
        
        # Initialize initial state
        initial_state: ResearchState = {
            "project_name": project_name,
            "research_topic": research_topic,
            "research_goals": research_goals,
            "rigor_level": rigor_level,
            "papers": [],
            "chunks": [],
            "knowledge_entities": [],
            "hyperedges": [],
            "audit_log": [],
        }

        # Build eager graph
        runner = build_research_graph(rigor_level=rigor_level)
        
        # Execute logic locally using EagerGraphRunner
        final_state = await runner.invoke(initial_state, config={"configurable": {"llm_deep": getattr(llm, "provider", None) if llm else None}})
        
        logger.info(f"Deterministic pipeline complete for {project_name}.")
        return final_state
