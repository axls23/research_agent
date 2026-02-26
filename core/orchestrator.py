"""
core/orchestrator.py
=====================
Backward-compatible orchestrator that wraps the new LangGraph pipeline.

Users can either:
- Use the LangGraph pipeline directly (``core.graph.run_research_pipeline``)
- Use this orchestrator for the legacy Workflow/Task API
- Use the new ReAct orchestrator for agentic mode (``build_orchestrator``)
"""

from __future__ import annotations

import copy
import logging
import os
import time
from typing import Any, Dict, List, Optional

from core.state import make_initial_state

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# LEGACY: Lightweight Workflow / Task / Context / Registry (replacing google.adk)
# ---------------------------------------------------------------------------


class Task:
    """A single task in a workflow."""

    def __init__(
        self,
        name: str,
        agent_type: str,
        parameters: Optional[Dict] = None,
        dependencies: Optional[List[str]] = None,
    ):
        self.name = name
        self.agent_type = agent_type
        self.parameters = parameters or {}
        self.dependencies = dependencies or []


class Workflow:
    """A sequence of tasks forming a research pipeline."""

    def __init__(self, name: str, description: str, tasks: Optional[List[Task]] = None):
        self.name = name
        self.description = description
        self.tasks = tasks or []


class Context:
    """Execution context for a workflow run."""

    def __init__(
        self,
        project_id: str = "",
        researcher_preferences: Optional[Dict] = None,
        knowledge_graph_id: Optional[str] = None,
        extra: Optional[Dict] = None,
        **kwargs,
    ):
        self.project_id = project_id
        self.researcher_preferences = researcher_preferences or {}
        self.knowledge_graph_id = knowledge_graph_id
        self.extra = extra if extra is not None else kwargs


class AgentRegistry:
    """Registry mapping agent names to instances."""

    def __init__(self):
        self._agents: Dict[str, object] = {}

    def register(self, name: str, agent: object) -> None:
        self._agents[name] = agent

    def get(self, name: str) -> Optional[object]:
        return self._agents.get(name)

    def list_agents(self) -> List[str]:
        return list(self._agents.keys())

    async def call_agent(self, name: str, input_data: Dict) -> Dict:
        agent = self._agents.get(name)
        if agent is None:
            raise KeyError(f"Agent '{name}' not registered")
        return await agent.process(input_data)


# ---------------------------------------------------------------------------
# LEGACY: Orchestrator
# ---------------------------------------------------------------------------


class ResearchWorkflowOrchestrator:
    """
    Multi-agent workflow orchestrator.

    Preserves the original API surface so existing tests pass.
    Internally can delegate to the LangGraph pipeline for actual execution.
    """

    def __init__(self, researcher_preferences: Dict):
        self.researcher_preferences = researcher_preferences
        self.active_research_projects: Dict[str, Dict] = {}
        self.workflow_templates = self._initialize_workflow_templates()
        self.agent_registry = AgentRegistry()
        self._register_agents()

    def _initialize_workflow_templates(self) -> Dict[str, Workflow]:
        templates = {}

        # Literature Review Workflow
        lit_review = Workflow(
            name="literature_review",
            description="Comprehensive Literature Review",
            tasks=[
                Task(
                    "query_formulation",
                    "LiteratureReviewAgent",
                    {"action": "formulate_search_query"},
                ),
                Task(
                    "paper_retrieval",
                    "LiteratureReviewAgent",
                    {"action": "retrieve_papers"},
                    ["query_formulation"],
                ),
                Task(
                    "paper_filtering",
                    "LiteratureReviewAgent",
                    {"action": "filter_papers"},
                    ["paper_retrieval"],
                ),
                Task(
                    "knowledge_extraction",
                    "KnowledgeGraphAgent",
                    {"action": "extract_knowledge"},
                    ["paper_filtering"],
                ),
                Task(
                    "synthesis",
                    "WritingAssistantAgent",
                    {"action": "synthesize_literature"},
                    ["knowledge_extraction"],
                ),
            ],
        )
        templates["literature_review"] = lit_review

        # Data Analysis Workflow
        data_analysis = Workflow(
            name="data_analysis",
            description="Complete Data Analysis Pipeline",
            tasks=[
                Task("data_prep", "DataProcessingAgent", {"action": "prepare_data"}),
                Task(
                    "exploratory_analysis",
                    "AnalysisAgent",
                    {"action": "explore_data"},
                    ["data_prep"],
                ),
                Task(
                    "statistical_testing",
                    "AnalysisAgent",
                    {"action": "run_statistical_tests"},
                    ["exploratory_analysis"],
                ),
                Task(
                    "visualization",
                    "AnalysisAgent",
                    {"action": "create_visualizations"},
                    ["statistical_testing"],
                ),
                Task(
                    "results_summary",
                    "WritingAssistantAgent",
                    {"action": "summarize_results"},
                    ["visualization"],
                ),
            ],
        )
        templates["data_analysis"] = data_analysis

        return templates

    def _register_agents(self) -> None:
        """Register all agent instances."""
        from agents.literature_review_agent import LiteratureReviewAgent
        from agents.data_processing_agent import DataProcessingAgent
        from agents.analysis_agent import AnalysisAgent
        from agents.writing_assistant_agent import WritingAssistantAgent
        from agents.knowledge_graph_agent import KnowledgeGraphAgent
        from agents.collaboration_agent import CollaborationAgent

        self.agent_registry.register("LiteratureReviewAgent", LiteratureReviewAgent())
        self.agent_registry.register("DataProcessingAgent", DataProcessingAgent())
        self.agent_registry.register("AnalysisAgent", AnalysisAgent())
        self.agent_registry.register("WritingAssistantAgent", WritingAssistantAgent())
        self.agent_registry.register("KnowledgeGraphAgent", KnowledgeGraphAgent())
        self.agent_registry.register("CollaborationAgent", CollaborationAgent())

    async def start_research_project(self, project_name: str, description: str) -> str:
        project_id = f"project_{len(self.active_research_projects) + 1}"

        # Initialize knowledge graph via agent
        kg_result = await self.agent_registry.call_agent(
            "KnowledgeGraphAgent",
            {"action": "initialize_graph", "project_name": project_name},
        )

        self.active_research_projects[project_id] = {
            "id": project_id,
            "name": project_name,
            "description": description,
            "workflows": {},
            "knowledge_graph": kg_result.get("graph_id"),
            "documents": [],
            "team_members": [],
            "created_at": time.time(),
        }
        return project_id

    async def start_research_workflow(
        self,
        project_id: str,
        workflow_type: str,
        custom_parameters: Optional[Dict] = None,
    ) -> str:
        if project_id not in self.active_research_projects:
            raise ValueError(f"Unknown project: {project_id}")
        if workflow_type not in self.workflow_templates:
            raise ValueError(f"Unknown workflow type: {workflow_type}")

        project = self.active_research_projects[project_id]
        workflow_instance = self._customize_workflow(
            self.workflow_templates[workflow_type], custom_parameters
        )
        instance_id = f"{project_id}_{workflow_type}_{len(project['workflows']) + 1}"

        context = Context(
            project_id=project_id,
            researcher_preferences=self.researcher_preferences,
            knowledge_graph_id=project["knowledge_graph"],
            **(custom_parameters or {}),
        )

        project["workflows"][instance_id] = {
            "type": workflow_type,
            "status": "started",
            "started_at": time.time(),
            "context": context,
        }
        return instance_id

    def _customize_workflow(
        self,
        workflow_template: Workflow,
        custom_parameters: Optional[Dict] = None,
    ) -> Workflow:
        customized = copy.deepcopy(workflow_template)
        if self.researcher_preferences.get("preferred_statistical_methods"):
            for task in customized.tasks:
                if (
                    task.agent_type == "AnalysisAgent"
                    and "run_statistical_tests" in task.parameters.get("action", "")
                ):
                    task.parameters["preferred_methods"] = self.researcher_preferences[
                        "preferred_statistical_methods"
                    ]
        return customized

    async def get_research_progress(self, project_id: str) -> Dict:
        if project_id not in self.active_research_projects:
            raise ValueError(f"Unknown project: {project_id}")
        project = self.active_research_projects[project_id]
        return {
            "project_id": project_id,
            "name": project["name"],
            "workflows": [
                {
                    "instance_id": i,
                    "type": w["type"],
                    "status": w["status"],
                    "started_at": w["started_at"],
                }
                for i, w in project["workflows"].items()
            ],
            "documents": len(project["documents"]),
        }

    async def suggest_next_steps(self, project_id: str) -> List[Dict]:
        if project_id not in self.active_research_projects:
            raise ValueError(f"Unknown project: {project_id}")
        project = self.active_research_projects[project_id]
        active_workflows = [
            w for w in project["workflows"].values() if w["status"] != "completed"
        ]
        completed_workflows = [
            w for w in project["workflows"].values() if w["status"] == "completed"
        ]

        suggestions = []
        workflow_checks = [
            (
                "literature_review",
                "No literature review has been initiated yet. This is a good first step.",
            ),
            (
                "data_analysis",
                "Literature review is complete. Next step is to analyze your data.",
            ),
            (
                "paper_writing",
                "Data analysis is complete. You can now draft your research paper.",
            ),
        ]

        for workflow_type, reason in workflow_checks:
            if not any(
                w["type"] == workflow_type
                for w in active_workflows + completed_workflows
            ):
                suggestions.append(
                    {
                        "action": "start_workflow",
                        "workflow_type": workflow_type,
                        "reason": reason,
                    }
                )
        return suggestions


# ===========================================================================
# NEW: ReAct Agent Orchestrator (Deep Agents SDK)
# ===========================================================================

# Maximum ReAct iterations before forcing completion
MAX_REACT_ITERATIONS = 15


# ---------------------------------------------------------------------------
# System Prompts
# ---------------------------------------------------------------------------

ORCHESTRATOR_SYSTEM_PROMPT = """\
You are a systematic review research orchestrator following PRISMA 2020 guidelines.
Your job is to conduct a complete systematic literature review by delegating work
to specialized subagents and tools.

## Available Subagents
- **literature-search**: Searches academic databases (ArXiv, Semantic Scholar, Crossref),
  performs PICO decomposition, and screens papers.
- **data-processing**: Processes PDF papers into text chunks for analysis.
- **knowledge-graph**: Extracts PRISMA-aligned entities using GLiNER + LLM, builds
  the Neo4j reasoning graph, and stores embeddings in Qdrant.
- **analysis**: Analyzes extracted evidence -- descriptive stats, LLM synthesis,
  and GraphRAG context retrieval.
- **writing**: Drafts academic sections and detects evidence gaps.

## Available Tools
- `qdrant_search(query, prisma_label, limit)` -- Search for semantically similar
  entities in the PRISMA knowledge base. Filter by label (objective, methodology,
  result, limitation, implication).
- `neo4j_query(cypher, params)` -- Run Cypher queries against the PRISMA graph.
  Use for structured reasoning paths like:
  `MATCH (p:Paper)-[:REPORTS_FINDING]->(r:Result) RETURN r.text`
- `validate_quality(stage, state_snapshot)` -- Validate pipeline output quality.

## Research Workflow
1. **Search**: Delegate literature search with PICO-decomposed queries
2. **Process**: Convert found papers into analysable chunks
3. **Extract**: Build PRISMA knowledge graph from chunks
4. **Assess Coverage**: Use qdrant_search to check coverage per PRISMA domain.
   If any domain has < 3 entities, search for MORE papers on that specific domain.
5. **Analyze**: Run evidence synthesis and pattern detection
6. **Write**: Draft sections, check for evidence gaps
7. **Iterate**: If gaps found, loop back to search with refined queries

## Key Principles
- Always check coverage BEFORE analysis. Use qdrant_search with prisma_label filters.
- If methodology coverage is thin, explicitly search for methodology-focused papers.
- Use neo4j_query to verify reasoning paths exist before synthesis.
- Cap total iterations at {max_iterations}. Prefer depth over breadth.
- Document your reasoning at each step.
""".format(
    max_iterations=MAX_REACT_ITERATIONS
)


SUBAGENT_PROMPTS = {
    "literature": (
        "You are a literature search specialist. Your job is to find relevant academic "
        "papers using PICO decomposition, multi-database search, and citation snowballing. "
        "Follow PRISMA guidelines for systematic search documentation."
    ),
    "data_processing": (
        "You are a document processing specialist. Your job is to extract text from "
        "research papers and split them into chunks suitable for embedding and "
        "knowledge extraction."
    ),
    "knowledge_graph": (
        "You are a PRISMA knowledge graph specialist. You build structured knowledge "
        "graphs aligned with the PRISMA 2020 ontology. Extract entities (Paper, Objective, "
        "Methodology, Result, Limitation, Implication) and relationships."
    ),
    "analysis": (
        "You are a systematic review analyst. Analyze extracted PRISMA entities to "
        "identify patterns, contradictions, and gaps. Use GraphRAG retrieval "
        "(qdrant_search -> neo4j_query) to build rich context before synthesis."
    ),
    "writing": (
        "You are an academic writing specialist drafting sections for a systematic review. "
        "Produce well-structured, evidence-based academic text. After drafting, check for "
        "evidence gaps that may require additional literature retrieval."
    ),
}


# ---------------------------------------------------------------------------
# Subagent Factory
# ---------------------------------------------------------------------------


def _build_subagent_configs() -> List[Dict[str, Any]]:
    """Build subagent configuration dicts for create_deep_agent."""
    from core.agent_tools import (
        search_literature,
        process_documents,
        extract_prisma_knowledge,
        qdrant_search,
        neo4j_query,
        analyze_evidence,
        draft_section,
        validate_quality,
    )

    return [
        {
            "name": "literature-search",
            "description": "Search academic databases with PICO decomposition and LLM screening.",
            "system_prompt": SUBAGENT_PROMPTS["literature"],
            "tools": [search_literature, validate_quality],
        },
        {
            "name": "data-processing",
            "description": "Process research papers into text chunks.",
            "system_prompt": SUBAGENT_PROMPTS["data_processing"],
            "tools": [process_documents],
        },
        {
            "name": "knowledge-graph",
            "description": "Extract PRISMA entities and build Neo4j/Qdrant stores.",
            "system_prompt": SUBAGENT_PROMPTS["knowledge_graph"],
            "tools": [extract_prisma_knowledge, qdrant_search, neo4j_query],
        },
        {
            "name": "analysis",
            "description": "Analyze evidence with GraphRAG retrieval and LLM synthesis.",
            "system_prompt": SUBAGENT_PROMPTS["analysis"],
            "tools": [analyze_evidence, qdrant_search, neo4j_query],
        },
        {
            "name": "writing",
            "description": "Draft academic sections and detect evidence gaps.",
            "system_prompt": SUBAGENT_PROMPTS["writing"],
            "tools": [draft_section, qdrant_search, validate_quality],
        },
    ]


# ---------------------------------------------------------------------------
# Orchestrator Builder
# ---------------------------------------------------------------------------


def build_orchestrator(
    model: str = "groq:llama-3.3-70b-versatile",
    model_provider: Optional[str] = None,
) -> Any:
    """
    Build the master ReAct orchestrator using LangChain Deep Agents SDK.

    Falls back to LangGraph create_react_agent if deepagents is not installed.

    Args:
        model: Model identifier (e.g. "groq:llama-3.1-70b-versatile").
        model_provider: Optional explicit model provider override.

    Returns:
        A compiled agent ready for .invoke() or .ainvoke().
    """
    from core.agent_tools import qdrant_search, neo4j_query, validate_quality

    subagents = _build_subagent_configs()

    try:
        from deepagents import create_deep_agent

        agent_kwargs: Dict[str, Any] = {
            "system_prompt": ORCHESTRATOR_SYSTEM_PROMPT,
            "tools": [qdrant_search, neo4j_query, validate_quality],
            "subagents": subagents,
        }

        # If provider is explicitly groq, or model string contains groq,
        # instantiate ChatGroq and pass it as the model instance.
        provider = model_provider or ("groq" if "groq" in model.lower() else None)
        model_name = model.split(":")[-1] if ":" in model else model

        if provider == "groq":
            from langchain_groq import ChatGroq

            agent_kwargs["model"] = ChatGroq(
                model=model_name, api_key=os.environ.get("GROQ_API_KEY", "")
            )
        else:
            # Fallback to passing the string
            agent_kwargs["model"] = model.replace(":", "/")

        orchestrator = create_deep_agent(**agent_kwargs)
        logger.info(f"Built ReAct orchestrator with model={model}")
        return orchestrator

    except ImportError:
        logger.warning(
            "deepagents not installed. Falling back to LangGraph ReAct agent."
        )
        return _build_langgraph_react_fallback(model)


def _build_langgraph_react_fallback(model: str = "llama-3.3-70b-versatile") -> Any:
    """Fallback ReAct agent using LangGraph create_react_agent."""
    from core.agent_tools import (
        qdrant_search,
        neo4j_query,
        validate_quality,
        search_literature,
        process_documents,
        extract_prisma_knowledge,
        analyze_evidence,
        draft_section,
    )

    try:
        from langgraph.prebuilt import create_react_agent
        from langchain_groq import ChatGroq

        model_name = model.split(":")[-1] if ":" in model else model
        llm = ChatGroq(
            model=model_name,
            api_key=os.environ.get("GROQ_API_KEY", ""),
        )

        agent = create_react_agent(
            llm,
            [
                qdrant_search,
                neo4j_query,
                search_literature,
                process_documents,
                extract_prisma_knowledge,
                analyze_evidence,
                draft_section,
                validate_quality,
            ],
            prompt=ORCHESTRATOR_SYSTEM_PROMPT,
        )
        logger.info(f"Built LangGraph ReAct fallback with model={model_name}")
        return agent

    except Exception as e:
        logger.error(f"Failed to build ReAct fallback: {e}")
        raise


# ---------------------------------------------------------------------------
# Agentic Pipeline Runner
# ---------------------------------------------------------------------------


async def run_agentic_pipeline(
    project_name: str,
    research_topic: str,
    research_goals: List[str],
    model: str = "groq:llama-3.3-70b-versatile",
) -> Dict[str, Any]:
    """
    Run the research pipeline in agentic mode using ReAct reasoning.

    Args:
        project_name: Name of the research project.
        research_topic: The main research question.
        research_goals: List of specific research objectives.
        model: Model identifier for the orchestrator.

    Returns:
        Dict with the final research results.
    """
    orchestrator = build_orchestrator(model=model)

    user_message = (
        f"Conduct a systematic literature review on: {research_topic}\n\n"
        f"Project: {project_name}\n"
        f"Research Goals:\n"
        + "\n".join(f"  - {g}" for g in research_goals)
        + "\n\nFollow the PRISMA 2020 workflow. Start by searching for "
        "literature, then process, extract, analyze, and write. "
        "Check coverage after extraction and loop back if needed."
    )

    try:
        result = await orchestrator.ainvoke(
            {"messages": [{"role": "user", "content": user_message}]}
        )
        logger.info("Agentic pipeline completed")
        return result
    except AttributeError:
        result = orchestrator.invoke(
            {"messages": [{"role": "user", "content": user_message}]}
        )
        logger.info("Agentic pipeline completed (sync)")
        return result
