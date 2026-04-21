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
import uuid
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
  the Neo4j reasoning graph, and stores embeddings using native Neo4j Vectors.
- **analysis**: Analyzes extracted evidence -- descriptive stats, LLM synthesis,
  and GraphRAG context retrieval.
- **writing**: Drafts academic sections and detects evidence gaps.

## Available Tools
- `neo4j_vector_search(query, prisma_label, limit)` -- Search for semantically similar
  entities in the PRISMA knowledge base using Neo4j native vector search. Filter by label (objective, methodology,
  result, limitation, implication).
- `neo4j_query(cypher, params)` -- Run Cypher queries against the PRISMA graph.
  Use for structured reasoning paths like:
  `MATCH (p:Paper)-[:REPORTS_FINDING]->(r:Result) RETURN r.text`
- `validate_quality(stage, state_snapshot)` -- Validate pipeline output quality.

## Research Workflow
0. **Plan (Mandatory)**: Call `deep-reasoner` first to create a concise execution plan
    before delegating retrieval and extraction tasks.
1. **Search**: Delegate literature search with PICO-decomposed queries
2. **Process**: Convert found papers into analysable chunks
3. **Extract**: Build PRISMA knowledge graph from chunks
4. **Assess Coverage**: Use neo4j_vector_search to check coverage per PRISMA domain.
   If any domain has < 3 entities, search for MORE papers on that specific domain.
5. **Analyze**: Run evidence synthesis and pattern detection
6. **Write**: Draft sections, check for evidence gaps
7. **Reasoning QA (Mandatory)**: Call `deep-reasoner` to perform a final
    consistency check before returning the answer.
8. **Iterate**: If gaps found, loop back to search with refined queries

## Key Principles
- Always check coverage BEFORE analysis. Use neo4j_vector_search with prisma_label filters.
- Treat `deep-reasoner` as the control-plane reasoning subagent:
    use it for planning and final validation.
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
        "(neo4j_vector_search -> neo4j_query) to build rich context before synthesis."
    ),
    "writing": (
        "You are an academic writing specialist drafting sections for a systematic review. "
        "Produce well-structured, evidence-based academic text. After drafting, check for "
        "evidence gaps that may require additional literature retrieval."
    ),
    "reasoning": (
        "You are a deep-reasoning agent with the ability to run Python code to solve "
        "complex mathematical, logical, or data-heavy problems. Use your Python sandbox "
        "to verify hypotheses or perform complex calculations when requested."
    ),
}


# ---------------------------------------------------------------------------
# Subagent Factory
# ---------------------------------------------------------------------------


def _build_subagent_configs(global_model: str = "ollama:qwen2.5:3b") -> List[Dict[str, Any]]:
    """Build subagent configuration dicts for create_deep_agent."""
    from core.agent_tools import (
        search_literature,
        process_documents,
        extract_prisma_knowledge,
        neo4j_vector_search,
        neo4j_query,
        analyze_evidence,
        draft_section,
        validate_quality,
    )
    
    # Route all subagents through the selected global model by default,
    # using concrete chat model instances where possible.
    subagent_model: Any = global_model
    deep_reasoner_model: Any = global_model

    model_name = global_model.split(":", 1)[1] if ":" in global_model else global_model
    model_lower = global_model.lower()

    if "ollama" in model_lower:
        from langchain_ollama import ChatOllama

        base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
        if base_url.endswith("/v1"):
            base_url = base_url[:-3]
        ollama_llm = ChatOllama(model=model_name, base_url=base_url)
        subagent_model = ollama_llm
        deep_reasoner_model = ollama_llm
    elif "groq" in model_lower:
        from langchain_groq import ChatGroq

        groq_llm = ChatGroq(model=model_name, api_key=os.environ.get("GROQ_API_KEY", ""))
        subagent_model = groq_llm
        deep_reasoner_model = groq_llm
    elif "fast_rlm" in model_lower or "fast-rlm" in model_lower:
        from core.llm_provider import ChatFastRLM

        deep_reasoner_model = ChatFastRLM(
            model_name=os.getenv("RLM_PRIMARY_MODEL", "Qwen/Qwen2.5-0.5B"),
            temperature=0.7,
        )
        subagent_model = deep_reasoner_model

    subagents = [
        {
            "name": "deep-reasoner",
            "description": "Call this agent for complex reasoning, math, or tasks requiring Python code execution.",
            "system_prompt": SUBAGENT_PROMPTS["reasoning"],
            "model": deep_reasoner_model,
            "tools": [validate_quality],
        },
        {
            "name": "literature-search",
            "description": "Search academic databases with PICO decomposition and LLM screening.",
            "system_prompt": SUBAGENT_PROMPTS["literature"],
            "model": subagent_model,
            "tools": [search_literature, validate_quality],
        },
        {
            "name": "data-processing",
            "description": "Process research papers into text chunks.",
            "system_prompt": SUBAGENT_PROMPTS["data_processing"],
            "model": subagent_model,
            "tools": [process_documents],
        },
        {
            "name": "knowledge-graph",
            "description": "Extract PRISMA entities and build Neo4j reasoning graphs.",
            "system_prompt": SUBAGENT_PROMPTS["knowledge_graph"],
            "model": subagent_model,
            "tools": [extract_prisma_knowledge, neo4j_vector_search, neo4j_query],
        },
        {
            "name": "analysis",
            "description": "Analyze evidence with GraphRAG retrieval and LLM synthesis.",
            "system_prompt": SUBAGENT_PROMPTS["analysis"],
            "model": subagent_model,
            "tools": [analyze_evidence, neo4j_vector_search, neo4j_query],
        },
        {
            "name": "writing",
            "description": "Draft academic sections and detect evidence gaps.",
            "system_prompt": SUBAGENT_PROMPTS["writing"],
            "model": subagent_model,
            "tools": [draft_section, neo4j_vector_search, validate_quality],
        },
    ]

    # Visibility for presentation/debugging: confirms subagents are configured and live.
    summary = []
    for cfg in subagents:
        model_ref = cfg.get("model")
        if isinstance(model_ref, str):
            model_desc = model_ref
        else:
            model_desc = type(model_ref).__name__
        summary.append(f"{cfg.get('name')}={model_desc}")
    logger.info("Subagents configured: %s", ", ".join(summary))

    return subagents


# ---------------------------------------------------------------------------
# Orchestrator Builder
# ---------------------------------------------------------------------------


def build_orchestrator(
    model: str = "ollama:qwen2.5:3b",
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
    from core.agent_tools import neo4j_vector_search, neo4j_query, validate_quality

    subagents = _build_subagent_configs(global_model=model)

    try:
        # LiteLLM (used by deepagents) does not support AirLLM natively.
        # Force fallback to our custom LangGraph wrapper for AirLLM.
        if "airllm" in model.lower():
            logger.info("AirLLM requested. Bypassing deepagents to use custom LangGraph wrapper.")
            return _build_langgraph_react_fallback(model)
            
        from deepagents import create_deep_agent
        
        agent_kwargs: Dict[str, Any] = {
            "system_prompt": ORCHESTRATOR_SYSTEM_PROMPT,
            "tools": [neo4j_vector_search, neo4j_query, validate_quality],
            "subagents": subagents,
        }

        # If provider is explicitly groq, or model string contains groq,
        # instantiate ChatGroq and pass it as the model instance.
        provider = model_provider
        if provider is None:
            if "groq" in model.lower():
                provider = "groq"
            elif "ollama" in model.lower():
                provider = "ollama"
        model_name = model.split(":", 1)[1] if ":" in model else model

        if provider == "groq":
            from langchain_groq import ChatGroq
            # DeepAgents handles raw model classes passing
            llm = ChatGroq(
                model=model_name, 
                api_key=os.environ.get("GROQ_API_KEY", "")
            )
            agent_kwargs["model"] = llm
        elif "fast_rlm" in model.lower() or "fast-rlm" in model.lower():
            # If fast_rlm requested, run the ORCHESTRATOR on vLLM (fast)
            # but it has access to the deep-reasoner subagent (RLM) created in configs.
            model_name = os.getenv("RLM_PRIMARY_MODEL", "Qwen/Qwen2.5-0.5B")
            from langchain_openai import ChatOpenAI
            llm = ChatOpenAI(
                model=model_name,
                base_url="http://172.30.177.136:8000/v1",
                api_key="dummy"
            )
            agent_kwargs["model"] = llm
        elif provider == "ollama" or model.lower().startswith("ollama:"):
            from langchain_ollama import ChatOllama
            base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
            if base_url.endswith("/v1"):
                base_url = base_url[:-3]
            llm = ChatOllama(
                model=model_name,
                base_url=base_url,
            )
            agent_kwargs["model"] = llm
        else:

            # Fallback to passing the generic string for liteLLM
            agent_kwargs["model"] = model.replace(":", "/")

        orchestrator = create_deep_agent(**agent_kwargs)
        logger.info(f"Built ReAct orchestrator with model={model}")
        return orchestrator

    except ImportError:
        logger.warning(
            "deepagents not installed. Falling back to LangGraph ReAct agent."
        )
        return _build_langgraph_react_fallback(model)


def _build_langgraph_react_fallback(model: str = "ollama:qwen2.5:3b") -> Any:
    """Fallback ReAct agent using LangGraph create_react_agent."""
    from core.agent_tools import (
        neo4j_vector_search,
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
        
        model_name = model.split(":", 1)[1] if ":" in model else model
        
        if "airllm" in model.lower():
            from langchain_core.language_models.chat_models import BaseChatModel
            from langchain_core.messages import BaseMessage, AIMessage
            from langchain_core.outputs import ChatResult, ChatGeneration
            from pydantic import PrivateAttr
            from typing import Optional, Any
            import asyncio
            
            class ChatAirLLM(BaseChatModel):
                model_name: str
                _provider: Any = PrivateAttr()

                def __init__(self, model_name: str, **kwargs):
                    super().__init__(model_name=model_name, **kwargs)
                    from core.llm_provider import AirLLMProvider
                    self._provider = AirLLMProvider(model=model_name)

                def bind_tools(self, tools: Any, **kwargs: Any) -> Any:
                    return self

                def _generate(self, messages: list[BaseMessage], stop: Optional[list[str]] = None, run_manager: Optional[Any] = None, **kwargs: Any) -> ChatResult:
                    prompt = "\n".join([f"{m.type}: {m.content}" for m in messages])
                    try:
                        loop = asyncio.get_event_loop()
                    except RuntimeError:
                        loop = asyncio.new_event_loop()
                        asyncio.set_event_loop(loop)
                    
                    if loop.is_running():
                        import nest_asyncio
                        nest_asyncio.apply()
                        
                    response = loop.run_until_complete(self._provider.generate(prompt=prompt))
                    message = AIMessage(content=response)
                    return ChatResult(generations=[ChatGeneration(message=message)])
                
                async def _agenerate(self, messages: list[BaseMessage], stop: Optional[list[str]] = None, run_manager: Optional[Any] = None, **kwargs: Any) -> ChatResult:
                    prompt = "\n".join([f"{m.type}: {m.content}" for m in messages])
                    response = await self._provider.generate(prompt=prompt)
                    message = AIMessage(content=response)
                    return ChatResult(generations=[ChatGeneration(message=message)])

                @property
                def _llm_type(self) -> str:
                    return "chat-airllm"

            llm = ChatAirLLM(model_name=model_name)
        elif "fast_rlm" in model.lower() or "fast-rlm" in model.lower():
            # Fallback orchestrator uses direct vLLM for the skeleton
            model_name = os.getenv("RLM_PRIMARY_MODEL", "Qwen/Qwen2.5-0.5B")
            from langchain_openai import ChatOpenAI
            llm = ChatOpenAI(
                model=model_name,
                base_url="http://172.30.177.136:8000/v1",
                api_key="dummy"
            )
        elif "ollama" in model.lower():
            from langchain_ollama import ChatOllama
            base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
            if base_url.endswith("/v1"):
                base_url = base_url[:-3]
            llm = ChatOllama(
                model=model_name,
                base_url=base_url,
            )
        else:
            from langchain_groq import ChatGroq
            llm = ChatGroq(
                model=model_name,
                api_key=os.environ.get("GROQ_API_KEY", ""),
            )

        agent = create_react_agent(
            llm,
            [
                neo4j_vector_search,
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
    model: Optional[str] = None,
    rigor_level: str = "exploratory",
) -> Dict[str, Any]:
    """
    Run the research pipeline in agentic mode using ReAct reasoning.

    Args:
        project_name: Name of the research project.
        research_topic: The main research question.
        research_goals: List of specific research objectives.
        model: Optional override model identifier.
    """
    configured_model = None
    try:
        import yaml
        from pathlib import Path

        path = Path("config/config.yaml")
        if path.exists():
            with open(path, encoding="utf-8") as f:
                cfg = yaml.safe_load(f) or {}
            configured_model = (
                cfg.get("llm", {}).get("tiers", {}).get("deep", {}).get("model")
                or cfg.get("llm", {}).get("model")
            )
    except Exception:
        configured_model = None

    default_ollama_model = os.getenv("OLLAMA_MODEL") or configured_model or "qwen2.5:3b"

    if model:
        requested = model.strip()
        lowered = requested.lower()

        if lowered.startswith("ollama:"):
            model = requested
        elif ":" in requested:
            prefix = requested.split(":", 1)[0].lower()
            known_providers = {
                "groq",
                "fast_rlm",
                "fast-rlm",
                "openai",
                "anthropic",
                "airllm",
                "ollama",
            }
            if prefix in known_providers and prefix != "ollama":
                logger.warning(
                    "Non-ollama agentic model '%s' requested; forcing Ollama model 'ollama:%s'.",
                    requested,
                    default_ollama_model,
                )
                model = f"ollama:{default_ollama_model}"
            elif prefix == "ollama":
                model = requested
            else:
                # Handles model tags like qwen2.5:3b that are valid Ollama names.
                model = f"ollama:{requested}"
        else:
            model = f"ollama:{requested}"
    else:
        model = f"ollama:{default_ollama_model}"

    logger.info("Agentic model resolved to %s", model)

    resolved_rigor = (rigor_level or "exploratory").strip().lower()
    if resolved_rigor not in {"exploratory", "prisma", "cochrane"}:
        logger.warning("Unknown rigor level '%s'; defaulting to exploratory", rigor_level)
        resolved_rigor = "exploratory"

    from core.agent_tools import begin_agentic_run, finish_agentic_run

    run_id = str(uuid.uuid4())
    previous_run_id = os.getenv("AGENTIC_RUN_ID")
    previous_rigor = os.getenv("RESEARCH_AGENT_RIGOR")

    begin_agentic_run(run_id)
    os.environ["AGENTIC_RUN_ID"] = run_id
    os.environ["RESEARCH_AGENT_RIGOR"] = resolved_rigor
    os.environ.setdefault("AGENTIC_FAIL_CLOSED", "true")

    orchestrator = build_orchestrator(model=model)

    workflow_instruction = (
        "Follow a rapid exploratory workflow."
        if resolved_rigor == "exploratory"
        else (
            "Follow the Cochrane workflow with strict methodological checks."
            if resolved_rigor == "cochrane"
            else "Follow the PRISMA 2020 workflow with strict validation gates."
        )
    )

    user_message = (
        f"Conduct a systematic literature review on: {research_topic}\n\n"
        f"Project: {project_name}\n"
        f"Research Goals:\n"
        + "\n".join(f"  - {g}" for g in research_goals)
        + f"\n\nRigor Level: {resolved_rigor}\n"
                + "Mandatory execution order: call deep-reasoner first for a short plan, "
                    "then run stage subagents, and call deep-reasoner again for final QA. "
        + workflow_instruction
        + " Start by searching for "
        "literature, then process, extract, analyze, and write. "
        "Check coverage after extraction and loop back if needed."
    )

    stage_summary: Dict[str, Any] = {"started_at": None, "stages": {}}
    try:
        try:
            result = await orchestrator.ainvoke(
                {"messages": [{"role": "user", "content": user_message}]}
            )
            logger.info("Agentic pipeline completed")
        except AttributeError:
            result = orchestrator.invoke(
                {"messages": [{"role": "user", "content": user_message}]}
            )
            logger.info("Agentic pipeline completed (sync)")
    finally:
        stage_summary = finish_agentic_run(run_id)

        if previous_run_id is None:
            os.environ.pop("AGENTIC_RUN_ID", None)
        else:
            os.environ["AGENTIC_RUN_ID"] = previous_run_id

        if previous_rigor is None:
            os.environ.pop("RESEARCH_AGENT_RIGOR", None)
        else:
            os.environ["RESEARCH_AGENT_RIGOR"] = previous_rigor

    fail_closed = (os.getenv("AGENTIC_FAIL_CLOSED") or "true").strip().lower() not in {
        "0",
        "false",
        "no",
        "off",
    }
    if resolved_rigor in {"prisma", "cochrane"} and fail_closed:
        required_stages = ["literature_review", "data_processing", "analysis"]
        stages = stage_summary.get("stages", {})

        missing_stages = [stage for stage in required_stages if stage not in stages]
        failed_stages: List[str] = []

        for stage in required_stages:
            stage_info = stages.get(stage) or {}
            validation = stage_info.get("validation") or {}
            if not validation.get("passed", False):
                issues = validation.get("issues") or ["validation did not pass"]
                failed_stages.append(f"{stage}: {'; '.join(str(i) for i in issues)}")

        if missing_stages or failed_stages:
            detail_parts = []
            if missing_stages:
                detail_parts.append(f"missing stages={missing_stages}")
            if failed_stages:
                detail_parts.append(f"failed validations={failed_stages}")
            raise RuntimeError(
                "Fail-closed agentic rigor enforcement triggered: "
                + " | ".join(detail_parts)
            )

    if isinstance(result, dict):
        result["agentic_validation"] = stage_summary
    return result
