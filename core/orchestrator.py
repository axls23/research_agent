"""
core/orchestrator.py
=====================
Active LangGraph ReAct orchestrator.
Orchestrates research subagents using the native LangGraph "agents as tools" supervisor pattern.
"""

from __future__ import annotations

import logging
import os
import uuid
from typing import Any, Dict, List, Optional

from langgraph.prebuilt import create_react_agent
from langchain_core.tools import tool

from core.llm_provider import local_only_enabled

logger = logging.getLogger(__name__)


def _enforce_local_model(model: str) -> str:
    if not local_only_enabled():
        return model
    lowered = (model or "").lower()
    if lowered.startswith("groq:") or lowered.startswith("openai:") or lowered.startswith("mistral:"):
        fallback = os.getenv("OLLAMA_MODEL", "qwen2.5:3b")
        logger.warning(
            "Model '%s' blocked by NEXUS_LOCAL_ONLY policy; using ollama:%s",
            model,
            fallback,
        )
        return f"ollama:{fallback}"
    return model


# Maximum ReAct iterations before forcing completion
MAX_REACT_ITERATIONS = 15


# ---------------------------------------------------------------------------
# System Prompts
# ---------------------------------------------------------------------------

ORCHESTRATOR_SYSTEM_PROMPT = """\
You are the NEXUS enterprise orchestrator for dark-data alignment across silos.
Your job is to run a local-first pipeline that ingests proprietary artifacts,
translates jargon into shared principles, and builds actionable graph intelligence.

CRITICAL: You must behaviorally enforce the requested Rigor Level (e.g. PRISMA 2020) across all your subagent delegations. You are the sovereign gatekeeper of methodology.

## Available Subagents
- **literature-search**: Searches external APIs (arXiv, Semantic Scholar) based on research topics.
- **dark-data-ingestion**: Ingests local enterprise artifacts from configured sources.
- **data-processing**: Processes PDF papers into text chunks for analysis.
- **rosetta-core**: Translates silo-specific jargon into core principles.
- **knowledge-graph**: Extracts structured entities, builds Neo4j reasoning graph,
    and stores embeddings using native Neo4j Vectors.
- **analysis**: Performs evidence synthesis, GraphRAG context retrieval, and statistics.
- **writing**: Drafts enterprise-ready findings and opportunity gaps.
- **deep-reasoner**: Performs planning and final validation consistency checks.

## Available Tools
- `neo4j_vector_search(query, prisma_label, limit)` -- Search for semantically similar
    entities in the knowledge base using Neo4j native vector search.
- `neo4j_query(cypher, params)` -- Run Cypher queries against the graph.
- `validate_quality(stage, state_snapshot)` -- Validate pipeline output quality.

## Research Workflow
0. **Plan (Mandatory)**: Call `deep-reasoner` first to create a concise execution plan.
1. **Topic Extraction & Search**: Extract core research topics from the user's prompt and call `literature-search` to hit external APIs.
2. **Ingest**: Ingest dark data from local connectors and file sources.
3. **Process**: Convert found papers into analysable chunks.
4. **Translate**: Run Rosetta core translation to abstract cross-domain principles.
5. **Extract**: Build the knowledge graph from chunks and translated principles.
6. **Assess Coverage**: Use `neo4j_vector_search` to check coverage per core domain.
7. **Analyze**: Run evidence synthesis and pattern detection.
8. **Write**: Draft sections and report opportunity gaps.
9. **Reasoning QA (Mandatory)**: Call `deep-reasoner` to perform a final
    consistency check before returning the answer.

Cap total iterations at {max_iterations}. Document your reasoning at each step.
""".format(
    max_iterations=MAX_REACT_ITERATIONS
)



SUBAGENT_PROMPTS = {
    "literature_search": (
        "You are an academic literature search specialist. Extract research topics from the orchestrator and query external databases (e.g., arXiv, Semantic Scholar) to retrieve papers."
    ),
    "dark_data_ingestion": (
        "You are an enterprise dark-data ingestion specialist. Load local artifacts "
        "from configured connectors and filesystem sources while preserving provenance."
    ),
    "data_processing": (
        "You are a document processing specialist. Your job is to extract text from "
        "enterprise documents and split them into chunks suitable for embedding and "
        "knowledge extraction."
    ),
    "rosetta_core": (
        "You are the Rosetta Core specialist. Translate silo-specific jargon into "
        "shared principles that can be mapped across departments and domains."
    ),
    "knowledge_graph": (
        "You are a knowledge graph specialist. You build structured knowledge "
        "graphs aligned with NEXUS principles. Extract entities (Paper, Objective, "
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
        "evidence gaps."
    ),
    "reasoning": (
        "You are a deep-reasoning agent with the ability to run Python code to solve "
        "complex mathematical, logical, or data-heavy problems. Use your Python sandbox "
        "to verify hypotheses or perform complex calculations when requested."
    ),
}


# ---------------------------------------------------------------------------
# Subagent Factory & Wrapping
# ---------------------------------------------------------------------------


def _build_subagent_configs(global_model: str = "ollama:qwen2.5:3b") -> List[Dict[str, Any]]:
    """Build subagent configuration dicts with resolved local model instances."""
    from core.agent_tools import (
        literature_search,
        ingest_dark_data,
        rosetta_translate,
        process_documents,
        extract_prisma_knowledge,
        neo4j_vector_search,
        neo4j_query,
        analyze_evidence,
        draft_section,
        validate_quality,
    )
    
    global_model = _enforce_local_model(global_model)

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
    elif "vllm" in model_lower:
        from langchain_openai import ChatOpenAI
        vllm_base = os.getenv("VLLM_BASE_URL", "http://localhost:8000/v1")
        vllm_llm = ChatOpenAI(
            model=model_name,
            base_url=vllm_base,
            api_key="vllm"
        )
        subagent_model = vllm_llm
        deep_reasoner_model = vllm_llm
    elif "fast_rlm" in model_lower or "fast-rlm" in model_lower:
        from core.llm_provider import ChatFastRLM

        deep_reasoner_model = ChatFastRLM(
            model_name=os.getenv("RLM_PRIMARY_MODEL", "Qwen/Qwen2.5-1.5B-Instruct"),
            temperature=0.7,
        )
        subagent_model = deep_reasoner_model
    else:
        from langchain_ollama import ChatOllama
        fallback_name = os.getenv("OLLAMA_MODEL", "qwen2.5:3b")
        base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
        if base_url.endswith("/v1"):
            base_url = base_url[:-3]
        ollama_llm = ChatOllama(model=fallback_name, base_url=base_url)
        subagent_model = ollama_llm
        deep_reasoner_model = ollama_llm

    subagents = [
        {
            "name": "deep-reasoner",
            "description": "Call this agent for complex reasoning, math, or planning and final validation consistency checks.",
            "system_prompt": SUBAGENT_PROMPTS["reasoning"],
            "model": deep_reasoner_model,
            "tools": [validate_quality],
        },
        {
            "name": "literature-search",
            "description": "Search external academic databases (arXiv, Semantic Scholar) based on extracted research topics.",
            "system_prompt": SUBAGENT_PROMPTS["literature_search"],
            "model": subagent_model,
            "tools": [literature_search, validate_quality],
        },
        {
            "name": "dark-data-ingestion",
            "description": "Ingest enterprise dark data from local files and staged directories.",
            "system_prompt": SUBAGENT_PROMPTS["dark_data_ingestion"],
            "model": subagent_model,
            "tools": [ingest_dark_data, validate_quality],
        },
        {
            "name": "data-processing",
            "description": "Process staged PDFs or text documents into chunks.",
            "system_prompt": SUBAGENT_PROMPTS["data_processing"],
            "model": subagent_model,
            "tools": [process_documents],
        },
        {
            "name": "rosetta-core",
            "description": "Translate domain-specific terminology/jargon into shared core engineering principles.",
            "system_prompt": SUBAGENT_PROMPTS["rosetta_core"],
            "model": subagent_model,
            "tools": [rosetta_translate],
        },
        {
            "name": "knowledge-graph",
            "description": "Extract entities and construct structured Neo4j reasoning graphs.",
            "system_prompt": SUBAGENT_PROMPTS["knowledge_graph"],
            "model": subagent_model,
            "tools": [extract_prisma_knowledge, neo4j_vector_search, neo4j_query],
        },
        {
            "name": "analysis",
            "description": "Run evidence synthesis and GraphRAG context retrieval.",
            "system_prompt": SUBAGENT_PROMPTS["analysis"],
            "model": subagent_model,
            "tools": [analyze_evidence, neo4j_vector_search, neo4j_query],
        },
        {
            "name": "writing",
            "description": "Draft academic/enterprise review sections.",
            "system_prompt": SUBAGENT_PROMPTS["writing"],
            "model": subagent_model,
            "tools": [draft_section, neo4j_vector_search, validate_quality],
        },
    ]

    summary = []
    for cfg in subagents:
        model_ref = cfg.get("model")
        model_desc = type(model_ref).__name__
        summary.append(f"{cfg.get('name')}={model_desc}")
    logger.info("Subagents configured: %s", ", ".join(summary))

    return subagents


def _make_subagent_tool(cfg: Dict[str, Any]) -> Any:
    """Compile subagent runnable and wrap as a tool for the supervisor orchestrator."""
    name = cfg["name"]
    description = cfg["description"]
    system_prompt = cfg["system_prompt"]
    sub_model = cfg["model"]
    sub_tools = cfg["tools"]

    subagent_runnable = create_react_agent(
        sub_model,
        sub_tools,
        prompt=system_prompt
    )

    def call_subagent(query: str) -> str:
        from core.job_queue import enqueue_job
        job_id = enqueue_job(name, {"query": query})
        return f"Task asynchronously dispatched to {name} subagent. Job ID: {job_id}. Do not wait for the result."

    func_name = name.replace("-", "_")
    call_subagent.__name__ = func_name
    call_subagent.__doc__ = f"Call subagent {name} for task execution. Input query describing target action. Description: {description}"
    
    return tool(call_subagent)


# ---------------------------------------------------------------------------
# Master Orchestrator Builder
# ---------------------------------------------------------------------------


def build_orchestrator(
    model: str = "ollama:qwen2.5:3b",
    model_provider: Optional[str] = None,
) -> Any:
    """
    Build the master ReAct orchestrator using native LangGraph subagents wrapped as tools.
    """
    from core.agent_tools import neo4j_vector_search, neo4j_query, validate_quality

    model = _enforce_local_model(model)
    subagent_configs = _build_subagent_configs(global_model=model)
    subagent_tools = [_make_subagent_tool(cfg) for cfg in subagent_configs]

    model_name = model.split(":", 1)[1] if ":" in model else model
    model_lower = model.lower()

    if "fast_rlm" in model_lower or "fast-rlm" in model_lower:
        model_name = os.getenv("RLM_PRIMARY_MODEL", "Qwen/Qwen2.5-1.5B-Instruct")
        from langchain_openai import ChatOpenAI
        vllm_base = os.getenv("VLLM_BASE_URL", "http://localhost:8000/v1")
        llm = ChatOpenAI(
            model=model_name,
            base_url=vllm_base,
            api_key="dummy"
        )
    elif "vllm" in model_lower:
        from langchain_openai import ChatOpenAI
        vllm_base = os.getenv("VLLM_BASE_URL", "http://localhost:8000/v1")
        llm = ChatOpenAI(
            model=model_name,
            base_url=vllm_base,
            api_key="vllm"
        )
    else:
        from langchain_ollama import ChatOllama
        base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
        if base_url.endswith("/v1"):
            base_url = base_url[:-3]
        llm = ChatOllama(
            model=model_name,
            base_url=base_url,
        )

    # Supervisor agent compiled using native create_react_agent
    orchestrator = create_react_agent(
        llm,
        tools=[neo4j_vector_search, neo4j_query, validate_quality] + subagent_tools,
        prompt=ORCHESTRATOR_SYSTEM_PROMPT,
    )
    logger.info(f"Built native LangGraph ReAct orchestrator with model={model_name}")
    return orchestrator


# ---------------------------------------------------------------------------
# Agentic Pipeline Runner
# ---------------------------------------------------------------------------


async def run_agentic_pipeline(
    project_name: str,
    research_topic: str,
    research_goals: List[str],
    model: Optional[str] = None,
    rigor_level: str = "prisma",
) -> Dict[str, Any]:
    """
    Run the research pipeline in agentic mode using ReAct reasoning.
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
                "fast_rlm",
                "fast-rlm",
                "openai",
                "ollama",
                "vllm",
            }
            if prefix in known_providers and prefix != "ollama":
                model = requested
            elif prefix == "ollama":
                model = requested
            else:
                model = f"ollama:{requested}"
        else:
            model = f"ollama:{requested}"
    else:
        model = f"ollama:{default_ollama_model}"

    model = _enforce_local_model(model)
    logger.info("Agentic model resolved to %s", model)

    resolved_rigor = (rigor_level or "prisma").strip().lower()
    if resolved_rigor not in {"exploratory", "prisma", "cochrane"}:
        logger.warning("Unknown rigor level '%s'; defaulting to prisma", rigor_level)
        resolved_rigor = "prisma"

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
        f"Conduct a NEXUS enterprise alignment run on: {research_topic}\n\n"
        f"Project: {project_name}\n"
        f"Strategic Goals:\n"
        + "\n".join(f"  - {g}" for g in research_goals)
        + f"\n\nRigor Level: {resolved_rigor}\n"
        + "Mandatory execution order: call deep-reasoner first for a short plan, "
          "then run stage subagents, and call deep-reasoner again for final QA. "
        + workflow_instruction
        + " Start by extracting topics for literature search, then ingest local dark data, then process, translate, extract, analyze, and write. "
        "Ensure the PRISMA checklist methodology is strictly enforced in all task delegations. Check coverage after extraction and loop back if needed."
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

    # Hardcoded validation checks removed in favor of behavioral rigor in the Orchestrator prompt.

    if isinstance(result, dict):
        result["agentic_validation"] = stage_summary
    return result
