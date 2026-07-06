"""
core/orchestrator.py
=====================
Active LangGraph ReAct orchestrator.
Orchestrates research subagents using the native LangGraph "agents as tools" supervisor pattern.
"""

from __future__ import annotations

import json
import logging
import os
import time
import uuid
from typing import Any, Dict, List, Optional

from langgraph.prebuilt import create_react_agent
from langchain_core.tools import tool

from core.capabilities import (
    AgentResult,
    all_capabilities,
    make_result,
    render_catalog,
    resolve_dispatch,
    resolve_tools,
)
from core.llm_provider import local_only_enabled

logger = logging.getLogger(__name__)


def _enforce_local_model(model: str) -> str:
    if not local_only_enabled():
        return model
    lowered = (model or "").lower()
    # llama.cpp / vllm are local OpenAI-compatible servers — allowed under air-gap.
    if lowered.startswith("llamacpp:") or lowered.startswith("vllm:"):
        return model
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

_ORCHESTRATOR_PROMPT_TEMPLATE = """\
You are the NEXUS orchestrator: an automated cross-domain scientific laboratory
for deep-tech R&D teams. Your job is to run a local-first pipeline that ingests
proprietary artifacts from different domain silos, translates each silo's jargon
into shared core principles, and builds graph intelligence that surfaces
isomorphic mappings — structurally identical problems and solutions hiding in
different fields — so teams can collaborate instead of reinventing the wheel.

CRITICAL: You must behaviorally enforce the requested Rigor Level (e.g. PRISMA 2020) across all your subagent delegations. You are the sovereign gatekeeper of methodology.

## Available Subagents
{subagent_catalog}

Every subagent call returns a JSON result envelope with fields: agent, status
("ok" | "error" | "queued"), summary, error, job_id, duration_ms. Inspect the
status and summary before deciding the next step; if status is "error", either
retry with a refined query or route around the failure explicitly.

## Available Tools
- `neo4j_vector_search(query, prisma_label, limit)` -- Search for semantically similar
    entities in the knowledge base using Neo4j native vector search.
- `neo4j_query(cypher, params)` -- Run Cypher queries against the graph.
- `validate_quality(stage, state_snapshot)` -- Validate pipeline output quality.

## Research Workflow
0. **Plan (Mandatory)**: Call `deep-reasoner` first to create a concise execution plan.
1. **Ingest (Primary)**: Ingest dark data from local connectors and file sources. Local enterprise artifacts are the system of record.
2. **Enrich (Optional)**: Call `literature-search` ONLY if the run explicitly requests external public literature.
3. **Process**: Convert ingested artifacts into analysable chunks.
4. **Translate**: Run Rosetta core translation to abstract silo jargon into shared cross-domain principles.
5. **Extract**: Build the knowledge graph from chunks and translated principles.
6. **Assess Coverage**: Use `neo4j_vector_search` to check coverage per core domain.
7. **Analyze**: Run evidence synthesis and cross-silo isomorphic pattern detection.
8. **Write**: Draft findings, cross-silo mapping alerts, and opportunity gaps.
9. **Reasoning QA (Mandatory)**: Call `deep-reasoner` to perform a final
    consistency check before returning the answer.

Cap total iterations at {max_iterations}. Document your reasoning at each step.
"""


def _render_orchestrator_prompt() -> str:
    """Render the supervisor prompt from the current capability registry."""
    return _ORCHESTRATOR_PROMPT_TEMPLATE.format(
        subagent_catalog=render_catalog(),
        max_iterations=MAX_REACT_ITERATIONS,
    )


# Rendered once at import for introspection/tests; build_orchestrator()
# re-renders so tiles registered at runtime appear in the catalog.
ORCHESTRATOR_SYSTEM_PROMPT = _render_orchestrator_prompt()


# ---------------------------------------------------------------------------
# Subagent Factory & Wrapping
# ---------------------------------------------------------------------------


def _build_subagent_configs(global_model: str = "ollama:qwen2.5:3b") -> List[Dict[str, Any]]:
    """Build subagent configuration dicts from the capability registry.

    Each dict keeps the legacy shape (name / description / system_prompt /
    model / tools) consumed by the nexus.py worker, plus a "capability"
    key carrying the full AgentCapability tile.
    """
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
    elif "llamacpp" in model_lower:
        from langchain_openai import ChatOpenAI
        llama_base = os.getenv("LLAMACPP_BASE_URL", "http://127.0.0.1:8001/v1")
        llama_llm = ChatOpenAI(
            model=model_name or os.getenv("LLAMACPP_MODEL", "local-model"),
            base_url=llama_base,
            api_key=os.getenv("LLAMACPP_API_KEY", "sk-no-key-required"),
        )
        subagent_model = llama_llm
        deep_reasoner_model = llama_llm
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
            "name": cap.name,
            "description": cap.description,
            "system_prompt": cap.system_prompt,
            "model": deep_reasoner_model if cap.model_tier == "deep" else subagent_model,
            "tools": resolve_tools(cap),
            "capability": cap,
        }
        for cap in all_capabilities()
    ]

    summary = []
    for cfg in subagents:
        model_ref = cfg.get("model")
        model_desc = type(model_ref).__name__
        summary.append(f"{cfg.get('name')}={model_desc}")
    logger.info("Subagents configured: %s", ", ".join(summary))

    return subagents


def _ledger_start(agent_name: str, query: str) -> Optional[int]:
    """Best-effort job-queue ledger entry; never blocks execution."""
    try:
        from core.job_queue import enqueue_job

        return enqueue_job(
            agent_name, {"query": query, "dispatch": "inline"}, status="IN_PROGRESS"
        )
    except Exception as exc:  # noqa: BLE001 — observability must not break runs
        logger.debug("Job ledger unavailable for %s: %s", agent_name, exc)
        return None


def _ledger_finish(job_id: Optional[int], ok: bool, payload: Any) -> None:
    if job_id is None:
        return
    try:
        from core.job_queue import complete_job, fail_job

        if ok:
            complete_job(job_id, {"result": payload})
        else:
            fail_job(job_id, str(payload))
    except Exception as exc:  # noqa: BLE001
        logger.debug("Job ledger update failed for job %s: %s", job_id, exc)


def _last_message_content(response: Any) -> str:
    messages = response.get("messages", []) if isinstance(response, dict) else []
    if not messages:
        return "No response"
    content = getattr(messages[-1], "content", messages[-1])
    if isinstance(content, dict):
        content = content.get("content", content)
    return content if isinstance(content, str) else str(content)


def _execute_subagent_inline(cfg: Dict[str, Any], query: str) -> AgentResult:
    """Run a subagent synchronously and return the structured result contract."""
    name = cfg["name"]
    cap = cfg.get("capability")
    version = cap.version if cap is not None else ""
    job_id = _ledger_start(name, query)
    start = time.perf_counter()

    try:
        runnable = cfg.get("runnable")
        if runnable is None:
            # Compile lazily and cache on the config so repeat calls reuse it.
            runnable = create_react_agent(
                cfg["model"], cfg["tools"], prompt=cfg["system_prompt"]
            )
            cfg["runnable"] = runnable

        response = runnable.invoke({"messages": [("user", query)]})
        summary = _last_message_content(response)
        duration_ms = int((time.perf_counter() - start) * 1000)
        _ledger_finish(job_id, ok=True, payload=summary)
        return make_result(
            agent=name,
            status="ok",
            summary=summary,
            job_id=job_id,
            duration_ms=duration_ms,
            capability_version=version,
        )
    except Exception as exc:  # noqa: BLE001 — supervisor must see the failure
        duration_ms = int((time.perf_counter() - start) * 1000)
        logger.exception("Subagent '%s' failed during inline execution", name)
        _ledger_finish(job_id, ok=False, payload=exc)
        return make_result(
            agent=name,
            status="error",
            summary=f"Subagent {name} failed.",
            error=str(exc),
            job_id=job_id,
            duration_ms=duration_ms,
            capability_version=version,
        )


def _dispatch_subagent_to_queue(cfg: Dict[str, Any], query: str) -> AgentResult:
    """Legacy dispatch: enqueue for an external `nexus.py worker` process."""
    name = cfg["name"]
    cap = cfg.get("capability")
    from core.job_queue import enqueue_job

    job_id = enqueue_job(name, {"query": query})
    return make_result(
        agent=name,
        status="queued",
        summary=(
            f"Task dispatched to the {name} job queue (job {job_id}). "
            "An external worker must process it; results are not available "
            "in this session."
        ),
        job_id=job_id,
        capability_version=cap.version if cap is not None else "",
    )


def _make_subagent_tool(cfg: Dict[str, Any]) -> Any:
    """Wrap a capability tile as a supervisor tool.

    Inline dispatch (the default) executes the subagent synchronously so the
    ReAct supervisor reasons over real outcomes — the agentic contract. Queue
    dispatch preserves the external-worker mode (NEXUS_AGENT_DISPATCH=queue
    or a per-tile override).
    """
    name = cfg["name"]
    description = cfg["description"]

    def call_subagent(query: str) -> str:
        if resolve_dispatch(cfg.get("capability")) == "queue":
            result = _dispatch_subagent_to_queue(cfg, query)
        else:
            result = _execute_subagent_inline(cfg, query)
        return json.dumps(result, ensure_ascii=False, default=str)

    func_name = name.replace("-", "_")
    call_subagent.__name__ = func_name
    call_subagent.__doc__ = (
        f"Call subagent {name} for task execution. Input query describing "
        f"target action. Returns a JSON result envelope (agent, status, "
        f"summary, error, job_id, duration_ms). Description: {description}"
    )

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
    elif "llamacpp" in model_lower:
        from langchain_openai import ChatOpenAI
        llama_base = os.getenv("LLAMACPP_BASE_URL", "http://127.0.0.1:8001/v1")
        llm = ChatOpenAI(
            model=model_name or os.getenv("LLAMACPP_MODEL", "local-model"),
            base_url=llama_base,
            api_key=os.getenv("LLAMACPP_API_KEY", "sk-no-key-required"),
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

    # Supervisor agent compiled using native create_react_agent.
    # Prompt is re-rendered so runtime-registered capability tiles are listed.
    orchestrator = create_react_agent(
        llm,
        tools=[neo4j_vector_search, neo4j_query, validate_quality] + subagent_tools,
        prompt=_render_orchestrator_prompt(),
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
                "llamacpp",
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
        f"Conduct a NEXUS cross-domain discovery run on: {research_topic}\n\n"
        f"Project: {project_name}\n"
        f"Strategic Goals:\n"
        + "\n".join(f"  - {g}" for g in research_goals)
        + f"\n\nRigor Level: {resolved_rigor}\n"
        + "Mandatory execution order: call deep-reasoner first for a short plan, "
          "then run stage subagents, and call deep-reasoner again for final QA. "
        + workflow_instruction
        + " Start by ingesting local dark data, then process, translate jargon into "
        "shared principles, extract the knowledge graph, analyze for cross-silo "
        "isomorphic mappings, and write findings. Use literature-search only if the "
        "goals explicitly require external public literature. "
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
