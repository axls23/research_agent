import os
import json
import asyncio
import sqlite3
import uuid
import re
from collections import deque
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional
from urllib import error, request
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
from dotenv import load_dotenv

from core.graph import run_research_pipeline
from core.capabilities import all_capabilities, resolve_dispatch
from core.job_queue import DB_PATH as JOB_QUEUE_DB_PATH, enqueue_job, complete_job, fail_job
from core.tools.extraction_tools import extract_paper_text, chunk_text

load_dotenv()

app = FastAPI(title="Research Agent API", version="0.1.0")

# Allow requests from Next.js frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class ChatRequest(BaseModel):
    message: str
    use_harness: bool = True
    rigor_level: Literal["exploratory", "prisma", "cochrane"] = "prisma"
    mode: Literal["agentic", "default", "langgraph"] = "agentic"
    research_goals: List[str] = Field(default_factory=list)
    allow_auto_override: bool = True
    # Opt-in to real external literature search (arXiv/Semantic Scholar/Crossref).
    # Defaults True for the chat flow so requests produce grounded output;
    # run_research_pipeline's own default stays False for other callers.
    enable_external_search: bool = True


class ChatSource(BaseModel):
    title: str
    paper_id: Optional[str] = None
    year: Optional[int] = None
    source_url: Optional[str] = None

class AgentStep(BaseModel):
    agent: str
    action: str
    color: str


class Hyperedge(BaseModel):
    hyperedge_id: str
    principle_name: str
    member_entity_ids: List[str] = Field(default_factory=list)
    domain_jargon: List[str] = Field(default_factory=list)
    weight: float = 0.0
    paper_ids: List[str] = Field(default_factory=list)
    checklist_tags: List[str] = Field(default_factory=list)
    source_entity_labels: List[str] = Field(default_factory=list)


class IsomorphicCluster(BaseModel):
    cluster_id: str
    shared_principle: str
    hyperedge_ids: List[str] = Field(default_factory=list)
    domains: List[str] = Field(default_factory=list)
    similarity_score: float = 0.0
    actionable_insight: Optional[str] = None


class KnowledgeEntity(BaseModel):
    entity_id: str
    label: str
    text: str
    paper_ids: List[str] = Field(default_factory=list)


class ChatResponse(BaseModel):
    content: str
    citations: List[str] = Field(default_factory=list)
    sources: List[ChatSource] = Field(default_factory=list)
    agentSteps: List[AgentStep] = Field(default_factory=list)
    hyperedges: List[Hyperedge] = Field(default_factory=list)
    isomorphicClusters: List[IsomorphicCluster] = Field(default_factory=list)
    knowledgeEntities: List[KnowledgeEntity] = Field(default_factory=list)


_NODE_TO_STAGE = {
    "literature_review": "search",
    "data_processing": "process",
    "knowledge_graph": "process",
    "analysis": "analysis",
    "writing": "writing",
}


def _get_agentic_ollama_model() -> str:
    """Resolve the model used by agentic chat routes to a local provider."""
    configured = os.getenv("AGENTIC_MODEL") or os.getenv("OLLAMA_MODEL") or "qwen2.5:3b"
    configured = configured.strip()
    default_model = os.getenv("OLLAMA_MODEL", "qwen2.5:3b").strip()
    if default_model.lower().startswith("ollama:"):
        default_model = default_model.split(":", 1)[1]

    if configured.lower().startswith("ollama:"):
        return configured
    if ":" in configured:
        prefix = configured.split(":", 1)[0].lower()
        # vllm/llamacpp: local OpenAI-compatible servers (vLLM, llama.cpp),
        # so they stay within the local-first policy.
        if prefix in {"vllm", "llamacpp"}:
            return configured
        if prefix in {"groq", "fast_rlm", "fast-rlm", "openai", "anthropic", "airllm"}:
            return f"ollama:{default_model}"
    return f"ollama:{configured}"


def _read_log_delta(file_path: Path, offset: int) -> tuple[int, List[str]]:
    if not file_path.exists() or not file_path.is_file():
        return offset, []

    try:
        with file_path.open("r", encoding="utf-8", errors="replace") as f:
            f.seek(offset)
            delta = f.read()
            new_offset = f.tell()
    except Exception:
        return offset, []

    if not delta:
        return new_offset, []
    return new_offset, [ln.strip() for ln in delta.splitlines() if ln.strip()]


def _line_to_stage_event(line: str) -> Optional[Dict[str, str]]:
    lowered = line.lower()

    m = re.search(r"eagerly executing \[\s*([^\]]+)\s*\]", lowered)
    if m:
        node = m.group(1).strip()
        stage = _NODE_TO_STAGE.get(node)
        if stage == "search":
            return {"stage": stage, "message": "Literature search started"}
        if stage == "process":
            if node == "data_processing":
                return {"stage": stage, "message": "Data processing and chunking started"}
            if node == "knowledge_graph":
                return {"stage": stage, "message": "Knowledge graph enrichment started"}
            return {"stage": stage, "message": "Processing stage started"}
        if stage == "analysis":
            return {"stage": stage, "message": "Evidence analysis started"}
        if stage == "writing":
            return {"stage": stage, "message": "Writing synthesis started"}

    if "generated" in lowered and "search queries" in lowered:
        return {"stage": "search", "message": line}
    if "total unique papers found" in lowered:
        return {"stage": "search", "message": line}
    if "processed " in lowered and "chunks" in lowered:
        return {"stage": "process", "message": line}
    if "computing embeddings" in lowered or "prisma entities" in lowered:
        return {"stage": "process", "message": line}
    if "llm synthesis failed" in lowered or "analysis" in lowered and "failed" in lowered:
        return {"stage": "analysis", "message": line}
    if "evidence gaps detected" in lowered or "no evidence gaps detected" in lowered:
        return {"stage": "writing", "message": line}

    return None


def _build_chat_payload(result_state: Dict[str, Any]) -> Dict[str, Any]:
    # Extract content
    final_text = ""
    if "draft_sections" in result_state and result_state["draft_sections"]:
        final_text = "\n\n".join(result_state["draft_sections"].values())
    elif result_state.get("analysis_results"):
        summaries = [
            r.get("result_summary", "")
            for r in result_state.get("analysis_results", [])
            if isinstance(r, dict) and r.get("result_summary")
        ]
        if summaries:
            final_text = "\n\n".join(summaries[:2])
    elif result_state.get("papers"):
        # No authored draft yet, but literature-search returned real results --
        # surface those directly instead of a canned line.
        papers = result_state["papers"]
        lines = [f"Found {len(papers)} paper(s) from external literature search:\n"]
        for p in papers[:10]:
            title = p.get("title", "Untitled")
            year = p.get("year")
            year_str = f" ({year})" if year else ""
            abstract = (p.get("abstract") or "").strip()
            snippet = (abstract[:280] + "…") if len(abstract) > 280 else abstract
            lines.append(f"- **{title}**{year_str}\n  {snippet}")
        final_text = "\n\n".join(lines)
    else:
        final_text = "I have completed analyzing the research related to your query."

    # Extract citations
    citations: List[str] = []
    sources: List[Dict[str, Any]] = []
    if "papers" in result_state:
        for p in result_state["papers"][:5]:
            citations.append(p.get("title", "Unknown Source"))
        for p in result_state["papers"][:8]:
            sources.append(
                {
                    "title": p.get("title", "Unknown Source"),
                    "paper_id": p.get("paper_id"),
                    "year": p.get("year"),
                    "source_url": p.get("source_url"),
                }
            )

    # Extract agent steps from audit_log
    agent_steps: List[Dict[str, str]] = []
    if "audit_log" in result_state:
        for log in result_state["audit_log"]:
            agent_name = log.get("agent", "System")
            color = "cyan"
            if "lit" in agent_name.lower() or "search" in agent_name.lower():
                color = "violet"
            elif "write" in agent_name.lower():
                color = "green"

            agent_steps.append(
                {
                    "agent": agent_name,
                    "action": log.get("output_summary", "Processing step"),
                    "color": color,
                }
            )

    # Extract NEXUS hypergraph/isomorphic-mapping data (additive; does not
    # change the existing content/citations/sources/agentSteps shape).
    hyperedges = result_state.get("hyperedges", []) or []
    isomorphic_clusters = result_state.get("isomorphic_clusters", []) or []
    knowledge_entities = result_state.get("knowledge_entities", []) or []

    return {
        "content": final_text,
        "citations": citations,
        "sources": sources,
        "agentSteps": agent_steps,
        "hyperedges": hyperedges,
        "isomorphicClusters": isomorphic_clusters,
        "knowledgeEntities": knowledge_entities,
    }


def _extract_topic_from_chat(message: str) -> str:
    """Derive a clean research topic string from a free-form chat message."""
    text = " ".join((message or "").strip().split())
    if not text:
        return "General Research Topic"

    explicit = re.search(r"(?:research\s+topic|topic)\s*[:=-]\s*(.+)", text, re.IGNORECASE)
    if explicit:
        text = explicit.group(1).strip()

    text = re.sub(
        r"^(can\s+you|could\s+you|please|i\s+want\s+to|i\s+need\s+to|help\s+me)\s+",
        "",
        text,
        flags=re.IGNORECASE,
    )
    text = re.sub(r"^(analy[sz]e|research|review|summari[sz]e|find\s+papers\s+on|find\s+literature\s+on)\s+", "", text, flags=re.IGNORECASE)
    text = re.sub(r"[?.!]+$", "", text).strip()

    if len(text) > 180:
        text = text[:180].rsplit(" ", 1)[0]

    return text or "General Research Topic"


def _build_pipeline_kwargs(req: ChatRequest) -> Dict[str, Any]:
    """Translate a chat request into a run_research_pipeline invocation."""
    use_harness = bool(req.use_harness)
    resolved_rigor = req.rigor_level if use_harness else "exploratory"
    topic = _extract_topic_from_chat(req.message)
    goals = list(req.research_goals or [])
    if not goals:
        goals = [
            f"Summarize key methods and evidence for: {topic}",
            f"Identify major limitations and open challenges for: {topic}",
        ]
        # Preserve full user phrasing when it carries extra intent.
        if req.message.strip() and req.message.strip() != topic:
            goals.insert(0, req.message.strip())

    return {
        "project_name": (
            f"HarnessChat_{uuid.uuid4().hex[:8]}"
            if use_harness
            else f"ChatSession_{uuid.uuid4().hex[:8]}"
        ),
        "research_topic": topic,
        "research_goals": goals,
        "rigor_level": resolved_rigor,
        "interactive": False,
        "allow_auto_override": bool(req.allow_auto_override),
        "mode": req.mode if use_harness else "agentic",
        "agentic_model": _get_agentic_ollama_model(),
        "enable_external_search": bool(req.enable_external_search),
    }


def _tail_lines(file_path: Path, max_lines: int = 120) -> List[str]:
    if not file_path.exists() or not file_path.is_file():
        return []

    lines = deque(maxlen=max_lines)
    try:
        with file_path.open("r", encoding="utf-8", errors="replace") as f:
            for line in f:
                lines.append(line.rstrip("\n"))
    except Exception:
        return []
    return list(lines)


def _check_ollama() -> Dict[str, Any]:
    base = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
    if base.endswith("/v1"):
        base = base[:-3]
    tags_url = f"{base}/api/tags"

    try:
        req = request.Request(tags_url, method="GET")
        with request.urlopen(req, timeout=2.5) as resp:
            payload = json.loads(resp.read().decode("utf-8", errors="replace"))
            models = payload.get("models", []) or []
            names = [m.get("name", "") for m in models if isinstance(m, dict)]
            return {
                "up": True,
                "base_url": base,
                "model_count": len(names),
                "models": names[:8],
            }
    except Exception as e:
        return {
            "up": False,
            "base_url": base,
            "error": str(e),
            "model_count": 0,
            "models": [],
        }


DEEP_AGENT_DEFINITIONS = [
    {
        "name": "deep-reasoner",
        "role": "Complex reasoning",
        "aliases": ["orchestrator", "deep-reasoner", "react"],
    },
    {
        "name": "literature-search",
        "role": "Search databases",
        "aliases": ["literature_review_node", "literature-search", "search"],
    },
    {
        "name": "data-processing",
        "role": "Chunk/process docs",
        "aliases": ["data_processing_node", "data-processing"],
    },
    {
        "name": "knowledge-graph",
        "role": "Build entities/relations",
        "aliases": ["knowledge_graph_node", "knowledge-graph", "neo4j", "vector-search"],
    },
    {
        "name": "analysis",
        "role": "Evidence synthesis",
        "aliases": ["analysis_node", "analysis"],
    },
    {
        "name": "writing",
        "role": "Draft sections",
        "aliases": ["writing_node", "writing"],
    },
]

NODE_AGENT_MAP = {
    "literature_review_node": "literature-search",
    "data_processing_node": "data-processing",
    "knowledge_graph_node": "knowledge-graph",
    "analysis_node": "analysis",
    "writing_node": "writing",
}


def _canonical_subagent(agent_name: str) -> Optional[str]:
    if not agent_name:
        return None

    direct = NODE_AGENT_MAP.get(agent_name)
    if direct:
        return direct

    lowered = agent_name.lower()
    if "literature" in lowered or "search" in lowered:
        return "literature-search"
    if "data_processing" in lowered or "data-processing" in lowered:
        return "data-processing"
    if "knowledge_graph" in lowered or "knowledge-graph" in lowered:
        return "knowledge-graph"
    if "analysis" in lowered:
        return "analysis"
    if "writing" in lowered:
        return "writing"
    if "orchestrator" in lowered or "deep-reasoner" in lowered or "react" in lowered:
        return "deep-reasoner"
    return None


def _load_latest_workflow() -> Dict[str, Any]:
    outputs_dir = Path("outputs")
    candidates = sorted(
        outputs_dir.glob("audit_log_*.json"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )

    if not candidates:
        return {
            "available": False,
            "source_file": None,
            "project_id": None,
            "project_name": None,
            "rigor_level": None,
            "completed_at": None,
            "step_count": 0,
            "steps": [],
            "summary": {},
        }

    latest = candidates[0]
    try:
        payload = json.loads(latest.read_text(encoding="utf-8", errors="replace"))
    except Exception as e:
        return {
            "available": False,
            "source_file": str(latest),
            "project_id": None,
            "project_name": None,
            "rigor_level": None,
            "completed_at": None,
            "step_count": 0,
            "steps": [],
            "summary": {},
            "error": str(e),
        }

    raw_steps = payload.get("audit_log", [])
    if not isinstance(raw_steps, list):
        raw_steps = []

    steps: List[Dict[str, Any]] = []
    for idx, entry in enumerate(raw_steps, start=1):
        if not isinstance(entry, dict):
            continue
        raw_agent = str(entry.get("agent", ""))
        canonical_agent = _canonical_subagent(raw_agent) or raw_agent
        steps.append(
            {
                "order": idx,
                "agent": canonical_agent,
                "agent_raw": raw_agent,
                "action": entry.get("action", ""),
                "timestamp": entry.get("timestamp"),
                "summary": entry.get("output_summary", ""),
            }
        )

    return {
        "available": True,
        "source_file": str(latest),
        "project_id": payload.get("project_id"),
        "project_name": payload.get("project_name"),
        "rigor_level": payload.get("rigor_level"),
        "completed_at": payload.get("completed_at"),
        "step_count": len(steps),
        "steps": steps,
        "summary": payload.get("summary", {}),
    }


def _subagent_runtime(workflow: Dict[str, Any], recent_logs: List[str]) -> List[Dict[str, Any]]:
    runtime: Dict[str, Dict[str, Any]] = {
        spec["name"]: {
            "name": spec["name"],
            "role": spec["role"],
            "live": False,
            "last_seen": None,
            "last_action": None,
            "last_summary": None,
        }
        for spec in DEEP_AGENT_DEFINITIONS
    }

    steps = workflow.get("steps", []) if isinstance(workflow, dict) else []
    if isinstance(steps, list):
        for step in steps:
            if not isinstance(step, dict):
                continue
            name = step.get("agent")
            if name in runtime:
                runtime[name]["live"] = True
                runtime[name]["last_seen"] = step.get("timestamp")
                runtime[name]["last_action"] = step.get("action")
                runtime[name]["last_summary"] = step.get("summary")

    log_blob = "\n".join(recent_logs[-120:]).lower()
    for spec in DEEP_AGENT_DEFINITIONS:
        name = spec["name"]
        aliases = spec.get("aliases", [])
        if any(alias in log_blob for alias in aliases):
            runtime[name]["live"] = True

    # In practice the orchestrator/deep-reasoner may not appear in audit steps
    # even when stage subagents are active. Infer liveness from active pipeline.
    if not runtime["deep-reasoner"]["live"]:
        stage_agents = [
            "literature-search",
            "data-processing",
            "knowledge-graph",
            "analysis",
            "writing",
        ]
        if any(runtime[a]["live"] for a in stage_agents if a in runtime):
            runtime["deep-reasoner"]["live"] = True
            runtime["deep-reasoner"]["last_seen"] = (
                runtime["deep-reasoner"].get("last_seen")
                or datetime.now(timezone.utc).isoformat()
            )
            runtime["deep-reasoner"]["last_action"] = (
                runtime["deep-reasoner"].get("last_action")
                or "orchestration_control"
            )
            runtime["deep-reasoner"]["last_summary"] = (
                runtime["deep-reasoner"].get("last_summary")
                or "Inferred from active downstream deep-agent stages"
            )

    return [runtime[spec["name"]] for spec in DEEP_AGENT_DEFINITIONS]

@app.get("/health")
async def health_check():
    return {"status": "ok"}


@app.get("/api/backend/monitor")
async def backend_monitor(lines: int = 120):
    lines = max(20, min(lines, 400))
    log_path = Path("logs") / "research_agent.log"
    recent = _tail_lines(log_path, max_lines=lines)

    level_counts = {"INFO": 0, "WARNING": 0, "ERROR": 0}
    last_error = None
    last_warning = None
    subagent_line = None

    for line in recent:
        if "[INFO]" in line:
            level_counts["INFO"] += 1
        if "[WARNING]" in line:
            level_counts["WARNING"] += 1
            last_warning = line
        if "[ERROR]" in line:
            level_counts["ERROR"] += 1
            last_error = line
        if "Subagents configured:" in line:
            subagent_line = line

    workflow = _load_latest_workflow()
    subagents = _subagent_runtime(workflow=workflow, recent_logs=recent)

    return {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "backend": {
            "up": True,
            "pid": os.getpid(),
            "cwd": str(Path.cwd()),
            "agentic_model": _get_agentic_ollama_model(),
        },
        "ollama": _check_ollama(),
        "subagents": subagents,
        "workflow": workflow,
        "stats": {
            "log_file": str(log_path),
            "total_lines_returned": len(recent),
            "level_counts": level_counts,
            "last_error": last_error,
            "last_warning": last_warning,
            "subagent_config_line": subagent_line,
        },
        "recent_logs": recent[-80:],
    }


@app.get("/api/capabilities")
async def get_capabilities():
    """Expose the real mosaic capability registry (core.capabilities) to the frontend."""
    capabilities = []
    for cap in all_capabilities():
        capabilities.append(
            {
                "name": cap.name,
                "description": cap.description,
                "tool_names": list(cap.tool_names),
                "model_tier": cap.model_tier,
                "dispatch": resolve_dispatch(cap),
                "catalog_note": cap.catalog_note,
                "version": cap.version,
            }
        )
    return {"capabilities": capabilities}


def _row_to_job_dict(row: tuple) -> Dict[str, Any]:
    job_id, agent_name, payload, status, created_at, updated_at, result = row

    def _try_json(raw: Optional[str]) -> Any:
        if raw is None:
            return None
        try:
            return json.loads(raw)
        except (TypeError, ValueError):
            return raw

    return {
        "id": job_id,
        "agent_name": agent_name,
        "status": status,
        "payload": _try_json(payload),
        "result": _try_json(result),
        "created_at": created_at,
        "updated_at": updated_at,
    }


@app.get("/api/jobs")
async def list_jobs(agent_name: Optional[str] = None, limit: int = 50):
    if not Path(JOB_QUEUE_DB_PATH).exists():
        return {"jobs": []}

    try:
        with sqlite3.connect(JOB_QUEUE_DB_PATH) as conn:
            cursor = conn.cursor()
            if agent_name:
                cursor.execute(
                    "SELECT id, agent_name, payload, status, created_at, updated_at, result "
                    "FROM jobs WHERE agent_name = ? ORDER BY created_at DESC LIMIT ?",
                    (agent_name, limit),
                )
            else:
                cursor.execute(
                    "SELECT id, agent_name, payload, status, created_at, updated_at, result "
                    "FROM jobs ORDER BY created_at DESC LIMIT ?",
                    (limit,),
                )
            rows = cursor.fetchall()
    except sqlite3.Error:
        return {"jobs": []}

    return {"jobs": [_row_to_job_dict(row) for row in rows]}


@app.get("/api/jobs/{job_id}")
async def get_job(job_id: int):
    if not Path(JOB_QUEUE_DB_PATH).exists():
        raise HTTPException(status_code=404, detail="Job not found")

    try:
        with sqlite3.connect(JOB_QUEUE_DB_PATH) as conn:
            cursor = conn.cursor()
            cursor.execute(
                "SELECT id, agent_name, payload, status, created_at, updated_at, result "
                "FROM jobs WHERE id = ?",
                (job_id,),
            )
            row = cursor.fetchone()
    except sqlite3.Error:
        raise HTTPException(status_code=404, detail="Job not found")

    if row is None:
        raise HTTPException(status_code=404, detail="Job not found")

    return _row_to_job_dict(row)


@app.get("/api/chat")
async def chat_endpoint_get():
    return {
        "status": "ok",
        "message": (
            "Use POST /api/chat or /api/chat/stream with JSON body: "
            "{\"message\": \"...\", \"use_harness\": true, "
            "\"rigor_level\": \"prisma\", \"mode\": \"default\"}"
        ),
    }


@app.post("/api/chat/stream")
async def chat_stream_endpoint(req: ChatRequest):
    def _sse(event: str, data: Dict[str, Any]) -> str:
        return f"event: {event}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"

    async def _event_stream():
        pipeline_kwargs = _build_pipeline_kwargs(req)
        mode_tag = (
            f"harness/{pipeline_kwargs['rigor_level']}"
            if req.use_harness
            else "chat/exploratory"
        )

        yield _sse(
            "status",
            {
                "phase": "started",
                "stage": "planner",
                "message": (
                    f"Planning workflow ({mode_tag}) for topic: "
                    f"{pipeline_kwargs.get('research_topic', 'N/A')}"
                ),
            },
        )

        log_path = Path("logs") / "research_agent.log"
        log_offset = log_path.stat().st_size if log_path.exists() else 0
        emitted = set()
        fallback_stages = [
            ("search", "Searching literature sources"),
            ("process", "Processing papers and preparing structured evidence"),
            ("analysis", "Analyzing evidence and synthesizing findings"),
            ("writing", "Drafting the final research response"),
        ]
        stage_rank = {"search": 0, "process": 1, "analysis": 2, "writing": 3}
        fallback_index = 0
        idle_loops = 0

        task = asyncio.create_task(
            run_research_pipeline(**pipeline_kwargs)
        )

        while not task.done():
            emitted_this_loop = False
            log_offset, new_lines = _read_log_delta(log_path, log_offset)
            for line in new_lines:
                evt = _line_to_stage_event(line)
                if not evt:
                    continue
                key = f"{evt['stage']}::{evt['message']}"
                if key in emitted:
                    continue
                emitted.add(key)
                emitted_this_loop = True
                idle_loops = 0
                if evt["stage"] in stage_rank:
                    fallback_index = max(fallback_index, stage_rank[evt["stage"]] + 1)
                yield _sse(
                    "status",
                    {
                        "phase": "stage",
                        "stage": evt["stage"],
                        "message": evt["message"],
                    },
                )

            if not emitted_this_loop:
                idle_loops += 1
                if idle_loops >= 3 and fallback_index < len(fallback_stages):
                    stage, message = fallback_stages[fallback_index]
                    key = f"{stage}::{message}"
                    if key not in emitted:
                        emitted.add(key)
                        yield _sse(
                            "status",
                            {
                                "phase": "stage",
                                "stage": stage,
                                "message": message,
                            },
                        )
                    fallback_index += 1
                    idle_loops = 0
            await asyncio.sleep(0.8)

        try:
            result_state = await task
        except Exception as e:
            yield _sse("error", {"message": str(e)})
            yield _sse("done", {"ok": False})
            return

        payload = _build_chat_payload(result_state)
        content = payload.get("content", "")

        yield _sse(
            "status",
            {
                "phase": "stage",
                "stage": "writing",
                "message": "Final response is being assembled",
            },
        )

        # Stream content in chunks to render progressively in UI.
        chunk_size = 180
        for i in range(0, len(content), chunk_size):
            yield _sse("content", {"delta": content[i : i + chunk_size]})
            await asyncio.sleep(0.02)

        yield _sse(
            "meta",
            {
                "citations": payload.get("citations", []),
                "sources": payload.get("sources", []),
                "agentSteps": payload.get("agentSteps", []),
                "hyperedges": payload.get("hyperedges", []),
                "isomorphicClusters": payload.get("isomorphicClusters", []),
                "knowledgeEntities": payload.get("knowledgeEntities", []),
            },
        )
        yield _sse("done", {"ok": True})

    headers = {
        "Cache-Control": "no-cache",
        "Connection": "keep-alive",
        "X-Accel-Buffering": "no",
    }
    return StreamingResponse(_event_stream(), media_type="text/event-stream", headers=headers)

@app.post("/api/chat", response_model=ChatResponse)
async def chat_endpoint(req: ChatRequest):
    try:
        result_state = await run_research_pipeline(**_build_pipeline_kwargs(req))

        payload = _build_chat_payload(result_state)
        return ChatResponse(**payload)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


def _persist_chunks_to_neo4j(chunks: List[Dict[str, Any]], paper_id: str) -> Dict[str, Any]:
    """Persist Chunk nodes only (no PRISMA entities/hyperedges — that needs the
    LLM-based extractor, which is out of scope for a raw upload). Mirrors the
    MERGE pattern used by knowledge_graph_node._persist_to_neo4j for chunks.
    """
    neo4j_uri = os.environ.get("NEO4J_URI", "bolt://localhost:7687")
    neo4j_user = os.environ.get("NEO4J_USER", "neo4j")
    neo4j_password = os.environ.get("NEO4J_PASSWORD", "")

    if not neo4j_password:
        return {"neo4j_status": "skipped", "reason": "no credentials"}

    try:
        from neo4j import GraphDatabase

        driver = GraphDatabase.driver(neo4j_uri, auth=(neo4j_user, neo4j_password))
        nodes_created = 0

        with driver.session() as session:
            session.run(
                "MERGE (p:Paper {paper_id: $pid}) "
                "ON CREATE SET p.created_at = datetime()",
                pid=paper_id,
            )

            for c in chunks:
                c_text = c.get("text")
                c_emb = c.get("embedding")
                if c_text and c_emb:
                    session.run(
                        "MERGE (ch:Chunk {text: $text}) "
                        "ON CREATE SET ch.paper_id = $pid, ch.embedding = $emb "
                        "ON MATCH SET ch.embedding = $emb",
                        text=c_text, pid=c.get("paper_id", paper_id), emb=c_emb,
                    )
                    nodes_created += 1

        driver.close()
        return {"neo4j_status": "success", "nodes_created": nodes_created}
    except Exception as e:
        return {"neo4j_status": "error", "error": str(e)}


@app.post("/api/upload")
async def upload_endpoint(file: UploadFile = File(...)):
    upload_dir = Path("uploads")
    upload_dir.mkdir(parents=True, exist_ok=True)

    filename = file.filename or f"upload_{uuid.uuid4().hex[:8]}"
    dest_path = upload_dir / filename

    contents = await file.read()
    with open(dest_path, "wb") as f:
        f.write(contents)

    paper_id = Path(filename).stem
    job_id = enqueue_job("upload-ingest", {"filename": filename, "path": str(dest_path)})

    try:
        extraction = await extract_paper_text(str(dest_path))
        full_text = extraction.get("full_text", "")
        chunks = chunk_text(full_text, paper_id) if full_text else []
    except Exception as e:
        fail_job(job_id, str(e))
        raise HTTPException(status_code=500, detail=f"Extraction/chunking failed: {e}")

    # Embedding and Neo4j persistence are best-effort: degrade gracefully
    # (matching _check_ollama()/_persist_to_neo4j() conventions) rather than
    # failing the whole upload if the model or database is unavailable.
    embed_status = "skipped"
    if chunks:
        try:
            from sentence_transformers import SentenceTransformer

            embed_model = SentenceTransformer("allenai/specter2_base")
            chunk_texts = [c["text"] for c in chunks]
            embeddings = embed_model.encode(chunk_texts, show_progress_bar=False)
            for c, emb in zip(chunks, embeddings):
                c["embedding"] = emb.tolist()
            embed_status = "success"
        except Exception as e:
            embed_status = f"error: {e}"

    neo4j_result = _persist_chunks_to_neo4j(chunks, paper_id)

    result = {
        "filename": filename,
        "paper_id": paper_id,
        "chunk_count": len(chunks),
        "extraction_method": extraction.get("method"),
        "embedding_status": embed_status,
        "neo4j": neo4j_result,
    }
    complete_job(job_id, result)

    return {"job_id": job_id, "filename": filename, "status": "COMPLETED"}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
