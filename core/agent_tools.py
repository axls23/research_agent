"""
core/agent_tools.py
====================
Tool wrappers that expose LangGraph node capabilities as callable
functions for the Deep Agents SDK.

Each tool wraps existing node logic so the ReAct orchestrator can
invoke them through the tool-calling interface. Tools are plain
functions with docstrings that describe their purpose and parameters
(Deep Agents auto-generates schemas from these).
"""

from __future__ import annotations

import json
import logging
import os
from datetime import datetime, timezone
from threading import Lock
from typing import Any, Dict, List, Literal, Optional

logger = logging.getLogger(__name__)


_VALIDATED_AGENTIC_STAGES = {
    "literature_review",
    "data_processing",
    "analysis",
}
_GREY_LITERATURE_SOURCES = {
    "clinicaltrials",
    "clinicaltrials.gov",
    "opengrey",
    "medrxiv",
    "biorxiv",
    "who_ictrp",
    "trialregister",
}

from core.state import ResearchState

_AGENTIC_RUNS: Dict[str, Dict[str, Any]] = {}
_AGENTIC_RUNS_LOCK = Lock()

_AGENTIC_STATE: ResearchState = {
    "papers": [],
    "chunks": [],
    "knowledge_entities": [],
    "hyperedges": [],
    "audit_log": [],
}
_AGENTIC_STATE_LOCK = Lock()


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _resolve_rigor_level() -> str:
    raw = (os.getenv("RESEARCH_AGENT_RIGOR") or "exploratory").strip().lower()
    if raw in {"prisma", "cochrane", "exploratory"}:
        return raw
    return "exploratory"


def _fail_closed_enabled(rigor_level: str) -> bool:
    if rigor_level == "exploratory":
        return False
    flag = (os.getenv("AGENTIC_FAIL_CLOSED") or "true").strip().lower()
    return flag not in {"0", "false", "no", "off"}


def begin_agentic_run(run_id: str) -> None:
    """Initialize per-run validation tracking used by strict agentic mode."""
    if not run_id:
        return
    with _AGENTIC_RUNS_LOCK:
        _AGENTIC_RUNS[run_id] = {
            "started_at": _utc_now_iso(),
            "stages": {},
        }


def finish_agentic_run(run_id: str) -> Dict[str, Any]:
    """Return and clear per-run validation tracking data."""
    if not run_id:
        return {"started_at": None, "stages": {}}

    with _AGENTIC_RUNS_LOCK:
        return _AGENTIC_RUNS.pop(run_id, {"started_at": None, "stages": {}})


def _record_stage_event(
    stage: str,
    payload: Dict[str, Any],
    validation: Optional[Dict[str, Any]] = None,
) -> None:
    run_id = (os.getenv("AGENTIC_RUN_ID") or "").strip()
    if not run_id:
        return

    summary: Dict[str, Any] = {"status": payload.get("status", "unknown")}
    for key in (
        "papers_found",
        "papers_processed",
        "chunk_count",
        "entity_count",
        "relations_count",
        "papers_included",
    ):
        if key in payload:
            summary[key] = payload.get(key)

    with _AGENTIC_RUNS_LOCK:
        run_record = _AGENTIC_RUNS.setdefault(
            run_id,
            {"started_at": _utc_now_iso(), "stages": {}},
        )
        run_record["stages"][stage] = {
            "updated_at": _utc_now_iso(),
            "summary": summary,
            "validation": validation,
        }


def _format_validation_failure(stage: str, validation: Dict[str, Any]) -> str:
    issues = validation.get("issues") or ["Unknown validation failure"]
    joined = "; ".join(str(issue) for issue in issues)
    return (
        f"Fail-closed validation blocked stage '{stage}'. "
        f"Issues: {joined}"
    )


async def _validate_stage_or_raise(
    stage: str,
    state_snapshot: Dict[str, Any],
) -> Dict[str, Any]:
    """Run gate validation for non-exploratory rigor and return structured errors instead of failing."""
    rigor_level = _resolve_rigor_level()

    if stage not in _VALIDATED_AGENTIC_STAGES or rigor_level == "exploratory":
        return {
            "status": "skipped",
            "passed": True,
            "score": 1.0,
            "issues": [],
            "recommendations": [],
        }

    snapshot = dict(state_snapshot)
    snapshot["rigor_level"] = rigor_level

    validation = await validate_quality(stage=stage, state_snapshot=snapshot)

    if _fail_closed_enabled(rigor_level) and not validation.get("passed", False):
        error_msg = _format_validation_failure(stage, validation)
        logger.warning(f"Validation failed for stage {stage}: {error_msg}")
        return {
            "status": "error",
            "error": error_msg,
            "passed": False,
            "issues": validation.get("issues", []),
            "recommendations": validation.get("recommendations", [])
        }

    return validation


# ---------------------------------------------------------------------------
# Literature Search Tool
# ---------------------------------------------------------------------------


async def search_literature(
    topic: str,
    research_goals: list[str],
    min_year: str = "2015",
    max_results_per_db: str = "5",
    snowball: str = "true",
) -> dict:
    """Search academic databases for papers relevant to a research topic.

    Uses PICO decomposition to formulate boolean queries, searches
    ArXiv, Semantic Scholar, and Crossref in parallel, deduplicates
    results, and optionally snowballs citations from top papers.

    Args:
        topic: The research question or topic to search for.
        research_goals: List of specific research objectives.
        min_year: Earliest publication year to include.
        max_results_per_db: Maximum papers per database per query.
        snowball: Whether to follow citations from top papers.

    Returns:
        Dict with keys: status, papers, papers_found, queries, search_log
    """
    from agents.literature_review_agent import LiteratureReviewAgent
    from core.llm_provider import create_llm_from_config

    llm = None
    try:
        llm = create_llm_from_config()
    except Exception as e:
        logger.warning(f"Failed to initialize LLM for literature search tool: {e}")

    agent = LiteratureReviewAgent(llm=llm)

    # Step 1: Formulate queries
    query_result = await agent.process(
        {
            "action": "formulate_search_query",
            "topic": topic,
            "research_goals": research_goals,
            "min_year": int(min_year),
        }
    )

    # Step 2: Retrieve papers
    retrieval_result = await agent.process(
        {
            "action": "retrieve_papers",
            "queries": query_result.get("queries", [topic]),
            "max_results_per_db": int(max_results_per_db),
            "snowball": snowball.lower() == "true",
        }
    )

    # Step 3: Filter papers
    filter_result = await agent.process(
        {
            "action": "filter_papers",
            "papers": retrieval_result.get("papers", []),
            "topic": topic,
            "research_goals": research_goals,
            "min_year": int(min_year),
        }
    )

    included = [
        p for p in filter_result.get("filtered_papers", []) if p.get("included", False)
    ]

    with _AGENTIC_STATE_LOCK:
        _AGENTIC_STATE["papers"].extend(included)

    databases_searched = retrieval_result.get("databases_searched", [])
    try:
        min_year_int = int(min_year)
    except (TypeError, ValueError):
        min_year_int = 2015
    search_date_range = {
        "min_year": min_year_int,
        "max_year": datetime.now(timezone.utc).year,
    }

    validation = await _validate_stage_or_raise(
        stage="literature_review",
        state_snapshot={
            "rigor_level": _resolve_rigor_level(),
            "search_queries": query_result.get("queries", []),
            "databases_searched": databases_searched,
            "papers_found": len(included),
            "papers": included,
            "search_date_range": search_date_range,
            "grey_literature_searched": any(
                (db or "").lower() in _GREY_LITERATURE_SOURCES
                for db in databases_searched
            ),
            "audit_log": [
                {
                    "timestamp": _utc_now_iso(),
                    "agent": "search_literature_tool",
                    "action": "search_literature",
                    "input_hash": "tool",
                    "output_summary": f"Found {len(included)} included papers",
                    "provenance": {
                        "search_date_range": search_date_range,
                        "databases_searched": databases_searched,
                    },
                }
            ],
        },
    )

    payload = {
        "status": "completed",
        "papers": included,
        "papers_found": len(included),
        "queries": query_result.get("queries", []),
        "databases_searched": databases_searched,
        "search_date_range": search_date_range,
        "validation": validation,
        "search_log": {
            "pico": query_result.get("pico", {}),
            "screening": filter_result.get("screening_log", {}),
        },
    }

    _record_stage_event("literature_review", payload, validation=validation)
    return payload


# ---------------------------------------------------------------------------
# Document Processing Tool
# ---------------------------------------------------------------------------


async def process_documents(
    papers: list[dict],
    chunk_size: str = "1000",
    chunk_overlap: str = "200",
) -> dict:
    """Process research papers into text chunks for downstream analysis.

    Extracts text from PDF documents, cleans and normalises content,
    then splits into overlapping chunks suitable for embedding and
    knowledge extraction.

    Args:
        papers: List of paper dicts with metadata (title, abstract, etc.).
        chunk_size: Target size of each text chunk in characters.
        chunk_overlap: Overlap between consecutive chunks.

    Returns:
        Dict with keys: status, chunks, chunk_count, papers_processed
    """
    from core.nodes.data_processing_node import data_processing_node

    with _AGENTIC_STATE_LOCK:
        combined_papers = list(papers) if papers else []
        # Add papers that are in global state but not in the passed list (by id/title)
        existing_ids = {p.get("paper_id") for p in combined_papers if p.get("paper_id")}
        for p in _AGENTIC_STATE["papers"]:
            if p.get("paper_id") and p["paper_id"] not in existing_ids:
                combined_papers.append(p)

    # Build minimal state for the node
    state = {
        "papers": combined_papers,
        "chunks": [],
        "audit_log": [],
    }

    result = await data_processing_node(state)
    chunks = result.get("chunks", [])
    processed_papers = result.get("papers", combined_papers)

    with _AGENTIC_STATE_LOCK:
        _AGENTIC_STATE["chunks"].extend(chunks)

    dual_extraction_performed = bool(result.get("dual_extraction_performed", False))
    if not dual_extraction_performed:
        dual_extraction_performed = any(
            bool((paper.get("annotations") or {}).get("dual_extraction"))
            for paper in processed_papers
            if isinstance(paper, dict)
        )

    validation = await _validate_stage_or_raise(
        stage="data_processing",
        state_snapshot={
            "rigor_level": _resolve_rigor_level(),
            "papers": processed_papers,
            "chunks": chunks,
            "total_tokens_extracted": result.get("total_tokens_extracted", 0),
            "dual_extraction_performed": dual_extraction_performed,
        },
    )

    payload = {
        "status": "completed",
        "chunks": chunks,
        "chunk_count": len(chunks),
        "papers_processed": len(papers),
        "dual_extraction_performed": dual_extraction_performed,
        "validation": validation,
    }

    _record_stage_event("data_processing", payload, validation=validation)
    return payload


# ---------------------------------------------------------------------------
# PRISMA Knowledge Extraction Tool
# ---------------------------------------------------------------------------


async def extract_prisma_knowledge(
    chunks: list[dict],
    llm: Optional[str] = None,
) -> dict:
    """Extract PRISMA-aligned knowledge entities from text chunks.

    Runs a 2-tier extraction pipeline:
      Tier 1: GLiNER zero-shot NER for grounded entity spans
      Tier 2: LLM + Pydantic for structured PRISMA property extraction

    Persists results to Neo4j (reasoning graph) and Qdrant (semantic retrieval).

    Args:
        chunks: List of text chunk dicts with 'text' and 'paper_id' keys.
        llm: Optional LLM provider for Tier 2 extraction.

    Returns:
        Dict with keys: status, entities, relations, entity_count,
                        neo4j_status
    """
    from core.nodes.knowledge_graph_node import knowledge_graph_node

    # Normalize chunks: handle both string and dict inputs
    normalized = []
    for c in chunks:
        if isinstance(c, str):
            normalized.append({"text": c, "chunk_index": len(normalized)})
        else:
            normalized.append(c)

    with _AGENTIC_STATE_LOCK:
        combined_chunks = list(normalized) if normalized else []
        # Fall back to global state chunks if none provided or to augment
        existing_texts = {c.get("text", "") for c in combined_chunks}
        for c in _AGENTIC_STATE["chunks"]:
            if c.get("text", "") not in existing_texts:
                combined_chunks.append(c)

    # Convert state to ResearchState expected by graph
    state = {
        "chunks": combined_chunks,
        "rigor_level": _resolve_rigor_level(),
        "audit_log": [],
    }

    config = {"configurable": {}}
    if llm:
        config["configurable"]["llm_deep"] = llm

    result = await knowledge_graph_node(state, config=config)
    
    with _AGENTIC_STATE_LOCK:
        _AGENTIC_STATE["knowledge_entities"].extend(result.get("knowledge_entities", []))
        _AGENTIC_STATE["hyperedges"].extend(result.get("hyperedges", []))
        if "audit_log" in result and result["audit_log"]:
            _AGENTIC_STATE["audit_log"].extend(result["audit_log"])

    summary = result.get("knowledge_graph_summary", {})

    payload = {
        "status": "completed",
        "entities": result.get("knowledge_entities", []),
        "entity_count": summary.get("entities_extracted", 0),
        "relations_count": summary.get("relations_extracted", 0),
        "entity_breakdown": summary.get("entity_breakdown", {}),
        "neo4j_status": summary.get("neo4j", {}).get("neo4j_status", "unknown"),
    }

    rigor_level = _resolve_rigor_level()
    if (
        rigor_level in {"prisma", "cochrane"}
        and _fail_closed_enabled(rigor_level)
        and payload["entity_count"] <= 0
    ):
        raise RuntimeError(
            "Fail-closed validation blocked stage 'knowledge_graph'. "
            "No PRISMA entities were extracted."
        )

    _record_stage_event(
        "knowledge_graph",
        payload,
        validation={
            "status": "completed",
            "passed": payload["entity_count"] > 0,
            "score": 1.0 if payload["entity_count"] > 0 else 0.0,
            "issues": [] if payload["entity_count"] > 0 else ["No entities extracted"],
            "recommendations": [],
        },
    )
    return payload


# ---------------------------------------------------------------------------
# Neo4j Semantic Search Tool
# ---------------------------------------------------------------------------


def neo4j_vector_search(
    query: str,
    prisma_label: Optional[str] = None,
    limit: str = "10",
) -> list[dict]:
    """Search Neo4j using native vector indexing for semantically similar PRISMA entities.

    Embeds the query using allenai/specter2_base and searches the
    prisma_embeddings vector index. Optionally filters by PRISMA label
    (objective, methodology, result, limitation, implication) using post-filtering.

    Args:
        query: Natural language search query.
        prisma_label: Optional PRISMA label to filter results.
        limit: Maximum number of results to return.

    Returns:
        List of matching entity dicts with text, label, score, and metadata.
    """
    neo4j_uri = os.environ.get("NEO4J_URI", "bolt://localhost:7687")
    neo4j_user = os.environ.get("NEO4J_USER", "neo4j")
    neo4j_password = os.environ.get("NEO4J_PASSWORD", "")

    if not neo4j_password:
        logger.warning("NEO4J_PASSWORD not set. Cannot run vector search.")
        return []

    try:
        from sentence_transformers import SentenceTransformer
        from neo4j import GraphDatabase

        model = SentenceTransformer("allenai/specter2_base")
        query_vector = model.encode([query], show_progress_bar=False)[0].tolist()

        driver = GraphDatabase.driver(neo4j_uri, auth=(neo4j_user, neo4j_password))
        records = []

        query_str = (
            "CALL db.index.vector.queryNodes('prisma_embeddings', $limit, $query_vector) "
            "YIELD node, score "
        )
        if prisma_label:
            query_str += "WHERE node.prisma_label = $prisma_label "

        query_str += "RETURN node, score ORDER BY score DESC"

        with driver.session() as session:
            result = session.run(
                query_str,
                limit=int(limit) * 2 if prisma_label else int(limit),
                query_vector=query_vector,
                prisma_label=prisma_label
            )

            for item in result:
                node = item["node"]
                records.append({
                    "text": node.get("text", ""),
                    "label": node.get("prisma_label", ""),
                    "paper_ids": node.get("paper_ids", []),
                    "score": round(item["score"], 4),
                    "prisma_properties": json.loads(node.get("prisma_properties", "{}")) if isinstance(node.get("prisma_properties"), str) else node.get("prisma_properties", {}),
                })
        
        driver.close()

        if prisma_label:
             records = records[:int(limit)]

        return records

    except Exception as e:
        logger.error(f"Neo4j vector search failed: {e}")
        return []


# ---------------------------------------------------------------------------
# Neo4j Graph Query Tool
# ---------------------------------------------------------------------------


def neo4j_query(
    cypher: str,
    params: Optional[dict] = None,
) -> list[dict]:
    """Execute a Cypher query against the Neo4j PRISMA knowledge graph.

    Use this to traverse the PRISMA ontology and find reasoning paths.
    The graph contains nodes: Paper, Objective, Methodology, Result,
    Limitation, Implication with relationships: INVESTIGATES,
    UTILIZES_METHOD, REPORTS_FINDING, HAS_LIMITATION, SUPPORTS_IMPLICATION.

    Args:
        cypher: A valid Cypher query string.
        params: Optional dict of query parameters.

    Returns:
        List of result records as dicts.
    """
    neo4j_uri = os.environ.get("NEO4J_URI", "bolt://localhost:7687")
    neo4j_user = os.environ.get("NEO4J_USER", "neo4j")
    neo4j_password = os.environ.get("NEO4J_PASSWORD", "")

    if not neo4j_password:
        logger.warning("NEO4J_PASSWORD not set. Cannot query.")
        return []

    try:
        from neo4j import GraphDatabase

        driver = GraphDatabase.driver(neo4j_uri, auth=(neo4j_user, neo4j_password))
        records = []

        with driver.session() as session:
            result = session.run(cypher, parameters=params or {})
            for record in result:
                records.append(dict(record))

        driver.close()
        return records

    except Exception as e:
        logger.error(f"Neo4j query failed: {e}")
        return []


# ---------------------------------------------------------------------------
# Analysis Tool
# ---------------------------------------------------------------------------


async def analyze_evidence(
    entities: list[dict],
    topic: str,
    papers: Optional[list[dict]] = None,
    llm: Optional[str] = None,
) -> dict:
    """Analyze extracted knowledge entities to find patterns and synthesize.

    Runs descriptive statistics, LLM-based synthesis with extended
    thinking, and GraphRAG context retrieval (Qdrant → Neo4j).

    Args:
        entities: List of PRISMA knowledge entities.
        topic: The research topic for contextualised analysis.
        papers: Optional list of paper dicts for counts.
        llm: Optional LLM provider for synthesis.

    Returns:
        Dict with keys: status, results, papers_included
    """
    from core.nodes.analysis_node import analysis_node

    state = {
        "knowledge_entities": entities,
        "papers": papers or [],
        "research_topic": topic,
        "analysis_results": [],
        "audit_log": [],
    }

    config = {"configurable": {}}
    if llm:
        config["configurable"]["llm_deep"] = llm

    result = await analysis_node(state, config=config)

    validation = await _validate_stage_or_raise(
        stage="analysis",
        state_snapshot={
            "rigor_level": _resolve_rigor_level(),
            "analysis_results": result.get("analysis_results", []),
            "knowledge_entities": entities,
            "papers": papers or [],
        },
    )

    payload = {
        "status": "completed",
        "results": result.get("analysis_results", []),
        "papers_included": result.get("papers_included", 0),
        "validation": validation,
    }

    _record_stage_event("analysis", payload, validation=validation)
    return payload


# ---------------------------------------------------------------------------
# Writing Tool
# ---------------------------------------------------------------------------


async def draft_section(
    topic: str,
    section_name: str = "literature_review",
    entities: Optional[list[dict]] = None,
    analysis_results: Optional[list[dict]] = None,
    llm: Optional[str] = None,
) -> dict:
    """Draft an academic section using extracted knowledge and analysis.

    Generates outlines, drafts key sections (Introduction, Literature
    Review, Methods, Results), and checks for evidence gaps that may
    require additional literature retrieval.

    Args:
        topic: The research topic.
        section_name: Which section to draft.
        entities: Optional knowledge entities for context.
        analysis_results: Optional analysis results for synthesis.
        llm: Optional LLM provider.

    Returns:
        Dict with keys: status, outline, draft_sections, needs_more_papers,
                        gap_analysis
    """
    from core.nodes.writing_node import writing_node

    state = {
        "research_topic": topic,
        "knowledge_entities": entities or [],
        "analysis_results": analysis_results or [],
        "draft_sections": {},
        "outline": None,
        "backtrack_count": 0,
        "audit_log": [],
    }

    config = {"configurable": {}}
    if llm:
        config["configurable"]["llm_deep"] = llm

    result = await writing_node(state, config=config)

    payload = {
        "status": "completed",
        "outline": result.get("outline", ""),
        "draft_sections": result.get("draft_sections", {}),
        "needs_more_papers": result.get("needs_more_papers", False),
        "gap_analysis": result.get("gap_analysis"),
    }

    _record_stage_event("writing", payload)
    return payload


# ---------------------------------------------------------------------------
# Quality Validation Tool
# ---------------------------------------------------------------------------


async def validate_quality(
    stage: str,
    state_snapshot: dict,
) -> dict:
    """Run quality validation on pipeline output at a given stage.

    Checks whether the pipeline output meets quality thresholds
    appropriate for the current rigor level (exploratory, prisma, cochrane).

    Args:
        stage: Pipeline stage being validated (e.g. 'literature_review',
               'data_processing', 'analysis').
        state_snapshot: Current pipeline state dict.

    Returns:
        Dict with keys: status, passed, score, issues, recommendations
    """
    from core.nodes.quality_validator_node import quality_validator_node

    state = {
        **state_snapshot,
        "current_node": stage,
        "audit_log": list(state_snapshot.get("audit_log", [])),
    }

    result = await quality_validator_node(state)

    return {
        "status": "completed",
        "passed": result.get("last_validation_passed", True),
        "score": result.get("validation_score", 0.0),
        "issues": result.get("validation_issues", []),
        "recommendations": result.get("validation_recommendations", []),
    }


# ---------------------------------------------------------------------------
# Skill Reading Tool
# ---------------------------------------------------------------------------


def read_skill(skill_name: str) -> str:
    """Read the instructions for a specific NEXUS or system skill.

    Use this tool when you need to understand the formal rules, ontology,
    or expected formats for a particular domain before writing queries or code.
    For example, reading 'nexus-neo4j-mapper' before writing Cypher queries.

    Args:
        skill_name: The name of the skill (e.g., 'nexus-neo4j-mapper').

    Returns:
        The markdown text containing the exact rules and instructions for the skill.
    """
    skill_path = os.path.join(
        os.path.dirname(__file__), "..", ".agents", "skills", skill_name, "SKILL.md"
    )
    
    try:
        if os.path.exists(skill_path):
            with open(skill_path, "r", encoding="utf-8") as f:
                return f.read()
        else:
            return f"Skill '{skill_name}' not found at {skill_path}. Cannot load rules."
    except Exception as e:
        logger.error(f"Failed to read skill '{skill_name}': {e}")
        return f"Error reading skill: {e}"


# ---------------------------------------------------------------------------
# Registry for Deep Agents
# ---------------------------------------------------------------------------

# Sync tools (can be called directly)
SYNC_TOOLS = [
    neo4j_vector_search,
    neo4j_query,
    read_skill,
]

# Async tools (need await)
ASYNC_TOOLS = [
    search_literature,
    process_documents,
    extract_prisma_knowledge,
    analyze_evidence,
    draft_section,
    validate_quality,
]

ALL_TOOLS = SYNC_TOOLS + ASYNC_TOOLS
