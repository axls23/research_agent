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
from typing import Any, Dict, List, Literal, Optional

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Literature Search Tool
# ---------------------------------------------------------------------------


async def search_literature(
    topic: str,
    research_goals: List[str],
    min_year: int = 2015,
    max_results_per_db: int = 30,
    snowball: bool = True,
) -> Dict[str, Any]:
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

    agent = LiteratureReviewAgent()

    # Step 1: Formulate queries
    query_result = await agent.process(
        {
            "action": "formulate_search_query",
            "topic": topic,
            "research_goals": research_goals,
            "min_year": min_year,
        }
    )

    # Step 2: Retrieve papers
    retrieval_result = await agent.process(
        {
            "action": "retrieve_papers",
            "queries": query_result.get("queries", [topic]),
            "max_results_per_db": max_results_per_db,
            "snowball": snowball,
        }
    )

    # Step 3: Filter papers
    filter_result = await agent.process(
        {
            "action": "filter_papers",
            "papers": retrieval_result.get("papers", []),
            "topic": topic,
            "research_goals": research_goals,
            "min_year": min_year,
        }
    )

    included = [
        p for p in filter_result.get("filtered_papers", []) if p.get("included", False)
    ]

    return {
        "status": "completed",
        "papers": included,
        "papers_found": len(included),
        "queries": query_result.get("queries", []),
        "search_log": {
            "pico": query_result.get("pico", {}),
            "screening": filter_result.get("screening_log", {}),
        },
    }


# ---------------------------------------------------------------------------
# Document Processing Tool
# ---------------------------------------------------------------------------


async def process_documents(
    papers: List[Dict[str, Any]],
    chunk_size: int = 1000,
    chunk_overlap: int = 200,
) -> Dict[str, Any]:
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

    # Build minimal state for the node
    state = {
        "papers": papers,
        "chunks": [],
        "audit_log": [],
    }

    result = await data_processing_node(state)
    chunks = result.get("chunks", [])

    return {
        "status": "completed",
        "chunks": chunks,
        "chunk_count": len(chunks),
        "papers_processed": len(papers),
    }


# ---------------------------------------------------------------------------
# PRISMA Knowledge Extraction Tool
# ---------------------------------------------------------------------------


async def extract_prisma_knowledge(
    chunks: List[Dict[str, Any]],
    llm: Any = None,
) -> Dict[str, Any]:
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
                        neo4j_status, qdrant_status
    """
    from core.nodes.knowledge_graph_node import knowledge_graph_node

    # Normalize chunks: handle both string and dict inputs
    normalized = []
    for c in chunks:
        if isinstance(c, str):
            normalized.append({"text": c, "chunk_index": len(normalized)})
        else:
            normalized.append(c)

    state = {
        "chunks": normalized,
        "audit_log": [],
    }

    config = {"configurable": {}}
    if llm:
        config["configurable"]["llm_deep"] = llm

    result = await knowledge_graph_node(state, config=config)
    summary = result.get("knowledge_graph_summary", {})

    return {
        "status": "completed",
        "entities": result.get("knowledge_entities", []),
        "entity_count": summary.get("entities_extracted", 0),
        "relations_count": summary.get("relations_extracted", 0),
        "entity_breakdown": summary.get("entity_breakdown", {}),
        "neo4j_status": summary.get("neo4j", {}).get("neo4j_status", "unknown"),
        "qdrant_status": summary.get("qdrant", {}).get("qdrant_status", "unknown"),
    }


# ---------------------------------------------------------------------------
# Qdrant Semantic Search Tool
# ---------------------------------------------------------------------------


def qdrant_search(
    query: str,
    prisma_label: Optional[str] = None,
    limit: int = 10,
) -> List[Dict[str, Any]]:
    """Search Qdrant for semantically similar PRISMA entities.

    Embeds the query using BAAI/bge-small-en-v1.5 and searches the
    research_entities collection. Optionally filters by PRISMA label
    (objective, methodology, result, limitation, implication).

    Args:
        query: Natural language search query.
        prisma_label: Optional PRISMA label to filter results.
        limit: Maximum number of results to return.

    Returns:
        List of matching entity dicts with text, label, score, and metadata.
    """
    qdrant_url = os.environ.get("QDRANT_URL")
    qdrant_api_key = os.environ.get("QDRANT_API_KEY")

    if not qdrant_url:
        logger.warning("QDRANT_URL not set. Cannot search.")
        return []

    try:
        from qdrant_client import QdrantClient
        from qdrant_client.http import models as qmodels
        from sentence_transformers import SentenceTransformer

        client = QdrantClient(url=qdrant_url, api_key=qdrant_api_key)
        model = SentenceTransformer("BAAI/bge-small-en-v1.5")
        query_vector = model.encode([query], show_progress_bar=False)[0].tolist()

        # Build filter if prisma_label specified
        search_filter = None
        if prisma_label:
            search_filter = qmodels.Filter(
                must=[
                    qmodels.FieldCondition(
                        key="prisma_label",
                        match=qmodels.MatchValue(value=prisma_label),
                    )
                ]
            )

        response = client.query_points(
            collection_name="research_entities",
            query=query_vector,
            query_filter=search_filter,
            limit=limit,
        )

        return [
            {
                "text": r.payload.get("text", ""),
                "label": r.payload.get("prisma_label", ""),
                "paper_ids": r.payload.get("paper_ids", []),
                "score": round(r.score, 4),
                "prisma_properties": r.payload.get("prisma_properties", {}),
            }
            for r in response.points
        ]

    except Exception as e:
        logger.error(f"Qdrant search failed: {e}")
        return []


# ---------------------------------------------------------------------------
# Neo4j Graph Query Tool
# ---------------------------------------------------------------------------


def neo4j_query(
    cypher: str,
    params: Optional[Dict[str, Any]] = None,
) -> List[Dict[str, Any]]:
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
    entities: List[Dict[str, Any]],
    topic: str,
    papers: Optional[List[Dict[str, Any]]] = None,
    llm: Any = None,
) -> Dict[str, Any]:
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

    return {
        "status": "completed",
        "results": result.get("analysis_results", []),
        "papers_included": result.get("papers_included", 0),
    }


# ---------------------------------------------------------------------------
# Writing Tool
# ---------------------------------------------------------------------------


async def draft_section(
    topic: str,
    section_name: str = "literature_review",
    entities: Optional[List[Dict[str, Any]]] = None,
    analysis_results: Optional[List[Dict[str, Any]]] = None,
    llm: Any = None,
) -> Dict[str, Any]:
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

    return {
        "status": "completed",
        "outline": result.get("outline", ""),
        "draft_sections": result.get("draft_sections", {}),
        "needs_more_papers": result.get("needs_more_papers", False),
        "gap_analysis": result.get("gap_analysis"),
    }


# ---------------------------------------------------------------------------
# Quality Validation Tool
# ---------------------------------------------------------------------------


async def validate_quality(
    stage: str,
    state_snapshot: Dict[str, Any],
) -> Dict[str, Any]:
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

    state = {**state_snapshot, "current_node": stage, "audit_log": []}

    result = await quality_validator_node(state)

    return {
        "status": "completed",
        "passed": result.get("last_validation_passed", True),
        "score": result.get("validation_score", 0.0),
        "issues": result.get("validation_issues", []),
        "recommendations": result.get("validation_recommendations", []),
    }


# ---------------------------------------------------------------------------
# Registry for Deep Agents
# ---------------------------------------------------------------------------

# Sync tools (can be called directly)
SYNC_TOOLS = [
    qdrant_search,
    neo4j_query,
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
