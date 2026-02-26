"""
core/nodes/analysis_node.py
=============================
LangGraph node that runs analysis on extracted knowledge.
"""

from __future__ import annotations

import logging
import os
from typing import Any, Dict, List

from core.state import ResearchState, append_audit

logger = logging.getLogger(__name__)


async def analysis_node(
    state: ResearchState,
    config: Dict[str, Any] | None = None,
) -> Dict[str, Any]:
    """
    LangGraph node: Analysis

    1. Summarise patterns across papers using LLM
    2. Generate descriptive statistics (paper counts, topic distribution)
    3. Optionally run risk-of-bias assessment
    """
    config = config or {}
    cfgr = config.get("configurable", {})
    llm = cfgr.get("llm_deep") or cfgr.get("llm")  # prefer deep tier

    entities = state.get("knowledge_entities", [])
    papers = state.get("papers", [])
    results = list(state.get("analysis_results", []))

    # ---- Descriptive summary ----
    included = [p for p in papers if p.get("included", True)]
    topic_dist: Dict[str, int] = {}
    for e in entities:
        label = e.get("label", "unknown")
        topic_dist[label] = topic_dist.get(label, 0) + 1

    results.append(
        {
            "method": "descriptive",
            "result_summary": (
                f"{len(included)} papers included, "
                f"{len(entities)} entities extracted, "
                f"distribution: {topic_dist}"
            ),
            "figures": [],
            "tables": [{"entity_distribution": topic_dist}],
            "statistical_output": {
                "papers_included": len(included),
                "entity_count": len(entities),
                "topic_distribution": topic_dist,
            },
        }
    )

    # ---- LLM synthesis (Extended Thinking ) ----
    if llm and entities:
        entity_texts = [e.get("text", "") for e in entities[:50]]
        system = (
            "You are a research analyst performing a systematic review. "
            "Think step-by-step through the following structured analysis:\n\n"
            "1. CATEGORIZE: Group the entities by type (concept, method, "
            "result, dataset). List the categories.\n"
            "2. PATTERNS: Identify recurring themes that appear across "
            "multiple papers. Note frequency.\n"
            "3. CONTRADICTIONS: Note any conflicting findings or "
            "methodological disagreements between papers.\n"
            "4. GAPS: What is missing from the literature? What questions "
            "remain unanswered?\n"
            "5. SYNTHESIS: Write your final structured analysis combining "
            "the above. Be specific and cite entity names.\n\n"
            "Show your reasoning for each step."
        )
        # ---- GraphRAG Integration: Qdrant + Neo4j Context Retrieval ----
        # Fetch deep structural context instead of just a flat list of entities
        graph_context = await _retrieve_graphrag_context(
            topic=state.get("research_topic", ""),
            entities=entities,
        )

        prompt = (
            f"Research topic: {state.get('research_topic', '')}\n"
            f"Papers included: {len(included)}\n\n"
            f"=== Extracted Knowledge ===\n{graph_context}\n"
        )

        try:
            synthesis = await llm.generate(
                prompt,
                system_prompt=system,
                temperature=0.5,
                max_tokens=4096,  # extended budget for step-by-step reasoning
            )
            results.append(
                {
                    "method": "llm_synthesis",
                    "result_summary": synthesis,
                    "figures": [],
                    "tables": [],
                    "statistical_output": None,
                }
            )
        except Exception as e:
            logger.warning(f"LLM synthesis failed: {e}")

    audit_log = append_audit(
        state,
        agent="analysis_node",
        action="run_analysis",
        inputs={"entity_count": len(entities), "paper_count": len(papers)},
        output_summary=f"Produced {len(results)} analysis results",
        provenance={
            "model_tier": "deep",
            "model": getattr(llm, "model", "unknown") if llm else None,
            "methods_run": [r.get("method", "") for r in results],
            "extended_thinking": llm is not None,
        },
    )

    return {
        "current_node": "analysis",
        "analysis_results": results,
        "papers_included": len(included),
        "audit_log": audit_log,
    }


# ---------------------------------------------------------------------------
# GraphRAG Retrieval (Qdrant -> Neo4j)
# ---------------------------------------------------------------------------


async def _retrieve_graphrag_context(topic: str, entities: List[Dict[str, Any]]) -> str:
    """
    1. Embeds the topic and queries Qdrant to find semantic entry points.
    2. Queries Neo4j to find structural 2-hop relationships around those entry points.
    Falls back to a flat list of entities if either DB is unavailable.
    """
    fallback_context = ", ".join([e.get("text", "") for e in entities[:50]])
    qdrant_url = os.environ.get("QDRANT_URL")
    qdrant_api_key = os.environ.get("QDRANT_API_KEY")
    neo4j_uri = os.environ.get("NEO4J_URI", "bolt://localhost:7687")
    neo4j_user = os.environ.get("NEO4J_USER", "neo4j")
    neo4j_password = os.environ.get("NEO4J_PASSWORD", "")

    if not qdrant_url or not neo4j_password:
        return fallback_context  # Fallback to local entities

    try:
        from qdrant_client import QdrantClient
        from sentence_transformers import SentenceTransformer
        from neo4j import GraphDatabase

        # Step 1: Semantic Entry Points (Qdrant)
        client = QdrantClient(url=qdrant_url, api_key=qdrant_api_key)
        model = SentenceTransformer("BAAI/bge-small-en-v1.5")
        query_vector = model.encode([topic], show_progress_bar=False)[0].tolist()

        search_result = client.search(
            collection_name="research_entities",
            query_vector=query_vector,
            limit=5,
        )
        entry_texts = [r.payload.get("text") for r in search_result if r.payload]

        if not entry_texts:
            return fallback_context

        # Step 2: Structural Traversal (Neo4j)
        driver = GraphDatabase.driver(neo4j_uri, auth=(neo4j_user, neo4j_password))
        graph_traces = []

        with driver.session() as session:
            for text in entry_texts:
                # 2-hop traversal query
                result = session.run(
                    "MATCH (start {text: $text})-[r1]->(mid)-[r2]->(end) "
                    "RETURN start.text AS s, type(r1) AS r1_type, mid.text AS m, "
                    "type(r2) AS r2_type, end.text AS e LIMIT 5",
                    text=text,
                )
                for record in result:
                    path = f"({record['s']}) -[{record['r1_type']}]-> ({record['m']}) -[{record['r2_type']}]-> ({record['e']})"
                    graph_traces.append(path)

                # Fallback purely to 1-hop if 2-hop is sparse
                if not graph_traces:
                    result1 = session.run(
                        "MATCH (start {text: $text})-[r1]->(mid) "
                        "RETURN start.text AS s, type(r1) AS r1_type, mid.text AS m LIMIT 5",
                        text=text,
                    )
                    for record in result1:
                        path = (
                            f"({record['s']}) -[{record['r1_type']}]-> ({record['m']})"
                        )
                        graph_traces.append(path)

        driver.close()

        if not graph_traces:
            return f"Entry Points: {', '.join(entry_texts)}"

        trace_blocks = "\\n".join([f"- {t}" for t in set(graph_traces)])
        return f"Semantic Entry Points:\\n{', '.join(entry_texts)}\\n\\nReasoning Paths (Neo4j Graph):\\n{trace_blocks}"

    except Exception as e:
        logger.warning(f"GraphRAG retrieval failed, using fallback: {e}")
        return fallback_context
