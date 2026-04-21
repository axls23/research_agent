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
        # Load the NEXUS ontology rules directly from the skill file so the runtime LLM understands the schema
        skill_rules = ""
        skill_path = os.path.join(os.path.dirname(__file__), "..", "..", ".agents", "skills", "nexus-neo4j-mapper", "SKILL.md")
        try:
            if os.path.exists(skill_path):
                with open(skill_path, "r", encoding="utf-8") as f:
                    skill_rules = f"\n\n=== ONTOLOGY RULES ===\n{f.read()}"
        except Exception as e:
            logger.warning(f"Failed to load NEXUS skill rules: {e}")

        system = (
            "You are the NEXUS Engine traversing a knowledge hypergraph. "
            "Think step-by-step through the following structured analysis:\n\n"
            "1. HYPEREDGE IDENTIFICATION: List the abstract 'Core Principles' discovered across the papers.\n"
            "2. ISOMORPHIC CLUSTERING: Identify situations where different domains or papers share the SAME Hyperedge (Core Principle). "
            "Highlight these structural intersections explicitly.\n"
            "3. CROSS-DOMAIN SOLUTIONS: Where Domains A and B share an isomorphic cluster, can a METHOD in Domain A "
            "solve a LIMITATION in Domain B? Propose actionable cross-silo insights.\n"
            "4. SYNTHESIS: Write your final structured action report combining "
            "the above into a 'Concept Canvas'. Be specific and cite the exact terms.\n\n"
            f"Show your reasoning for each step.{skill_rules}"
        )
        # ---- GraphRAG Integration: Neo4j Vector + Structural Traversal ----
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
    1. Embeds the topic and queries Neo4j vector index to find semantic entry points.
    2. Queries Neo4j to find structural 2-hop relationships AND Hyperedge Intersections.
    Falls back to a flat list of entities if Neo4j is unavailable.
    """
    fallback_context = ", ".join([e.get("text", "") for e in entities[:50]])
    neo4j_uri = os.environ.get("NEO4J_URI", "bolt://localhost:7687")
    neo4j_user = os.environ.get("NEO4J_USER", "neo4j")
    neo4j_password = os.environ.get("NEO4J_PASSWORD", "")

    if not neo4j_password:
        return fallback_context  # Fallback to local entities

    try:
        from sentence_transformers import SentenceTransformer
        from neo4j import GraphDatabase

        # Step 1: Semantic Entry Points (Neo4j Vector Search)
        model = SentenceTransformer("allenai/specter2_base")
        query_vector = model.encode([topic], show_progress_bar=False)[0].tolist()

        driver = GraphDatabase.driver(neo4j_uri, auth=(neo4j_user, neo4j_password))
        entry_texts = []
        graph_traces = []

        with driver.session() as session:
            # Query Neo4j Native Vector Index
            vector_result = session.run(
                "CALL db.index.vector.queryNodes('prisma_embeddings', 5, $query_vector) "
                "YIELD node, score "
                "RETURN node.text AS text "
                "ORDER BY score DESC",
                query_vector=query_vector
            )
            for record in vector_result:
                if record["text"]:
                    entry_texts.append(record["text"])

            if not entry_texts:
                driver.close()
                return fallback_context

            # Step 2: Structural Traversal (Neo4j)
            for text in entry_texts:
                
                # NEXUS Hyperedge Intersection Query (Isomorphic Mapping)
                iso_result = session.run(
                    "MATCH (e1 {text: $text})-[:IN_HYPEREDGE]->(h:Hyperedge)<-[:IN_HYPEREDGE]-(e2) "
                    "WHERE elementId(e1) <> elementId(e2) "
                    "RETURN e1.text AS source, h.principle_name AS principle, e2.text AS target LIMIT 10",
                    text=text,
                )
                for record in iso_result:
                    path = f"[ISOMORPHIC MATCH]: '{record['source']}' AND '{record['target']}' SHARE PRINCIPLE: '{record['principle']}'"
                    graph_traces.append(path)

                # Standard 2-hop traversal query (fallback context)
                result = session.run(
                    "MATCH (start {text: $text})-[r1]->(mid)-[r2]->(end) "
                    "RETURN start.text AS s, type(r1) AS r1_type, mid.text AS m, "
                    "type(r2) AS r2_type, end.text AS e LIMIT 5",
                    text=text,
                )
                for record in result:
                    path = f"({record['s']}) -[{record['r1_type']}]-> ({record['m']}) -[{record['r2_type']}]-> ({record['e']})"
                    graph_traces.append(path)

                # Fallback purely to 1-hop if graph is sparse
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
