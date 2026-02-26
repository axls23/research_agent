"""
core/nodes/knowledge_graph_node.py
===================================
LangGraph node — PRISMA 2020-aligned knowledge graph construction.

Pipeline:
  1. Tier 1 (GLiNER) — zero-shot NER for grounded entity spans (local CPU)
  2. Tier 2 (LLM + Pydantic) — structured PRISMA extraction via Groq (remote)
  3. Neo4j — persist PRISMA ontology as reasoning graph
  4. Qdrant — embed entities with PRISMA-tagged payloads for semantic retrieval
  5. JSON export — for visualization
"""

from __future__ import annotations

import json
import logging
import os
import uuid
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from core.state import ResearchState, append_audit, KnowledgeEntity
from core.nodes.prisma_extractor import (
    extract_prisma_entities,
    extract_entities_gliner,
    PRISMA_ENTITY_TYPES,
    PRISMA_RELATION_MAP,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Neo4j graph persistence — PRISMA ontology
# ---------------------------------------------------------------------------


def _persist_to_neo4j(
    entities: List[Dict[str, Any]],
    relations: List[Tuple[str, str, str]],
    paper_id: str = "",
) -> Dict[str, Any]:
    """
    Persist PRISMA entities and relations to Neo4j.

    Creates nodes with PRISMA labels (Paper, Objective, Methodology, Result,
    Limitation, Implication) and PRISMA relationships (INVESTIGATES,
    UTILIZES_METHOD, REPORTS_FINDING, HAS_LIMITATION, SUPPORTS_IMPLICATION).

    Falls back gracefully if Neo4j is not available.
    """
    neo4j_uri = os.environ.get("NEO4J_URI", "bolt://localhost:7687")
    neo4j_user = os.environ.get("NEO4J_USER", "neo4j")
    neo4j_password = os.environ.get("NEO4J_PASSWORD", "")

    if not neo4j_password:
        logger.warning(
            "NEO4J_PASSWORD not set. Skipping Neo4j persistence. "
            "Set NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD to enable."
        )
        return {"neo4j_status": "skipped", "reason": "no credentials"}

    try:
        from neo4j import GraphDatabase

        driver = GraphDatabase.driver(neo4j_uri, auth=(neo4j_user, neo4j_password))
        nodes_created = 0
        rels_created = 0

        with driver.session() as session:
            # Ensure Paper node exists
            if paper_id:
                session.run(
                    "MERGE (p:Paper {paper_id: $pid}) "
                    "ON CREATE SET p.created_at = datetime()",
                    pid=paper_id,
                )
                nodes_created += 1

            # Create PRISMA entity nodes
            for ent in entities:
                label = ent["label"].capitalize()
                text = ent["text"]
                prisma_props = ent.get("prisma_properties", {})

                # Use parameterized label via APOC or fallback to safe labels
                if label in {
                    "Paper",
                    "Objective",
                    "Methodology",
                    "Result",
                    "Limitation",
                    "Implication",
                }:
                    # MERGE entity node with PRISMA properties
                    props_json = json.dumps(prisma_props) if prisma_props else "{}"
                    session.run(
                        f"MERGE (e:{label} {{text: $text}}) "
                        f"ON CREATE SET e.entity_id = $eid, "
                        f"  e.paper_ids = $pids, "
                        f"  e.prisma_properties = $props "
                        f"ON MATCH SET e.paper_ids = "
                        f"  [x IN e.paper_ids + $pids WHERE x IS NOT NULL]",
                        text=text,
                        eid=ent["entity_id"],
                        pids=ent["paper_ids"],
                        props=props_json,
                    )
                    nodes_created += 1

            # Create PRISMA relationships
            for subj_text, rel_type, obj_text in relations:
                # Validate relationship type
                if rel_type in {
                    "INVESTIGATES",
                    "UTILIZES_METHOD",
                    "REPORTS_FINDING",
                    "HAS_LIMITATION",
                    "SUPPORTS_IMPLICATION",
                }:
                    session.run(
                        f"MATCH (a {{text: $subj}}) "
                        f"MATCH (b {{text: $obj}}) "
                        f"MERGE (a)-[r:{rel_type}]->(b) "
                        f"ON CREATE SET r.weight = 1 "
                        f"ON MATCH SET r.weight = r.weight + 1",
                        subj=subj_text,
                        obj=obj_text,
                    )
                    rels_created += 1

        driver.close()
        logger.info(
            f"Neo4j: created/merged {nodes_created} PRISMA nodes, "
            f"{rels_created} relationships"
        )
        return {
            "neo4j_status": "success",
            "nodes_created": nodes_created,
            "relationships_created": rels_created,
        }

    except Exception as e:
        logger.error(f"Neo4j persistence failed: {e}")
        return {"neo4j_status": "error", "error": str(e)}


# ---------------------------------------------------------------------------
# Qdrant + Transformer vector embedding — PRISMA-tagged payloads
# ---------------------------------------------------------------------------


def _embed_and_store_qdrant(
    entities: List[Dict[str, Any]],
    vector_size: int = 384,
) -> Dict[str, Any]:
    """
    Generate embeddings for PRISMA entities and upsert to Qdrant.

    Payloads include PRISMA label, prisma_properties, and paper_ids
    to enable filtered semantic search by PRISMA domain.

    Uses BAAI/bge-small-en-v1.5 (local CPU, no API calls).
    """
    if not entities:
        return {"qdrant_status": "skipped", "reason": "no entities"}

    try:
        from qdrant_client import QdrantClient
        from qdrant_client.http import models

        # Prepare texts for embedding
        texts = [ent["text"] for ent in entities if ent.get("text")]
        if not texts:
            logger.warning("No texts to embed, skipping Qdrant")
            return {"qdrant_status": "skipped", "reason": "no entity texts"}

        from sentence_transformers import SentenceTransformer

        logger.info("Loading BAAI/bge-small-en-v1.5 embedding model...")
        model = SentenceTransformer("BAAI/bge-small-en-v1.5")

        logger.info(f"Computing embeddings for {len(texts)} PRISMA entities...")
        embeddings = model.encode(texts, show_progress_bar=False)

        # Build points with PRISMA-enriched payloads
        points = []
        valid_entities = [ent for ent in entities if ent.get("text")]
        for i, (ent, embedding) in enumerate(zip(valid_entities, embeddings)):
            payload = {
                "text": ent["text"],
                "label": ent.get("label", "objective"),
                "prisma_label": ent.get("label", "objective"),
                "paper_ids": ent.get("paper_ids", []),
                "entity_id": ent.get("entity_id", ""),
            }
            # Add PRISMA properties to payload for filtered queries
            prisma_props = ent.get("prisma_properties", {})
            if prisma_props:
                payload["prisma_properties"] = prisma_props

            points.append(
                models.PointStruct(
                    id=i,
                    vector=embedding.tolist(),
                    payload=payload,
                )
            )

        # Connect to Qdrant
        qdrant_url = os.environ.get("QDRANT_URL")
        qdrant_api_key = os.environ.get("QDRANT_API_KEY")
        collection_name = "research_entities"

        if qdrant_url:
            client = QdrantClient(url=qdrant_url, api_key=qdrant_api_key)
        else:
            client = QdrantClient(location=":memory:")
            logger.info(
                "Using in-memory Qdrant (set QDRANT_URL for persistent storage)"
            )

        # Recreate collection
        client.recreate_collection(
            collection_name=collection_name,
            vectors_config=models.VectorParams(
                size=vector_size,
                distance=models.Distance.COSINE,
            ),
        )

        # Upsert in batches
        batch_size = 100
        for i in range(0, len(points), batch_size):
            client.upsert(
                collection_name=collection_name,
                points=points[i : i + batch_size],
            )

        logger.info(
            f"Qdrant: upserted {len(points)} PRISMA entity vectors "
            f"to '{collection_name}'"
        )

        return {
            "qdrant_status": "success",
            "vectors_stored": len(points),
            "collection": collection_name,
            "model": "BAAI/bge-small-en-v1.5",
            "prisma_labels": {
                label: sum(1 for e in valid_entities if e.get("label") == label)
                for label in PRISMA_ENTITY_TYPES
            },
        }

    except Exception as e:
        logger.error(f"Qdrant/Transformer embedding failed: {e}")
        return {"qdrant_status": "error", "error": str(e)}


# ---------------------------------------------------------------------------
# Export graph as JSON — PRISMA schema
# ---------------------------------------------------------------------------


def _export_graph_json(
    entities: List[Dict[str, Any]],
    relations: List[Tuple[str, str, str]],
    output_dir: str = "outputs",
) -> str:
    """Export the PRISMA knowledge graph as a JSON file for visualization."""
    os.makedirs(output_dir, exist_ok=True)

    nodes = []
    node_texts = set()
    for ent in entities:
        if ent["text"] not in node_texts:
            nodes.append(
                {
                    "id": ent["entity_id"],
                    "text": ent["text"],
                    "label": ent.get("label", "objective"),
                    "paper_ids": ent.get("paper_ids", []),
                    "prisma_properties": ent.get("prisma_properties", {}),
                }
            )
            node_texts.add(ent["text"])

    edges = [
        {"source": subj, "relation": rel, "target": obj} for subj, rel, obj in relations
    ]

    graph_data = {
        "schema": "prisma_2020",
        "nodes": nodes,
        "edges": edges,
        "stats": {
            "total_nodes": len(nodes),
            "total_edges": len(edges),
            "entity_types": {
                t: sum(1 for n in nodes if n["label"] == t) for t in PRISMA_ENTITY_TYPES
            },
        },
    }

    path = os.path.join(output_dir, "knowledge_graph.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(graph_data, f, indent=2, ensure_ascii=False)

    logger.info(f"Exported PRISMA knowledge graph to {path}")
    return path


# ---------------------------------------------------------------------------
# LangGraph Node
# ---------------------------------------------------------------------------


async def knowledge_graph_node(
    state: ResearchState,
    config: Dict[str, Any] | None = None,
) -> Dict[str, Any]:
    """
    LangGraph node: PRISMA-Aligned Knowledge Graph Construction

    Pipeline:
      1. Tier 1 (GLiNER) — zero-shot NER → grounded entity spans
      2. Tier 2 (LLM + Pydantic) — structured PRISMA extraction via Groq
      3. Neo4j — persist PRISMA ontology as reasoning graph
      4. Qdrant — embed entities with PRISMA payloads for semantic retrieval
      5. JSON export — for visualization
    """
    config = config or {}
    cfgr = config.get("configurable", {})
    llm = cfgr.get("llm_deep") or cfgr.get("llm")  # prefer deep tier

    chunks = state.get("chunks", [])
    all_entities: List[Dict[str, Any]] = []
    all_relations: List[Tuple[str, str, str]] = []

    if not chunks:
        logger.warning("No chunks available for PRISMA knowledge extraction")
        audit_log = append_audit(
            state,
            agent="knowledge_graph_node",
            action="extract_prisma_entities",
            inputs={"chunk_count": 0},
            output_summary="No chunks to process",
        )
        return {"current_node": "knowledge_graph", "audit_log": audit_log}

    # ---- Phase 1: Extract PRISMA entities & relations ----
    sample_size = min(len(chunks), 50)
    for chunk_data in chunks[:sample_size]:
        text = chunk_data.get("text", "")
        paper_id = chunk_data.get("paper_id", "")

        # Run 2-tier extraction pipeline
        entities, relations = await extract_prisma_entities(text, paper_id, llm=llm)
        all_entities.extend(entities)
        all_relations.extend(relations)

    logger.info(
        f"Extracted {len(all_entities)} PRISMA entities and "
        f"{len(all_relations)} relations from {sample_size} chunks"
    )

    # ---- Phase 2: Persist to Neo4j (reasoning graph) ----
    # Collect unique paper_ids for paper node creation
    paper_ids = set()
    for ent in all_entities:
        paper_ids.update(ent.get("paper_ids", []))

    neo4j_result = {"neo4j_status": "skipped"}
    for pid in paper_ids:
        pid_entities = [e for e in all_entities if pid in e.get("paper_ids", [])]
        pid_relations = [r for r in all_relations if pid in str(r[0])]
        result = _persist_to_neo4j(pid_entities, pid_relations, paper_id=pid)
        if result.get("neo4j_status") == "success":
            neo4j_result = result

    # ---- Phase 3: Embed in Qdrant (semantic retrieval) ----
    qdrant_result = _embed_and_store_qdrant(all_entities)

    # ---- Phase 4: Export graph JSON ----
    graph_path = _export_graph_json(all_entities, all_relations)

    # ---- Build summary ----
    summary = {
        "schema": "prisma_2020",
        "entities_extracted": len(all_entities),
        "relations_extracted": len(all_relations),
        "chunks_processed": sample_size,
        "entity_breakdown": {
            label: sum(1 for e in all_entities if e.get("label") == label)
            for label in PRISMA_ENTITY_TYPES
        },
        "neo4j": neo4j_result,
        "qdrant": qdrant_result,
        "graph_export_path": graph_path,
    }

    audit_log = append_audit(
        state,
        agent="knowledge_graph_node",
        action="build_prisma_knowledge_graph",
        inputs={"chunk_count": len(chunks), "sample_size": sample_size},
        output_summary=(
            f"PRISMA extraction: {len(all_entities)} entities, "
            f"{len(all_relations)} relations. "
            f"Neo4j: {neo4j_result.get('neo4j_status')}. "
            f"Qdrant: {qdrant_result.get('qdrant_status')}."
        ),
    )

    return {
        "current_node": "knowledge_graph",
        "knowledge_entities": all_entities,
        "knowledge_graph_summary": summary,
        "audit_log": audit_log,
    }
