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
    hyperedges: List[Dict[str, Any]] = None,
    paper_id: str = "",
    chunks: List[Dict[str, Any]] = None,
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
            # Create vector indexes if they don't exist
            # Note: Requires Neo4j 5.x+
            session.run(
                "CREATE VECTOR INDEX prisma_embeddings IF NOT EXISTS "
                "FOR (n:PRISMAEntity) ON (n.embedding) "
                "OPTIONS {indexConfig: { "
                " `vector.dimensions`: 768, "
                " `vector.similarity_function`: 'cosine' "
                "}}"
            )
            session.run(
                "CREATE VECTOR INDEX chunk_embeddings IF NOT EXISTS "
                "FOR (c:Chunk) ON (c.embedding) "
                "OPTIONS {indexConfig: { "
                " `vector.dimensions`: 768, "
                " `vector.similarity_function`: 'cosine' "
                "}}"
            )

            # Ensure Paper node exists
            if paper_id:
                session.run(
                    "MERGE (p:Paper {paper_id: $pid}) "
                    "ON CREATE SET p.created_at = datetime()",
                    pid=paper_id,
                )
                nodes_created += 1

            # Persist Chunk nodes
            if chunks:
                for c in chunks:
                    c_text = c.get("text")
                    c_emb = c.get("embedding")
                    if c_text and c_emb:
                        session.run(
                            "MERGE (ch:Chunk {text: $text}) "
                            "ON CREATE SET ch.paper_id = $pid, ch.embedding = $emb "
                            "ON MATCH SET ch.embedding = $emb",
                            text=c_text, pid=c.get("paper_id", paper_id), emb=c_emb
                        )
                        nodes_created += 1

            # Create PRISMA entity nodes
            for ent in entities:
                label = ent["label"].capitalize()
                text = ent["text"]
                prisma_props = ent.get("prisma_properties", {})
                emb = ent.get("embedding", [])

                # Use parameterized label via APOC or fallback to safe labels
                if label in {
                    "Paper",
                    "Objective",
                    "Methodology",
                    "Result",
                    "Limitation",
                    "Implication",
                }:
                    # MERGE entity node with PRISMA properties and Vector Embedding
                    props_json = json.dumps(prisma_props) if prisma_props else "{}"
                    session.run(
                        f"MERGE (e:{label}:PRISMAEntity {{text: $text}}) "
                        f"ON CREATE SET e.entity_id = $eid, "
                        f"  e.paper_ids = $pids, "
                        f"  e.prisma_properties = $props, "
                        f"  e.prisma_label = $prisma_label, "
                        f"  e.embedding = $emb "
                        f"ON MATCH SET e.paper_ids = "
                        f"  [x IN e.paper_ids + $pids WHERE x IS NOT NULL], "
                        f"  e.embedding = $emb",
                        text=text,
                        eid=ent["entity_id"],
                        pids=ent["paper_ids"],
                        props=props_json,
                        emb=emb,
                        prisma_label=ent.get("label", "objective")
                    )
                    nodes_created += 1

                    # Ground the entity back to its exact source chunk (Provenance routing)
                    source_chunk = ent.get("source_chunk_text")
                    if source_chunk:
                        session.run(
                            f"MATCH (e:{label}:PRISMAEntity {{text: $ent_text}}) "
                            f"MATCH (ch:Chunk {{text: $chunk_text}}) "
                            f"MERGE (e)-[r:EXTRACTED_FROM]->(ch)",
                            ent_text=text, chunk_text=source_chunk
                        )
                        rels_created += 1

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

            # Create Hyperedges
            if hyperedges:
                for he in hyperedges:
                    # Create the central Hyperedge node
                    session.run(
                        "MERGE (h:Hyperedge {hyperedge_id: $hid}) "
                        "ON CREATE SET h.principle_name = $pname, h.weight = $weight, h.paper_ids = $pids "
                        "ON MATCH SET h.paper_ids = [x IN h.paper_ids + $pids WHERE x IS NOT NULL]",
                        hid=he["hyperedge_id"],
                        pname=he["principle_name"],
                        weight=he["weight"],
                        pids=he["paper_ids"]
                    )
                    nodes_created += 1

                    # Connect members to it via IN_HYPEREDGE
                    for member_id in he["member_entity_ids"]:
                        session.run(
                            "MATCH (h:Hyperedge {hyperedge_id: $hid}) "
                            "MATCH (e {entity_id: $eid}) "
                            "MERGE (e)-[r:IN_HYPEREDGE]->(h)",
                            hid=he["hyperedge_id"],
                            eid=member_id
                        )
                        rels_created += 1

        driver.close()
        logger.info(
            f"Neo4j: created/merged {nodes_created} PRISMA/NEXUS nodes, "
            f"{rels_created} relationships/hyperedge-links"
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
      4. Neo4j Vector — Embed entities using Specter2 for semantic search
      5. JSON export — for visualization
    """
    config = config or {}
    cfgr = config.get("configurable", {})
    llm = cfgr.get("llm_deep") or cfgr.get("llm")  # prefer deep tier

    chunks = state.get("chunks", [])
    all_entities: List[Dict[str, Any]] = []
    all_relations: List[Tuple[str, str, str]] = []
    all_hyperedges: List[Dict[str, Any]] = []

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
    rigor_level = state.get("rigor_level", "exploratory")
    is_dual_pass = (rigor_level == "cochrane")
    
    for chunk_data in chunks[:sample_size]:
        text = chunk_data.get("text", "")
        paper_id = chunk_data.get("paper_id", "")

        entities, relations, hyperedges = await extract_prisma_entities(
            text, paper_id, llm=llm, dual_pass=is_dual_pass, rigor_level=rigor_level
        )
        
        # Ground extracted entities to the specific chunk for UI tracing
        for e in entities:
            e["source_chunk_text"] = text

        all_entities.extend(entities)
        all_relations.extend(relations)
        all_hyperedges.extend(hyperedges)

    logger.info(
        f"Extracted {len(all_entities)} PRISMA entities, "
        f"{len(all_relations)} relations, and {len(all_hyperedges)} hyperedges from {sample_size} chunks"
    )

    # ---- Phase 1.5: Embed using Specter2 (Rigor alignment) ----
    embed_result = {"status": "skipped"}
    try:
        from sentence_transformers import SentenceTransformer
        logger.info("Loading allenai/specter2_base embedding model for rigorous vector space...")
        # Local model loading, this requires memory but avoids API calls
        embed_model = SentenceTransformer("allenai/specter2_base")

        # 1. Embed Chunks
        valid_chunks = [c for c in chunks[:sample_size] if c.get("text")]
        if valid_chunks:
            chunk_texts = [c["text"] for c in valid_chunks]
            chunk_embeddings = embed_model.encode(chunk_texts, show_progress_bar=False)
            for c, emb in zip(valid_chunks, chunk_embeddings):
                c["embedding"] = emb.tolist()

        # 2. Embed Entities
        valid_entities = [e for e in all_entities if e.get("text")]
        if valid_entities:
            entity_texts = [e["text"] for e in valid_entities]
            entity_embeddings = embed_model.encode(entity_texts, show_progress_bar=False)
            for e, emb in zip(valid_entities, entity_embeddings):
                e["embedding"] = emb.tolist()

        embed_result = {"status": "success", "model": "allenai/specter2_base"}
    except Exception as e:
        logger.error(f"Specter2 embedding sequence failed: {e}")
        embed_result = {"status": "error", "error": str(e)}

    # ---- Phase 2: Persist to Neo4j (reasoning graph) ----
    # Collect unique paper_ids for paper node creation
    paper_ids = set()
    for ent in all_entities:
        paper_ids.update(ent.get("paper_ids", []))

    neo4j_result = {"neo4j_status": "skipped"}
    for pid in paper_ids:
        pid_entities = [e for e in all_entities if pid in e.get("paper_ids", [])]
        pid_relations = [r for r in all_relations if pid in str(r[0])]
        pid_hyperedges = [h for h in all_hyperedges if pid in h.get("paper_ids", [])]
        pid_chunks = [c for c in chunks[:sample_size] if c.get("paper_id") == pid]
        result = _persist_to_neo4j(pid_entities, pid_relations, hyperedges=pid_hyperedges, paper_id=pid, chunks=pid_chunks)
        if result.get("neo4j_status") == "success":
            neo4j_result = result

    # ---- Phase 3: Export graph JSON ----
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
        "embedding": embed_result,
        "graph_export_path": graph_path,
    }

    audit_log = append_audit(
        state,
        agent="knowledge_graph_node",
        action="build_prisma_knowledge_graph",
        inputs={"chunk_count": len(chunks), "sample_size": sample_size},
        output_summary=(
            f"PRISMA extraction: {len(all_entities)} entities, "
            f"{len(all_relations)} relations, {len(all_hyperedges)} hyperedges. "
            f"Neo4j: {neo4j_result.get('neo4j_status')}. "
            f"Embeddings: {embed_result.get('status')}."
        ),
    )

    return {
        "current_node": "knowledge_graph",
        "knowledge_entities": all_entities,
        "hyperedges": all_hyperedges,
        "knowledge_graph_summary": summary,
        "audit_log": audit_log,
    }
