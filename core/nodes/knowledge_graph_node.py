"""
core/nodes/knowledge_graph_node.py
===================================
LangGraph node that extracts entities + relationships from paper chunks,
stores them in Neo4j as a relational knowledge graph, and embeds them in
Qdrant vector space using Word2Vec for cross-paper semantic similarity.
"""

from __future__ import annotations

import json
import logging
import os
import re
import uuid
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from core.state import ResearchState, append_audit, KnowledgeEntity

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Entity / Relation extraction
# ---------------------------------------------------------------------------

ENTITY_TYPES = {"concept", "method", "result", "dataset"}

RELATION_PATTERNS = [
    # (regex_pattern, relation_type)
    (r"(\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b)\s+(?:uses?|utiliz\w+|employ\w+)\s+(.*?)(?:\.|,|$)", "USES"),
    (r"(\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b)\s+(?:achiev\w+|obtain\w+|reach\w+)\s+(.*?)(?:\.|,|$)", "ACHIEVES"),
    (r"(\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b)\s+(?:extend\w*|build\w*\s+(?:on|upon))\s+(.*?)(?:\.|,|$)", "EXTENDS"),
    (r"(\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b)\s+(?:evaluat\w+|test\w+|benchmark\w+)\s+(?:on\s+)?(.*?)(?:\.|,|$)", "EVALUATED_ON"),
]


def _extract_entities_rule_based(text: str, paper_id: str) -> List[Dict[str, Any]]:
    """Extract entities using regex patterns (no LLM required)."""
    entities = []
    # Multi-word capitalised phrases as concepts
    concept_matches = re.findall(r'\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)+)\b', text)
    for c in set(concept_matches):
        if len(c) > 4:  # skip very short matches
            entities.append({
                "entity_id": str(uuid.uuid4())[:8],
                "label": "concept",
                "text": c.strip(),
                "paper_ids": [paper_id],
            })
    # Acronyms as methods/concepts
    acronym_matches = re.findall(r'\b([A-Z]{2,6})\b', text)
    for a in set(acronym_matches):
        if a not in {"THE", "AND", "FOR", "ARE", "NOT", "BUT", "HAS", "WAS", "ALL", "CAN"}:
            entities.append({
                "entity_id": str(uuid.uuid4())[:8],
                "label": "method",
                "text": a,
                "paper_ids": [paper_id],
            })
    return entities


def _extract_relations_rule_based(text: str) -> List[Tuple[str, str, str]]:
    """Extract (subject, relation, object) triples using regex."""
    triples = []
    for pattern, rel_type in RELATION_PATTERNS:
        for match in re.finditer(pattern, text, re.IGNORECASE):
            subj = match.group(1).strip()
            obj = match.group(2).strip()[:80]  # cap object length
            if subj and obj and len(subj) > 2 and len(obj) > 2:
                triples.append((subj, rel_type, obj))
    return triples


async def _extract_entities_llm(text: str, paper_id: str, llm: Any) -> Tuple[List[Dict], List[Tuple]]:
    """Use LLM to extract structured entity-relation triples."""
    system = (
        "You are a research knowledge extractor. Given a text chunk from an "
        "academic paper, extract:\n"
        "1. Entities with 'label' (concept/method/result/dataset) and 'text'\n"
        "2. Relationships as [subject_text, relation_type, object_text] where "
        "relation_type is one of: USES, ACHIEVES, EXTENDS, EVALUATED_ON\n\n"
        "Return JSON: {\"entities\": [...], \"relations\": [...]}"
    )
    try:
        raw = await llm.generate(
            f"Extract entities and relations:\n\n{text[:2000]}",
            system_prompt=system,
            temperature=0.2,
            max_tokens=1024,
        )
        cleaned = raw.strip()
        if cleaned.startswith("```"):
            cleaned = cleaned.split("\n", 1)[1]
            if cleaned.endswith("```"):
                cleaned = cleaned[:-3]

        result = json.loads(cleaned)
        entities = [{
            "entity_id": str(uuid.uuid4())[:8],
            "label": e.get("label", "concept"),
            "text": e.get("text", ""),
            "paper_ids": [paper_id],
        } for e in result.get("entities", [])]

        relations = [
            (r[0], r[1], r[2])
            for r in result.get("relations", [])
            if len(r) == 3
        ]
        return entities, relations

    except Exception as e:
        logger.warning(f"LLM entity extraction failed: {e}")
        return [], []


# ---------------------------------------------------------------------------
# Neo4j graph persistence
# ---------------------------------------------------------------------------

def _persist_to_neo4j(
    entities: List[Dict[str, Any]],
    relations: List[Tuple[str, str, str]],
) -> Dict[str, Any]:
    """
    Persist entities and relations to Neo4j.
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
            # Merge entities as nodes (dedup by text+label)
            # Allowed labels to prevent Cypher injection
            allowed_labels = {"Concept", "Method", "Result", "Dataset"}
            for ent in entities:
                label = ent["label"].capitalize()
                if label not in allowed_labels:
                    label = "Concept"
                session.run(
                    f"MERGE (e:{label} {{text: $text}}) "
                    f"ON CREATE SET e.entity_id = $eid, e.paper_ids = $pids "
                    f"ON MATCH SET e.paper_ids = e.paper_ids + $pids",
                    text=ent["text"],
                    eid=ent["entity_id"],
                    pids=ent["paper_ids"],
                )
                nodes_created += 1

            # Create relationships
            allowed_rels = {"USES", "ACHIEVES", "EXTENDS", "EVALUATED_ON"}
            for subj, rel_type, obj in relations:
                if rel_type not in allowed_rels:
                    continue
                session.run(
                    "MATCH (a {text: $subj}), (b {text: $obj}) "
                    f"MERGE (a)-[r:{rel_type}]->(b) "
                    "ON CREATE SET r.weight = 1 "
                    "ON MATCH SET r.weight = r.weight + 1",
                    subj=subj,
                    obj=obj,
                )
                rels_created += 1

        driver.close()
        logger.info(f"Neo4j: created/merged {nodes_created} nodes, {rels_created} relationships")
        return {
            "neo4j_status": "success",
            "nodes_created": nodes_created,
            "relationships_created": rels_created,
        }

    except Exception as e:
        logger.error(f"Neo4j persistence failed: {e}")
        return {"neo4j_status": "error", "error": str(e)}


# ---------------------------------------------------------------------------
# Qdrant + SentenceTransformers vector embedding
# ---------------------------------------------------------------------------

def _embed_and_store_qdrant(
    entities: List[Dict[str, Any]],
    vector_size: int = 384,
) -> Dict[str, Any]:
    """
    Embed entity texts using SentenceTransformers (BAAI/bge-small-en-v1.5)
    and upsert to Qdrant for semantic similarity search.
    Uses in-memory Qdrant when no QDRANT_URL is set.
    """
    if not entities:
        return {"qdrant_status": "skipped", "reason": "no entities"}

    try:
        from qdrant_client import QdrantClient
        from qdrant_client.http import models

        # Prepare inputs
        texts = [ent["text"] for ent in entities if ent.get("text")]
        if not texts:
            logger.warning("No texts to embed, skipping Qdrant")
            return {"qdrant_status": "skipped", "reason": "no entity texts"}

        from sentence_transformers import SentenceTransformer
        logger.info("Loading BAAI/bge-small-en-v1.5 embedding model...")
        model = SentenceTransformer("BAAI/bge-small-en-v1.5")
        
        logger.info(f"Computing embeddings for {len(texts)} entities...")
        embeddings = model.encode(texts, show_progress_bar=False)

        # Generate points
        points = []
        valid_entities = [ent for ent in entities if ent.get("text")]
        for i, (ent, embedding) in enumerate(zip(valid_entities, embeddings)):
            points.append(
                models.PointStruct(
                    id=i,
                    vector=embedding.tolist(),
                        payload={
                            "text": ent["text"],
                            "label": ent.get("label", "concept"),
                            "paper_ids": ent.get("paper_ids", []),
                            "entity_id": ent.get("entity_id", ""),
                        },
                    )
                )

        # Connect to Qdrant (in-memory if no URL configured)
        qdrant_url = os.environ.get("QDRANT_URL")
        qdrant_api_key = os.environ.get("QDRANT_API_KEY")
        collection_name = "research_entities"

        if qdrant_url:
            client = QdrantClient(url=qdrant_url, api_key=qdrant_api_key)
        else:
            client = QdrantClient(location=":memory:")
            logger.info("Using in-memory Qdrant (set QDRANT_URL for persistent storage)")

        # Create collection if it doesn't exist (preserve existing data)
        try:
            client.get_collection(collection_name=collection_name)
        except Exception:
            client.create_collection(
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

        logger.info(f"Qdrant: upserted {len(points)} entity vectors to '{collection_name}'")

        return {
            "qdrant_status": "success",
            "vectors_stored": len(points),
            "collection": collection_name,
            "model": "BAAI/bge-small-en-v1.5",
        }

    except Exception as e:
        logger.error(f"Qdrant/Word2Vec embedding failed: {e}")
        return {"qdrant_status": "error", "error": str(e)}


# ---------------------------------------------------------------------------
# Export graph as JSON
# ---------------------------------------------------------------------------

def _export_graph_json(
    entities: List[Dict[str, Any]],
    relations: List[Tuple[str, str, str]],
    output_dir: str = "outputs",
) -> str:
    """Export the knowledge graph as a JSON file for visualization."""
    os.makedirs(output_dir, exist_ok=True)

    nodes = []
    node_texts = set()
    for ent in entities:
        if ent["text"] not in node_texts:
            nodes.append({
                "id": ent["entity_id"],
                "text": ent["text"],
                "label": ent.get("label", "concept"),
                "paper_ids": ent.get("paper_ids", []),
            })
            node_texts.add(ent["text"])

    edges = [
        {"source": subj, "relation": rel, "target": obj}
        for subj, rel, obj in relations
    ]

    graph_data = {
        "nodes": nodes,
        "edges": edges,
        "stats": {
            "total_nodes": len(nodes),
            "total_edges": len(edges),
            "entity_types": {
                t: sum(1 for n in nodes if n["label"] == t)
                for t in ENTITY_TYPES
            },
        },
    }

    path = os.path.join(output_dir, "knowledge_graph.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(graph_data, f, indent=2, ensure_ascii=False)

    logger.info(f"Exported knowledge graph to {path}")
    return path


# ---------------------------------------------------------------------------
# LangGraph Node
# ---------------------------------------------------------------------------

async def knowledge_graph_node(
    state: ResearchState,
    config: Dict[str, Any] | None = None,
) -> Dict[str, Any]:
    """
    LangGraph node: Knowledge Graph Construction

    1. Extracts (entity, relation, entity) triples from chunks
    2. Persists to Neo4j as a relational graph
    3. Embeds entities in Qdrant via SentenceTransformers for semantic similarity
    4. Exports graph JSON for visualization
    """
    config = config or {}
    llm = config.get("configurable", {}).get("llm")

    chunks = state.get("chunks", [])
    all_entities: List[Dict[str, Any]] = []
    all_relations: List[Tuple[str, str, str]] = []

    if not chunks:
        logger.warning("No chunks available for knowledge extraction")
        audit_log = append_audit(
            state,
            agent="knowledge_graph_node",
            action="extract_entities",
            inputs={"chunk_count": 0},
            output_summary="No chunks to process",
        )
        return {"current_node": "knowledge_graph", "audit_log": audit_log}

    # ---- Phase 1: Extract entities & relations ----
    sample_size = min(len(chunks), 50)
    for chunk_data in chunks[:sample_size]:
        text = chunk_data.get("text", "")
        paper_id = chunk_data.get("paper_id", "")

        if llm:
            ents, rels = await _extract_entities_llm(text, paper_id, llm)
            if ents:
                all_entities.extend(ents)
                all_relations.extend(rels)
            else:
                # Fallback if LLM returns nothing
                all_entities.extend(_extract_entities_rule_based(text, paper_id))
                all_relations.extend(_extract_relations_rule_based(text))
        else:
            all_entities.extend(_extract_entities_rule_based(text, paper_id))
            all_relations.extend(_extract_relations_rule_based(text))

    logger.info(
        f"Extracted {len(all_entities)} entities and "
        f"{len(all_relations)} relations from {sample_size} chunks"
    )

    # ---- Phase 2: Persist to Neo4j ----
    neo4j_result = _persist_to_neo4j(all_entities, all_relations)

    # ---- Phase 3: Embed in Qdrant via SentenceTransformers ----
    qdrant_result = _embed_and_store_qdrant(all_entities)

    # ---- Phase 4: Export graph JSON ----
    graph_path = _export_graph_json(all_entities, all_relations)

    # ---- Build summary ----
    graph_id = f"kg_{state.get('project_id', 'unknown')}_{len(all_entities)}"
    summary = {
        "entities_extracted": len(all_entities),
        "relations_extracted": len(all_relations),
        "chunks_processed": sample_size,
        "neo4j": neo4j_result,
        "qdrant": qdrant_result,
        "graph_export_path": graph_path,
    }

    audit_log = append_audit(
        state,
        agent="knowledge_graph_node",
        action="build_knowledge_graph",
        inputs={"chunk_count": len(chunks), "sample_size": sample_size},
        output_summary=(
            f"Extracted {len(all_entities)} entities, {len(all_relations)} relations. "
            f"Neo4j: {neo4j_result.get('neo4j_status')}. "
            f"Qdrant: {qdrant_result.get('qdrant_status')}."
        ),
    )

    return {
        "current_node": "knowledge_graph",
        "knowledge_entities": all_entities,
        "knowledge_graph_id": graph_id,
        "knowledge_graph_summary": summary,
        "audit_log": audit_log,
    }
