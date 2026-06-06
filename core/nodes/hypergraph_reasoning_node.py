"""core/nodes/hypergraph_reasoning_node.py
==========================================
Builds isomorphic clusters from extracted hyperedges.
"""

from __future__ import annotations

import logging
import os
import uuid
from typing import Any, Dict, List

from core.state import ResearchState, append_audit

logger = logging.getLogger(__name__)


def _jaccard(a: set[str], b: set[str]) -> float:
    if not a and not b:
        return 0.0
    inter = len(a.intersection(b))
    union = len(a.union(b))
    return inter / union if union else 0.0


def _persist_minimap_to_neo4j(
    session_id: str,
    clusters: List[Dict[str, Any]],
) -> Dict[str, Any]:
    neo4j_uri = os.environ.get("NEO4J_URI", "bolt://localhost:7687")
    neo4j_user = os.environ.get("NEO4J_USER", "neo4j")
    neo4j_password = os.environ.get("NEO4J_PASSWORD", "")

    if not neo4j_password:
        return {"status": "skipped", "reason": "no credentials"}

    try:
        from neo4j import GraphDatabase

        driver = GraphDatabase.driver(neo4j_uri, auth=(neo4j_user, neo4j_password))
        cluster_count = 0
        links_count = 0

        with driver.session() as session:
            session.run(
                "MERGE (s:ReasoningSession {session_id: $sid}) "
                "ON CREATE SET s.created_at = datetime() "
                "SET s.updated_at = datetime()",
                sid=session_id,
            )

            session.run(
                "MERGE (m:ReasoningMiniMap {session_id: $sid}) "
                "ON CREATE SET m.created_at = datetime() "
                "SET m.updated_at = datetime()",
                sid=session_id,
            )
            session.run(
                "MATCH (s:ReasoningSession {session_id: $sid}) "
                "MATCH (m:ReasoningMiniMap {session_id: $sid}) "
                "MERGE (s)-[:HAS_MINIMAP]->(m)",
                sid=session_id,
            )

            for idx, cluster in enumerate(clusters):
                session.run(
                    "MERGE (c:IsomorphicCluster {cluster_id: $cid}) "
                    "ON CREATE SET c.shared_principle = $principle, c.created_at = datetime() "
                    "SET c.domains = $domains, c.similarity_score = $score, c.actionable_insight = $insight, "
                    "    c.session_id = $sid, c.rank = $rank, c.updated_at = datetime()",
                    cid=cluster.get("cluster_id"),
                    principle=cluster.get("shared_principle", ""),
                    domains=cluster.get("domains", []),
                    score=float(cluster.get("similarity_score", 0.0)),
                    insight=cluster.get("actionable_insight"),
                    sid=session_id,
                    rank=idx,
                )
                cluster_count += 1

                session.run(
                    "MATCH (m:ReasoningMiniMap {session_id: $sid}) "
                    "MATCH (c:IsomorphicCluster {cluster_id: $cid}) "
                    "MERGE (m)-[:CONTAINS_CLUSTER]->(c)",
                    sid=session_id,
                    cid=cluster.get("cluster_id"),
                )

                for hid in cluster.get("hyperedge_ids", []):
                    session.run(
                        "MATCH (c:IsomorphicCluster {cluster_id: $cid}) "
                        "MATCH (h:Hyperedge {hyperedge_id: $hid}) "
                        "MERGE (c)-[:CLUSTERS]->(h)",
                        cid=cluster.get("cluster_id"),
                        hid=hid,
                    )
                    links_count += 1

        driver.close()
        return {
            "status": "success",
            "clusters_persisted": cluster_count,
            "links_persisted": links_count,
        }
    except Exception as exc:
        logger.error("Failed to persist session mini map: %s", exc)
        return {"status": "error", "error": str(exc)}


async def hypergraph_reasoning_node(
    state: ResearchState,
    config: Dict[str, Any] | None = None,
) -> Dict[str, Any]:
    """Create isomorphic clusters using hyperedge principle and structural similarity."""
    config = config or {}
    cfgr = config.get("configurable", {})

    hyperedges = list(state.get("hyperedges", []))
    existing_clusters = list(state.get("isomorphic_clusters", []))
    session_id = (
        cfgr.get("session_id")
        or os.getenv("AGENTIC_RUN_ID")
        or state.get("project_id")
        or "default-session"
    )

    if not hyperedges:
        audit_log = append_audit(
            state,
            agent="hypergraph_reasoning_node",
            action="build_isomorphic_clusters",
            inputs={"hyperedge_count": 0},
            output_summary="No hyperedges available for clustering",
        )
        return {
            "current_node": "hypergraph_reasoning",
            "audit_log": audit_log,
        }

    # Group by abstract principle first, then assess structural overlap.
    by_principle: Dict[str, List[Dict[str, Any]]] = {}
    for he in hyperedges:
        principle = (he.get("principle_name") or "generalized_system_principle").strip()
        by_principle.setdefault(principle, []).append(he)

    new_clusters: List[Dict[str, Any]] = []

    for principle, group in by_principle.items():
        if len(group) < 2:
            continue

        member_ids = [g.get("hyperedge_id", "") for g in group if g.get("hyperedge_id")]
        jargon_sets = [set(g.get("domain_jargon", [])) for g in group]
        label_sets = [set(g.get("source_entity_labels", [])) for g in group]

        # Approximate cross-domain structure similarity from jargon and source-label overlap.
        similarities = []
        for i in range(len(group)):
            for j in range(i + 1, len(group)):
                jaccard_jargon = _jaccard(jargon_sets[i], jargon_sets[j])
                jaccard_labels = _jaccard(label_sets[i], label_sets[j])
                similarities.append((0.6 * jaccard_labels) + (0.4 * jaccard_jargon))

        similarity_score = sum(similarities) / len(similarities) if similarities else 0.0

        # Derive pseudo-domain facets from jargon/source labels while proper domain typing is added.
        domains = sorted(
            {
                *[j for s in jargon_sets for j in s if j],
                *[l for s in label_sets for l in s if l],
            }
        )[:8]

        cluster = {
            "cluster_id": f"iso_{uuid.uuid4().hex[:10]}",
            "shared_principle": principle,
            "hyperedge_ids": member_ids,
            "domains": domains,
            "similarity_score": round(float(similarity_score), 4),
            "actionable_insight": (
                f"Principle '{principle}' recurs across {len(member_ids)} hyperedges; "
                f"review transfer opportunities between mapped domains."
            ),
        }
        new_clusters.append(cluster)

    merged_clusters = [*existing_clusters, *new_clusters]
    minimap_result = _persist_minimap_to_neo4j(session_id=session_id, clusters=new_clusters)

    audit_log = append_audit(
        state,
        agent="hypergraph_reasoning_node",
        action="build_isomorphic_clusters",
        inputs={"hyperedge_count": len(hyperedges), "principle_groups": len(by_principle)},
        output_summary=(
            f"Generated {len(new_clusters)} new isomorphic clusters "
            f"from {len(hyperedges)} hyperedges"
        ),
        provenance={
            "clustering_strategy": "principle_grouping_plus_structural_jaccard",
            "session_id": session_id,
            "minimap_persistence": minimap_result,
            "mini_map_query": (
                "MATCH (m:ReasoningMiniMap {session_id: $sid})-[:CONTAINS_CLUSTER]->(c:IsomorphicCluster) "
                "OPTIONAL MATCH (c)-[:CLUSTERS]->(h:Hyperedge) "
                "RETURN m, c, h"
            ),
        },
    )

    logger.info(
        "Hypergraph reasoning created %d clusters from %d hyperedges",
        len(new_clusters),
        len(hyperedges),
    )

    return {
        "current_node": "hypergraph_reasoning",
        "isomorphic_clusters": merged_clusters,
        "audit_log": audit_log,
    }
