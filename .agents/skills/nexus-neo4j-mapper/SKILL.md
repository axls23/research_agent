---
name: nexus-neo4j-mapper
description: Handle the translation of extracted Hyperedge and IsomorphicCluster structures into optimal Cypher queries for persistence in the NEXUS system. Use this skill whenever the user asks to update graph storage logic, edit knowledge_graph_node.py, add new node or edge types to Neo4j, or write Cypher queries for the LangGraph pipeline.
---

# NEXUS Neo4j Mapper Skill

The NEXUS system uses a LangGraph pipeline to extract and store complex mathematical structures, specifically `Hyperedge`s and `IsomorphicCluster`s, into Neo4j. 
Because Neo4j does not natively support hyperedges (edges connecting more than 2 nodes), NEXUS uses a specific modelling pattern where the Hyperedge itself is a Node, and member entities connect to it via `IN_HYPEREDGE` relationships.

When updating Neo4j persistence logic (usually in `core/nodes/knowledge_graph_node.py`), you must adhere to the following strict patterns.

## 1. Graph Entity Models

The primary structures are defined in `core.state`.

### Hyperedge Pattern
In Neo4j, a Hyperedge is represented as a Node with the label `:Hyperedge`.
It has properties: `hyperedge_id`, `principle_name`, `weight`, `paper_ids`.
Member entities connect to the Hyperedge node using the `[:IN_HYPEREDGE]` directed relationship.

Example Cypher for storing a Hyperedge:
```cypher
// 1. Merge the Hyperedge node
MERGE (h:Hyperedge {hyperedge_id: $hid})
ON CREATE SET h.principle_name = $pname, 
              h.weight = $weight, 
              h.paper_ids = $pids
ON MATCH SET h.paper_ids = [x IN h.paper_ids + $pids WHERE x IS NOT NULL]

// 2. Connect members
MATCH (h:Hyperedge {hyperedge_id: $hid})
MATCH (e {entity_id: $member_id})
MERGE (e)-[r:IN_HYPEREDGE]->(h)
```

## 2. Isomorphic Cluster Pattern
Isomorphic Clusters represent mappings between multiple Hyperedges across different domains.
It is represented as a Node with the label `:IsomorphicCluster`.
Properties: `cluster_id`, `shared_principle`, `domains`, `similarity_score`.
Matched Hyperedges connect to the cluster via `[:PART_OF_CLUSTER]`.

Example Cypher:
```cypher
// 1. Merge the cluster node
MERGE (c:IsomorphicCluster {cluster_id: $cid})
ON CREATE SET c.shared_principle = $principle,
              c.domains = $domains,
              c.similarity_score = $score

// 2. Connect hyperedges
MATCH (c:IsomorphicCluster {cluster_id: $cid})
MATCH (h:Hyperedge {hyperedge_id: $hid})
MERGE (h)-[r:PART_OF_CLUSTER]->(c)
```

## 3. Python Driver Usage Rules

When writing Python code using the `neo4j` official driver:
- **Parameterization:** NEVER use string formatting/concatenation (f-strings) for query property values. Always use parameters (`$param_name`) to prevent Cypher injection and improve query cache performance.
- **Labels are Exceptions:** Only use f-strings for dynamic Node Labels or Relationship Types containing checked alphanumeric strings, because Neo4j does not allow parameterising labels/types.
- **List Appending:** ALWAYS use `ON MATCH SET` for lists (like `paper_ids` or `domains`) to append new values while preserving existing ones, ensuring no duplicates. 
  Example: `ON MATCH SET c.domains = [x IN c.domains + $domains WHERE NOT x IN c.domains]`
- **Avoid Duplicates Check:** Do not create blind `CREATE` statements. Always use `MERGE` with `ON CREATE SET` and `ON MATCH SET` to handle idempotency correctly, as the LangGraph pipeline may replay nodes during back-tracking or human interventions.
