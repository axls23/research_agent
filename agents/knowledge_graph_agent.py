"""Knowledge Graph Agent — PRISMA-aligned knowledge graph construction
and traversal using Deep Agents subagent pattern.

Wraps the 2-tier extraction pipeline (GLiNER + LLM) and provides
Qdrant/Neo4j query capabilities for agentic reasoning.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from core.base_agent import ResearchAgent
from core.agent_tools import (
    extract_prisma_knowledge,
    qdrant_search,
    neo4j_query,
)


class KnowledgeGraphAgent(ResearchAgent):
    """Agent responsible for PRISMA knowledge graph construction and querying.

    In agentic mode, this agent's methods are exposed as tools to the
    orchestrator's ReAct loop. The agent can:
      - Extract entities aligned with PRISMA 2020 ontology
      - Query Qdrant for semantically similar entities (with PRISMA label filter)
      - Traverse Neo4j graph for structured reasoning paths
      - Check coverage per PRISMA domain before analysis

    Supported actions:
      - ``extract_knowledge``   — run 2-tier extraction (GLiNER + LLM)
      - ``search_entities``     — semantic search in Qdrant
      - ``query_graph``         — Cypher traversal in Neo4j
      - ``check_coverage``      — assess entity coverage per PRISMA domain
    """

    def __init__(self, llm=None):
        super().__init__(
            name="knowledge_graph",
            description=(
                "Builds and queries a PRISMA 2020-aligned knowledge graph "
                "using GLiNER + LLM extraction, Neo4j reasoning graph, "
                "and Qdrant semantic retrieval."
            ),
            llm=llm,
        )

    def get_required_fields(self) -> List[str]:
        return ["action"]

    async def process(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        await super().process(input_data)
        action = input_data["action"]

        if action == "initialize_graph":
            return await self._initialize_graph(input_data)
        elif action == "extract_knowledge":
            return await self._extract_knowledge(input_data)
        elif action == "search_entities":
            return await self._search_entities(input_data)
        elif action == "query_graph":
            return await self._query_graph(input_data)
        elif action == "check_coverage":
            return await self._check_coverage(input_data)
        else:
            raise ValueError(f"Unknown action: {action}")

    async def _initialize_graph(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Create a new project knowledge graph (legacy compat)."""
        project_name = data.get("project_name", "")
        self.logger.info(f"Initializing knowledge graph for project: {project_name}")
        graph_id = f"kg_{project_name.lower().replace(' ', '_')}"
        return {"status": "completed", "graph_id": graph_id}

    async def _extract_knowledge(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Run 2-tier PRISMA extraction on text chunks."""
        chunks = data.get("chunks", [])
        self.logger.info(f"Extracting PRISMA knowledge from {len(chunks)} chunks")
        return await extract_prisma_knowledge(chunks, llm=self.llm)

    async def _search_entities(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Search Qdrant for semantically similar PRISMA entities."""
        query = data.get("query", "")
        prisma_label = data.get("prisma_label")
        limit = data.get("limit", 10)

        self.logger.info(
            f"Searching entities: {query!r} " f"(label={prisma_label}, limit={limit})"
        )
        results = qdrant_search(query, prisma_label=prisma_label, limit=limit)
        return {"status": "completed", "results": results, "count": len(results)}

    async def _query_graph(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Execute a Cypher query against the Neo4j PRISMA graph."""
        cypher = data.get("cypher", data.get("query", ""))
        params = data.get("params", {})

        self.logger.info(f"Running Cypher: {cypher[:100]}...")
        records = neo4j_query(cypher, params=params)
        return {
            "status": "completed",
            "results": records,
            "query": cypher,
            "count": len(records),
        }

    async def _check_coverage(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Check entity coverage per PRISMA domain.

        Returns coverage counts and flags domains with < min_threshold entities.
        """
        topic = data.get("topic", "")
        min_threshold = data.get("min_threshold", 3)

        prisma_labels = [
            "objective",
            "methodology",
            "result",
            "limitation",
            "implication",
        ]
        coverage: Dict[str, int] = {}
        thin_domains: List[str] = []

        for label in prisma_labels:
            results = qdrant_search(topic, prisma_label=label, limit=20)
            coverage[label] = len(results)
            if len(results) < min_threshold:
                thin_domains.append(label)

        self.logger.info(f"Coverage: {coverage} | Thin: {thin_domains}")
        return {
            "status": "completed",
            "coverage": coverage,
            "thin_domains": thin_domains,
            "sufficient": len(thin_domains) == 0,
        }
