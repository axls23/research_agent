"""Analysis Agent — systematic review analysis using GraphRAG retrieval
and LLM synthesis via the Deep Agents subagent pattern.

Connects Qdrant (semantic entry points) and Neo4j (reasoning paths)
for rich context before analysis.
"""

from __future__ import annotations

from typing import Any, Dict, List

from core.base_agent import ResearchAgent
from core.agent_tools import (
    analyze_evidence,
    qdrant_search,
    neo4j_query,
)


class AnalysisAgent(ResearchAgent):
    """Agent for evidence analysis, pattern detection, and synthesis.

    In agentic mode, uses GraphRAG retrieval (Qdrant → Neo4j) to build
    rich context before running LLM-based synthesis with extended thinking.

    Supported actions:
      - ``analyze``               — full analysis pipeline
      - ``graphrag_retrieve``     — Qdrant → Neo4j context retrieval
      - ``explore_data``          — descriptive statistics
    """

    def __init__(self, llm=None):
        super().__init__(
            name="analysis",
            description=(
                "Analyzes research evidence using GraphRAG retrieval, "
                "statistical methods, and LLM-based synthesis."
            ),
            llm=llm,
        )

    def get_required_fields(self) -> List[str]:
        return ["action"]

    async def process(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        await super().process(input_data)
        action = input_data["action"]

        if action == "analyze":
            return await self._analyze(input_data)
        elif action == "graphrag_retrieve":
            return await self._graphrag_retrieve(input_data)
        elif action == "explore_data":
            return await self._explore_data(input_data)
        elif action == "run_statistical_tests":
            return await self._run_statistical_tests(input_data)
        elif action == "create_visualizations":
            return await self._create_visualizations(input_data)
        else:
            raise ValueError(f"Unknown action: {action}")

    async def _analyze(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Run full analysis pipeline with GraphRAG context."""
        entities = data.get("entities", [])
        topic = data.get("topic", "")
        papers = data.get("papers", [])

        self.logger.info(
            f"Running analysis: {len(entities)} entities, {len(papers)} papers"
        )
        return await analyze_evidence(entities, topic, papers=papers, llm=self.llm)

    async def _graphrag_retrieve(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Retrieve GraphRAG context: Qdrant entry → Neo4j traversal."""
        topic = data.get("topic", "")

        # Step 1: Semantic entry points from Qdrant
        entry_points = qdrant_search(topic, limit=5)
        entry_texts = [r["text"] for r in entry_points]

        if not entry_texts:
            return {
                "status": "completed",
                "context": "No semantic matches found.",
                "paths": [],
            }

        # Step 2: Neo4j 2-hop traversal from entry points
        paths = []
        for text in entry_texts:
            records = neo4j_query(
                "MATCH (start {text: $text})-[r1]->(mid)-[r2]->(end) "
                "RETURN start.text AS s, type(r1) AS r1, mid.text AS m, "
                "type(r2) AS r2, end.text AS e LIMIT 5",
                params={"text": text},
            )
            for rec in records:
                path = (
                    f"({rec.get('s')}) -[{rec.get('r1')}]-> "
                    f"({rec.get('m')}) -[{rec.get('r2')}]-> ({rec.get('e')})"
                )
                paths.append(path)

        return {
            "status": "completed",
            "entry_points": entry_texts,
            "paths": paths,
            "context": "\n".join(paths) if paths else "No graph paths found.",
        }

    async def _explore_data(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Run exploratory data analysis on entities."""
        entities = data.get("entities", [])
        self.logger.info(f"Running EDA on {len(entities)} entities")

        label_dist = {}
        for e in entities:
            label = e.get("label", "unknown")
            label_dist[label] = label_dist.get(label, 0) + 1

        return {
            "status": "completed",
            "entity_count": len(entities),
            "label_distribution": label_dist,
        }

    async def _run_statistical_tests(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Execute statistical tests (legacy compat)."""
        preferred_methods = data.get("preferred_methods", ["descriptive"])
        self.logger.info(f"Running statistical tests: {preferred_methods}")
        return {"status": "completed", "results": {}, "methods": preferred_methods}

    async def _create_visualizations(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Generate visualizations from analysis results (legacy compat)."""
        results = data.get("results", {})
        self.logger.info("Creating visualizations")
        return {"status": "completed", "figures": []}
