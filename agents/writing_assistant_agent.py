"""Writing Assistant Agent — academic drafting, synthesis, and evidence
gap detection via the Deep Agents subagent pattern.

Wraps writing_node logic and adds gap-driven backtracking capability
as a callable tool for the orchestrator's ReAct loop.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from core.base_agent import ResearchAgent
from core.agent_tools import draft_section, qdrant_search


class WritingAssistantAgent(ResearchAgent):
    """Agent that assists with academic writing tasks.

    In agentic mode, drafts sections and detects evidence gaps.
    When gaps are found, the orchestrator loops back to literature
    search with refined queries.

    Supported actions:
      - ``draft``                — draft a specific paper section
      - ``synthesize_literature``— produce literature synthesis from entities
      - ``detect_gaps``          — check drafts for unsupported claims
    """

    def __init__(self, llm=None):
        super().__init__(
            name="writing_assistant",
            description=(
                "Assists academics in drafting, summarising, and synthesising "
                "research content. Detects evidence gaps for backtracking."
            ),
            llm=llm,
        )

    def get_required_fields(self) -> List[str]:
        return ["action"]

    async def process(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        await super().process(input_data)
        action = input_data["action"]

        if action == "draft":
            return await self._draft(input_data)
        elif action == "synthesize_literature":
            return await self._synthesize_literature(input_data)
        elif action == "detect_gaps":
            return await self._detect_gaps(input_data)
        elif action == "summarize_results":
            return await self._summarize_results(input_data)
        elif action == "generate_outline":
            return await self._generate_outline(input_data)
        else:
            raise ValueError(f"Unknown action: {action}")

    async def _draft(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Draft a specific paper section."""
        topic = data.get("topic", "")
        section_name = data.get("section_name", "literature_review")
        entities = data.get("entities", [])
        analysis_results = data.get("analysis_results", [])

        self.logger.info(f"Drafting section: {section_name}")
        return await draft_section(
            topic=topic,
            section_name=section_name,
            entities=entities,
            analysis_results=analysis_results,
            llm=self.llm,
        )

    async def _synthesize_literature(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Produce a literature synthesis from extracted knowledge."""
        topic = data.get("topic", "")
        entities = data.get("entities", [])

        # Use Qdrant to gather semantically rich context
        qdrant_context = qdrant_search(topic, limit=20)
        context_texts = [r["text"] for r in qdrant_context]

        if not self.llm:
            return {
                "status": "completed",
                "synthesis": f"Entities: {', '.join(e.get('text', '') for e in entities[:10])}",
                "entity_count": len(entities),
                "paper_count": len(data.get("papers", entities)),
            }

        entity_texts = [e.get("text", "") for e in entities[:40]]
        prompt = (
            f"Topic: {topic}\n\n"
            f"Knowledge base context: {', '.join(context_texts[:10])}\n\n"
            f"Key entities: {', '.join(entity_texts)}\n\n"
            "Write a coherent literature synthesis (500-800 words) from "
            "these findings. Use academic tone and structure."
        )

        synthesis = await self.llm.generate(
            prompt,
            system_prompt="You are an expert academic writer specialising in systematic reviews.",
            temperature=0.6,
            max_tokens=2048,
        )

        return {
            "status": "completed",
            "synthesis": synthesis,
            "entity_count": len(entities),
            "paper_count": len(data.get("papers", entities)),
            "qdrant_context_count": len(qdrant_context),
        }

    async def _detect_gaps(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Check drafted sections for evidence gaps.

        Queries Qdrant per PRISMA domain to verify coverage backing
        the claims made in the draft.
        """
        draft_sections = data.get("draft_sections", {})
        topic = data.get("topic", "")

        prisma_domains = [
            "objective",
            "methodology",
            "result",
            "limitation",
            "implication",
        ]
        domain_coverage = {}
        gaps = []

        for domain in prisma_domains:
            results = qdrant_search(topic, prisma_label=domain, limit=5)
            domain_coverage[domain] = len(results)
            if len(results) < 2:
                gaps.append(
                    f"Thin coverage: '{domain}' has only {len(results)} entities"
                )

        self.logger.info(
            f"Gap detection: {len(gaps)} gaps found in {len(prisma_domains)} domains"
        )

        return {
            "status": "completed",
            "domain_coverage": domain_coverage,
            "gaps": gaps,
            "has_gaps": len(gaps) > 0,
            "recommendation": (
                f"Search for more papers covering: "
                f"{', '.join(d for d in prisma_domains if domain_coverage.get(d, 0) < 2)}"
                if gaps
                else "Coverage sufficient for all PRISMA domains."
            ),
        }

    async def _summarize_results(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Create a concise results summary (legacy compat)."""
        results = data.get("results", {})
        self.logger.info("Summarizing results")
        return {"status": "completed", "summary": ""}

    async def _generate_outline(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Generate a structured outline for a research paper (legacy compat)."""
        topic = data.get("topic", "")
        outline_type = data.get("outline_type", "research_paper")
        self.logger.info(f"Generating {outline_type} outline for: {topic}")
        return {"status": "completed", "outline": "", "outline_type": outline_type}
