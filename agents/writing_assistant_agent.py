"""Writing Assistant Agent – helps academics draft, summarise, and
synthesise research content in proper academic style."""

from core.base_agent import ResearchAgent
from typing import Dict, Any, List


class WritingAssistantAgent(ResearchAgent):
    """Agent that assists with academic writing tasks.

    Supported actions:
      - ``synthesize_literature`` – produce a literature synthesis from papers
      - ``summarize_results``     – create a results summary section
      - ``generate_outline``      – generate a structured paper outline
    """

    def __init__(self):
        super().__init__(
            name="writing_assistant",
            description=(
                "Assists academics in drafting, summarising, and synthesising "
                "research content to accelerate the writing process"
            ),
        )

    def get_required_fields(self) -> List[str]:
        return ["action"]

    async def process(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        await super().process(input_data)
        action = input_data["action"]

        if action == "synthesize_literature":
            return await self._synthesize_literature(input_data)
        elif action == "summarize_results":
            return await self._summarize_results(input_data)
        elif action == "generate_outline":
            return await self._generate_outline(input_data)
        else:
            raise ValueError(f"Unknown action: {action}")

    async def _synthesize_literature(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Produce a coherent literature synthesis from extracted papers."""
        papers = data.get("papers", [])
        topic = data.get("topic", "")
        self.logger.info(f"Synthesizing literature for topic: {topic}")
        # TODO: integrate with LLM for synthesis
        return {"status": "completed", "synthesis": "", "paper_count": len(papers)}

    async def _summarize_results(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Create a concise results summary section."""
        results = data.get("results", {})
        self.logger.info("Summarizing results")
        # TODO: integrate with LLM for summarisation
        return {"status": "completed", "summary": ""}

    async def _generate_outline(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Generate a structured outline for a research paper."""
        topic = data.get("topic", "")
        outline_type = data.get("outline_type", "research_paper")
        self.logger.info(f"Generating {outline_type} outline for: {topic}")
        # TODO: integrate with LLM for outline generation
        return {"status": "completed", "outline": "", "outline_type": outline_type}
