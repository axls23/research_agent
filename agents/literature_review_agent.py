"""Literature Review Agent – helps academics discover, filter, and
summarise relevant papers to reduce information overload."""

from core.base_agent import ResearchAgent
from typing import Dict, Any, List


class LiteratureReviewAgent(ResearchAgent):
    """Agent responsible for literature search, paper filtering, and
    search-query formulation.

    Supported actions (passed via *input_data["action"]*):
      - ``formulate_search_query`` – generate structured search queries
      - ``retrieve_papers``        – fetch candidate papers for a query
      - ``filter_papers``          – rank / filter papers by relevance
    """

    def __init__(self):
        super().__init__(
            name="literature_review",
            description=(
                "Discovers, retrieves, and filters academic papers to help "
                "researchers manage information overload"
            ),
        )

    def get_required_fields(self) -> List[str]:
        return ["action"]

    async def process(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        await super().process(input_data)
        action = input_data["action"]

        if action == "formulate_search_query":
            return await self._formulate_search_query(input_data)
        elif action == "retrieve_papers":
            return await self._retrieve_papers(input_data)
        elif action == "filter_papers":
            return await self._filter_papers(input_data)
        else:
            raise ValueError(f"Unknown action: {action}")

    # ------------------------------------------------------------------
    # Action handlers
    # ------------------------------------------------------------------

    async def _formulate_search_query(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Generate structured search queries from a research topic."""
        topic = data.get("topic", "")
        goals = data.get("research_goals", [])
        self.logger.info(f"Formulating search queries for topic: {topic}")
        queries = [f"{topic} {goal}" for goal in goals] if goals else [topic]
        return {"status": "completed", "queries": queries}

    async def _retrieve_papers(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Retrieve candidate papers matching the search queries."""
        queries = data.get("queries", [])
        self.logger.info(f"Retrieving papers for {len(queries)} queries")
        # TODO: integrate with ArXiv / Semantic Scholar APIs
        return {"status": "completed", "papers": [], "query_count": len(queries)}

    async def _filter_papers(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Rank and filter papers by relevance to reduce overload."""
        papers = data.get("papers", [])
        self.logger.info(f"Filtering {len(papers)} papers")
        # TODO: implement relevance scoring
        return {"status": "completed", "filtered_papers": papers}
