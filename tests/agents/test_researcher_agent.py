"""Tests for the autonomous ResearcherAgent."""

import sys
from pathlib import Path
import asyncio

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from core.researcher_agent import ResearcherAgent


def _run(coro):
    return asyncio.get_event_loop().run_until_complete(coro)


class TestResearcherAgent:
    def test_process_builds_plan_and_insights(self, tmp_path: Path):
        agent = ResearcherAgent()
        memory_path = tmp_path / "researcher_memory.json"

        result = _run(
            agent.process(
                {
                    "research_topic": "LLM safety",
                    "research_goals": ["robustness evaluation"],
                    "execute_pipeline": False,  # keep tests lightweight
                    "memory_path": memory_path,
                }
            )
        )

        assert result["plan"]["workflow"] in {"literature_review", "data_analysis"}
        assert result["findings"]["status"] == "skipped"
        assert result["analysis"]["status"] == "completed"
        assert len(result["insights"]) >= 1
        assert memory_path.exists()

    def test_memory_is_capped_and_persists(self, tmp_path: Path):
        agent = ResearcherAgent()
        memory_path = tmp_path / "researcher_memory.json"

        for i in range(agent.MEMORY_LIMIT + 5):
            _run(
                agent.process(
                    {
                        "research_topic": f"Topic {i}",
                        "research_goals": ["goal"],
                        "execute_pipeline": False,
                        "memory_path": memory_path,
                    }
                )
            )

        history = agent._load_memory(memory_path)
        assert len(history) == agent.MEMORY_LIMIT
