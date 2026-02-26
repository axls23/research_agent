"""Tests for all AI agent components.

Validates that each agent:
  - can be instantiated
  - has the correct name, description, and required fields
  - processes supported actions correctly
  - rejects unknown actions
  - aligns with the academic research assistant theme
"""

import sys
from pathlib import Path
import asyncio
import pytest

# Ensure the project root is on sys.path so that "agents" / "core" resolve.
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from agents.literature_review_agent import LiteratureReviewAgent
from agents.data_processing_agent import DataProcessingAgent
from agents.analysis_agent import AnalysisAgent
from agents.writing_assistant_agent import WritingAssistantAgent
from agents.knowledge_graph_agent import KnowledgeGraphAgent
from agents.collaboration_agent import CollaborationAgent

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _run(coro):
    """Run an async coroutine synchronously for test convenience."""
    return asyncio.get_event_loop().run_until_complete(coro)


# ---------------------------------------------------------------------------
# LiteratureReviewAgent
# ---------------------------------------------------------------------------


class TestLiteratureReviewAgent:
    def test_instantiation(self):
        agent = LiteratureReviewAgent()
        assert agent.name == "literature_review"
        assert (
            "paper" in agent.description.lower()
            or "information overload" in agent.description.lower()
        )

    def test_required_fields(self):
        agent = LiteratureReviewAgent()
        assert "action" in agent.get_required_fields()

    def test_formulate_search_query(self):
        agent = LiteratureReviewAgent()
        result = _run(
            agent.process(
                {
                    "action": "formulate_search_query",
                    "topic": "deep learning",
                    "research_goals": ["accuracy", "speed"],
                }
            )
        )
        assert result["status"] == "completed"
        assert isinstance(result["queries"], list)
        assert len(result["queries"]) >= 1

    def test_retrieve_papers(self):
        agent = LiteratureReviewAgent()
        result = _run(
            agent.process(
                {"action": "retrieve_papers", "queries": ["deep learning accuracy"]}
            )
        )
        assert result["status"] == "completed"

    def test_filter_papers(self):
        agent = LiteratureReviewAgent()
        result = _run(
            agent.process(
                {"action": "filter_papers", "papers": [{"title": "A"}, {"title": "B"}]}
            )
        )
        assert result["status"] == "completed"

    def test_unknown_action_raises(self):
        agent = LiteratureReviewAgent()
        with pytest.raises(ValueError, match="Unknown action"):
            _run(agent.process({"action": "nonexistent"}))


# ---------------------------------------------------------------------------
# DataProcessingAgent
# ---------------------------------------------------------------------------


class TestDataProcessingAgent:
    def test_instantiation(self):
        agent = DataProcessingAgent()
        assert agent.name == "data_processing"

    def test_prepare_data(self):
        agent = DataProcessingAgent()
        result = _run(
            agent.process(
                {"action": "prepare_data", "documents": ["doc1.pdf"], "chunk_size": 500}
            )
        )
        assert result["status"] == "completed"
        assert result["document_count"] == 1

    def test_extract_text(self):
        agent = DataProcessingAgent()
        result = _run(
            agent.process({"action": "extract_text", "file_path": "sample.pdf"})
        )
        assert result["status"] == "completed"

    def test_unknown_action_raises(self):
        agent = DataProcessingAgent()
        with pytest.raises(ValueError, match="Unknown action"):
            _run(agent.process({"action": "bad"}))


# ---------------------------------------------------------------------------
# AnalysisAgent
# ---------------------------------------------------------------------------


class TestAnalysisAgent:
    def test_instantiation(self):
        agent = AnalysisAgent()
        assert agent.name == "analysis"

    def test_explore_data(self):
        agent = AnalysisAgent()
        result = _run(agent.process({"action": "explore_data", "dataset": {"a": 1}}))
        assert result["status"] == "completed"

    def test_run_statistical_tests(self):
        agent = AnalysisAgent()
        result = _run(
            agent.process(
                {
                    "action": "run_statistical_tests",
                    "preferred_methods": ["descriptive"],
                }
            )
        )
        assert result["status"] == "completed"
        assert result["methods"] == ["descriptive"]

    def test_create_visualizations(self):
        agent = AnalysisAgent()
        result = _run(agent.process({"action": "create_visualizations", "results": {}}))
        assert result["status"] == "completed"

    def test_unknown_action_raises(self):
        agent = AnalysisAgent()
        with pytest.raises(ValueError, match="Unknown action"):
            _run(agent.process({"action": "bad"}))


# ---------------------------------------------------------------------------
# WritingAssistantAgent
# ---------------------------------------------------------------------------


class TestWritingAssistantAgent:
    def test_instantiation(self):
        agent = WritingAssistantAgent()
        assert agent.name == "writing_assistant"

    def test_synthesize_literature(self):
        agent = WritingAssistantAgent()
        result = _run(
            agent.process(
                {
                    "action": "synthesize_literature",
                    "papers": [{"title": "Paper A"}],
                    "topic": "NLP",
                }
            )
        )
        assert result["status"] == "completed"
        assert result["paper_count"] == 1

    def test_summarize_results(self):
        agent = WritingAssistantAgent()
        result = _run(
            agent.process({"action": "summarize_results", "results": {"key": "value"}})
        )
        assert result["status"] == "completed"

    def test_generate_outline(self):
        agent = WritingAssistantAgent()
        result = _run(
            agent.process(
                {
                    "action": "generate_outline",
                    "topic": "AI Ethics",
                    "outline_type": "survey",
                }
            )
        )
        assert result["status"] == "completed"
        assert result["outline_type"] == "survey"

    def test_unknown_action_raises(self):
        agent = WritingAssistantAgent()
        with pytest.raises(ValueError, match="Unknown action"):
            _run(agent.process({"action": "bad"}))


# ---------------------------------------------------------------------------
# KnowledgeGraphAgent
# ---------------------------------------------------------------------------


class TestKnowledgeGraphAgent:
    def test_instantiation(self):
        agent = KnowledgeGraphAgent()
        assert agent.name == "knowledge_graph"

    def test_initialize_graph(self):
        agent = KnowledgeGraphAgent()
        result = _run(
            agent.process(
                {"action": "initialize_graph", "project_name": "Test Project"}
            )
        )
        assert result["status"] == "completed"
        assert "graph_id" in result

    def test_extract_knowledge(self):
        agent = KnowledgeGraphAgent()
        result = _run(
            agent.process(
                {"action": "extract_knowledge", "chunks": ["chunk1", "chunk2"]}
            )
        )
        assert result["status"] == "completed"

    def test_query_graph(self):
        agent = KnowledgeGraphAgent()
        result = _run(
            agent.process({"action": "query_graph", "query": "machine learning"})
        )
        assert result["status"] == "completed"
        assert result["query"] == "machine learning"

    def test_unknown_action_raises(self):
        agent = KnowledgeGraphAgent()
        with pytest.raises(ValueError, match="Unknown action"):
            _run(agent.process({"action": "bad"}))


# ---------------------------------------------------------------------------
# CollaborationAgent
# ---------------------------------------------------------------------------


class TestCollaborationAgent:
    def test_instantiation(self):
        agent = CollaborationAgent()
        assert agent.name == "collaboration"

    def test_assign_task(self):
        agent = CollaborationAgent()
        result = _run(
            agent.process(
                {"action": "assign_task", "member": "alice", "task": "review section 3"}
            )
        )
        assert result["status"] == "completed"
        assert result["member"] == "alice"

    def test_track_progress(self):
        agent = CollaborationAgent()
        result = _run(
            agent.process({"action": "track_progress", "project_id": "project_1"})
        )
        assert result["status"] == "completed"

    def test_share_context(self):
        agent = CollaborationAgent()
        result = _run(
            agent.process({"action": "share_context", "context_data": {"key": "value"}})
        )
        assert result["status"] == "completed"
        assert result["shared"] is True

    def test_unknown_action_raises(self):
        agent = CollaborationAgent()
        with pytest.raises(ValueError, match="Unknown action"):
            _run(agent.process({"action": "bad"}))


# ---------------------------------------------------------------------------
# Base Agent contract
# ---------------------------------------------------------------------------


class TestBaseAgentContract:
    """Verify that every agent follows the ResearchAgent contract."""

    AGENT_CLASSES = [
        LiteratureReviewAgent,
        DataProcessingAgent,
        AnalysisAgent,
        WritingAssistantAgent,
        KnowledgeGraphAgent,
        CollaborationAgent,
    ]

    def test_all_agents_have_name(self):
        for cls in self.AGENT_CLASSES:
            agent = cls()
            assert isinstance(agent.name, str) and len(agent.name) > 0

    def test_all_agents_have_description(self):
        for cls in self.AGENT_CLASSES:
            agent = cls()
            assert isinstance(agent.description, str) and len(agent.description) > 0

    def test_all_agents_require_action_field(self):
        for cls in self.AGENT_CLASSES:
            agent = cls()
            assert "action" in agent.get_required_fields()

    def test_all_agents_reject_missing_action(self):
        for cls in self.AGENT_CLASSES:
            agent = cls()
            with pytest.raises(ValueError):
                _run(agent.process({}))

    def test_initialize_and_cleanup(self):
        for cls in self.AGENT_CLASSES:
            agent = cls()
            _run(agent.initialize({"project": "test"}))
            assert agent.context is not None
            _run(agent.cleanup())
            assert agent.context is None
