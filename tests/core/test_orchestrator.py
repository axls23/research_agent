"""Tests for the multi-agent orchestrator, registry, workflow, and context.

Validates that:
  - The orchestrator instantiates and registers all six agents
  - Projects and workflows can be created and tracked
  - The agent registry can call any registered agent
  - Workflow templates contain the expected tasks
  - suggest_next_steps returns sensible recommendations
  - Error cases (unknown project / workflow) raise properly
"""

import sys
from pathlib import Path
import asyncio
import pytest

# Ensure the project root is on sys.path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from core.orchestrator import ResearchWorkflowOrchestrator
from core.workflow import Workflow, Task
from core.context import Context
from core.registry import AgentRegistry


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _run(coro):
    """Run an async coroutine synchronously for test convenience."""
    return asyncio.get_event_loop().run_until_complete(coro)


PREFS = {"citation_format": "apa"}


# ---------------------------------------------------------------------------
# Workflow & Task data classes
# ---------------------------------------------------------------------------


class TestWorkflowAndTask:
    def test_task_creation(self):
        task = Task("my_task", "SomeAgent", {"action": "do_stuff"}, ["dep1"])
        assert task.name == "my_task"
        assert task.agent_type == "SomeAgent"
        assert task.parameters == {"action": "do_stuff"}
        assert task.dependencies == ["dep1"]

    def test_task_defaults(self):
        task = Task("t", "A")
        assert task.parameters == {}
        assert task.dependencies == []

    def test_workflow_creation(self):
        wf = Workflow("wf", "A workflow", [Task("t1", "A"), Task("t2", "B")])
        assert wf.name == "wf"
        assert len(wf.tasks) == 2


# ---------------------------------------------------------------------------
# Context
# ---------------------------------------------------------------------------


class TestContext:
    def test_defaults(self):
        ctx = Context()
        assert ctx.project_id == ""
        assert ctx.knowledge_graph_id is None
        assert ctx.extra == {}

    def test_custom(self):
        ctx = Context(
            project_id="p1",
            knowledge_graph_id="kg_1",
            researcher_preferences={"style": "apa"},
            extra={"focus": "ml"},
        )
        assert ctx.project_id == "p1"
        assert ctx.extra["focus"] == "ml"


# ---------------------------------------------------------------------------
# AgentRegistry
# ---------------------------------------------------------------------------


class TestAgentRegistry:
    def test_register_and_list(self):
        from agents.literature_review_agent import LiteratureReviewAgent

        reg = AgentRegistry()
        reg.register("LitReview", LiteratureReviewAgent())
        assert "LitReview" in reg.list_agents()

    def test_get_unknown_returns_none(self):
        reg = AgentRegistry()
        assert reg.get("nonexistent") is None

    def test_call_unknown_raises(self):
        reg = AgentRegistry()
        with pytest.raises(KeyError, match="not registered"):
            _run(reg.call_agent("missing", {}))

    def test_call_agent(self):
        from agents.knowledge_graph_agent import KnowledgeGraphAgent

        reg = AgentRegistry()
        reg.register("KG", KnowledgeGraphAgent())
        result = _run(
            reg.call_agent(
                "KG",
                {
                    "action": "initialize_graph",
                    "project_name": "Test",
                },
            )
        )
        assert result["status"] == "completed"
        assert "graph_id" in result


# ---------------------------------------------------------------------------
# Orchestrator – instantiation
# ---------------------------------------------------------------------------


class TestOrchestratorInit:
    def test_creates_with_preferences(self):
        orch = ResearchWorkflowOrchestrator(PREFS)
        assert orch.researcher_preferences == PREFS

    def test_registers_all_agents(self):
        orch = ResearchWorkflowOrchestrator(PREFS)
        names = orch.agent_registry.list_agents()
        assert "LiteratureReviewAgent" in names
        assert "DataProcessingAgent" in names
        assert "AnalysisAgent" in names
        assert "WritingAssistantAgent" in names
        assert "KnowledgeGraphAgent" in names
        assert "CollaborationAgent" in names

    def test_has_workflow_templates(self):
        orch = ResearchWorkflowOrchestrator(PREFS)
        assert "literature_review" in orch.workflow_templates
        assert "data_analysis" in orch.workflow_templates


# ---------------------------------------------------------------------------
# Orchestrator – project lifecycle
# ---------------------------------------------------------------------------


class TestOrchestratorProjects:
    def test_start_project(self):
        orch = ResearchWorkflowOrchestrator(PREFS)
        pid = _run(orch.start_research_project("AI Research", "desc"))
        assert pid == "project_1"
        assert pid in orch.active_research_projects
        assert orch.active_research_projects[pid]["knowledge_graph"] is not None

    def test_start_workflow(self):
        orch = ResearchWorkflowOrchestrator(PREFS)
        pid = _run(orch.start_research_project("P1", "d"))
        wid = _run(orch.start_research_workflow(pid, "literature_review"))
        assert "literature_review" in wid
        assert wid in orch.active_research_projects[pid]["workflows"]

    def test_start_workflow_with_custom_params(self):
        orch = ResearchWorkflowOrchestrator(PREFS)
        pid = _run(orch.start_research_project("P1", "d"))
        wid = _run(
            orch.start_research_workflow(pid, "data_analysis", {"focus": "stats"})
        )
        wf = orch.active_research_projects[pid]["workflows"][wid]
        assert wf["context"].extra == {"focus": "stats"}

    def test_unknown_project_raises(self):
        orch = ResearchWorkflowOrchestrator(PREFS)
        with pytest.raises(ValueError, match="Unknown project"):
            _run(orch.start_research_workflow("bad", "literature_review"))

    def test_unknown_workflow_type_raises(self):
        orch = ResearchWorkflowOrchestrator(PREFS)
        pid = _run(orch.start_research_project("P", "d"))
        with pytest.raises(ValueError, match="Unknown workflow type"):
            _run(orch.start_research_workflow(pid, "nonexistent"))


# ---------------------------------------------------------------------------
# Orchestrator – progress & suggestions
# ---------------------------------------------------------------------------


class TestOrchestratorProgress:
    def test_get_progress(self):
        orch = ResearchWorkflowOrchestrator(PREFS)
        pid = _run(orch.start_research_project("P", "d"))
        _run(orch.start_research_workflow(pid, "literature_review"))
        progress = _run(orch.get_research_progress(pid))
        assert progress["project_id"] == pid
        assert len(progress["workflows"]) == 1

    def test_progress_unknown_project_raises(self):
        orch = ResearchWorkflowOrchestrator(PREFS)
        with pytest.raises(ValueError, match="Unknown project"):
            _run(orch.get_research_progress("bad"))

    def test_suggest_next_steps(self):
        orch = ResearchWorkflowOrchestrator(PREFS)
        pid = _run(orch.start_research_project("P", "d"))
        suggestions = _run(orch.suggest_next_steps(pid))
        types = [s["workflow_type"] for s in suggestions]
        assert "literature_review" in types
        assert "data_analysis" in types

    def test_suggest_excludes_started(self):
        orch = ResearchWorkflowOrchestrator(PREFS)
        pid = _run(orch.start_research_project("P", "d"))
        _run(orch.start_research_workflow(pid, "literature_review"))
        suggestions = _run(orch.suggest_next_steps(pid))
        types = [s["workflow_type"] for s in suggestions]
        assert "literature_review" not in types
        assert "data_analysis" in types

    def test_suggest_unknown_project_raises(self):
        orch = ResearchWorkflowOrchestrator(PREFS)
        with pytest.raises(ValueError, match="Unknown project"):
            _run(orch.suggest_next_steps("bad"))


# ---------------------------------------------------------------------------
# Workflow template content
# ---------------------------------------------------------------------------


class TestWorkflowTemplates:
    def test_literature_review_pipeline(self):
        orch = ResearchWorkflowOrchestrator(PREFS)
        wf = orch.workflow_templates["literature_review"]
        task_names = [t.name for t in wf.tasks]
        assert task_names == [
            "query_formulation",
            "paper_retrieval",
            "paper_filtering",
            "knowledge_extraction",
            "synthesis",
        ]

    def test_data_analysis_pipeline(self):
        orch = ResearchWorkflowOrchestrator(PREFS)
        wf = orch.workflow_templates["data_analysis"]
        task_names = [t.name for t in wf.tasks]
        assert task_names == [
            "data_prep",
            "exploratory_analysis",
            "statistical_testing",
            "visualization",
            "results_summary",
        ]

    def test_customize_injects_preferred_methods(self):
        prefs = {"preferred_statistical_methods": ["t-test", "anova"]}
        orch = ResearchWorkflowOrchestrator(prefs)
        wf = orch._customize_workflow(orch.workflow_templates["data_analysis"])
        stat_task = next((t for t in wf.tasks if t.name == "statistical_testing"), None)
        assert stat_task is not None, "statistical_testing task not found"
        assert stat_task.parameters["preferred_methods"] == ["t-test", "anova"]
