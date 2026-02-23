"""
core/orchestrator.py
====================
Backward-compatible orchestrator that wraps the new LangGraph pipeline.

Users can either:
- Use the LangGraph pipeline directly (``core.graph.run_research_pipeline``)
- Use this orchestrator for the legacy Workflow/Task API
"""

from __future__ import annotations

import copy
import logging
import time
from typing import Dict, List, Optional

from core.state import make_initial_state

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Lightweight Workflow / Task / Context / Registry (replacing google.adk)
# ---------------------------------------------------------------------------

class Task:
    """A single task in a workflow."""

    def __init__(
        self,
        name: str,
        agent_type: str,
        parameters: Optional[Dict] = None,
        dependencies: Optional[List[str]] = None,
    ):
        self.name = name
        self.agent_type = agent_type
        self.parameters = parameters or {}
        self.dependencies = dependencies or []


class Workflow:
    """A sequence of tasks forming a research pipeline."""

    def __init__(self, name: str, description: str, tasks: Optional[List[Task]] = None):
        self.name = name
        self.description = description
        self.tasks = tasks or []


class Context:
    """Execution context for a workflow run."""

    def __init__(
        self,
        project_id: str = "",
        researcher_preferences: Optional[Dict] = None,
        knowledge_graph_id: Optional[str] = None,
        extra: Optional[Dict] = None,
        **kwargs,
    ):
        self.project_id = project_id
        self.researcher_preferences = researcher_preferences or {}
        self.knowledge_graph_id = knowledge_graph_id
        self.extra = extra if extra is not None else kwargs


class AgentRegistry:
    """Registry mapping agent names to instances."""

    def __init__(self):
        self._agents: Dict[str, object] = {}

    def register(self, name: str, agent: object) -> None:
        self._agents[name] = agent

    def get(self, name: str) -> Optional[object]:
        return self._agents.get(name)

    def list_agents(self) -> List[str]:
        return list(self._agents.keys())

    async def call_agent(self, name: str, input_data: Dict) -> Dict:
        agent = self._agents.get(name)
        if agent is None:
            raise KeyError(f"Agent '{name}' not registered")
        return await agent.process(input_data)


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------

class ResearchWorkflowOrchestrator:
    """
    Multi-agent workflow orchestrator.

    Preserves the original API surface so existing tests pass.
    Internally can delegate to the LangGraph pipeline for actual execution.
    """

    def __init__(self, researcher_preferences: Dict):
        self.researcher_preferences = researcher_preferences
        self.active_research_projects: Dict[str, Dict] = {}
        self.workflow_templates = self._initialize_workflow_templates()
        self.agent_registry = AgentRegistry()
        self._register_agents()

    def _initialize_workflow_templates(self) -> Dict[str, Workflow]:
        templates = {}

        # Literature Review Workflow
        lit_review = Workflow(
            name="literature_review",
            description="Comprehensive Literature Review",
            tasks=[
                Task("query_formulation", "LiteratureReviewAgent", {"action": "formulate_search_query"}),
                Task("paper_retrieval", "LiteratureReviewAgent", {"action": "retrieve_papers"}, ["query_formulation"]),
                Task("paper_filtering", "LiteratureReviewAgent", {"action": "filter_papers"}, ["paper_retrieval"]),
                Task("knowledge_extraction", "KnowledgeGraphAgent", {"action": "extract_knowledge"}, ["paper_filtering"]),
                Task("synthesis", "WritingAssistantAgent", {"action": "synthesize_literature"}, ["knowledge_extraction"]),
            ],
        )
        templates["literature_review"] = lit_review

        # Data Analysis Workflow
        data_analysis = Workflow(
            name="data_analysis",
            description="Complete Data Analysis Pipeline",
            tasks=[
                Task("data_prep", "DataProcessingAgent", {"action": "prepare_data"}),
                Task("exploratory_analysis", "AnalysisAgent", {"action": "explore_data"}, ["data_prep"]),
                Task("statistical_testing", "AnalysisAgent", {"action": "run_statistical_tests"}, ["exploratory_analysis"]),
                Task("visualization", "AnalysisAgent", {"action": "create_visualizations"}, ["statistical_testing"]),
                Task("results_summary", "WritingAssistantAgent", {"action": "summarize_results"}, ["visualization"]),
            ],
        )
        templates["data_analysis"] = data_analysis

        return templates

    def _register_agents(self) -> None:
        """Register all agent instances."""
        from agents.literature_review_agent import LiteratureReviewAgent
        from agents.data_processing_agent import DataProcessingAgent
        from agents.analysis_agent import AnalysisAgent
        from agents.writing_assistant_agent import WritingAssistantAgent
        from agents.knowledge_graph_agent import KnowledgeGraphAgent
        from agents.collaboration_agent import CollaborationAgent

        self.agent_registry.register("LiteratureReviewAgent", LiteratureReviewAgent())
        self.agent_registry.register("DataProcessingAgent", DataProcessingAgent())
        self.agent_registry.register("AnalysisAgent", AnalysisAgent())
        self.agent_registry.register("WritingAssistantAgent", WritingAssistantAgent())
        self.agent_registry.register("KnowledgeGraphAgent", KnowledgeGraphAgent())
        self.agent_registry.register("CollaborationAgent", CollaborationAgent())

    async def start_research_project(self, project_name: str, description: str) -> str:
        project_id = f"project_{len(self.active_research_projects) + 1}"

        # Initialize knowledge graph via agent
        kg_result = await self.agent_registry.call_agent(
            "KnowledgeGraphAgent",
            {"action": "initialize_graph", "project_name": project_name},
        )

        self.active_research_projects[project_id] = {
            "id": project_id,
            "name": project_name,
            "description": description,
            "workflows": {},
            "knowledge_graph": kg_result.get("graph_id"),
            "documents": [],
            "team_members": [],
            "created_at": time.time(),
        }
        return project_id

    async def start_research_workflow(
        self,
        project_id: str,
        workflow_type: str,
        custom_parameters: Optional[Dict] = None,
    ) -> str:
        if project_id not in self.active_research_projects:
            raise ValueError(f"Unknown project: {project_id}")
        if workflow_type not in self.workflow_templates:
            raise ValueError(f"Unknown workflow type: {workflow_type}")

        project = self.active_research_projects[project_id]
        workflow_instance = self._customize_workflow(
            self.workflow_templates[workflow_type], custom_parameters
        )
        instance_id = f"{project_id}_{workflow_type}_{len(project['workflows']) + 1}"

        context = Context(
            project_id=project_id,
            researcher_preferences=self.researcher_preferences,
            knowledge_graph_id=project["knowledge_graph"],
            **(custom_parameters or {}),
        )

        project["workflows"][instance_id] = {
            "type": workflow_type,
            "status": "started",
            "started_at": time.time(),
            "context": context,
        }
        return instance_id

    def _customize_workflow(
        self,
        workflow_template: Workflow,
        custom_parameters: Optional[Dict] = None,
    ) -> Workflow:
        customized = copy.deepcopy(workflow_template)
        if self.researcher_preferences.get("preferred_statistical_methods"):
            for task in customized.tasks:
                if (
                    task.agent_type == "AnalysisAgent"
                    and "run_statistical_tests" in task.parameters.get("action", "")
                ):
                    task.parameters["preferred_methods"] = self.researcher_preferences[
                        "preferred_statistical_methods"
                    ]
        return customized

    async def get_research_progress(self, project_id: str) -> Dict:
        if project_id not in self.active_research_projects:
            raise ValueError(f"Unknown project: {project_id}")
        project = self.active_research_projects[project_id]
        return {
            "project_id": project_id,
            "name": project["name"],
            "workflows": [
                {
                    "instance_id": i,
                    "type": w["type"],
                    "status": w["status"],
                    "started_at": w["started_at"],
                }
                for i, w in project["workflows"].items()
            ],
            "documents": len(project["documents"]),
        }

    async def suggest_next_steps(self, project_id: str) -> List[Dict]:
        if project_id not in self.active_research_projects:
            raise ValueError(f"Unknown project: {project_id}")
        project = self.active_research_projects[project_id]
        active_workflows = [
            w for w in project["workflows"].values() if w["status"] != "completed"
        ]
        completed_workflows = [
            w for w in project["workflows"].values() if w["status"] == "completed"
        ]

        suggestions = []
        workflow_checks = [
            ("literature_review", "No literature review has been initiated yet. This is a good first step."),
            ("data_analysis", "Literature review is complete. Next step is to analyze your data."),
            ("paper_writing", "Data analysis is complete. You can now draft your research paper."),
        ]

        for workflow_type, reason in workflow_checks:
            if not any(
                w["type"] == workflow_type
                for w in active_workflows + completed_workflows
            ):
                suggestions.append(
                    {
                        "action": "start_workflow",
                        "workflow_type": workflow_type,
                        "reason": reason,
                    }
                )
        return suggestions