"""Multi-agent research workflow orchestrator.

Coordinates the six specialized agents through configurable research
pipelines (literature review, data analysis, …) while keeping each
agent focused on a single responsibility.

Why multi-agent?
    Academic research involves distinct phases — discovery, processing,
    analysis, writing, and knowledge management.  Mapping each phase to
    a dedicated agent keeps prompt context small, makes each agent
    independently testable, and lets workflows compose them in any
    order the researcher needs.
"""

from __future__ import annotations

import copy
import logging
import time
from typing import Dict, List, Optional

from core.context import Context
from core.registry import AgentRegistry
from core.workflow import Task, Workflow

logger = logging.getLogger(__name__)


class ResearchWorkflowOrchestrator:
    """Top-level orchestrator that manages research projects, registers
    agents, and instantiates workflow pipelines.

    Parameters
    ----------
    researcher_preferences:
        User preferences (citation style, preferred statistical methods, …)
        that are threaded through every workflow context.
    """

    def __init__(self, researcher_preferences: Dict) -> None:
        self.researcher_preferences = researcher_preferences
        self.active_research_projects: Dict[str, Dict] = {}
        self.workflow_templates = self._initialize_workflow_templates()
        self.agent_registry = AgentRegistry()
        self._register_agents()

    # ------------------------------------------------------------------
    # Workflow templates
    # ------------------------------------------------------------------

    @staticmethod
    def _initialize_workflow_templates() -> Dict[str, Workflow]:
        templates: Dict[str, Workflow] = {}

        # Literature Review Workflow
        templates["literature_review"] = Workflow(
            name="literature_review",
            description="Comprehensive Literature Review",
            tasks=[
                Task("query_formulation", "LiteratureReviewAgent",
                     {"action": "formulate_search_query"}),
                Task("paper_retrieval", "LiteratureReviewAgent",
                     {"action": "retrieve_papers"},
                     ["query_formulation"]),
                Task("paper_filtering", "LiteratureReviewAgent",
                     {"action": "filter_papers"},
                     ["paper_retrieval"]),
                Task("knowledge_extraction", "KnowledgeGraphAgent",
                     {"action": "extract_knowledge"},
                     ["paper_filtering"]),
                Task("synthesis", "WritingAssistantAgent",
                     {"action": "synthesize_literature"},
                     ["knowledge_extraction"]),
            ],
        )

        # Data Analysis Workflow
        templates["data_analysis"] = Workflow(
            name="data_analysis",
            description="Complete Data Analysis Pipeline",
            tasks=[
                Task("data_prep", "DataProcessingAgent",
                     {"action": "prepare_data"}),
                Task("exploratory_analysis", "AnalysisAgent",
                     {"action": "explore_data"},
                     ["data_prep"]),
                Task("statistical_testing", "AnalysisAgent",
                     {"action": "run_statistical_tests"},
                     ["exploratory_analysis"]),
                Task("visualization", "AnalysisAgent",
                     {"action": "create_visualizations"},
                     ["statistical_testing"]),
                Task("results_summary", "WritingAssistantAgent",
                     {"action": "summarize_results"},
                     ["visualization"]),
            ],
        )

        return templates

    # ------------------------------------------------------------------
    # Agent registration
    # ------------------------------------------------------------------

    def _register_agents(self) -> None:
        """Instantiate and register all specialized agents.

        Imports are deferred to method scope to avoid circular imports
        (agents depend on ``core.base_agent`` which lives in the same
        ``core`` package as the orchestrator).
        """
        from agents.literature_review_agent import LiteratureReviewAgent
        from agents.data_processing_agent import DataProcessingAgent
        from agents.analysis_agent import AnalysisAgent
        from agents.writing_assistant_agent import WritingAssistantAgent
        from agents.knowledge_graph_agent import KnowledgeGraphAgent
        from agents.collaboration_agent import CollaborationAgent

        agent_classes = {
            "LiteratureReviewAgent": LiteratureReviewAgent,
            "DataProcessingAgent": DataProcessingAgent,
            "AnalysisAgent": AnalysisAgent,
            "WritingAssistantAgent": WritingAssistantAgent,
            "KnowledgeGraphAgent": KnowledgeGraphAgent,
            "CollaborationAgent": CollaborationAgent,
        }
        for name, cls in agent_classes.items():
            self.agent_registry.register_class(name, cls)

    # ------------------------------------------------------------------
    # Project management
    # ------------------------------------------------------------------

    async def start_research_project(
        self, project_name: str, description: str
    ) -> str:
        """Create a new research project and initialize its knowledge graph."""
        project_id = f"project_{len(self.active_research_projects) + 1}"

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

    # ------------------------------------------------------------------
    # Workflow execution
    # ------------------------------------------------------------------

    async def start_research_workflow(
        self,
        project_id: str,
        workflow_type: str,
        custom_parameters: Optional[Dict] = None,
    ) -> str:
        """Start a workflow of the given *workflow_type* inside *project_id*."""
        if project_id not in self.active_research_projects:
            raise ValueError(f"Unknown project: {project_id}")
        if workflow_type not in self.workflow_templates:
            raise ValueError(f"Unknown workflow type: {workflow_type}")

        project = self.active_research_projects[project_id]
        self._customize_workflow(
            self.workflow_templates[workflow_type], custom_parameters
        )
        instance_id = (
            f"{project_id}_{workflow_type}_{len(project['workflows']) + 1}"
        )

        context = Context(
            project_id=project_id,
            researcher_preferences=self.researcher_preferences,
            knowledge_graph_id=project.get("knowledge_graph"),
            extra=custom_parameters or {},
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
        """Return a deep copy of *workflow_template* with researcher
        preferences injected into matching tasks."""
        customized = copy.deepcopy(workflow_template)
        preferred = self.researcher_preferences.get(
            "preferred_statistical_methods"
        )
        if preferred:
            for task in customized.tasks:
                if (
                    task.agent_type == "AnalysisAgent"
                    and "run_statistical_tests"
                    in task.parameters.get("action", "")
                ):
                    task.parameters["preferred_methods"] = preferred
        return customized

    # ------------------------------------------------------------------
    # Progress & suggestions
    # ------------------------------------------------------------------

    async def get_research_progress(self, project_id: str) -> Dict:
        """Return a progress summary for *project_id*."""
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
        """Suggest next research workflow(s) based on current progress."""
        if project_id not in self.active_research_projects:
            raise ValueError(f"Unknown project: {project_id}")
        project = self.active_research_projects[project_id]
        all_workflows = list(project["workflows"].values())
        active = [w for w in all_workflows if w["status"] != "completed"]
        completed = [w for w in all_workflows if w["status"] == "completed"]

        suggestions: List[Dict] = []
        checks = [
            (
                "literature_review",
                "No literature review has been initiated yet. "
                "This is a good first step.",
            ),
            (
                "data_analysis",
                "Literature review is complete. "
                "Next step is to analyze your data.",
            ),
            (
                "paper_writing",
                "Data analysis is complete. "
                "You can now draft your research paper.",
            ),
        ]
        for wf_type, reason in checks:
            if not any(w["type"] == wf_type for w in active + completed):
                suggestions.append(
                    {
                        "action": "start_workflow",
                        "workflow_type": wf_type,
                        "reason": reason,
                    }
                )
        return suggestions 