from google.adk.agents import Agents, AgentRegistry, Workflow, Task, Context
from typing import Dict, List, Optional
import time
import copy

class ResearchWorkflowOrchestrator(Agents):
    def __init__(self, researcher_preferences: Dict):
        super().__init__("ResearchOrchestrator")
        self.researcher_preferences = researcher_preferences
        self.active_research_projects = {}
        self.workflow_templates = self._initialize_workflow_templates()
        self.agent_registry = AgentRegistry()
        self._register_agents()
        
    def _initialize_workflow_templates(self) -> Dict:
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
                Task("synthesis", "WritingAssistantAgent", {"action": "synthesize_literature"}, ["knowledge_extraction"])
            ]
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
                Task("results_summary", "WritingAssistantAgent", {"action": "summarize_results"}, ["visualization"])
            ]
        )
        templates["data_analysis"] = data_analysis
        return templates
    
    def _register_agents(self):
        agents = [
            "LiteratureReviewAgent", "DataProcessingAgent", "ExperimentDesignAgent",
            "AnalysisAgent", "WritingAssistantAgent", "CollaborationAgent", "KnowledgeGraphAgent"
        ]
        for agent in agents:
            self.agent_registry.register(agent, f"https://{agent.lower().replace('agent', '-agent')}.example.com/a2a")
    
    async def start_research_project(self, project_name: str, description: str) -> str:
        project_id = f"project_{len(self.active_research_projects) + 1}"
        kg_result = await self.agent_registry.call_agent(
            "KnowledgeGraphAgent",
            {"action": "initialize_graph", "project_name": project_name}
        )
        self.active_research_projects[project_id] = {
            "id": project_id, "name": project_name, "description": description,
            "workflows": {}, "knowledge_graph": kg_result.get("graph_id"),
            "documents": [], "team_members": [], "created_at": time.time()
        }
        return project_id
    
    async def start_research_workflow(self, project_id: str, workflow_type: str, custom_parameters: Optional[Dict] = None) -> str:
        if project_id not in self.active_research_projects or workflow_type not in self.workflow_templates:
            raise Exception(f"Unknown {'project' if project_id not in self.active_research_projects else 'workflow type'}")
        
        project = self.active_research_projects[project_id]
        workflow_instance = self._customize_workflow(self.workflow_templates[workflow_type], custom_parameters)
        instance_id = f"{project_id}_{workflow_type}_{len(project['workflows']) + 1}"
        
        context = Context(
            project_id=project_id,
            researcher_preferences=self.researcher_preferences,
            knowledge_graph_id=project["knowledge_graph"],
            **(custom_parameters or {})
        )
        
        project["workflows"][instance_id] = {
            "type": workflow_type,
            "status": "started",
            "started_at": time.time(),
            "context": context
        }
        return instance_id
    
    def _customize_workflow(self, workflow_template: Workflow, custom_parameters: Optional[Dict] = None) -> Workflow:
        customized = copy.deepcopy(workflow_template)
        if self.researcher_preferences.get("preferred_statistical_methods"):
            for task in customized.tasks:
                if task.agent_type == "AnalysisAgent" and "run_statistical_tests" in task.parameters.get("action", ""):
                    task.parameters["preferred_methods"] = self.researcher_preferences["preferred_statistical_methods"]
        return customized
    
    async def get_research_progress(self, project_id: str) -> Dict:
        if project_id not in self.active_research_projects:
            raise Exception(f"Unknown project: {project_id}")
        project = self.active_research_projects[project_id]
        return {
            "project_id": project_id,
            "name": project["name"],
            "workflows": [{"instance_id": i, "type": w["type"], "status": w["status"], "started_at": w["started_at"]}
                         for i, w in project["workflows"].items()],
            "documents": len(project["documents"])
        }
    
    async def suggest_next_steps(self, project_id: str) -> List[Dict]:
        if project_id not in self.active_research_projects:
            raise Exception(f"Unknown project: {project_id}")
        project = self.active_research_projects[project_id]
        active_workflows = [w for w in project["workflows"].values() if w["status"] != "completed"]
        completed_workflows = [w for w in project["workflows"].values() if w["status"] == "completed"]
        
        suggestions = []
        workflow_checks = [
            ("literature_review", "No literature review has been initiated yet. This is a good first step."),
            ("data_analysis", "Literature review is complete. Next step is to analyze your data."),
            ("paper_writing", "Data analysis is complete. You can now draft your research paper.")
        ]
        
        for workflow_type, reason in workflow_checks:
            if not any(w["type"] == workflow_type for w in active_workflows + completed_workflows):
                suggestions.append({"action": "start_workflow", "workflow_type": workflow_type, "reason": reason})
        return suggestions 