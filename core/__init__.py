"""
core — research agent core components.

Provides:
- ResearchAgent base class
- Orchestrator (legacy workflow API)
- LangGraph pipeline (graph.py)
- State management
- LLM providers
"""
from core.base_agent import ResearchAgent
from core.orchestrator import (
    ResearchWorkflowOrchestrator,
    Task,
    Workflow,
    Context,
    AgentRegistry,
)