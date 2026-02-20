# Research Agent System

An AI-powered academic research assistant that uses a **multi-agent
architecture** to help researchers reduce information overload.

## Why Multi-Agent?

Academic research spans distinct phases — literature discovery, data
processing, analysis, writing, and knowledge management.  A multi-agent
design maps each phase to a **dedicated, independently testable agent**
coordinated by a lightweight orchestrator:

| Agent | Responsibility |
|---|---|
| **LiteratureReviewAgent** | Search-query formulation, paper retrieval & relevance filtering |
| **DataProcessingAgent** | Document ingestion, text extraction, chunking |
| **AnalysisAgent** | Exploratory data analysis, statistical testing, visualization |
| **WritingAssistantAgent** | Literature synthesis, results summarization, outline generation |
| **KnowledgeGraphAgent** | Knowledge-graph creation, entity extraction, cross-paper querying |
| **CollaborationAgent** | Task delegation, progress tracking, shared context |

Benefits:
- **Separation of concerns** – each agent has a small, focused prompt context
- **Composable workflows** – the orchestrator chains agents into configurable
  research pipelines (e.g. *literature review → knowledge extraction → synthesis*)
- **Independent testing** – every agent is unit-testable in isolation (60 tests)
- **Extensibility** – new agents or workflows can be added without touching
  existing ones

## Project Structure

```
research_agent/
├── agents/                    # Specialized research agents
│   ├── literature_review_agent.py
│   ├── data_processing_agent.py
│   ├── analysis_agent.py
│   ├── writing_assistant_agent.py
│   ├── knowledge_graph_agent.py
│   └── collaboration_agent.py
├── core/                      # Core system components
│   ├── base_agent.py         # ResearchAgent base class
│   ├── orchestrator.py       # Multi-agent workflow orchestrator
│   ├── workflow.py           # Workflow & Task definitions
│   ├── context.py            # Execution context
│   └── registry.py           # Agent registry
├── utils/                     # Utility functions
│   ├── logger.py             # Logging configuration
│   ├── config.py             # Configuration management
│   └── helpers.py            # Helper functions
├── data/                      # Data storage
│   ├── chunks/               # Research paper chunks
│   └── knowledge_graph/      # Knowledge graph data
├── tests/                     # Test suite (60 tests)
│   ├── agents/
│   ├── core/
│   └── utils/
└── docs/                      # Documentation
    ├── api/
    └── examples/
```

## Features

- **Workflow Management**: Define and execute complex research workflows
- **Agent Integration**: Seamless integration with specialized research agents
- **Knowledge Graph**: Maintain and query research knowledge
- **Collaboration**: Support for team-based research projects
- **Progress Tracking**: Monitor research progress and suggest next steps

## Installation

```bash
pip install -e .
```

## Usage

```python
from core.orchestrator import ResearchWorkflowOrchestrator

# Initialize orchestrator with researcher preferences
orchestrator = ResearchWorkflowOrchestrator({"citation_format": "apa"})

# Start a research project
project_id = await orchestrator.start_research_project(
    "AI Research Project",
    "Analysis of recent AI developments"
)

# Start a workflow pipeline
workflow_id = await orchestrator.start_research_workflow(
    project_id,
    "literature_review",
    custom_parameters={"focus_area": "machine_learning"}
)

# Check progress and get suggestions
progress = await orchestrator.get_research_progress(project_id)
next_steps = await orchestrator.suggest_next_steps(project_id)
```

## Development

1. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

2. Install development dependencies:
```bash
pip install -r requirements-dev.txt
```

3. Run tests:
```bash
pytest
```

## Contributing

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request

## License
[MIT License](https://github.com/axls23/research_agent/blob/main/LICENSE)