# Research Agent System

A sophisticated research workflow orchestration system built on Google's Agent Development Kit (ADK).

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
│   ├── orchestrator.py        # Main workflow orchestrator
│   ├── workflow.py           # Workflow definitions
│   ├── context.py            # Context management
│   └── registry.py           # Agent registry
├── utils/                     # Utility functions
│   ├── logger.py             # Logging configuration
│   ├── config.py             # Configuration management
│   └── helpers.py            # Helper functions
├── data/                      # Data storage
│   ├── chunks/               # Research paper chunks
│   └── knowledge_graph/      # Knowledge graph data
├── tests/                     # Test suite
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
from research_agent.core import ResearchWorkflowOrchestrator

# Initialize orchestrator
orchestrator = ResearchWorkflowOrchestrator(researcher_preferences)

# Start a research project
project_id = await orchestrator.start_research_project(
    "AI Research Project",
    "Analysis of recent AI developments"
)

# Start a workflow
workflow_id = await orchestrator.start_research_workflow(
    project_id,
    "literature_review",
    custom_parameters={"focus_area": "machine_learning"}
)
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