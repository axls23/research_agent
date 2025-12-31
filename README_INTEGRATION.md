# Research Agent Integration Guide

## Overview

This document explains how the Research Agent system integrates the frontend, research_agent core, and paperconstructor components with optimal token usage and infinite loop prevention.

## Architecture

```
┌─────────────────┐     ┌──────────────────┐     ┌────────────────┐
│    Frontend     │────►│   API Server     │────►│ Research Agent │
│   (HTML/JS)     │◄────│   (FastAPI)      │◄────│     Core       │
└─────────────────┘     └──────────────────┘     └────────────────┘
                                │                          │
                                │                          │
                                ▼                          ▼
                        ┌──────────────────┐     ┌────────────────┐
                        │ Paper Constructor│     │ Context Manager│
                        │   Integration    │     │ (Token & Loop) │
                        └──────────────────┘     └────────────────┘
```

## Key Components

### 1. Frontend (`frontend/`)
- **index.html**: Main UI with project creation, progress tracking, and paper construction
- **api-bridge.js**: JavaScript API client for backend communication
- **Features**:
  - Real-time progress updates via WebSocket
  - Token usage visualization
  - Automatic status polling
  - Error handling and user feedback

### 2. API Server (`research_agent/api/`)
- **server.py**: FastAPI server with REST endpoints and WebSocket support
- **models.py**: Pydantic models for request/response validation
- **integration.py**: Bridge between research_agent and paperconstructor
- **Endpoints**:
  - `POST /api/research/create`: Create new research project
  - `GET /api/research/{id}/status`: Get project status
  - `POST /api/research/{id}/construct-paper`: Generate paper
  - `GET /api/research/{id}/context`: Get context state
  - `WS /ws/{id}`: WebSocket for real-time updates

### 3. Context Management (`research_agent/utils/context_manager.py`)
- **Token Budget Management**:
  - Tracks token usage across all operations
  - Implements pruning strategies (LRU, relevance, age)
  - Prevents token budget overflow
- **Infinite Loop Prevention**:
  - Detects execution patterns
  - Limits iterations per agent/action
  - Identifies 2-step and 3-step cyclic patterns
  - Maintains execution history with sliding window

### 4. Integration Layer (`research_agent/api/integration.py`)
- Connects research_agent workflows with paperconstructor
- Manages agent execution with loop prevention
- Handles paper construction requests
- Provides progress callbacks

## How to Run

1. **Install Dependencies**:
```bash
pip install -e .
```

2. **Set Environment Variables**:
```bash
export OPENAI_API_KEY="your-openai-api-key"
```

3. **Run the System**:
```bash
python run_research_agent.py
```

This will:
- Start the FastAPI server on http://localhost:8000
- Open the frontend in your browser
- Enable all integrations

## Token Management

The system implements smart token management:

1. **Budget Tracking**: Each project has a 100k token limit
2. **Usage Monitoring**: Real-time token usage displayed in UI
3. **Automatic Pruning**: When approaching limit, least relevant context is removed
4. **Warning System**: Users warned at 80% usage

## Loop Prevention

Multiple strategies prevent infinite loops:

1. **Execution Counting**: Max 10 iterations per unique agent/action/input
2. **Pattern Detection**: Identifies repeating sequences
3. **Time Windows**: 20-execution sliding window for pattern analysis
4. **Graceful Handling**: Loops are broken with informative messages

## Usage Example

1. **Create Research Project**:
   - Open frontend
   - Enter project name and description
   - Select task type (Literature Review, etc.)
   - Click "Start Research"

2. **Monitor Progress**:
   - Watch real-time updates in progress panel
   - Monitor token usage meter
   - View completion status

3. **Construct Paper**:
   - Once research completes, enter paper details
   - Select citation style and format
   - Click "Construct Paper"
   - Download generated document

## API Usage

```javascript
// Create project
const response = await fetch('http://localhost:8000/api/research/create', {
    method: 'POST',
    headers: {'Content-Type': 'application/json'},
    body: JSON.stringify({
        project_name: "Climate Change Research",
        description: "Analyze recent climate change papers",
        task_type: "literature_review"
    })
});

// Connect WebSocket for updates
const ws = new WebSocket(`ws://localhost:8000/ws/${projectId}`);
ws.onmessage = (event) => {
    const update = JSON.parse(event.data);
    console.log('Progress:', update);
};
```

## Configuration

Edit `research_agent/config.yaml` for:
- Token limits
- Loop detection thresholds
- API settings
- Agent configurations

## Troubleshooting

1. **Token Limit Exceeded**:
   - System will automatically prune old context
   - Consider breaking large tasks into smaller ones

2. **Loop Detected**:
   - Check agent logic for circular dependencies
   - Review workflow definitions

3. **Connection Issues**:
   - Ensure API server is running
   - Check firewall settings
   - Verify environment variables

## Development

To extend the system:

1. **Add New Agents**: Create in `research_agent/agents/`
2. **Add Workflows**: Define in `research_agent/core/workflow.py`
3. **Extend API**: Add endpoints in `research_agent/api/server.py`
4. **Update Frontend**: Modify `frontend/index.html` and `api-bridge.js`
