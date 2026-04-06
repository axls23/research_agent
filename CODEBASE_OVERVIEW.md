# Research Agent System - Codebase Overview

## 🎯 Project Summary

**Name:** Research Agent System
**Repository:** axls23/research_agent
**Purpose:** AI-powered academic research assistant using multi-agent architecture to reduce information overload

---

## 🏗️ Multi-Agent Architecture

The system employs a **specialized multi-agent design** where each agent handles a distinct phase of the research workflow:

### Core Agents

| Agent | Primary Responsibility | Key Features |
|-------|----------------------|--------------|
| **LiteratureReviewAgent** | Search & retrieval | PICO decomposition, multi-database search (ArXiv, S2, Crossref), citation snowballing, two-pass filtering |
| **DataProcessingAgent** | Document processing | PDF ingestion, text extraction, intelligent chunking (300 tokens default) |
| **AnalysisAgent** | Data analysis | Exploratory analysis, statistical testing, visualization, pattern detection |
| **WritingAssistantAgent** | Content synthesis | Literature synthesis, results summarization, outline generation, gap detection |
| **KnowledgeGraphAgent** | Knowledge management | Entity extraction, graph creation, cross-paper querying, PRISMA alignment |
| **CollaborationAgent** | Coordination | Task delegation, progress tracking, shared context management |

### Benefits of Multi-Agent Design

- **Separation of Concerns** – Small, focused prompt contexts per agent
- **Composable Workflows** – Orchestrator chains agents into configurable pipelines
- **Independent Testing** – Each agent is unit-testable in isolation (154 tests)
- **Extensibility** – New agents/workflows added without touching existing code

---

## 💻 Technical Stack

### AI & LLM Infrastructure

**Primary Models:**
- **Groq LLM:** `llama-3.3-70b-versatile` (upgraded from 3.1)
- **Tiered Routing:** Fast 8B models for retrieval/screening, 70B for synthesis (~5x cost reduction)
- **Orchestration:** LangGraph StateGraph + Deep Agents SDK (ReAct reasoning loop)

**Document Processing:**
- **OCR:** Mistral Document AI (structured annotation for tables, equations, figures)
- **PDF Processing:** PyPDF2 + ArXiv paper constructor
- **Chunking:** Configurable size with metadata preservation

### Knowledge Infrastructure

**Vector Database:**
- **Platform:** Qdrant Cloud
- **Embedding Model:** BAAI/bge-small-en-v1.5
- **Dimensions:** 384
- **Distance:** COSINE similarity
- **Status:** 1,792 semantic concepts successfully indexed

**Graph Database:**
- **Platform:** Neo4j
- **Ontology:** PRISMA 2020-aligned
- **Entity Types:** Paper, Objective, Methodology, Result, Limitation, Implication
- **Relationships:** INVESTIGATES, UTILIZES_METHOD, REPORTS_FINDING

**Entity Extraction:**
- **NER Engine:** GLiNER (urchade/gliner_mediumv2.1)
- **Architecture:** 2-Tier pipeline
  - Tier 1: Local zero-shot NER for grounded span detection
  - Tier 2: LLM + Pydantic schema validation against PRISMA rules

---

## 🔬 Research Standards & Methodologies

### PRISMA 2020 Compliance

The system implements **PRISMA (Preferred Reporting Items for Systematic Reviews and Meta-Analyses) 2020** guidelines:

- Quality validation gates at each pipeline stage
- Self-critique loops for quality assurance
- Confidence scoring with thresholds
- Human-in-the-loop review flags
- Rich provenance tracking with audit trails

### Knowledge Graph Structure

**PRISMA-Aligned Entity Extraction:**
```
Paper → INVESTIGATES → Objective
Paper → UTILIZES_METHOD → Methodology
Paper → REPORTS_FINDING → Result
Paper → HAS_LIMITATION → Limitation
Paper → SUGGESTS → Implication
```

### Cochrane Integration

Additional methodological rigor workflows for systematic reviews following Cochrane standards.

---

## 🔄 Research Workflows

### Standard Workflow Templates

#### 1. Literature Review Pipeline
```
Query Formulation → Paper Retrieval → Filtering →
Knowledge Extraction → Synthesis
```

**Features:**
- PICO (Population, Intervention, Comparison, Outcome) decomposition
- Multi-database search with citation snowballing
- Heuristic + LLM batch screening
- Automatic relevance filtering

#### 2. Data Analysis Pipeline
```
Data Prep → Exploratory Analysis → Statistical Testing →
Visualization → Results Summary
```

**Features:**
- Automated statistical testing
- Pattern and contradiction detection
- Gap identification
- Evidence synthesis

### Agentic Mode (ReAct Pipeline)

**Enhanced workflow with autonomous decision-making:**

1. **Search:** PICO-decomposed queries across databases
2. **Process:** Convert papers into analyzable chunks
3. **Extract:** Build PRISMA knowledge graph (Neo4j + Qdrant)
4. **Assess Coverage:** Check domain coverage via vector search
5. **Analyze:** Evidence synthesis with GraphRAG retrieval
6. **Write:** Draft sections, detect evidence gaps
7. **Iterate:** Loop back if coverage gaps found (max 15 iterations)

**GraphRAG Retrieval Pattern:**
```
User Query → Qdrant Vector Search (entry nodes) →
Neo4j Graph Traversal (reasoning paths) → Context Assembly
```

---

## 🎨 Frontend & API

### Backend API
- **Framework:** FastAPI
- **URL:** http://localhost:8000
- **Health Check:** /health endpoint
- **CORS:** Configured for frontend origins

### Frontend Interface
- **Framework:** React-based UI
- **URL:** http://localhost:8080
- **Features:**
  - Real-time progress tracking via SSE (Server-Sent Events)
  - Interactive research dashboard
  - Project creation and management
  - Live graph visualization (React Flow/React Force-Graph planned)

---

## 📊 Project Status

### Test Coverage
✅ **154/154** baseline unit tests passing
✅ **23/24** Deep Agents E2E tests passing
✅ **4/4** Integration tests verified
✅ **4/4** Groq integration tests passing

### Infrastructure Status
✅ Qdrant Cloud connected and operational
✅ Groq API integrated with llama-3.3-70b
✅ Frontend-Backend connection verified
✅ Virtual environment configured
⚠️ Neo4j credentials need configuration

### Performance Metrics
- **Cost Optimization:** ~5x reduction via tiered model routing
- **Vector Space:** 1,792 unique semantic concepts extracted
- **Embedding Efficiency:** 384-dimensional vectors (compact & fast)

---

## 🗂️ Project Structure

```
research_agent/
├── agents/                          # Specialized research agents
│   ├── literature_review_agent.py   # Search & retrieval
│   ├── data_processing_agent.py     # Document processing
│   ├── analysis_agent.py            # Data analysis
│   ├── writing_assistant_agent.py   # Content synthesis
│   ├── knowledge_graph_agent.py     # Graph management
│   └── collaboration_agent.py       # Coordination
├── core/                            # Core system components
│   ├── base_agent.py                # ResearchAgent base class
│   ├── orchestrator.py              # Multi-agent orchestrator + ReAct
│   ├── state.py                     # ResearchState TypedDict
│   ├── graph.py                     # LangGraph StateGraph builder
│   ├── llm_provider.py              # LLM brain (Groq/Mistral/Ollama)
│   ├── agent_tools.py               # Tool wrappers for Deep Agents
│   ├── nodes/                       # LangGraph nodes
│   │   ├── literature_review_node.py
│   │   ├── data_processing_node.py
│   │   ├── knowledge_graph_node.py
│   │   ├── prisma_extractor.py      # GLiNER + LLM entity extraction
│   │   ├── analysis_node.py
│   │   ├── writing_node.py
│   │   ├── quality_validator_node.py
│   │   ├── human_intervention_node.py
│   │   └── audit_formatter_node.py
│   └── tools/                       # Utility tools
│       ├── search_tools.py          # ArXiv, S2, Crossref
│       ├── extraction_tools.py      # Mistral AI + PyPDF2
│       └── validation_tools.py      # PRISMA/Cochrane
├── paperconstructor/                # Paper construction
│   ├── constructors.py              # Arxiv class
│   └── utility/templates.py         # Extraction templates
├── frontend/                        # Legacy frontend
├── frontend-next/                   # React frontend
├── tests/                           # Test suite (154 tests)
│   ├── agents/
│   ├── core/
│   └── integration/
├── config/                          # Configuration
│   └── config.yaml                  # Model configs, API keys
└── docs/                            # Documentation
```

---

## 🚀 Key Innovations

### 1. Two-Tier Entity Extraction
- **Tier 1:** Local GLiNER for fast, grounded span detection (no API calls)
- **Tier 2:** LLM validation with Pydantic schemas for PRISMA compliance
- **Result:** High accuracy with controlled costs

### 2. Tiered Model Routing
- **Fast Models (8B):** Query formulation, paper screening, heuristic filtering
- **Deep Models (70B):** Synthesis, writing, complex reasoning
- **Impact:** ~5x cost reduction while maintaining quality

### 3. ReAct Orchestrator Loop
- **Deep Agents SDK:** Autonomous decision-making with tool use
- **Backtracking:** Automatic loop-back on evidence gaps
- **Coverage Checks:** Validates PRISMA domain coverage before synthesis

### 4. GraphRAG Integration
- **Hybrid Retrieval:** Vector search (Qdrant) + Graph traversal (Neo4j)
- **Reasoning Paths:** Follow relationships for rich context assembly
- **Semantic + Structural:** Best of both worlds

### 5. Quality Assurance
- **Self-Critique:** LLM evaluates its own outputs
- **Confidence Scoring:** Flags low-confidence results for human review
- **PRISMA Validation:** Rule-based + qualitative assessment
- **Audit Trails:** Complete provenance tracking

---

## 📈 Development Progress

### Session 5 (Latest) - Deep Agents Integration
- ✅ PRISMA Knowledge Graph with GLiNER
- ✅ Deep Agents SDK + ReAct orchestrator
- ✅ Tool wrappers with auto-schema generation
- ✅ Subagents rewritten for ReAct logic
- ✅ Model upgrade to llama-3.3-70b

### Session 4 - Multi-Agent Enrichment
- ✅ Tiered LLM routing (8B/70B)
- ✅ Self-critique & confidence scoring
- ✅ Backtrack loops for evidence gaps
- ✅ Extended thinking in analysis
- ✅ Rich provenance tracking

### Session 3 - Vector Database
- ✅ Qdrant Cloud setup
- ✅ BAAI/bge-small-en-v1.5 embeddings
- ✅ t-SNE visualization tools
- ✅ 1,792 concepts successfully indexed

### Session 2 - LangGraph Architecture
- ✅ Removed google.adk dependency
- ✅ Built LangGraph StateGraph
- ✅ Integrated Groq/Mistral/Ollama
- ✅ PRISMA/Cochrane workflows
- ✅ 60/60 tests passing (zero regressions)

---

## 🎯 Use Cases

1. **Systematic Literature Reviews:** PRISMA-compliant reviews with automated search, screening, and synthesis
2. **Meta-Analysis:** Extract structured data from papers for quantitative synthesis
3. **Research Gap Identification:** Detect understudied areas through knowledge graph analysis
4. **Paper Construction:** Generate draft papers from research results
5. **Knowledge Management:** Build and query research knowledge graphs
6. **Collaboration:** Team-based research with progress tracking

---

## 🔧 Installation & Usage

### Quick Start
```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install dependencies
pip install -e .

# Run research agent
python run_research_agent.py
```

### API Usage
```python
from core.orchestrator import ResearchWorkflowOrchestrator

# Initialize with preferences
orchestrator = ResearchWorkflowOrchestrator({
    "citation_format": "apa",
    "preferred_statistical_methods": ["t-test", "anova"]
})

# Start research project
project_id = await orchestrator.start_research_project(
    "AI Ethics Research",
    "Analysis of ethical considerations in AI systems"
)

# Start literature review workflow
workflow_id = await orchestrator.start_research_workflow(
    project_id,
    "literature_review",
    custom_parameters={"focus_area": "machine_learning"}
)
```

### Agentic Mode (ReAct)
```python
from core.orchestrator import run_agentic_pipeline

result = await run_agentic_pipeline(
    project_name="AI Safety Review",
    research_topic="Safety considerations in large language models",
    research_goals=[
        "Identify key safety challenges",
        "Review mitigation strategies",
        "Analyze regulatory frameworks"
    ],
    model="groq:llama-3.3-70b-versatile"
)
```

---

## 📝 License

MIT License

---

**Last Updated:** April 6, 2026
**Repository:** https://github.com/axls23/research_agent
