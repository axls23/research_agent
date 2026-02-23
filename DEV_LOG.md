# Research Agent — Development Log

> Track development progress per session. Append new entries at the top.

---

## Session 3 — 2026-02-24

**Focus**: Knowledge Graph Semantic Embedding & Qdrant Cloud Integration

### What Got Built
| Component | Status | Details |
|-----------|--------|---------|
| Qdrant Cloud Setup | ✅ Done | Replaced in-memory Qdrant with persistent Cloud cluster |
| Embedding Engine | ✅ Done | Ripped out Word2Vec (Gensim), added `BAAI/bge-small-en-v1.5` via `sentence-transformers` |
| `setup_qdrant.py` | ✅ Done | Securely accepts `.env` credentials & initializes vector space (`size=384`, `COSINE`) |
| `visualize_qdrant.py` | ✅ Done | Pulls vectors from Cloud, generates a 2D interactive `t-SNE` HTML scatter plot map |

### Key Results
- Full LangGraph pipeline successfully extracted 1,792 unique semantic concepts from PDFs.
- 100% of concept vectors successfully upserted to remote Qdrant Cloud cleanly.
- Pipeline execution verified with `Exit Code 0` using Qdrant remote credentials.

### Architectural Decisions (Live Graph UI)
- **Live Graph Viewer**: Selected React Flow/React Force-Graph + SSE (Backend FastApi) as the chosen path forward. This will allow the LLM to output JSON node IDs during iteration, seamlessly glowing nodes on the frontend UI. 
- Discarded direct Excalidraw integration for the graph itself due to lack of auto-physics layout for 30+ node subgraphs.

### What's Left (Next Session)
- [ ] Connect Neo4j credentials to inject nodes + edges and enable GraphRAG queries.
- [ ] Build the GraphRAG Retriever: Query Qdrant for the entry-node, then query Neo4j for the connected reasoning path.
- [ ] Build the SSE FastApi Route to pipe LLM Graph-search events to the frontend.

---

## Session 2 — 2026-02-21 (00:30 – 01:06 IST)

**Focus**: LangGraph Multi-Agent Architecture Implementation

### Decisions Made
- **OCR**: Mistral Document AI (not docling/marker) — structured annotation for tables, equations, figures
- **LLM Provider**: Groq primary, Mistral for Document AI, Ollama local fallback
- **google.adk**: Fully removed — replaced with pure LangGraph + local Task/Workflow/Context classes

### What Got Built
| Component | File | Status |
|-----------|------|--------|
| ResearchState TypedDict | `core/state.py` | ✅ Done |
| LLM Brain (Groq/Mistral/Ollama) | `core/llm_provider.py` | ✅ Done |
| Base Agent (no google.adk) | `core/base_agent.py` | ✅ Rewritten |
| Orchestrator (no google.adk) | `core/orchestrator.py` | ✅ Rewritten |
| Search Tools (ArXiv/S2/Crossref) | `core/tools/search_tools.py` | ✅ Done |
| Extraction Tools (Mistral AI + PyPDF2) | `core/tools/extraction_tools.py` | ✅ Done |
| Validation Tools (PRISMA/Cochrane) | `core/tools/validation_tools.py` | ✅ Done |
| Literature Review Node | `core/nodes/literature_review_node.py` | ✅ Done |
| Data Processing Node | `core/nodes/data_processing_node.py` | ✅ Done |
| Knowledge Graph Node | `core/nodes/knowledge_graph_node.py` | ✅ Done |
| Analysis Node | `core/nodes/analysis_node.py` | ✅ Done |
| Writing Node | `core/nodes/writing_node.py` | ✅ Done |
| Quality Validator Node | `core/nodes/quality_validator_node.py` | ✅ Done |
| Human Intervention Node | `core/nodes/human_intervention_node.py` | ✅ Done |
| Audit Formatter Node | `core/nodes/audit_formatter_node.py` | ✅ Done |
| StateGraph Builder | `core/graph.py` | ✅ Done |
| Exploratory Workflow | `core/workflows/exploratory.yaml` | ✅ Done |
| PRISMA Workflow | `core/workflows/prisma.yaml` | ✅ Done |
| Cochrane Workflow | `core/workflows/cochrane.yaml` | ✅ Done |
| Backward-compat shims | `core/workflow.py`, `core/context.py`, `core/registry.py` | ✅ Done |

### Test Results
- **60/60 existing tests pass** — zero regressions
- Warning: deprecated `asyncio.get_event_loop()` in test helper (pre-existing)

### What's Left
- [x] Update `run_research_agent.py` — add `--mode langgraph --rigor prisma` CLI
- [ ] New unit tests for state, graph compilation, validation gates
- [ ] Integration test with live Groq/Mistral API
- [ ] Update `requirements-phase1.txt` with new deps (`langgraph`, `mistralai`, `langchain-groq`)

---

## Session 1 — 2026-02-20 (16:16 – 17:00 IST)

**Focus**: Baseline Analysis & Architecture Planning

### What Happened
- Explored full codebase: 6 agents, orchestrator, knowledge_graph.py, paperconstructor
- Identified gaps: no LLM brain, no OCR, no methodological rigor
- Drafted implementation plan with LangGraph StateGraph, PRISMA/Cochrane validation gates, human-in-the-loop
- User approved plan with two changes: Mistral Document AI for OCR, remove google.adk entirely

### Key Findings
- All agents use action-dispatch pattern (`process()` → `_action_handler()`)
- `paperconstructor` uses PyPDF2 (fails on scanned PDFs)
- `knowledge_graph.py` uses Qdrant + Word2Vec + LangChain
- `orchestrator.py` imports from `google.adk.agents` (custom/private SDK)
- Config already has Groq model list in `config/config.yaml`
