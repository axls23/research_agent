# Research Agent Architecture

This document outlines the system architecture, data flow, and state management unique to the Research Agent system. The system uses a **LangGraph-based multi-agent architecture** incorporating methodological rigor (PRISMA/Cochrane standards), validation gates, human-in-the-loop interventions, and full auditability.

## 1. High-Level System Design

The Research Agent can operate in two primary modes:
1. **Deterministic Mode**: A fixed `StateGraph` pipeline that enforces a strict linear workflow with optional human-in-the-loop validation gates.
2. **Agentic Mode**: A ReAct-based orchestrator that dynamically decides the flow of execution based on the research goals.

### Key Components
- **Orchestration**: LangGraph (`StateGraph`)
- **LLM Routing**: Tiered LLM approach utilizing "Fast" models (e.g., 8B parameters for rapid screening/validation) and "Deep" models (e.g., 70B parameters for synthesis and writing).
- **Knowledge Representation**: 
  - **Vector Store**: Qdrant Cloud (for semantic search and chunk embeddings).
  - **Graph Database**: Neo4j Aura (for PRISMA-compliant knowledge entities and relationships).
- **Audit & Compliance**: Append-only audit logs capturing cryptographic hashes of all inputs/outputs to guarantee reproducibility.

## 2. Research Pipeline Workflow (Deterministic)

The deterministic pipeline models the research process as a directed graph. Validation gates are injected periodically depending on the requested `rigor_level` (Exploratory vs. PRISMA/Cochrane).

```mermaid
graph TD
    %% Nodes
    LitReview[Literature Review Node]
    DataProc[Data Processing Node]
    KG[Knowledge Graph Node]
    Analysis[Analysis Node]
    Writing[Writing Node]
    Audit[Audit Formatter Node]
    
    %% Validation & Human Gates
    Val1{Validator<br>(Post-Lit)}
    Val2{Validator<br>(Post-Data)}
    Val3{Validator<br>(Post-Analysis)}
    
    Hum1[Human Intervention]
    Hum2[Human Intervention]
    Hum3[Human Intervention]
    
    %% Edges
    LitReview --> Val1
    Val1 -- Pass --> DataProc
    Val1 -- Fail --> Hum1
    Hum1 -- Override/Retry --> DataProc
    Hum1 -- Abort --> Audit
    
    DataProc --> Val2
    Val2 -- Pass --> KG
    Val2 -- Fail --> Hum2
    Hum2 -- Override/Retry --> KG
    Hum2 -- Abort --> Audit
    
    KG --> Analysis
    Analysis --> Val3
    Val3 -- Pass --> Writing
    Val3 -- Fail --> Hum3
    Hum3 -- Override/Retry --> Writing
    Hum3 -- Abort --> Audit
    
    Writing --> CheckBacktrack{Needs more<br>papers?}
    CheckBacktrack -- Yes (Backtrack) --> LitReview
    CheckBacktrack -- No --> Audit
    Audit --> End([END])
```

## 3. State Management (`ResearchState`)

The LangGraph operates over a shared `TypedDict` state (`ResearchState`) that passes through all nodes. Nodes mutate specific fields or append to logs.

### Core State Segments:

1. **Project Metadata**: Tracks `project_id`, `research_topic`, `research_goals`, and the chosen `rigor_level`.
2. **Literature Review**: Manages PRISMA metric counts (`papers_found`, `papers_screened`, `papers_included`) and holds the core `PaperRecord` structures.
3. **Data Processing**: Holds text `chunks` extracted from full-text PDFs using tools like Mistral Document AI.
4. **Knowledge Graph**: Maintains semantic `KnowledgeEntity` objects following the PRISMA 2020 ontology (e.g., *paper*, *objective*, *methodology*, *result*, *limitation*).
5. **Writing & Analysis**: Holds `analysis_results`, generating statistical outputs or charts, and maintains `draft_sections` mapped by section names.
6. **Audit & Compliance**:
   - `audit_log`: **Append-only** list of `AuditEntry` records containing timestamps, action names, input SHA-256 hashes, and output summaries. 
   - `validation_reports`: Appended by the `quality_validator_node`.
   - `human_decisions`: Appended by the `human_intervention_node`.

## 4. Node Definitions

- **Literature Review**: Queries databases (e.g., ArXiv, PubMed) based on defined search strategies. Screens abstracts for inclusion.
- **Data Processing**: Extracts text and metadata from selected papers, chunks text, and prepares material for embedding.
- **Knowledge Graph**: Generates structured entities and loads them into Neo4j and Qdrant.
- **Analysis**: Conducts thematic synthesis, statistical evaluation, and prepares tables/figures.
- **Writing**: Synthesizes the extracted findings into a structured markdown document, automatically citing papers from the `ResearchState`. Has the ability to "backtrack" to the Literature Review if evidence gaps are detected.
- **Audit Formatter**: Finalizes the PRISMA flow diagram and exports an immutable JSON audit trail of the entire run.
- **Quality Validator**: Automatically verifies if the output of the preceding node meets the rigorous demands of the `rigor_level`. Emits model critiques.
- **Human Intervention**: Pauses graph execution when validation fails, waiting for user input (`retry`, `override`, or `abort`).
