"""
core/state.py
=============
Shared ResearchState TypedDict that flows through all LangGraph nodes.
Includes full audit trail, PRISMA counts, validation reports, and
human-decision records for reproducible, certifiable systematic reviews.
"""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from typing import Any, Dict, List, Literal, Optional
from typing_extensions import TypedDict


# ---------------------------------------------------------------------------
# Sub-types
# ---------------------------------------------------------------------------

class AuditEntry(TypedDict):
    """Immutable record appended by every agent node."""
    timestamp: str          # ISO-8601 UTC
    agent: str              # e.g. "literature_review_node"
    action: str             # e.g. "search_arxiv"
    input_hash: str         # SHA-256 of serialised inputs (reproducibility)
    output_summary: str     # Short human-readable description of output
    provenance: Dict[str, Any]  # Raw pointers (paper IDs, file paths, etc.)


class ValidationReport(TypedDict):
    """Produced by quality_validator_node at each gate."""
    gate_name: str              # e.g. "post_literature_review"
    rigor_level: str            # "exploratory" | "prisma" | "cochrane"
    passed: bool
    criteria: Dict[str, Any]    # Loaded from YAML workflow config
    failures: List[str]         # Human-readable failure reasons
    timestamp: str


class HumanDecision(TypedDict):
    """Records a human-in-the-loop intervention."""
    gate_name: str
    validation_failures: List[str]
    decision: Literal["retry", "override", "abort"]
    reason: str                 # Free-text justification
    timestamp: str


class PaperRecord(TypedDict):
    """Metadata + extraction result for one paper."""
    paper_id: str               # ArXiv ID, DOI, or internal UUID
    title: str
    authors: List[str]
    year: Optional[int]
    abstract: str
    source_url: str
    databases: List[str]        # Which databases returned this paper
    # Post-extraction
    full_text: Optional[str]    # Extracted via Mistral Document AI
    annotations: Optional[Dict[str, Any]]  # Mistral structured annotations
    quality_score: Optional[float]
    included: bool              # PRISMA: included in final review
    exclusion_reason: Optional[str]


class Chunk(TypedDict):
    """Text chunk from a paper, ready for embedding."""
    chunk_id: str
    paper_id: str
    text: str
    token_count: int
    page_range: Optional[str]


class KnowledgeEntity(TypedDict):
    """A node in the knowledge graph."""
    entity_id: str
    label: str       # e.g. "concept", "method", "result"
    text: str
    paper_ids: List[str]


class AnalysisResult(TypedDict):
    """Output from the analysis node."""
    method: str
    result_summary: str
    figures: List[str]     # File paths to generated charts
    tables: List[Dict[str, Any]]
    statistical_output: Optional[Dict[str, Any]]


# ---------------------------------------------------------------------------
# Main State
# ---------------------------------------------------------------------------

class ResearchState(TypedDict):
    """
    Complete shared state for the LangGraph research pipeline.

    Design principles:
    - ``audit_log`` is append-only — nodes must never mutate existing entries.
    - ``validation_reports`` is appended by quality_validator_node.
    - ``human_decisions`` is appended by human_intervention_node.
    - PRISMA counts (found / screened / included) are updated incrementally.
    """

    # ---- Project metadata ----
    project_id: str
    project_name: str
    research_topic: str
    research_goals: List[str]
    rigor_level: Literal["exploratory", "prisma", "cochrane"]

    # ---- Workflow control ----
    current_node: str          # Name of the node currently executing
    last_validation_passed: bool
    abort: bool                # Set True by human_intervention to stop graph

    # ---- Literature Review ----
    search_queries: List[str]
    databases_searched: List[str]   # PRISMA: which DBs were queried
    papers_found: int               # PRISMA: total raw hits
    papers_screened: int            # PRISMA: after title/abstract screen
    papers_included: int            # PRISMA: after full-text inclusion
    papers: List[PaperRecord]

    # ---- Data Processing ----
    chunks: List[Chunk]
    total_tokens_extracted: int

    # ---- Knowledge Graph ----
    knowledge_entities: List[KnowledgeEntity]
    knowledge_graph_id: Optional[str]
    knowledge_graph_summary: Optional[Dict[str, Any]]  # Neo4j/Qdrant stats

    # ---- Analysis ----
    analysis_results: List[AnalysisResult]

    # ---- Writing ----
    outline: Optional[str]
    draft_sections: Dict[str, str]  # section_name -> text
    methods_section: Optional[str]  # Auto-generated from audit trail

    # ---- Audit & Compliance ----
    audit_log: List[AuditEntry]         # APPEND ONLY
    validation_reports: List[ValidationReport]
    human_decisions: List[HumanDecision]

    # ---- Output artifacts ----
    prisma_flow_diagram: Optional[str]   # Path or ASCII string
    audit_export_path: Optional[str]     # Path to JSON export


# ---------------------------------------------------------------------------
# Helper constructors
# ---------------------------------------------------------------------------

def make_initial_state(
    project_id: str,
    project_name: str,
    research_topic: str,
    research_goals: List[str],
    rigor_level: Literal["exploratory", "prisma", "cochrane"] = "exploratory",
) -> ResearchState:
    """Create a fresh ResearchState with sensible defaults."""
    return ResearchState(
        project_id=project_id,
        project_name=project_name,
        research_topic=research_topic,
        research_goals=research_goals,
        rigor_level=rigor_level,
        current_node="",
        last_validation_passed=True,
        abort=False,
        search_queries=[],
        databases_searched=[],
        papers_found=0,
        papers_screened=0,
        papers_included=0,
        papers=[],
        chunks=[],
        total_tokens_extracted=0,
        knowledge_entities=[],
        knowledge_graph_id=None,
        knowledge_graph_summary=None,
        analysis_results=[],
        outline=None,
        draft_sections={},
        methods_section=None,
        audit_log=[],
        validation_reports=[],
        human_decisions=[],
        prisma_flow_diagram=None,
        audit_export_path=None,
    )


def append_audit(
    state: ResearchState,
    agent: str,
    action: str,
    inputs: Any,
    output_summary: str,
    provenance: Optional[Dict[str, Any]] = None,
) -> List[AuditEntry]:
    """
    Return a new audit_log list with one entry appended.

    Usage inside a node::

        state["audit_log"] = append_audit(
            state, "literature_review_node", "search_arxiv",
            inputs={"query": q}, output_summary=f"Found {n} papers"
        )
    """
    input_hash = hashlib.sha256(
        json.dumps(inputs, sort_keys=True, default=str).encode()
    ).hexdigest()[:16]

    entry: AuditEntry = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "agent": agent,
        "action": action,
        "input_hash": input_hash,
        "output_summary": output_summary,
        "provenance": provenance or {},
    }
    return [*state["audit_log"], entry]
