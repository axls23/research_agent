"""Tests for core/state.py — ResearchState, sub-types, and helper functions.

Validates:
  - make_initial_state returns a complete state with correct defaults
  - append_audit is append-only, deterministic, and preserves existing entries
  - Sub-type dicts (PaperRecord, ValidationReport, HumanDecision) have expected structure
"""

import sys
from datetime import datetime
from pathlib import Path

import pytest

# Ensure the project root is on sys.path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from core.state import (
    AuditEntry,
    Chunk,
    HumanDecision,
    KnowledgeEntity,
    PaperRecord,
    ResearchState,
    ValidationReport,
    append_audit,
    make_initial_state,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_test_state(**overrides) -> ResearchState:
    """Create a minimal ResearchState with test defaults, applying overrides."""
    defaults = dict(
        project_id="test-001",
        project_name="Test Project",
        research_topic="Machine Learning",
        research_goals=["accuracy", "speed"],
        rigor_level="exploratory",
    )
    defaults.update(overrides)
    return make_initial_state(**defaults)


# ---------------------------------------------------------------------------
# make_initial_state
# ---------------------------------------------------------------------------


class TestMakeInitialState:
    """Verify make_initial_state returns a fully populated ResearchState."""

    def test_returns_all_keys(self):
        state = _make_test_state()
        expected_keys = {
            "project_id",
            "project_name",
            "research_topic",
            "research_goals",
            "rigor_level",
            "current_node",
            "last_validation_passed",
            "abort",
            "search_queries",
            "databases_searched",
            "papers_found",
            "papers_screened",
            "papers_included",
            "papers",
            "chunks",
            "total_tokens_extracted",
            "knowledge_entities",
            "knowledge_graph_id",
            "knowledge_graph_summary",
            "analysis_results",
            "outline",
            "draft_sections",
            "needs_more_papers",
            "gap_analysis",
            "backtrack_count",
            "methods_section",
            "audit_log",
            "validation_reports",
            "human_decisions",
            "prisma_flow_diagram",
            "audit_export_path",
        }
        assert set(state.keys()) == expected_keys

    def test_default_values(self):
        state = _make_test_state()
        assert state["abort"] is False
        assert state["last_validation_passed"] is True
        assert state["current_node"] == ""
        assert state["papers"] == []
        assert state["chunks"] == []
        assert state["audit_log"] == []
        assert state["validation_reports"] == []
        assert state["human_decisions"] == []
        assert state["papers_found"] == 0
        assert state["papers_screened"] == 0
        assert state["papers_included"] == 0
        assert state["total_tokens_extracted"] == 0
        assert state["knowledge_graph_id"] is None
        assert state["knowledge_graph_summary"] is None
        assert state["outline"] is None
        assert state["methods_section"] is None
        assert state["prisma_flow_diagram"] is None
        assert state["audit_export_path"] is None
        assert state["draft_sections"] == {}

    @pytest.mark.parametrize("rigor", ["exploratory", "prisma", "cochrane"])
    def test_rigor_level_propagation(self, rigor):
        state = _make_test_state(rigor_level=rigor)
        assert state["rigor_level"] == rigor

    def test_project_metadata_round_trip(self):
        state = _make_test_state(
            project_id="proj-42",
            project_name="Deep Review",
            research_topic="NLP Transformers",
            research_goals=["interpretability"],
        )
        assert state["project_id"] == "proj-42"
        assert state["project_name"] == "Deep Review"
        assert state["research_topic"] == "NLP Transformers"
        assert state["research_goals"] == ["interpretability"]


# ---------------------------------------------------------------------------
# append_audit
# ---------------------------------------------------------------------------


class TestAppendAudit:
    """Verify append_audit is append-only and produces valid AuditEntry dicts."""

    def test_first_entry_on_empty_state(self):
        state = _make_test_state()
        new_log = append_audit(
            state,
            agent="test_agent",
            action="test_action",
            inputs={"key": "value"},
            output_summary="Did a thing",
        )
        assert len(new_log) == 1

    def test_preserves_existing_entries(self):
        state = _make_test_state()
        # Append first
        state["audit_log"] = append_audit(
            state,
            agent="a1",
            action="act1",
            inputs={},
            output_summary="First",
        )
        assert len(state["audit_log"]) == 1
        # Append second
        new_log = append_audit(
            state,
            agent="a2",
            action="act2",
            inputs={},
            output_summary="Second",
        )
        assert len(new_log) == 2
        assert new_log[0]["agent"] == "a1"
        assert new_log[1]["agent"] == "a2"

    def test_entry_fields_populated(self):
        state = _make_test_state()
        new_log = append_audit(
            state,
            agent="lit_node",
            action="search_arxiv",
            inputs={"q": "ML"},
            output_summary="Found 5 papers",
        )
        entry = new_log[0]
        assert entry["agent"] == "lit_node"
        assert entry["action"] == "search_arxiv"
        assert entry["output_summary"] == "Found 5 papers"
        assert len(entry["input_hash"]) > 0
        assert len(entry["timestamp"]) > 0

    def test_input_hash_deterministic(self):
        state = _make_test_state()
        inputs = {"query": "deep learning", "limit": 10}
        log1 = append_audit(state, "a", "x", inputs, "s1")
        log2 = append_audit(state, "a", "x", inputs, "s2")
        assert log1[0]["input_hash"] == log2[0]["input_hash"]

    def test_input_hash_differs_for_different_inputs(self):
        state = _make_test_state()
        log1 = append_audit(state, "a", "x", {"q": "alpha"}, "s")
        log2 = append_audit(state, "a", "x", {"q": "beta"}, "s")
        assert log1[0]["input_hash"] != log2[0]["input_hash"]

    def test_provenance_defaults_to_empty_dict(self):
        state = _make_test_state()
        log = append_audit(state, "a", "x", {}, "s")
        assert log[0]["provenance"] == {}

    def test_provenance_passed_through(self):
        state = _make_test_state()
        prov = {"paper_ids": ["p1", "p2"]}
        log = append_audit(state, "a", "x", {}, "s", provenance=prov)
        assert log[0]["provenance"] == prov

    def test_timestamp_is_valid_iso8601(self):
        state = _make_test_state()
        log = append_audit(state, "a", "x", {}, "s")
        ts = log[0]["timestamp"]
        parsed = datetime.fromisoformat(ts)
        assert parsed is not None

    def test_original_state_unchanged(self):
        state = _make_test_state()
        original_len = len(state["audit_log"])
        _ = append_audit(state, "a", "x", {}, "s")
        assert len(state["audit_log"]) == original_len


# ---------------------------------------------------------------------------
# Sub-type structural checks
# ---------------------------------------------------------------------------


class TestSubTypes:
    """Verify TypedDict sub-types can be instantiated with expected keys."""

    def test_paper_record_structure(self):
        paper: PaperRecord = {
            "paper_id": "arxiv:1234",
            "title": "Test Paper",
            "authors": ["Author A"],
            "year": 2024,
            "abstract": "An abstract",
            "source_url": "https://arxiv.org/abs/1234",
            "databases": ["arxiv"],
            "full_text": None,
            "annotations": None,
            "quality_score": None,
            "included": True,
            "exclusion_reason": None,
        }
        assert paper["paper_id"] == "arxiv:1234"
        assert paper["included"] is True

    def test_validation_report_structure(self):
        report: ValidationReport = {
            "gate_name": "post_literature_review",
            "rigor_level": "prisma",
            "passed": False,
            "criteria": {"min_databases": 3},
            "failures": ["Only 1 DB searched"],
            "timestamp": "2026-01-01T00:00:00+00:00",
        }
        assert report["passed"] is False
        assert len(report["failures"]) == 1

    def test_human_decision_structure(self):
        decision: HumanDecision = {
            "gate_name": "post_analysis",
            "validation_failures": ["Missing method"],
            "decision": "override",
            "reason": "Acceptable for exploratory",
            "timestamp": "2026-01-01T00:00:00+00:00",
        }
        assert decision["decision"] in {"retry", "override", "abort"}

    def test_chunk_structure(self):
        chunk: Chunk = {
            "chunk_id": "c1",
            "paper_id": "p1",
            "text": "Some text content",
            "token_count": 42,
            "page_range": "1-3",
        }
        assert chunk["token_count"] == 42

    def test_knowledge_entity_structure(self):
        entity: KnowledgeEntity = {
            "entity_id": "e1",
            "label": "objective",
            "text": "Evaluate drug efficacy",
            "paper_ids": ["p1", "p2"],
            "prisma_properties": {
                "pico": {"population": "adults", "intervention": "drug X"},
                "extraction_tier": "llm",
            },
        }
        assert entity["label"] == "objective"
        assert len(entity["paper_ids"]) == 2
        assert entity["prisma_properties"]["extraction_tier"] == "llm"

    def test_knowledge_entity_without_prisma_properties(self):
        entity: KnowledgeEntity = {
            "entity_id": "e2",
            "label": "result",
            "text": "Significant improvement observed",
            "paper_ids": ["p3"],
            "prisma_properties": None,
        }
        assert entity["label"] == "result"
        assert entity["prisma_properties"] is None
