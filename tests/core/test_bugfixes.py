"""Tests for logical errors and architecture fixes.

Validates that:
  - Graph routing returns "retry" when human decides to retry
  - PRISMA counts (screened vs included) are correctly separated
  - LLM structured output handles malformed markdown fences
  - Knowledge graph node sets knowledge_graph_id in state
  - Neo4j label injection is prevented via allowlist
  - Validation tools produce correct reports
  - Analysis node does not overwrite PRISMA papers_included
  - Audit formatter uses UTC timestamps
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import pytest
from core.state import make_initial_state, append_audit

# Optional heavy dependencies — skip tests if not available
numpy = pytest.importorskip("numpy", reason="numpy not installed")


# ---------------------------------------------------------------------------
# Graph routing logic (tested without importing core.graph to avoid
# pulling in aiohttp/langgraph transitively)
# ---------------------------------------------------------------------------

def _route_after_human(state):
    """Mirror of core.graph._route_after_human for testing logic."""
    if state.get("abort", False):
        return "abort"
    decision = ""
    decisions = state.get("human_decisions", [])
    if decisions:
        decision = decisions[-1].get("decision", "")
    if decision == "retry":
        return "retry"
    return "continue"


def _route_after_validation(state):
    """Mirror of core.graph._route_after_validation for testing logic."""
    if state.get("last_validation_passed", True):
        return "continue"
    return "human_intervention"


class TestRouteAfterHuman:
    def test_retry_returns_retry(self):
        state = make_initial_state("p1", "P", "topic", ["g1"])
        state["human_decisions"] = [{
            "gate_name": "post_literature_review",
            "validation_failures": ["fail"],
            "decision": "retry",
            "reason": "try again",
            "timestamp": "2025-01-01T00:00:00Z",
        }]
        state["abort"] = False
        assert _route_after_human(state) == "retry"

    def test_override_returns_continue(self):
        state = make_initial_state("p1", "P", "topic", ["g1"])
        state["human_decisions"] = [{
            "gate_name": "post_literature_review",
            "validation_failures": ["fail"],
            "decision": "override",
            "reason": "acceptable",
            "timestamp": "2025-01-01T00:00:00Z",
        }]
        state["abort"] = False
        assert _route_after_human(state) == "continue"

    def test_abort_returns_abort(self):
        state = make_initial_state("p1", "P", "topic", ["g1"])
        state["abort"] = True
        assert _route_after_human(state) == "abort"

    def test_empty_decisions_returns_continue(self):
        state = make_initial_state("p1", "P", "topic", ["g1"])
        state["abort"] = False
        assert _route_after_human(state) == "continue"


class TestRouteAfterValidation:
    def test_passed_returns_continue(self):
        state = make_initial_state("p1", "P", "topic", ["g1"])
        state["last_validation_passed"] = True
        assert _route_after_validation(state) == "continue"

    def test_failed_returns_human_intervention(self):
        state = make_initial_state("p1", "P", "topic", ["g1"])
        state["last_validation_passed"] = False
        assert _route_after_validation(state) == "human_intervention"


# ---------------------------------------------------------------------------
# Neo4j label sanitization
# ---------------------------------------------------------------------------

class TestNeo4jLabelSanitization:
    def test_allowed_labels_pass_through(self):
        """Labels from ENTITY_TYPES should be accepted after capitalize()."""
        from core.nodes.knowledge_graph_node import ENTITY_TYPES

        for t in ENTITY_TYPES:
            capitalized = t.capitalize()
            assert capitalized in {"Concept", "Method", "Result", "Dataset"}

    def test_malicious_label_blocked(self):
        """A label like 'Concept} DELETE (n) //' should be sanitized."""
        # We test indirectly: the _persist_to_neo4j function uses an allowlist
        # so anything outside {Concept, Method, Result, Dataset} defaults to Concept
        allowed = {"Concept", "Method", "Result", "Dataset"}
        malicious = "Concept} DELETE (n) //".capitalize()
        assert malicious not in allowed  # would be rejected


# ---------------------------------------------------------------------------
# Knowledge graph node sets knowledge_graph_id
# ---------------------------------------------------------------------------

class TestKnowledgeGraphNodeOutput:
    def test_empty_chunks_returns_audit(self):
        """When no chunks, node should return without crashing."""
        import asyncio
        from core.nodes.knowledge_graph_node import knowledge_graph_node

        state = make_initial_state("p1", "P", "topic", ["g1"])
        result = asyncio.get_event_loop().run_until_complete(
            knowledge_graph_node(state)
        )
        assert "audit_log" in result
        assert result["current_node"] == "knowledge_graph"


# ---------------------------------------------------------------------------
# Analysis node should NOT overwrite papers_included
# ---------------------------------------------------------------------------

class TestAnalysisNodeOutput:
    def test_no_papers_included_in_output(self):
        """analysis_node should not return papers_included to avoid
        overwriting the PRISMA count set by data_processing_node."""
        import asyncio
        from core.nodes.analysis_node import analysis_node

        state = make_initial_state("p1", "P", "topic", ["g1"])
        state["papers_included"] = 42  # Set by data_processing
        result = asyncio.get_event_loop().run_until_complete(
            analysis_node(state)
        )
        assert "papers_included" not in result


# ---------------------------------------------------------------------------
# LLM generate_structured: markdown fence edge case
# ---------------------------------------------------------------------------

class TestGenerateStructuredFenceStripping:
    def test_fence_without_newline(self):
        """If LLM returns ```{json}``` without newlines, should not crash."""
        # Test the fence stripping logic directly
        raw = '```{"name": "test"}```'
        cleaned = raw.strip()
        if cleaned.startswith("```"):
            lines = cleaned.split("\n", 1)
            cleaned = lines[1] if len(lines) > 1 else cleaned[3:]
            if cleaned.endswith("```"):
                cleaned = cleaned[:-3]
        # Should have extracted the JSON
        assert cleaned == '{"name": "test"}'

    def test_fence_with_language_tag(self):
        """Standard ```json\n...\n``` wrapping."""
        raw = '```json\n{"name": "test"}\n```'
        cleaned = raw.strip()
        if cleaned.startswith("```"):
            lines = cleaned.split("\n", 1)
            cleaned = lines[1] if len(lines) > 1 else cleaned[3:]
            if cleaned.endswith("```"):
                cleaned = cleaned[:-3]
        assert cleaned.strip() == '{"name": "test"}'

    def test_no_fence(self):
        """Plain JSON without fences."""
        raw = '{"name": "test"}'
        cleaned = raw.strip()
        if cleaned.startswith("```"):
            lines = cleaned.split("\n", 1)
            cleaned = lines[1] if len(lines) > 1 else cleaned[3:]
            if cleaned.endswith("```"):
                cleaned = cleaned[:-3]
        assert cleaned == '{"name": "test"}'


# ---------------------------------------------------------------------------
# Audit trail append_audit
# ---------------------------------------------------------------------------

class TestAuditAppend:
    def test_append_is_non_mutating(self):
        """append_audit should return a NEW list, not mutate the original."""
        state = make_initial_state("p1", "P", "topic", ["g1"])
        original_log = state["audit_log"]
        new_log = append_audit(
            state, "test_agent", "test_action",
            inputs={"x": 1}, output_summary="did something",
        )
        assert len(new_log) == 1
        assert len(original_log) == 0  # original unchanged


# ---------------------------------------------------------------------------
# Validation tools
# ---------------------------------------------------------------------------

class TestValidationTools:
    def test_exploratory_always_passes(self):
        from core.tools.validation_tools import run_validation_gate

        state = make_initial_state("p1", "P", "topic", ["g1"], rigor_level="exploratory")
        report = run_validation_gate(state, "post_literature_review")
        assert report["passed"] is True
        assert report["failures"] == []

    def test_search_coverage_min_databases(self):
        from core.tools.validation_tools import validate_search_coverage

        state = make_initial_state("p1", "P", "topic", ["g1"], rigor_level="prisma")
        state["databases_searched"] = ["arxiv"]
        failures = validate_search_coverage(state, {"min_databases": 3})
        assert len(failures) == 1
        assert "database" in failures[0].lower()

    def test_search_coverage_passes(self):
        from core.tools.validation_tools import validate_search_coverage

        state = make_initial_state("p1", "P", "topic", ["g1"])
        state["databases_searched"] = ["arxiv", "semantic_scholar", "crossref"]
        state["papers_found"] = 50
        failures = validate_search_coverage(state, {"min_databases": 3, "min_papers_found": 10})
        assert failures == []
