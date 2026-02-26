"""Tests for core/graph.py — routing functions and graph compilation.

Validates:
  - Routing functions return correct path for each state condition
  - Graph compiles for all rigor levels without error
  - Exploratory graph skips validator nodes; prisma/cochrane include them
  - Entry point is literature_review; terminal edge is audit_formatter → END
"""

import sys
from pathlib import Path

import pytest

# Ensure the project root is on sys.path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from core.graph import (
    _route_after_validation,
    _route_after_human,
    _should_validate,
    build_research_graph,
)
from core.state import make_initial_state


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _state(**overrides) -> dict:
    """Minimal state dict for routing function tests."""
    base = {
        "last_validation_passed": True,
        "abort": False,
        "rigor_level": "exploratory",
    }
    base.update(overrides)
    return base


# ---------------------------------------------------------------------------
# Routing: _route_after_validation
# ---------------------------------------------------------------------------


class TestRouteAfterValidation:
    def test_validation_passed_continues(self):
        assert (
            _route_after_validation(_state(last_validation_passed=True)) == "continue"
        )

    def test_validation_failed_goes_to_human(self):
        assert (
            _route_after_validation(_state(last_validation_passed=False))
            == "human_intervention"
        )

    def test_missing_key_defaults_to_continue(self):
        assert _route_after_validation({}) == "continue"


# ---------------------------------------------------------------------------
# Routing: _route_after_human
# ---------------------------------------------------------------------------


class TestRouteAfterHuman:
    def test_abort_routes_to_abort(self):
        assert _route_after_human(_state(abort=True)) == "abort"

    def test_no_abort_continues(self):
        assert _route_after_human(_state(abort=False)) == "continue"

    def test_missing_abort_key_continues(self):
        assert _route_after_human({}) == "continue"


# ---------------------------------------------------------------------------
# Routing: _should_validate
# ---------------------------------------------------------------------------


class TestShouldValidate:
    def test_exploratory_skips(self):
        assert _should_validate(_state(rigor_level="exploratory")) == "skip"

    def test_prisma_validates(self):
        assert _should_validate(_state(rigor_level="prisma")) == "validate"

    def test_cochrane_validates(self):
        assert _should_validate(_state(rigor_level="cochrane")) == "validate"

    def test_missing_rigor_defaults_to_skip(self):
        assert _should_validate({}) == "skip"


# ---------------------------------------------------------------------------
# Graph compilation
# ---------------------------------------------------------------------------

langgraph = pytest.importorskip("langgraph", reason="langgraph not installed")


class TestGraphCompilation:
    """Verify build_research_graph produces valid compiled graphs."""

    def test_exploratory_compiles(self):
        graph = build_research_graph(rigor_level="exploratory")
        assert graph is not None

    def test_prisma_compiles(self):
        graph = build_research_graph(rigor_level="prisma")
        assert graph is not None

    def test_cochrane_compiles(self):
        graph = build_research_graph(rigor_level="cochrane")
        assert graph is not None

    def test_exploratory_node_names(self):
        """Exploratory graph should still have core nodes."""
        graph = build_research_graph(rigor_level="exploratory")
        graph_repr = graph.get_graph()
        node_ids = set(graph_repr.nodes)
        assert "literature_review" in node_ids
        assert "data_processing" in node_ids
        assert "knowledge_graph" in node_ids
        assert "analysis" in node_ids
        assert "writing" in node_ids
        assert "audit_formatter" in node_ids

    def test_prisma_has_validator_and_human_nodes(self):
        """PRISMA graph should include validator and human intervention nodes."""
        graph = build_research_graph(rigor_level="prisma")
        graph_repr = graph.get_graph()
        node_ids = set(graph_repr.nodes)
        assert "validator_post_lit" in node_ids
        assert "validator_post_data" in node_ids
        assert "validator_post_analysis" in node_ids
        assert "human_post_lit" in node_ids
        assert "human_post_data" in node_ids
        assert "human_post_analysis" in node_ids

    def test_entry_point_is_literature_review(self):
        """All graphs start at literature_review."""
        graph = build_research_graph(rigor_level="exploratory")
        graph_repr = graph.get_graph()
        # The __start__ node should have an edge to literature_review
        start_edges = [e for e in graph_repr.edges if e.source == "__start__"]
        assert any(e.target == "literature_review" for e in start_edges)

    def test_terminal_edge_audit_formatter_to_end(self):
        """All graphs end with audit_formatter → __end__."""
        graph = build_research_graph(rigor_level="exploratory")
        graph_repr = graph.get_graph()
        end_edges = [
            e
            for e in graph_repr.edges
            if e.source == "audit_formatter" and e.target == "__end__"
        ]
        assert len(end_edges) == 1
