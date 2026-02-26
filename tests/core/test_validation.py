"""Tests for validation tools and quality_validator_node.

Validates:
  - validate_search_coverage checks DB count, search log, paper count
  - validate_extraction_completeness checks extraction rate, quality scores
  - validate_analysis_assumptions checks required methods
  - run_validation_gate orchestrates validators and produces ValidationReport
  - quality_validator_node async LangGraph node integrates correctly
"""

import sys
from pathlib import Path
from unittest.mock import patch
import asyncio

import pytest

# Ensure the project root is on sys.path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from core.state import make_initial_state, ResearchState, ValidationReport
from core.tools.validation_tools import (
    validate_search_coverage,
    validate_extraction_completeness,
    validate_analysis_assumptions,
    run_validation_gate,
)
from core.nodes.quality_validator_node import quality_validator_node


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _run(coro):
    """Run an async coroutine synchronously for test convenience."""
    return asyncio.get_event_loop().run_until_complete(coro)


def _make_state(**overrides) -> ResearchState:
    defaults = dict(
        project_id="test-v",
        project_name="Validation Test",
        research_topic="Testing",
        research_goals=["coverage"],
        rigor_level="prisma",
    )
    defaults.update(overrides)
    return make_initial_state(**defaults)


def _make_paper(paper_id="p1", included=True, full_text=None, quality_score=None):
    return {
        "paper_id": paper_id,
        "title": f"Paper {paper_id}",
        "authors": ["Author"],
        "year": 2024,
        "abstract": "Abstract",
        "source_url": "https://example.com",
        "databases": ["arxiv"],
        "full_text": full_text,
        "annotations": None,
        "quality_score": quality_score,
        "included": included,
        "exclusion_reason": None,
    }


# ---------------------------------------------------------------------------
# validate_search_coverage
# ---------------------------------------------------------------------------


class TestValidateSearchCoverage:
    def test_passes_with_sufficient_databases(self):
        state = _make_state()
        state["databases_searched"] = ["arxiv", "semantic_scholar", "crossref"]
        failures = validate_search_coverage(state, {"min_databases": 2})
        assert failures == []

    def test_fails_with_insufficient_databases(self):
        state = _make_state()
        state["databases_searched"] = ["arxiv"]
        failures = validate_search_coverage(state, {"min_databases": 3})
        assert len(failures) == 1
        assert "database" in failures[0].lower()

    def test_search_log_required_and_present(self):
        state = _make_state()
        state["databases_searched"] = ["arxiv"]
        state["audit_log"] = [
            {
                "timestamp": "",
                "agent": "a",
                "action": "search_arxiv",
                "input_hash": "",
                "output_summary": "",
                "provenance": {},
            }
        ]
        failures = validate_search_coverage(state, {"require_search_log": True})
        assert failures == []

    def test_search_log_required_but_missing(self):
        state = _make_state()
        state["databases_searched"] = ["arxiv"]
        state["audit_log"] = []
        failures = validate_search_coverage(state, {"require_search_log": True})
        assert len(failures) == 1
        assert "search" in failures[0].lower()

    def test_min_papers_satisfied(self):
        state = _make_state()
        state["databases_searched"] = ["arxiv"]
        state["papers_found"] = 15
        failures = validate_search_coverage(state, {"min_papers_found": 10})
        assert failures == []

    def test_min_papers_below_threshold(self):
        state = _make_state()
        state["databases_searched"] = ["arxiv"]
        state["papers_found"] = 3
        failures = validate_search_coverage(state, {"min_papers_found": 10})
        assert len(failures) == 1
        assert "paper" in failures[0].lower()

    def test_multiple_failures_compound(self):
        state = _make_state()
        state["databases_searched"] = []
        state["papers_found"] = 0
        state["audit_log"] = []
        criteria = {
            "min_databases": 3,
            "min_papers_found": 10,
            "require_search_log": True,
        }
        failures = validate_search_coverage(state, criteria)
        assert len(failures) >= 2

    def test_empty_criteria_passes(self):
        state = _make_state()
        state["databases_searched"] = ["arxiv"]
        failures = validate_search_coverage(state, {})
        assert failures == []


# ---------------------------------------------------------------------------
# validate_extraction_completeness
# ---------------------------------------------------------------------------


class TestValidateExtractionCompleteness:
    def test_no_papers_fails(self):
        state = _make_state()
        state["papers"] = []
        failures = validate_extraction_completeness(state, {})
        assert len(failures) == 1
        assert "no papers" in failures[0].lower()

    def test_no_included_papers_fails(self):
        state = _make_state()
        state["papers"] = [_make_paper(included=False)]
        failures = validate_extraction_completeness(state, {})
        assert len(failures) == 1
        assert "included" in failures[0].lower()

    def test_below_min_extraction_rate(self):
        state = _make_state()
        state["papers"] = [
            _make_paper("p1", full_text="text"),
            _make_paper("p2", full_text=None),
            _make_paper("p3", full_text=None),
            _make_paper("p4", full_text=None),
        ]
        failures = validate_extraction_completeness(state, {"min_extraction_rate": 0.8})
        assert len(failures) == 1
        assert "extraction rate" in failures[0].lower()

    def test_above_min_extraction_rate(self):
        state = _make_state()
        state["papers"] = [
            _make_paper("p1", full_text="text1"),
            _make_paper("p2", full_text="text2"),
            _make_paper("p3", full_text="text3"),
        ]
        failures = validate_extraction_completeness(state, {"min_extraction_rate": 0.8})
        assert failures == []

    def test_below_min_quality_score(self):
        state = _make_state()
        state["papers"] = [
            _make_paper("p1", full_text="t", quality_score=0.3),
            _make_paper("p2", full_text="t", quality_score=0.2),
        ]
        failures = validate_extraction_completeness(state, {"min_quality_score": 0.6})
        assert len(failures) == 1
        assert "quality" in failures[0].lower()

    def test_above_min_quality_score(self):
        state = _make_state()
        state["papers"] = [
            _make_paper("p1", full_text="t", quality_score=0.9),
            _make_paper("p2", full_text="t", quality_score=0.8),
        ]
        failures = validate_extraction_completeness(state, {"min_quality_score": 0.6})
        assert failures == []

    def test_mixed_rate_fail_quality_pass(self):
        state = _make_state()
        state["papers"] = [
            _make_paper("p1", full_text="text", quality_score=0.9),
            _make_paper("p2", full_text=None, quality_score=0.8),
            _make_paper("p3", full_text=None, quality_score=0.7),
        ]
        failures = validate_extraction_completeness(
            state, {"min_extraction_rate": 0.8, "min_quality_score": 0.5}
        )
        # Only extraction rate should fail
        assert len(failures) == 1
        assert "extraction rate" in failures[0].lower()


# ---------------------------------------------------------------------------
# validate_analysis_assumptions
# ---------------------------------------------------------------------------


class TestValidateAnalysisAssumptions:
    def test_required_methods_present(self):
        state = _make_state()
        state["analysis_results"] = [{"method": "descriptive", "result_summary": ""}]
        failures = validate_analysis_assumptions(
            state, {"require_methods": ["descriptive"]}
        )
        assert failures == []

    def test_missing_required_method(self):
        state = _make_state()
        state["analysis_results"] = []
        failures = validate_analysis_assumptions(
            state, {"require_methods": ["risk_of_bias"]}
        )
        assert len(failures) == 1
        assert "risk_of_bias" in failures[0]

    def test_multiple_missing_methods(self):
        state = _make_state()
        state["analysis_results"] = []
        failures = validate_analysis_assumptions(
            state, {"require_methods": ["descriptive", "risk_of_bias"]}
        )
        assert len(failures) == 1
        assert "descriptive" in failures[0]
        assert "risk_of_bias" in failures[0]

    def test_no_required_methods_passes(self):
        state = _make_state()
        state["analysis_results"] = []
        failures = validate_analysis_assumptions(state, {"require_methods": []})
        assert failures == []


# ---------------------------------------------------------------------------
# run_validation_gate
# ---------------------------------------------------------------------------


class TestRunValidationGate:
    def test_exploratory_always_passes(self):
        state = _make_state(rigor_level="exploratory")
        report = run_validation_gate(state, "post_literature_review")
        assert report["passed"] is True
        assert report["failures"] == []
        assert report["rigor_level"] == "exploratory"

    @patch("core.tools.validation_tools.load_workflow_config")
    def test_prisma_post_lit_calls_search_coverage(self, mock_config):
        mock_config.return_value = {
            "validation": {
                "search_coverage": {"min_databases": 3, "min_papers_found": 10},
            }
        }
        state = _make_state(rigor_level="prisma")
        state["databases_searched"] = ["arxiv"]
        state["papers_found"] = 2
        report = run_validation_gate(state, "post_literature_review")
        assert report["passed"] is False
        assert len(report["failures"]) >= 1
        mock_config.assert_called_once_with("prisma")

    @patch("core.tools.validation_tools.load_workflow_config")
    def test_prisma_post_data_calls_extraction(self, mock_config):
        mock_config.return_value = {
            "validation": {
                "extraction": {"min_extraction_rate": 0.8},
            }
        }
        state = _make_state(rigor_level="prisma")
        state["papers"] = [
            _make_paper("p1", full_text=None),
            _make_paper("p2", full_text=None),
        ]
        report = run_validation_gate(state, "post_data_processing")
        assert report["passed"] is False

    @patch("core.tools.validation_tools.load_workflow_config")
    def test_prisma_post_analysis_calls_analysis_assumptions(self, mock_config):
        mock_config.return_value = {
            "validation": {
                "analysis": {"require_methods": ["descriptive"]},
            }
        }
        state = _make_state(rigor_level="prisma")
        state["analysis_results"] = []
        report = run_validation_gate(state, "post_analysis")
        assert report["passed"] is False
        assert "descriptive" in report["failures"][0]

    @patch("core.tools.validation_tools.load_workflow_config")
    def test_unknown_gate_passes(self, mock_config):
        mock_config.return_value = {"validation": {}}
        state = _make_state(rigor_level="prisma")
        report = run_validation_gate(state, "post_something_unknown")
        assert report["passed"] is True

    def test_report_has_required_fields(self):
        state = _make_state(rigor_level="exploratory")
        report = run_validation_gate(state, "post_literature_review")
        assert "gate_name" in report
        assert "rigor_level" in report
        assert "passed" in report
        assert "criteria" in report
        assert "failures" in report
        assert "timestamp" in report
        assert report["gate_name"] == "post_literature_review"


# ---------------------------------------------------------------------------
# quality_validator_node (async LangGraph node)
# ---------------------------------------------------------------------------


class TestQualityValidatorNode:
    @patch("core.nodes.quality_validator_node.run_validation_gate")
    def test_returns_validation_passed(self, mock_gate):
        mock_gate.return_value = ValidationReport(
            gate_name="post_literature_review",
            rigor_level="prisma",
            passed=True,
            criteria={},
            failures=[],
            timestamp="2026-01-01T00:00:00+00:00",
        )
        state = _make_state(rigor_level="prisma")
        state["current_node"] = "literature_review"
        result = _run(quality_validator_node(state))
        assert result["last_validation_passed"] is True

    @patch("core.nodes.quality_validator_node.run_validation_gate")
    def test_returns_validation_failed(self, mock_gate):
        mock_gate.return_value = ValidationReport(
            gate_name="post_literature_review",
            rigor_level="prisma",
            passed=False,
            criteria={},
            failures=["Not enough DBs"],
            timestamp="2026-01-01T00:00:00+00:00",
        )
        state = _make_state(rigor_level="prisma")
        state["current_node"] = "literature_review"
        result = _run(quality_validator_node(state))
        assert result["last_validation_passed"] is False

    @patch("core.nodes.quality_validator_node.run_validation_gate")
    def test_appends_to_validation_reports(self, mock_gate):
        mock_gate.return_value = ValidationReport(
            gate_name="post_data_processing",
            rigor_level="prisma",
            passed=True,
            criteria={},
            failures=[],
            timestamp="2026-01-01T00:00:00+00:00",
        )
        state = _make_state(rigor_level="prisma")
        state["current_node"] = "data_processing"
        state["validation_reports"] = []
        result = _run(quality_validator_node(state))
        assert len(result["validation_reports"]) == 1

    @patch("core.nodes.quality_validator_node.run_validation_gate")
    def test_appends_audit_entry(self, mock_gate):
        mock_gate.return_value = ValidationReport(
            gate_name="post_analysis",
            rigor_level="cochrane",
            passed=False,
            criteria={},
            failures=["Missing method"],
            timestamp="2026-01-01T00:00:00+00:00",
        )
        state = _make_state(rigor_level="cochrane")
        state["current_node"] = "analysis"
        result = _run(quality_validator_node(state))
        assert len(result["audit_log"]) >= 1
        assert result["audit_log"][-1]["agent"] == "quality_validator_node"

    @patch("core.nodes.quality_validator_node.run_validation_gate")
    def test_gate_name_mapping(self, mock_gate):
        """current_node 'literature_review' maps to gate 'post_literature_review'."""
        mock_gate.return_value = ValidationReport(
            gate_name="post_literature_review",
            rigor_level="prisma",
            passed=True,
            criteria={},
            failures=[],
            timestamp="2026-01-01T00:00:00+00:00",
        )
        state = _make_state(rigor_level="prisma")
        state["current_node"] = "literature_review"
        _run(quality_validator_node(state))
        mock_gate.assert_called_once()
        call_gate = mock_gate.call_args[0][1]
        assert call_gate == "post_literature_review"
