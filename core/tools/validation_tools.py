"""
core/tools/validation_tools.py
===============================
Validation functions that check research state against
PRISMA / Cochrane methodological criteria loaded from YAML.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml

from core.state import ResearchState, ValidationReport

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Load workflow config
# ---------------------------------------------------------------------------

def load_workflow_config(rigor_level: str) -> Dict[str, Any]:
    """
    Load validation criteria from a YAML workflow file.

    Searches ``core/workflows/{rigor_level}.yaml`` relative to this file.
    """
    base_dir = Path(__file__).parent.parent / "workflows"
    yaml_path = base_dir / f"{rigor_level}.yaml"

    if not yaml_path.exists():
        logger.warning(f"Workflow config not found: {yaml_path}, using empty config")
        return {}

    with open(yaml_path) as f:
        return yaml.safe_load(f) or {}


# ---------------------------------------------------------------------------
# Individual validation checks
# ---------------------------------------------------------------------------

def validate_search_coverage(
    state: ResearchState,
    criteria: Dict[str, Any],
) -> List[str]:
    """
    Check that the literature search meets coverage requirements.

    Criteria keys:
    - ``min_databases``: minimum number of databases searched
    - ``require_search_log``: if True, audit_log must have search entries
    - ``min_papers_found``: minimum raw hit count
    """
    failures: List[str] = []

    min_dbs = criteria.get("min_databases", 1)
    if len(state.get("databases_searched", [])) < min_dbs:
        failures.append(
            f"Only {len(state.get('databases_searched', []))} database(s) searched; "
            f"minimum is {min_dbs}"
        )

    if criteria.get("require_search_log", False):
        search_entries = [
            e for e in state.get("audit_log", [])
            if "search" in e.get("action", "").lower()
        ]
        if not search_entries:
            failures.append("No search activity found in audit log")

    min_papers = criteria.get("min_papers_found", 0)
    if state.get("papers_found", 0) < min_papers:
        failures.append(
            f"Only {state.get('papers_found', 0)} papers found; "
            f"minimum is {min_papers}"
        )

    return failures


def validate_extraction_completeness(
    state: ResearchState,
    criteria: Dict[str, Any],
) -> List[str]:
    """
    Check that paper extraction is complete enough for the review.

    Criteria keys:
    - ``min_extraction_rate``: fraction of included papers that must have text
    - ``min_quality_score``: minimum average quality score
    """
    failures: List[str] = []
    papers = state.get("papers", [])

    if not papers:
        failures.append("No papers in state to validate extraction")
        return failures

    included = [p for p in papers if p.get("included", True)]
    if not included:
        failures.append("No included papers")
        return failures

    # Check extraction rate
    extracted = [p for p in included if p.get("full_text")]
    rate = len(extracted) / len(included) if included else 0
    min_rate = criteria.get("min_extraction_rate", 0.5)
    if rate < min_rate:
        failures.append(
            f"Extraction rate {rate:.0%} is below minimum {min_rate:.0%} "
            f"({len(extracted)}/{len(included)} papers extracted)"
        )

    # Check quality scores
    scored = [p for p in included if p.get("quality_score") is not None]
    if scored:
        avg_quality = sum(p["quality_score"] for p in scored) / len(scored)
        min_quality = criteria.get("min_quality_score", 0.0)
        if avg_quality < min_quality:
            failures.append(
                f"Average quality score {avg_quality:.2f} is below "
                f"minimum {min_quality:.2f}"
            )

    return failures


def validate_analysis_assumptions(
    state: ResearchState,
    criteria: Dict[str, Any],
) -> List[str]:
    """
    Check that analysis methods are appropriate.

    Criteria keys:
    - ``require_methods``: list of method names that must be present
    """
    failures: List[str] = []
    results = state.get("analysis_results", [])

    required = criteria.get("require_methods", [])
    present = {r.get("method", "") for r in results}
    missing = set(required) - present
    if missing:
        failures.append(f"Missing required analysis methods: {missing}")

    return failures


# ---------------------------------------------------------------------------
# Master validation runner
# ---------------------------------------------------------------------------

def run_validation_gate(
    state: ResearchState,
    gate_name: str,
) -> ValidationReport:
    """
    Run all validation checks for a specific gate using the criteria
    defined in the rigor-level's YAML config.

    Gate names map to YAML sections::

        validation:
          search_coverage:
            min_databases: 3
          extraction:
            min_extraction_rate: 0.8
          analysis:
            require_methods: ["descriptive"]

    For "exploratory" rigor, validation always passes (no gates).
    """
    rigor = state.get("rigor_level", "exploratory")

    # Exploratory = no validation gates
    if rigor == "exploratory":
        return ValidationReport(
            gate_name=gate_name,
            rigor_level=rigor,
            passed=True,
            criteria={},
            failures=[],
            timestamp=datetime.now(timezone.utc).isoformat(),
        )

    config = load_workflow_config(rigor)
    validation_config = config.get("validation", {})

    all_failures: List[str] = []

    # Map gate names to validation functions and their config sections
    gate_map = {
        "post_literature_review": [
            (validate_search_coverage, validation_config.get("search_coverage", {})),
        ],
        "post_data_processing": [
            (validate_extraction_completeness, validation_config.get("extraction", {})),
        ],
        "post_analysis": [
            (validate_analysis_assumptions, validation_config.get("analysis", {})),
        ],
    }

    checks = gate_map.get(gate_name, [])
    used_criteria: Dict[str, Any] = {}

    for validate_fn, criteria in checks:
        used_criteria[validate_fn.__name__] = criteria
        failures = validate_fn(state, criteria)
        all_failures.extend(failures)

    return ValidationReport(
        gate_name=gate_name,
        rigor_level=rigor,
        passed=len(all_failures) == 0,
        criteria=used_criteria,
        failures=all_failures,
        timestamp=datetime.now(timezone.utc).isoformat(),
    )
