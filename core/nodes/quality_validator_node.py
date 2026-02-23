"""
core/nodes/quality_validator_node.py
=====================================
LangGraph node that checks the current state against
PRISMA/Cochrane criteria and sets routing flags.
"""

from __future__ import annotations

import logging
from typing import Any, Dict

from core.state import ResearchState, append_audit
from core.tools.validation_tools import run_validation_gate

logger = logging.getLogger(__name__)


async def quality_validator_node(
    state: ResearchState,
    config: Dict[str, Any] | None = None,
) -> Dict[str, Any]:
    """
    LangGraph node: Quality Validator

    Runs validation checks based on the current node and rigor level.
    Sets ``last_validation_passed`` flag for conditional routing.
    """
    current = state.get("current_node", "")

    # Map the current node to a gate name
    gate_map = {
        "literature_review": "post_literature_review",
        "data_processing": "post_data_processing",
        "analysis": "post_analysis",
    }
    gate_name = gate_map.get(current, f"post_{current}")

    # Run validation
    report = run_validation_gate(state, gate_name)

    # Append to validation reports
    reports = list(state.get("validation_reports", []))
    reports.append(dict(report))

    if report["passed"]:
        logger.info(f"✅ Validation gate '{gate_name}' PASSED")
    else:
        logger.warning(
            f"❌ Validation gate '{gate_name}' FAILED: "
            f"{report['failures']}"
        )

    audit_log = append_audit(
        state,
        agent="quality_validator_node",
        action=f"validate_{gate_name}",
        inputs={"gate_name": gate_name, "rigor_level": state.get("rigor_level")},
        output_summary=(
            f"Gate '{gate_name}': {'PASSED' if report['passed'] else 'FAILED'}"
            + (f" — {report['failures']}" if not report["passed"] else "")
        ),
    )

    return {
        "last_validation_passed": report["passed"],
        "validation_reports": reports,
        "audit_log": audit_log,
    }
