"""
core/nodes/human_intervention_node.py
======================================
LangGraph node that pauses for human-in-the-loop decisions
when validation fails.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Any, Dict

from core.state import ResearchState, HumanDecision, append_audit

logger = logging.getLogger(__name__)


def _prompt_for_decision(gate_name: str) -> tuple[str, str]:
    """Block on stdin for a human decision at a failed validation gate."""
    valid = {"retry", "override", "abort"}
    while True:
        try:
            raw = input(
                f"Decision for gate '{gate_name}' [retry/override/abort]: "
            ).strip().lower()
        except (EOFError, KeyboardInterrupt):
            logger.warning("No stdin available for gate '%s'; aborting.", gate_name)
            return "abort", "stdin_unavailable"
        if raw in valid:
            reason = input("Reason (optional): ").strip()
            return raw, reason
        print(f"Invalid choice '{raw}'. Enter one of: retry, override, abort.")


async def human_intervention_node(
    state: ResearchState,
    config: Dict[str, Any] | None = None,
) -> Dict[str, Any]:
    """
    LangGraph node: Human Intervention

    Called when a quality validation gate fails. Presents the
    failures to the user and waits for a decision:
      - ``retry``    → re-run the previous node
      - ``override`` → continue despite failures
      - ``abort``    → stop the pipeline

    In non-interactive mode (e.g. CI/tests), this node aborts by default.
    Auto-override must be explicitly enabled via runtime config.
    """
    config = config or {}
    cfgr = config.get("configurable", {})
    interactive = cfgr.get("interactive", True)
    allow_auto_override = bool(cfgr.get("allow_auto_override", False))
    auto_override_gates = {
        str(g).strip().lower()
        for g in cfgr.get("auto_override_gates", [])
        if str(g).strip()
    }

    # Get the latest validation report
    reports = state.get("validation_reports", [])
    latest = reports[-1] if reports else {}
    failures = latest.get("failures", ["Unknown validation failure"])
    gate_name = latest.get("gate_name", "unknown")
    retry_target = state.get("last_failed_node")
    reason = ""

    # ---- Present to user ----
    print("\n" + "=" * 60)
    print(f"[!] VALIDATION FAILED at gate: {gate_name}")
    print("=" * 60)
    for i, failure in enumerate(failures, 1):
        print(f"  {i}. {failure}")
    print()

    if interactive:
        decision, reason = _prompt_for_decision(gate_name)
    elif allow_auto_override or gate_name.strip().lower() in auto_override_gates:
        decision = "override"
        reason = "auto_override_enabled"
        logger.warning(
            "Non-interactive override policy applied at gate '%s'.", gate_name
        )
    else:
        decision = "abort"
        reason = "non_interactive_default_abort"
        logger.warning(
            "Non-interactive run with no override policy: aborting at gate '%s'.",
            gate_name,
        )

    # Record decision
    human_decision: HumanDecision = {
        "gate_name": gate_name,
        "validation_failures": failures,
        "decision": decision,
        "reason": reason or f"Auto-{decision}",
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }

    decisions = list(state.get("human_decisions", []))
    decisions.append(human_decision)

    audit_log = append_audit(
        state,
        agent="human_intervention_node",
        action=f"human_{decision}",
        inputs={"gate_name": gate_name, "failures": failures},
        output_summary=f"Human decision at '{gate_name}': {decision}",
        provenance={"reason": reason, "override_reason": reason or None},
    )

    logger.info(f"Human decision at '{gate_name}': {decision}")

    return {
        "human_decisions": decisions,
        "human_decision": decision,
        "retry_target": retry_target if decision == "retry" else None,
        "abort": decision == "abort",
        "last_validation_passed": decision == "override",
        "audit_log": audit_log,
    }
