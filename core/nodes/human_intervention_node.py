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

    In non-interactive mode (e.g. CI/tests), auto-overrides.
    """
    config = config or {}
    interactive = config.get("configurable", {}).get("interactive", True)

    # Get the latest validation report
    reports = state.get("validation_reports", [])
    latest = reports[-1] if reports else {}
    failures = latest.get("failures", ["Unknown validation failure"])
    gate_name = latest.get("gate_name", "unknown")

    # ---- Present to user ----
    print("\n" + "=" * 60)
    print(f"[!] VALIDATION FAILED at gate: {gate_name}")
    print("=" * 60)
    for i, failure in enumerate(failures, 1):
        print(f"  {i}. {failure}")
    print()

    if interactive:
        while True:
            choice = (
                input("Decision — [R]etry / [O]verride / [A]bort: ").strip().lower()
            )
            if choice in ("r", "retry"):
                decision = "retry"
                break
            elif choice in ("o", "override"):
                decision = "override"
                break
            elif choice in ("a", "abort"):
                decision = "abort"
                break
            else:
                print("Invalid choice. Please enter R, O, or A.")
    else:
        # Non-interactive: auto-override
        logger.info("Non-interactive mode — auto-overriding validation failure")
        decision = "override"

    # Get optional reason
    reason = ""
    if interactive and decision != "abort":
        reason = input("Reason (optional, press Enter to skip): ").strip()

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
        provenance={"reason": reason},
    )

    logger.info(f"Human decision at '{gate_name}': {decision}")

    return {
        "human_decisions": decisions,
        "abort": decision == "abort",
        "last_validation_passed": decision == "override",
        "audit_log": audit_log,
    }
