"""Tests for non-interactive policy behavior in human_intervention_node."""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from core.nodes.human_intervention_node import human_intervention_node
from core.state import make_initial_state


@pytest.mark.asyncio
async def test_non_interactive_defaults_to_abort_without_override_policy():
    state = make_initial_state(
        project_id="t-1",
        project_name="Test",
        research_topic="Topic",
        research_goals=["Goal"],
        rigor_level="prisma",
    )
    state["validation_reports"] = [
        {
            "gate_name": "post_data_processing",
            "rigor_level": "prisma",
            "passed": False,
            "criteria": {},
            "failures": ["Extraction below threshold"],
            "timestamp": "2026-01-01T00:00:00+00:00",
            "model_critique": None,
        }
    ]

    result = await human_intervention_node(
        state,
        config={"configurable": {"interactive": False}},
    )

    assert result["human_decision"] == "abort"
    assert result["abort"] is True
    assert result["last_validation_passed"] is False


@pytest.mark.asyncio
async def test_non_interactive_can_override_when_policy_enabled():
    state = make_initial_state(
        project_id="t-2",
        project_name="Test",
        research_topic="Topic",
        research_goals=["Goal"],
        rigor_level="prisma",
    )
    state["validation_reports"] = [
        {
            "gate_name": "post_data_processing",
            "rigor_level": "prisma",
            "passed": False,
            "criteria": {},
            "failures": ["Extraction below threshold"],
            "timestamp": "2026-01-01T00:00:00+00:00",
            "model_critique": None,
        }
    ]

    result = await human_intervention_node(
        state,
        config={
            "configurable": {
                "interactive": False,
                "allow_auto_override": True,
            }
        },
    )

    assert result["human_decision"] == "override"
    assert result["abort"] is False
    assert result["last_validation_passed"] is True

    decision = result["human_decisions"][-1]
    assert decision["reason"] == "auto_override_enabled"


@pytest.mark.asyncio
async def test_non_interactive_gate_specific_override_policy():
    state = make_initial_state(
        project_id="t-3",
        project_name="Test",
        research_topic="Topic",
        research_goals=["Goal"],
        rigor_level="prisma",
    )
    state["validation_reports"] = [
        {
            "gate_name": "post_data_processing",
            "rigor_level": "prisma",
            "passed": False,
            "criteria": {},
            "failures": ["Extraction below threshold"],
            "timestamp": "2026-01-01T00:00:00+00:00",
            "model_critique": None,
        }
    ]

    result = await human_intervention_node(
        state,
        config={
            "configurable": {
                "interactive": False,
                "auto_override_gates": ["post_data_processing"],
            }
        },
    )

    assert result["human_decision"] == "override"
    assert result["abort"] is False
