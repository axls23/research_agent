"""
core/nodes/quality_validator_node.py
=====================================
LangGraph node that checks the current state against
PRISMA/Cochrane criteria and sets routing flags.

LLM self-critique: after rule-based
checks, the model reviews the research stage for qualitative
blind spots that rules can't catch.
"""

from __future__ import annotations

import logging
from typing import Any, Dict

from core.state import ResearchState, append_audit
from core.tools.validation_tools import run_validation_gate

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Self-critique prompt templates (per gate)
# ---------------------------------------------------------------------------

_CRITIQUE_TEMPLATES: Dict[str, str] = {
    "post_literature_review": (
        "Review this systematic search for methodological blind spots:\n"
        "- Databases searched: {databases_searched}\n"
        "- Papers found: {papers_found}\n"
        "- Search queries used: {search_queries}\n"
        "- Rigor level: {rigor_level}\n\n"
        "Check for: missing databases, narrow search terms, publication bias, "
        "language bias, date range gaps. Be concise and specific."
    ),
    "post_data_processing": (
        "Review this data extraction step:\n"
        "- Papers included: {papers_included}\n"
        "- Chunks extracted: {chunk_count}\n"
        "- Tokens extracted: {total_tokens}\n\n"
        "Check for: low extraction rate, missing full-text papers, "
        "over-reliance on abstracts, inconsistent chunking. Be concise."
    ),
    "post_analysis": (
        "Review this analysis step:\n"
        "- Analysis methods: {methods}\n"
        "- Papers analysed: {papers_included}\n"
        "- Entities extracted: {entity_count}\n\n"
        "Check for: missing statistical methods, selection bias in entity "
        "extraction, over-generalisation from small samples. Be concise."
    ),
}


async def quality_validator_node(
    state: ResearchState,
    config: Dict[str, Any] | None = None,
) -> Dict[str, Any]:
    """
    LangGraph node: Quality Validator

    1. Runs rule-based validation checks (PRISMA/Cochrane criteria)
    2. Runs LLM self-critique for qualitative gap detection
    3. Sets ``last_validation_passed`` flag for conditional routing
    """
    config = config or {}
    cfgr = config.get("configurable", {})
    llm = cfgr.get("llm_fast") or cfgr.get("llm")  # prefer fast tier

    current = state.get("current_node", "")

    # Map the current node to a gate name
    gate_map = {
        "literature_review": "post_literature_review",
        "data_processing": "post_data_processing",
        "analysis": "post_analysis",
    }
    gate_name = gate_map.get(current, f"post_{current}")

    # ---- Step 1: Rule-based validation ----
    report = run_validation_gate(state, gate_name)
    report_dict = dict(report)

    # ---- Step 2: LLM self-critique  ----
    model_critique = None
    rigor = state.get("rigor_level", "exploratory")

    if llm and rigor != "exploratory" and gate_name in _CRITIQUE_TEMPLATES:
        model_critique = await _run_self_critique(llm, gate_name, state)
        report_dict["model_critique"] = model_critique
    else:
        report_dict["model_critique"] = None

    # Append to validation reports
    reports = list(state.get("validation_reports", []))
    reports.append(report_dict)

    if report["passed"]:
        logger.info(f"Validation gate '{gate_name}' PASSED")
    else:
        logger.warning(
            f" Validation gate '{gate_name}' FAILED: " f"{report['failures']}"
        )

    if model_critique:
        logger.info(f" Model critique for '{gate_name}': {model_critique[:120]}...")

    audit_log = append_audit(
        state,
        agent="quality_validator_node",
        action=f"validate_{gate_name}",
        inputs={"gate_name": gate_name, "rigor_level": rigor},
        output_summary=(
            f"Gate '{gate_name}': {'PASSED' if report['passed'] else 'FAILED'}"
            + (f" — {report['failures']}" if not report["passed"] else "")
            + (f" | Critique: {model_critique[:80]}..." if model_critique else "")
        ),
        provenance={
            "model_tier": "fast",
            "critique_generated": model_critique is not None,
        },
    )

    return {
        "last_validation_passed": report["passed"],
        "validation_reports": reports,
        "audit_log": audit_log,
    }


async def _run_self_critique(
    llm: Any,
    gate_name: str,
    state: ResearchState,
) -> str | None:
    """Run LLM self-critique for the given validation gate."""
    template = _CRITIQUE_TEMPLATES.get(gate_name)
    if not template:
        return None

    # Build context from state
    context = {
        "databases_searched": state.get("databases_searched", []),
        "papers_found": state.get("papers_found", 0),
        "search_queries": state.get("search_queries", [])[:5],  # cap
        "rigor_level": state.get("rigor_level", ""),
        "papers_included": state.get("papers_included", 0),
        "chunk_count": len(state.get("chunks", [])),
        "total_tokens": state.get("total_tokens_extracted", 0),
        "methods": [r.get("method", "") for r in state.get("analysis_results", [])],
        "entity_count": len(state.get("knowledge_entities", [])),
    }

    prompt = template.format(**context)
    system = (
        "You are a methodological reviewer for systematic reviews. "
        "Critique the research step objectively. Identify 2-3 specific "
        "blind spots or gaps. If everything looks sound, say so briefly. "
        "Keep your response under 150 words."
    )

    try:
        critique = await llm.generate(
            prompt, system_prompt=system, temperature=0.3, max_tokens=512
        )
        return critique.strip()
    except Exception as e:
        logger.warning(f"Self-critique failed: {e}")
        return None
