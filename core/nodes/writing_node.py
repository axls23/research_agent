"""
core/nodes/writing_node.py
===========================
LangGraph node that generates outlines, draft sections,
and synthesised literature summaries.

Includes gap detection: after drafting, the
model checks for unsupported claims and signals the graph to
backtrack to literature_review if evidence gaps are found.
"""

from __future__ import annotations

import logging
from typing import Any, Dict

from core.state import ResearchState, append_audit

logger = logging.getLogger(__name__)

MAX_BACKTRACKS = 2  # Cap loops to prevent infinite cycles


async def writing_node(
    state: ResearchState,
    config: Dict[str, Any] | None = None,
) -> Dict[str, Any]:
    """
    LangGraph node: Writing Assistant

    1. Generate paper outline from analysis results
    2. Draft key sections (Introduction, Literature Review, Methods, Results)
    3. Produce synthesis text from knowledge entities
    4. Check for evidence gaps → signal backtrack if needed
    """
    config = config or {}
    cfgr = config.get("configurable", {})
    llm = cfgr.get("llm_deep") or cfgr.get("llm")  # prefer deep tier

    topic = state.get("research_topic", "")
    results = state.get("analysis_results", [])
    entities = state.get("knowledge_entities", [])
    draft_sections: Dict[str, str] = dict(state.get("draft_sections", {}))
    outline = state.get("outline")
    backtrack_count = state.get("backtrack_count", 0)

    if not llm:
        # Stub output without LLM
        outline = f"# {topic}\n## Introduction\n## Literature Review\n## Methods\n## Results\n## Discussion"
        draft_sections["stub"] = "LLM not available — manual writing required."

        audit_log = append_audit(
            state,
            agent="writing_node",
            action="generate_stub",
            inputs={"topic": topic},
            output_summary="Generated stub outline (no LLM)",
            provenance={"model_tier": None},
        )
        return {
            "current_node": "writing",
            "outline": outline,
            "draft_sections": draft_sections,
            "needs_more_papers": False,
            "audit_log": audit_log,
        }

    # ---- Generate outline ----
    system_outline = (
        "You are an academic writing assistant. Generate a detailed paper "
        "outline for a systematic review on the given topic. Include main "
        "sections and subsections. Format as markdown headers."
    )
    analysis_summary = "\n".join(r.get("result_summary", "") for r in results)[:3000]

    outline = await llm.generate(
        f"Topic: {topic}\n\nAnalysis findings:\n{analysis_summary}",
        system_prompt=system_outline,
        temperature=0.5,
        max_tokens=2048,
    )

    # ---- Draft Literature Review ----
    entity_texts = [e.get("text", "") for e in entities[:40]]
    lit_review = await llm.generate(
        (
            f"Topic: {topic}\n\n"
            f"Key concepts and methods: {', '.join(entity_texts)}\n\n"
            f"Analysis: {analysis_summary[:2000]}\n\n"
            "Write a coherent literature review section (500-800 words) "
            "synthesising these findings. Use academic tone."
        ),
        system_prompt="You are an expert academic writer.",
        temperature=0.6,
        max_tokens=2048,
    )
    draft_sections["literature_review"] = lit_review

    # ---- Draft Introduction ----
    intro = await llm.generate(
        (
            f"Topic: {topic}\n"
            f"Number of papers reviewed: {state.get('papers_included', 0)}\n"
            f"Key themes: {', '.join(entity_texts[:10])}\n\n"
            "Write a concise introduction (200-300 words) for this systematic "
            "review. Include the research question, significance, and scope."
        ),
        system_prompt="You are an expert academic writer.",
        temperature=0.6,
        max_tokens=1024,
    )
    draft_sections["introduction"] = intro

    # ---- Evidence Gap Detection (backtracking) ----
    needs_more = False
    gap_analysis = None

    if backtrack_count < MAX_BACKTRACKS:
        gap_analysis = await _detect_evidence_gaps(llm, topic, draft_sections, state)
        if gap_analysis:
            needs_more = True
            logger.info(
                f"📝 Evidence gaps detected (backtrack {backtrack_count + 1}/"
                f"{MAX_BACKTRACKS}): {gap_analysis[:120]}..."
            )

    # ---- Audit ----
    audit_log = append_audit(
        state,
        agent="writing_node",
        action="draft_sections",
        inputs={"topic": topic, "section_count": len(draft_sections)},
        output_summary=(
            f"Generated outline + {len(draft_sections)} draft sections"
            + (f" | Gaps detected, requesting backtrack" if needs_more else "")
        ),
        provenance={
            "model_tier": "deep",
            "model": getattr(llm, "model", "unknown"),
            "sections_drafted": list(draft_sections.keys()),
            "backtrack_requested": needs_more,
            "backtrack_count": backtrack_count,
        },
    )

    return {
        "current_node": "writing",
        "outline": outline,
        "draft_sections": draft_sections,
        "needs_more_papers": needs_more,
        "gap_analysis": gap_analysis,
        "backtrack_count": backtrack_count + (1 if needs_more else 0),
        "audit_log": audit_log,
    }


async def _detect_evidence_gaps(
    llm: Any,
    topic: str,
    draft_sections: Dict[str, str],
    state: ResearchState,
) -> str | None:
    """Ask the LLM to identify unsupported claims in the draft."""
    # Combine draft sections for review
    draft_text = "\n\n".join(
        f"## {name}\n{text[:1500]}" for name, text in draft_sections.items()
    )

    system = (
        "You are a critical academic reviewer. Analyze the draft below for "
        "claims that lack supporting evidence from the reviewed papers. "
        "Identify specific topics or claims that need additional literature. "
        "If the draft is well-supported, respond with exactly: NO_GAPS\n\n"
        "If gaps exist, list 2-3 specific missing topics as bullet points."
    )

    prompt = (
        f"Research topic: {topic}\n"
        f"Papers reviewed: {state.get('papers_included', 0)}\n"
        f"Research goals: {state.get('research_goals', [])}\n\n"
        f"Draft:\n{draft_text[:4000]}"
    )

    try:
        response = await llm.generate(
            prompt, system_prompt=system, temperature=0.2, max_tokens=512
        )
        response = response.strip()

        if "NO_GAPS" in response.upper():
            logger.info("No evidence gaps detected in draft")
            return None

        return response

    except Exception as e:
        logger.warning(f"Gap detection failed: {e}")
        return None
