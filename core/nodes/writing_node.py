"""
core/nodes/writing_node.py
===========================
LangGraph node that generates outlines, draft sections,
and synthesised literature summaries.
"""

from __future__ import annotations

import logging
from typing import Any, Dict

from core.state import ResearchState, append_audit

logger = logging.getLogger(__name__)


async def writing_node(
    state: ResearchState,
    config: Dict[str, Any] | None = None,
) -> Dict[str, Any]:
    """
    LangGraph node: Writing Assistant

    1. Generate paper outline from analysis results
    2. Draft key sections (Introduction, Literature Review, Methods, Results)
    3. Produce synthesis text from knowledge entities
    """
    config = config or {}
    llm = config.get("configurable", {}).get("llm")

    topic = state.get("research_topic", "")
    results = state.get("analysis_results", [])
    entities = state.get("knowledge_entities", [])
    draft_sections: Dict[str, str] = dict(state.get("draft_sections", {}))
    outline = state.get("outline")

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
        )
        return {
            "current_node": "writing",
            "outline": outline,
            "draft_sections": draft_sections,
            "audit_log": audit_log,
        }

    # ---- Generate outline ----
    system_outline = (
        "You are an academic writing assistant. Generate a detailed paper "
        "outline for a systematic review on the given topic. Include main "
        "sections and subsections. Format as markdown headers."
    )
    analysis_summary = "\n".join(
        r.get("result_summary", "") for r in results
    )[:3000]

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

    audit_log = append_audit(
        state,
        agent="writing_node",
        action="draft_sections",
        inputs={"topic": topic, "section_count": len(draft_sections)},
        output_summary=f"Generated outline + {len(draft_sections)} draft sections",
    )

    return {
        "current_node": "writing",
        "outline": outline,
        "draft_sections": draft_sections,
        "audit_log": audit_log,
    }
