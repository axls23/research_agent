"""
core/nodes/analysis_node.py
=============================
LangGraph node that runs analysis on extracted knowledge.
"""

from __future__ import annotations

import logging
from typing import Any, Dict

from core.state import ResearchState, append_audit

logger = logging.getLogger(__name__)


async def analysis_node(
    state: ResearchState,
    config: Dict[str, Any] | None = None,
) -> Dict[str, Any]:
    """
    LangGraph node: Analysis

    1. Summarise patterns across papers using LLM
    2. Generate descriptive statistics (paper counts, topic distribution)
    3. Optionally run risk-of-bias assessment
    """
    config = config or {}
    llm = config.get("configurable", {}).get("llm")

    entities = state.get("knowledge_entities", [])
    papers = state.get("papers", [])
    results = list(state.get("analysis_results", []))

    # ---- Descriptive summary ----
    included = [p for p in papers if p.get("included", True)]
    topic_dist: Dict[str, int] = {}
    for e in entities:
        label = e.get("label", "unknown")
        topic_dist[label] = topic_dist.get(label, 0) + 1

    results.append({
        "method": "descriptive",
        "result_summary": (
            f"{len(included)} papers included, "
            f"{len(entities)} entities extracted, "
            f"distribution: {topic_dist}"
        ),
        "figures": [],
        "tables": [{"entity_distribution": topic_dist}],
        "statistical_output": {
            "papers_included": len(included),
            "entity_count": len(entities),
            "topic_distribution": topic_dist,
        },
    })

    # ---- LLM synthesis ----
    if llm and entities:
        entity_texts = [e.get("text", "") for e in entities[:50]]
        system = (
            "You are a research analyst. Given a list of key concepts/methods/results "
            "extracted from academic papers, identify the main themes, gaps, and "
            "emerging trends. Be concise and structured."
        )
        prompt = f"Entities: {', '.join(entity_texts)}"

        try:
            synthesis = await llm.generate(
                prompt, system_prompt=system, temperature=0.5, max_tokens=2048,
            )
            results.append({
                "method": "llm_synthesis",
                "result_summary": synthesis,
                "figures": [],
                "tables": [],
                "statistical_output": None,
            })
        except Exception as e:
            logger.warning(f"LLM synthesis failed: {e}")

    audit_log = append_audit(
        state,
        agent="analysis_node",
        action="run_analysis",
        inputs={"entity_count": len(entities), "paper_count": len(papers)},
        output_summary=f"Produced {len(results)} analysis results",
    )

    return {
        "current_node": "analysis",
        "analysis_results": results,
        "audit_log": audit_log,
    }
