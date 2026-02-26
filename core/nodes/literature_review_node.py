"""
core/nodes/literature_review_node.py
====================================
LangGraph node that runs literature search using LLM-powered
query formulation + multi-database search.
"""

from __future__ import annotations

import logging
from typing import Any, Dict

from core.state import ResearchState, append_audit
from core.tools.search_tools import search_multiple_databases

logger = logging.getLogger(__name__)


async def literature_review_node(
    state: ResearchState,
    config: Dict[str, Any] | None = None,
) -> Dict[str, Any]:
    """
    LangGraph node: Literature Review

    1. Uses LLM to formulate optimised search queries from topic + goals
    2. Searches multiple databases concurrently
    3. Updates PRISMA counts
    4. Appends audit entries
    """
    config = config or {}
    cfgr = config.get("configurable", {})
    llm = cfgr.get("llm_fast") or cfgr.get("llm")  # prefer fast tier

    topic = state["research_topic"]
    goals = state.get("research_goals", [])

    # ---- Step 1: Formulate search queries ----
    if llm:
        system = (
            "You are a research librarian. Generate 3-5 precise search queries "
            "for academic databases (ArXiv, Semantic Scholar, Crossref) given "
            "a research topic and goals. Return one query per line, nothing else."
        )
        prompt = f"Topic: {topic}\nGoals: {', '.join(goals)}"
        raw = await llm.generate(prompt, system_prompt=system, temperature=0.4)
        queries = [q.strip() for q in raw.strip().split("\n") if q.strip()]
    else:
        # Fallback: simple concatenation (matches old agent behavior)
        queries = [f"{topic} {g}" for g in goals] if goals else [topic]

    logger.info(f"Generated {len(queries)} search queries")

    # ---- Step 2: Search databases ----
    databases = config.get("configurable", {}).get(
        "databases", ["arxiv", "semantic_scholar", "crossref"]
    )
    all_papers = []
    all_dbs_searched = set(state.get("databases_searched", []))

    for query in queries:
        papers, dbs = await search_multiple_databases(
            query=query,
            databases=databases,
            max_results_per_db=20,
        )
        all_papers.extend(papers)
        all_dbs_searched.update(dbs)

    # Deduplicate across queries
    seen_ids = set()
    unique_papers = []
    for p in all_papers:
        key = p.paper_id or p.title[:60].lower()
        if key not in seen_ids:
            seen_ids.add(key)
            unique_papers.append(p.to_dict())

    logger.info(f"Total unique papers found: {len(unique_papers)}")

    # ---- Step 3: Update state ----
    audit_log = append_audit(
        state,
        agent="literature_review_node",
        action="multi_database_search",
        inputs={"queries": queries, "databases": databases},
        output_summary=f"Found {len(unique_papers)} papers across {list(all_dbs_searched)}",
        provenance={
            "model_tier": "fast",
            "model": getattr(llm, "model", "unknown") if llm else None,
            "query_formulation": "llm" if llm else "fallback",
            "raw_hits": len(all_papers),
            "after_dedup": len(unique_papers),
            "databases_searched": list(all_dbs_searched),
            "paper_ids": [p["paper_id"] for p in unique_papers[:20]],
        },
    )

    return {
        "current_node": "literature_review",
        "search_queries": queries,
        "databases_searched": list(all_dbs_searched),
        "papers_found": len(unique_papers),
        "papers": unique_papers,
        "audit_log": audit_log,
    }
