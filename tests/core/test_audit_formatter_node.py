"""Tests for PRISMA metric mapping and invariants in audit_formatter_node."""

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from core.nodes.audit_formatter_node import audit_formatter_node
from core.state import make_initial_state


@pytest.mark.asyncio
async def test_audit_summary_enforces_prisma_invariants(tmp_path: Path):
    state = make_initial_state(
        project_id="prisma-map-test",
        project_name="PRISMA Mapping",
        research_topic="Topic",
        research_goals=["Goal"],
        rigor_level="prisma",
    )

    state["papers_found"] = 12
    state["papers_screened"] = 8  # legacy value that can conflict with papers list
    state["papers_included"] = 12  # intentionally inconsistent legacy value
    state["databases_searched"] = ["arxiv", "crossref", "semantic_scholar"]

    state["papers"] = [
        {
            "paper_id": f"p{i}",
            "title": f"Paper {i}",
            "authors": ["A"],
            "year": 2024,
            "abstract": "...",
            "source_url": "https://example.com",
            "databases": ["arxiv"],
            "full_text": "text" if i <= 8 else None,
            "annotations": {},
            "quality_score": None,
            "included": True,
            "exclusion_reason": None,
            "needs_human_review": None,
        }
        for i in range(1, 13)
    ]

    result = await audit_formatter_node(
        state,
        config={"configurable": {"output_dir": str(tmp_path)}},
    )

    export_path = Path(result["audit_export_path"])
    payload = json.loads(export_path.read_text(encoding="utf-8"))
    summary = payload["summary"]

    assert summary["papers_found"] == 12
    assert summary["papers_screened"] == 12
    assert summary["full_text_assessed"] == 8
    assert summary["papers_included"] == 8
    assert summary["papers_included"] <= summary["papers_screened"]
