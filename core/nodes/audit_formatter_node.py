"""
core/nodes/audit_formatter_node.py
====================================
Final LangGraph node that generates PRISMA flow diagrams,
Methods sections, and exports the complete audit trail.
"""

from __future__ import annotations

import json
import logging
import os
from datetime import datetime, timezone
from typing import Any, Dict

from core.state import ResearchState, append_audit

logger = logging.getLogger(__name__)


def _compute_prisma_metrics(state: ResearchState) -> Dict[str, int]:
    """Compute PRISMA counters from stage artifacts with invariant guards.

    Mapping decision:
    - identified_records: papers found in search
    - screened_records: records retained after dedup/search stage
    - full_text_assessed: records with extracted text available
    - studies_included: included records with extracted text
    """
    papers = [p for p in state.get("papers", []) if isinstance(p, dict)]

    identified_records = max(int(state.get("papers_found", 0) or 0), len(papers))
    screened_records = max(int(state.get("papers_screened", 0) or 0), len(papers))

    included_candidates = [p for p in papers if p.get("included", True)]
    full_text_assessed = sum(1 for p in papers if p.get("full_text"))
    included_with_full_text = sum(
        1 for p in included_candidates if p.get("full_text")
    )

    if papers:
        studies_included = included_with_full_text
    else:
        studies_included = int(state.get("papers_included", 0) or 0)

    # Enforce PRISMA invariants to avoid negative exclusions and impossible counts.
    screened_records = min(screened_records, identified_records)
    full_text_assessed = min(full_text_assessed, screened_records)
    studies_included = min(studies_included, full_text_assessed)

    excluded_at_screening = max(identified_records - screened_records, 0)
    excluded_at_full_text = max(full_text_assessed - studies_included, 0)

    return {
        "identified_records": identified_records,
        "screened_records": screened_records,
        "full_text_assessed": full_text_assessed,
        "studies_included": studies_included,
        "excluded_at_screening": excluded_at_screening,
        "excluded_at_full_text": excluded_at_full_text,
    }


def _generate_prisma_diagram(state: ResearchState) -> str:
    """
    Generate an ASCII PRISMA 2020 flow diagram from the state.

    Uses the standard PRISMA boxes:
    - Identification → Screening → Included
    """
    metrics = _compute_prisma_metrics(state)
    found = metrics["identified_records"]
    screened = metrics["screened_records"]
    full_text_assessed = metrics["full_text_assessed"]
    included = metrics["studies_included"]
    dbs = ", ".join(state.get("databases_searched", ["N/A"]))
    excluded_screen = metrics["excluded_at_screening"]
    excluded_full = metrics["excluded_at_full_text"]

    diagram = f"""
+=======================================================+
|                   PRISMA 2020 Flow                    |
+=======================================================+
|                                                       |
|  IDENTIFICATION                                       |
|  +-------------------------------------+              |
|  | Records identified through          |              |
|  | database searching: {found:<16}|              |
|  | Databases: {dbs:<25} |              |
|  +------------------+------------------+              |
|                     |                                 |
|                     v                                 |
|  SCREENING                                            |
|  +-------------------------------------+              |
|  | Records screened: {screened:<18}|              |
|  | Records excluded: {excluded_screen:<18}|              |
|  +------------------+------------------+              |
|                     |                                 |
|                     v                                 |
|  INCLUDED                                             |
|  +-------------------------------------+              |
|  | Full-text assessed: {full_text_assessed:<16}|              |
|  | Excluded (full-text): {excluded_full:<13}|              |
|  | Studies included: {included:<18}|              |
|  +-------------------------------------+              |
|                                                       |
+=======================================================+
"""
    return diagram.strip()


def _generate_methods_section(state: ResearchState) -> str:
    """Generate a Methods section text from the audit trail."""
    audit = state.get("audit_log", [])
    dbs = ", ".join(state.get("databases_searched", []))
    queries = state.get("search_queries", [])
    metrics = _compute_prisma_metrics(state)
    found = metrics["identified_records"]
    included = metrics["studies_included"]

    # Extract validation decisions
    decisions = state.get("human_decisions", [])
    decision_text = ""
    if decisions:
        decision_text = (
            "\n\nHuman review was required at the following gates:\n"
            + "\n".join(
                f"- {d['gate_name']}: {d['decision']} ({d.get('reason', 'N/A')})"
                for d in decisions
            )
        )

    methods = f"""## Methods

### Search Strategy
A systematic search was conducted across the following databases: {dbs}.
The search used {len(queries)} query formulations derived from the research
topic and goals.

### Search Queries
{chr(10).join(f'- `{q}`' for q in queries)}

### Screening
A total of {found} records were identified. After title and abstract
screening and full-text review, {included} studies were included
in the final synthesis.

### Data Extraction
Text was extracted using automated tools (Mistral Document AI / PyPDF2).
Documents were chunked using tiktoken-based recursive splitting.

### Quality Assessment
Validation gates were applied at key pipeline stages according to the
selected rigor level ({state.get('rigor_level', 'N/A')}).
{decision_text}

### Audit Trail
This review was conducted using the Research Agent automated pipeline.
A complete audit log containing {len(audit)} entries is available
in the exported JSON file.
"""
    return methods.strip()


async def audit_formatter_node(
    state: ResearchState,
    config: Dict[str, Any] | None = None,
) -> Dict[str, Any]:
    """
    LangGraph node: Audit Formatter (final node)

    1. Generate PRISMA flow diagram
    2. Generate Methods section
    3. Export complete audit log as JSON
    """
    config = config or {}
    output_dir = config.get("configurable", {}).get("output_dir", "outputs")
    os.makedirs(output_dir, exist_ok=True)

    rigor = state.get("rigor_level", "exploratory")
    metrics = _compute_prisma_metrics(state)

    # ---- PRISMA diagram ----
    prisma_diagram = None
    if rigor in ("prisma", "cochrane"):
        prisma_diagram = _generate_prisma_diagram(state)
        logger.info("Generated PRISMA flow diagram")
        print("\n" + prisma_diagram)

    # ---- Methods section ----
    methods = None
    if rigor in ("prisma", "cochrane"):
        methods = _generate_methods_section(state)
        logger.info("Generated Methods section")

    # ---- Export audit log ----
    export_path = os.path.join(
        output_dir,
        f"audit_log_{state.get('project_id', 'unknown')}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
    )

    audit_data = {
        "project_id": state.get("project_id"),
        "project_name": state.get("project_name"),
        "rigor_level": rigor,
        "completed_at": datetime.now(timezone.utc).isoformat(),
        "summary": {
            "papers_found": metrics["identified_records"],
            "papers_screened": metrics["screened_records"],
            "papers_included": metrics["studies_included"],
            "full_text_assessed": metrics["full_text_assessed"],
            "databases": state.get("databases_searched", []),
            "entities_extracted": len(state.get("knowledge_entities", [])),
            "validation_gates": len(state.get("validation_reports", [])),
            "human_interventions": len(state.get("human_decisions", [])),
        },
        "prisma_mapping": {
            "identified_records": "papers_found",
            "screened_records": "max(papers_screened, len(papers)) clamped to identified",
            "full_text_assessed": "count(papers where full_text is present)",
            "studies_included": "count(included papers with full_text), clamped to full_text_assessed",
        },
        "audit_log": state.get("audit_log", []),
        "validation_reports": state.get("validation_reports", []),
        "human_decisions": state.get("human_decisions", []),
    }

    with open(export_path, "w") as f:
        json.dump(audit_data, f, indent=2, default=str)
    logger.info(f"Exported audit log to {export_path}")

    audit_log = append_audit(
        state,
        agent="audit_formatter_node",
        action="export_audit",
        inputs={},
        output_summary=f"Exported audit log to {export_path}",
        provenance={"export_path": export_path},
    )

    return {
        "current_node": "audit_formatter",
        "prisma_flow_diagram": prisma_diagram,
        "methods_section": methods,
        "audit_export_path": export_path,
        "audit_log": audit_log,
    }
