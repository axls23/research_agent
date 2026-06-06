"""core/nodes/dark_data_ingestion_node.py
======================================
Local-first ingestion node for enterprise dark data.
"""

from __future__ import annotations

import csv
import json
import logging
from pathlib import Path
from typing import Any, Dict, List

from core.state import ResearchState, append_audit

logger = logging.getLogger(__name__)


def read_single_path(path: Path, max_chars: int = 12000) -> Dict[str, Any] | None:
    """Read local CSV, JSON, TXT, PDF metadata/content and map to canonical record structures."""
    suffix = path.suffix.lower()
    try:
        if suffix in {".txt", ".md", ".log", ".yaml", ".yml"}:
            text = path.read_text(encoding="utf-8", errors="replace")[:max_chars]
            return {
                "paper_id": path.stem,
                "title": path.stem.replace("_", " "),
                "abstract": text,
                "source_url": str(path),
                "databases": ["local_filesystem"],
                "included": True,
            }

        if suffix == ".json":
            raw = path.read_text(encoding="utf-8", errors="replace")
            payload = json.loads(raw)
            text = json.dumps(payload, ensure_ascii=True)[:max_chars]
            return {
                "paper_id": path.stem,
                "title": path.stem.replace("_", " "),
                "abstract": text,
                "source_url": str(path),
                "databases": ["local_filesystem"],
                "included": True,
            }

        if suffix == ".csv":
            with path.open("r", encoding="utf-8", errors="replace", newline="") as f:
                reader = csv.DictReader(f)
                rows = []
                for i, row in enumerate(reader):
                    if i >= 25:
                        break
                    rows.append(row)
            text = json.dumps(rows, ensure_ascii=True)[:max_chars]
            return {
                "paper_id": path.stem,
                "title": path.stem.replace("_", " "),
                "abstract": text,
                "source_url": str(path),
                "databases": ["local_filesystem"],
                "included": True,
            }

        if suffix == ".pdf":
            # Staged metadata representation. Full parsing is deferred to downstream document processing.
            return {
                "paper_id": path.stem,
                "title": path.stem.replace("_", " "),
                "abstract": f"PDF source staged for extraction: {path.name}",
                "source_url": str(path),
                "databases": ["local_filesystem"],
                "included": True,
            }
    except Exception as e:
        logger.error("Failed to read path %s: %s", path, e)
    return None


def normalize_record(record: Dict[str, Any]) -> Dict[str, Any]:
    """Enforce standard schema structure for all paper records in the harness."""
    return {
        "paper_id": str(record.get("paper_id") or record.get("id") or "unknown"),
        "title": str(record.get("title") or "Untitled Artifact"),
        "authors": list(record.get("authors") or []),
        "year": record.get("year"),
        "abstract": str(record.get("abstract") or record.get("text") or ""),
        "source_url": str(record.get("source_url") or record.get("source") or ""),
        "databases": list(record.get("databases") or ["local_filesystem"]),
        "full_text": record.get("full_text"),
        "annotations": record.get("annotations"),
        "quality_score": record.get("quality_score"),
        "included": bool(record.get("included", True)),
        "exclusion_reason": record.get("exclusion_reason"),
        "needs_human_review": bool(record.get("needs_human_review", False)),
    }


def ingest_local_sources(
    source_patterns: List[str],
    max_files: int = 120,
    max_chars: int = 12000,
) -> List[Dict[str, Any]]:
    """Scan globs and ingest staged local enterprise data files."""
    records: List[Dict[str, Any]] = []
    seen: set[str] = set()

    for pattern in source_patterns:
        for path in Path(".").glob(pattern):
            if len(records) >= max_files:
                break
            if not path.is_file() or str(path) in seen:
                continue
            seen.add(str(path))

            rec = read_single_path(path, max_chars=max_chars)
            if rec:
                records.append(rec)
    return records


async def dark_data_ingestion_node(
    state: ResearchState,
    config: Dict[str, Any] | None = None,
) -> Dict[str, Any]:
    """Ingest local artifacts and map them into canonical paper records."""
    config = config or {}
    cfgr = config.get("configurable", {})

    source_patterns = cfgr.get(
        "dark_data_sources",
        ["data/**/*", "inputs/**/*", "papers/**/*", "chunks/**/*"],
    )
    max_files = int(cfgr.get("dark_data_max_files", 120))

    raw_records = ingest_local_sources(source_patterns, max_files=max_files)
    records = [normalize_record(r) for r in raw_records]

    audit_log = append_audit(
        state,
        agent="dark_data_ingestion_node",
        action="ingest_local_dark_data",
        inputs={"source_patterns": source_patterns, "max_files": max_files},
        output_summary=f"Ingested {len(records)} local artifacts",
        provenance={"sources": source_patterns, "ingestion_mode": "local_only"},
    )

    return {
        "current_node": "dark_data_ingestion",
        "papers": records,
        "papers_found": len(records),
        "papers_screened": len(records),
        "papers_included": len([r for r in records if r.get("included", True)]),
        "databases_searched": ["local_filesystem"],
        "search_queries": [state.get("research_topic", "")],
        "grey_literature_searched": False,
        "audit_log": audit_log,
    }
