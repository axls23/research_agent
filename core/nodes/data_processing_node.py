"""
core/nodes/data_processing_node.py
===================================
LangGraph node that downloads PDFs, extracts text (Mistral AI / PyPDF2),
and chunks documents for downstream analysis.
"""

from __future__ import annotations

import copy
import logging
from typing import Any, Dict

from core.state import ResearchState, append_audit, Chunk
from core.tools.extraction_tools import (
    download_pdf,
    extract_paper_text,
    chunk_text,
)

logger = logging.getLogger(__name__)


async def data_processing_node(
    state: ResearchState,
    config: Dict[str, Any] | None = None,
) -> Dict[str, Any]:
    """
    LangGraph node: Data Processing

    1. Download PDFs for included papers
    2. Extract text using Mistral Document AI (or PyPDF2 fallback)
    3. Chunk extracted text
    4. Update PRISMA screening counts
    5. Append audit entries
    """
    config = config or {}
    use_mistral = config.get("configurable", {}).get("use_mistral_ocr", True)

    papers = [copy.deepcopy(p) for p in state.get("papers", [])]
    all_chunks: list = list(state.get("chunks", []))
    total_tokens = state.get("total_tokens_extracted", 0)
    extracted_count = 0
    screened_count = 0

    for i, paper in enumerate(papers):
        if not paper.get("included", True):
            continue

        # Every included paper counts as screened (title/abstract reviewed)
        screened_count += 1

        if paper.get("full_text"):
            # Already extracted
            extracted_count += 1
            continue

        source_url = paper.get("source_url", "")
        paper_id = paper.get("paper_id", f"paper_{i}")

        if not source_url:
            logger.warning(f"No source URL for paper {paper_id}, skipping")
            continue

        # ---- Download ----
        pdf_path = await download_pdf(source_url)
        if not pdf_path:
            logger.warning(f"Failed to download {paper_id}")
            continue

        # ---- Extract ----
        extraction = await extract_paper_text(pdf_path, use_mistral=use_mistral)

        full_text = extraction.get("full_text", "")
        if not full_text:
            logger.warning(f"Empty extraction for {paper_id}")
            continue

        # Update paper record (working on a shallow copy)
        paper["full_text"] = full_text
        paper["annotations"] = extraction.get("annotations", {})
        extracted_count += 1

        # ---- Chunk ----
        paper_chunks = chunk_text(
            text=full_text,
            paper_id=paper_id,
            chunk_size=1000,
            chunk_overlap=200,
        )
        all_chunks.extend(paper_chunks)
        total_tokens += sum(c.get("token_count", 0) for c in paper_chunks)

        logger.info(
            f"Processed {paper_id}: "
            f"{len(full_text)} chars, {len(paper_chunks)} chunks "
            f"({extraction.get('method', 'unknown')})"
        )

    # ---- Update state ----
    audit_log = append_audit(
        state,
        agent="data_processing_node",
        action="extract_and_chunk",
        inputs={"paper_count": len(papers), "use_mistral": use_mistral},
        output_summary=(
            f"Screened {screened_count} papers, extracted {extracted_count}, "
            f"produced {len(all_chunks)} chunks, "
            f"{total_tokens} total tokens"
        ),
    )

    return {
        "current_node": "data_processing",
        "papers": papers,
        "papers_screened": screened_count,
        "papers_included": extracted_count,
        "chunks": all_chunks,
        "total_tokens_extracted": total_tokens,
        "audit_log": audit_log,
    }
