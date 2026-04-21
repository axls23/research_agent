"""
core/nodes/data_processing_node.py
===================================
LangGraph node that downloads PDFs, extracts text (Mistral AI / PyPDF2),
and chunks documents for downstream analysis.
"""

from __future__ import annotations

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
    dual_extraction_performed = bool(
        config.get("configurable", {}).get("dual_extraction_performed", False)
    )

    papers = list(state.get("papers", []))
    all_chunks: list = list(state.get("chunks", []))
    total_tokens = state.get("total_tokens_extracted", 0)
    extracted_count = 0

    for i, paper in enumerate(papers):
        if not paper.get("included", True):
            continue

        source_url = paper.get("source_url", "")
        paper_id = paper.get("paper_id", f"paper_{i}")
        abstract_text = (paper.get("abstract") or "").strip()

        existing_chunks = [c for c in all_chunks if c.get("paper_id") == paper_id]

        if paper.get("full_text"):
            # Full text may already exist (e.g., retry runs). Ensure it is chunked once.
            if existing_chunks:
                continue

            paper_chunks = chunk_text(
                text=paper["full_text"],
                paper_id=paper_id,
                chunk_size=1000,
                chunk_overlap=200,
            )
            all_chunks.extend(paper_chunks)
            total_tokens += sum(c.get("token_count", 0) for c in paper_chunks)
            extracted_count += 1
            logger.info(
                f"Chunked existing full_text for {paper_id}: {len(paper_chunks)} chunks"
            )
            continue

        if not source_url:
            if abstract_text:
                paper["full_text"] = abstract_text
                paper["annotations"] = {"method": "abstract_fallback"}
                paper_chunks = chunk_text(
                    text=abstract_text,
                    paper_id=paper_id,
                    chunk_size=600,
                    chunk_overlap=100,
                )
                all_chunks.extend(paper_chunks)
                total_tokens += sum(c.get("token_count", 0) for c in paper_chunks)
                extracted_count += 1
                logger.warning(
                    f"No source URL for {paper_id}; used abstract fallback ({len(paper_chunks)} chunks)"
                )
            else:
                logger.warning(f"No source URL for paper {paper_id}, skipping")
            continue

        # ---- Download ----
        pdf_path = await download_pdf(source_url)
        if not pdf_path:
            if abstract_text:
                paper["full_text"] = abstract_text
                paper["annotations"] = {"method": "abstract_fallback"}
                paper_chunks = chunk_text(
                    text=abstract_text,
                    paper_id=paper_id,
                    chunk_size=600,
                    chunk_overlap=100,
                )
                all_chunks.extend(paper_chunks)
                total_tokens += sum(c.get("token_count", 0) for c in paper_chunks)
                extracted_count += 1
                logger.warning(
                    f"Failed to download {paper_id}; used abstract fallback ({len(paper_chunks)} chunks)"
                )
            else:
                logger.warning(f"Failed to download {paper_id}")
            continue

        # ---- Extract ----
        extraction = await extract_paper_text(pdf_path, use_mistral=use_mistral)

        full_text = extraction.get("full_text", "")
        if not full_text:
            if abstract_text:
                paper["full_text"] = abstract_text
                paper["annotations"] = {"method": "abstract_fallback"}
                paper_chunks = chunk_text(
                    text=abstract_text,
                    paper_id=paper_id,
                    chunk_size=600,
                    chunk_overlap=100,
                )
                all_chunks.extend(paper_chunks)
                total_tokens += sum(c.get("token_count", 0) for c in paper_chunks)
                extracted_count += 1
                logger.warning(
                    f"Empty extraction for {paper_id}; used abstract fallback ({len(paper_chunks)} chunks)"
                )
            else:
                logger.warning(f"Empty extraction for {paper_id}")
            continue

        # Update paper record
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
            f"Extracted {extracted_count} papers, "
            f"produced {len(all_chunks)} chunks, "
            f"{total_tokens} total tokens"
        ),
    )

    return {
        "current_node": "data_processing",
        "papers": papers,
        "papers_screened": extracted_count,
        "chunks": all_chunks,
        "total_tokens_extracted": total_tokens,
        "dual_extraction_performed": dual_extraction_performed,
        "audit_log": audit_log,
    }
