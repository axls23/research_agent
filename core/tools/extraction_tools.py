"""
core/tools/extraction_tools.py
===============================
Paper text extraction using Mistral Document AI for OCR/annotation,
with PyPDF2 as a lightweight fallback.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Mistral Document AI extraction (primary)
# ---------------------------------------------------------------------------


async def extract_with_mistral(
    pdf_path: str,
    mistral_api_key: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Extract text and annotations from a PDF using Mistral Document AI.

    Returns::

        {
            "full_text": str,
            "annotations": {...},
            "method": "mistral_document_ai",
        }

    Ref: https://docs.mistral.ai/capabilities/document_ai/annotations
    """
    from core.llm_provider import MistralProvider

    api_key = mistral_api_key or os.getenv("MISTRAL_API_KEY", "")
    if not api_key:
        logger.warning("MISTRAL_API_KEY not set, falling back to PyPDF2")
        return await extract_with_pypdf2(pdf_path)

    provider = MistralProvider(api_key=api_key)

    try:
        result = await provider.annotate_document(
            file_path=pdf_path,
            query=(
                "Extract the complete text of this document. "
                "For tables, preserve the structure as markdown tables. "
                "For equations, output LaTeX notation. "
                "For figures, describe their content and caption."
            ),
        )

        return {
            "full_text": result.get("full_text", ""),
            "annotations": result.get("annotations", {}),
            "method": "mistral_document_ai",
        }

    except Exception as e:
        logger.error(f"Mistral extraction failed: {e}, falling back to PyPDF2")
        return await extract_with_pypdf2(pdf_path)


# ---------------------------------------------------------------------------
# PyPDF2 fallback (lightweight, no API)
# ---------------------------------------------------------------------------


async def extract_with_pypdf2(pdf_path: str) -> Dict[str, Any]:
    """
    Extract text from PDF using PyPDF2 (no OCR).

    Good for born-digital PDFs. Will return empty text for scanned docs.
    """
    import PyPDF2

    text_pages: List[str] = []

    try:
        with open(pdf_path, "rb") as f:
            reader = PyPDF2.PdfReader(f)
            for page in reader.pages:
                text_pages.append(page.extract_text() or "")

        full_text = "\n".join(text_pages)
        logger.info(f"PyPDF2 extracted {len(full_text)} chars from {pdf_path}")

        return {
            "full_text": full_text,
            "annotations": {},
            "method": "pypdf2",
        }

    except Exception as e:
        logger.error(f"PyPDF2 extraction failed for {pdf_path}: {e}")
        return {
            "full_text": "",
            "annotations": {},
            "method": "pypdf2_error",
        }


# ---------------------------------------------------------------------------
# Smart extractor (auto-selects method)
# ---------------------------------------------------------------------------


async def extract_paper_text(
    pdf_path: str,
    use_mistral: bool = True,
) -> Dict[str, Any]:
    """
    Top-level extraction function:
    - Uses Mistral Document AI if ``use_mistral=True`` and API key is set
    - Falls back to PyPDF2 otherwise
    """
    if use_mistral and os.getenv("MISTRAL_API_KEY"):
        return await extract_with_mistral(pdf_path)
    return await extract_with_pypdf2(pdf_path)


# ---------------------------------------------------------------------------
# Chunking (tiktoken-aware)
# ---------------------------------------------------------------------------


def chunk_text(
    text: str,
    paper_id: str,
    chunk_size: int = 1000,
    chunk_overlap: int = 200,
) -> List[Dict[str, Any]]:
    """
    Split text into chunks using LangChain's RecursiveCharacterTextSplitter
    with tiktoken-based length function.

    Returns list of Chunk-compatible dicts.
    """
    from langchain_text_splitters import RecursiveCharacterTextSplitter

    try:
        import tiktoken

        tokenizer = tiktoken.get_encoding("cl100k_base")

        def length_fn(t: str) -> int:
            return len(tokenizer.encode(t, disallowed_special=()))

    except ImportError:
        logger.warning("tiktoken not available, using character count")
        length_fn = len

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=length_fn,
        separators=["\n\n", "\n", ". ", " ", ""],
    )

    raw_chunks = splitter.split_text(text)

    return [
        {
            "chunk_id": f"{paper_id}_chunk_{i}",
            "paper_id": paper_id,
            "text": chunk,
            "token_count": length_fn(chunk),
            "page_range": None,
        }
        for i, chunk in enumerate(raw_chunks)
    ]


# ---------------------------------------------------------------------------
# Download helper
# ---------------------------------------------------------------------------


async def download_pdf(url: str, output_dir: str = "papers") -> str:
    """Download a PDF from URL and return the local file path."""
    import aiohttp
    import re

    os.makedirs(output_dir, exist_ok=True)

    # Check if this is an ArXiv paper (by URL or DOI)
    # arxiv URLs look like: http://arxiv.org/abs/2102.12206 or https://arxiv.org/pdf/2102.12206.pdf
    # or DOI: 10.48550/arXiv.2102.12206
    arxiv_id_match = re.search(
        r"arxiv\.org/(?:abs|pdf)/(\d+\.\d+)", url.lower()
    ) or re.search(r"arxiv\.(\d+\.\d+)", url.lower())

    if arxiv_id_match:
        arxiv_id = arxiv_id_match.group(1)
        logger.info(
            f"Detected ArXiv paper ID: {arxiv_id}. Using arxiv package to fetch PDF."
        )
        output_path = os.path.join(output_dir, f"{arxiv_id}.pdf")

        if os.path.exists(output_path):
            logger.info(f"PDF already exists: {output_path}")
            return output_path

        try:
            import arxiv

            # Run the synchronous arxiv client in a thread pool so we don't block the async loop
            import asyncio

            loop = asyncio.get_running_loop()

            def _fetch_arxiv():
                client = arxiv.Client()
                search = arxiv.Search(id_list=[arxiv_id])
                results = list(client.results(search))
                if not results:
                    raise ValueError(f"No ArXiv paper found with ID {arxiv_id}")
                paper = results[0]
                # Download to the output_dir
                paper.download_pdf(dirpath=output_dir, filename=f"{arxiv_id}.pdf")
                return output_path

            return await loop.run_in_executor(None, _fetch_arxiv)

        except Exception as e:
            logger.error(f"ArXiv PDF download error for {arxiv_id}: {e}")
            # Fall through to generic grab if arxiv package fails for some reason

    # Generic download for non-ArXiv (or if ArXiv package failed)
    filename = url.split("/")[-1]
    if "?" in filename:
        filename = filename.split("?")[0]
    if not filename.endswith(".pdf"):
        filename += ".pdf"

    # sanitize filename
    filename = "".join(
        c for c in filename if c.isalnum() or c in ("-", "_", ".")
    ).rstrip()

    output_path = os.path.join(output_dir, filename)

    if os.path.exists(output_path):
        logger.info(f"PDF already exists: {output_path}")
        return output_path

    try:
        async with aiohttp.ClientSession() as session:
            # We add a common user agent to prevent basic 403s on landing pages
            headers = {
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
            }
            async with session.get(
                url, headers=headers, timeout=aiohttp.ClientTimeout(total=60)
            ) as resp:
                if resp.status == 200:
                    with open(output_path, "wb") as f:
                        f.write(await resp.read())
                    logger.info(f"Downloaded PDF to {output_path}")
                else:
                    logger.error(f"Failed to download {url}: HTTP {resp.status}")
                    return ""
    except Exception as e:
        logger.error(f"PDF download error: {e}")
        return ""

    return output_path
