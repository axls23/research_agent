"""
core/tools/search_tools.py
==========================
Paper search across multiple databases: ArXiv, Semantic Scholar, Crossref.
Wraps existing paperconstructor.Arxiv and adds new API integrations.
"""

from __future__ import annotations

import asyncio
import logging
import re
from typing import Any, Dict, List, Optional

import aiohttp

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data model (maps to state.PaperRecord)
# ---------------------------------------------------------------------------

class PaperMeta:
    """Lightweight paper metadata returned by search functions."""

    def __init__(
        self,
        paper_id: str,
        title: str,
        authors: List[str],
        year: Optional[int],
        abstract: str,
        source_url: str,
        database: str,
    ):
        self.paper_id = paper_id
        self.title = title
        self.authors = authors
        self.year = year
        self.abstract = abstract
        self.source_url = source_url
        self.database = database

    def to_dict(self) -> Dict[str, Any]:
        return {
            "paper_id": self.paper_id,
            "title": self.title,
            "authors": self.authors,
            "year": self.year,
            "abstract": self.abstract,
            "source_url": self.source_url,
            "databases": [self.database],
            "full_text": None,
            "annotations": None,
            "quality_score": None,
            "included": True,
            "exclusion_reason": None,
        }


# ---------------------------------------------------------------------------
# ArXiv search (wraps existing paperconstructor + arxiv library)
# ---------------------------------------------------------------------------

async def search_arxiv(
    query: str,
    max_results: int = 50,
) -> List[PaperMeta]:
    """
    Search ArXiv for papers matching the query.

    Uses the ``arxiv`` Python library for structured API access.
    """
    import arxiv

    papers: List[PaperMeta] = []

    try:
        search = arxiv.Search(
            query=query,
            max_results=max_results,
            sort_by=arxiv.SortCriterion.Relevance,
        )

        # arxiv library is synchronous — run in executor
        loop = asyncio.get_event_loop()
        results = await loop.run_in_executor(None, lambda: list(search.results()))

        for r in results:
            paper_id = re.sub(r"v\d+$", "", r.entry_id.split("/")[-1])
            papers.append(
                PaperMeta(
                    paper_id=paper_id,
                    title=r.title,
                    authors=[a.name for a in r.authors],
                    year=r.published.year if r.published else None,
                    abstract=r.summary,
                    source_url=r.pdf_url or r.entry_id,
                    database="arxiv",
                )
            )
        logger.info(f"ArXiv search for '{query}': {len(papers)} results")

    except Exception as e:
        logger.error(f"ArXiv search failed: {e}")

    return papers


# ---------------------------------------------------------------------------
# Semantic Scholar search
# ---------------------------------------------------------------------------

async def search_semantic_scholar(
    query: str,
    max_results: int = 50,
) -> List[PaperMeta]:
    """
    Search Semantic Scholar API (free, no key required for basic tier).

    Endpoint: https://api.semanticscholar.org/graph/v1/paper/search
    """
    papers: List[PaperMeta] = []
    url = "https://api.semanticscholar.org/graph/v1/paper/search"
    params = {
        "query": query,
        "limit": min(max_results, 100),
        "fields": "paperId,title,authors,year,abstract,url,externalIds",
    }

    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(url, params=params, timeout=aiohttp.ClientTimeout(total=30)) as resp:
                if resp.status != 200:
                    logger.warning(f"Semantic Scholar returned {resp.status}")
                    return papers

                data = await resp.json()

        for item in data.get("data", []):
            ext = item.get("externalIds") or {}
            papers.append(
                PaperMeta(
                    paper_id=ext.get("DOI") or item.get("paperId", ""),
                    title=item.get("title", ""),
                    authors=[
                        a.get("name", "") for a in (item.get("authors") or [])
                    ],
                    year=item.get("year"),
                    abstract=item.get("abstract") or "",
                    source_url=item.get("url") or "",
                    database="semantic_scholar",
                )
            )
        logger.info(
            f"Semantic Scholar search for '{query}': {len(papers)} results"
        )

    except Exception as e:
        logger.error(f"Semantic Scholar search failed: {e}")

    return papers


# ---------------------------------------------------------------------------
# Crossref search
# ---------------------------------------------------------------------------

async def search_crossref(
    query: str,
    max_results: int = 50,
) -> List[PaperMeta]:
    """
    Search Crossref for DOI-registered works.

    Endpoint: https://api.crossref.org/works
    """
    papers: List[PaperMeta] = []
    url = "https://api.crossref.org/works"
    params = {
        "query": query,
        "rows": min(max_results, 100),
        "select": "DOI,title,author,published-print,abstract,URL",
    }
    headers = {
        "User-Agent": "ResearchAgent/1.0 (mailto:research@example.com)",
    }

    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(
                url,
                params=params,
                headers=headers,
                timeout=aiohttp.ClientTimeout(total=30),
            ) as resp:
                if resp.status != 200:
                    logger.warning(f"Crossref returned {resp.status}")
                    return papers

                data = await resp.json()

        for item in data.get("message", {}).get("items", []):
            title_list = item.get("title", [])
            title = title_list[0] if title_list else ""

            pub = item.get("published-print") or item.get("published-online") or {}
            year = None
            if pub.get("date-parts"):
                year = pub["date-parts"][0][0] if pub["date-parts"][0] else None

            papers.append(
                PaperMeta(
                    paper_id=item.get("DOI", ""),
                    title=title,
                    authors=[
                        f"{a.get('given', '')} {a.get('family', '')}".strip()
                        for a in (item.get("author") or [])
                    ],
                    year=year,
                    abstract=item.get("abstract") or "",
                    source_url=item.get("URL") or "",
                    database="crossref",
                )
            )
        logger.info(f"Crossref search for '{query}': {len(papers)} results")

    except Exception as e:
        logger.error(f"Crossref search failed: {e}")

    return papers


# ---------------------------------------------------------------------------
# Multi-database search (orchestrates all)
# ---------------------------------------------------------------------------

async def search_multiple_databases(
    query: str,
    databases: Optional[List[str]] = None,
    max_results_per_db: int = 50,
) -> tuple[List[PaperMeta], List[str]]:
    """
    Search across multiple databases concurrently.

    Returns ``(papers, databases_searched)`` — the list of databases
    that were actually queried (for PRISMA audit).
    """
    if databases is None:
        databases = ["arxiv", "semantic_scholar", "crossref"]

    db_map = {
        "arxiv": search_arxiv,
        "semantic_scholar": search_semantic_scholar,
        "crossref": search_crossref,
    }

    tasks = []
    searched = []
    for db_name in databases:
        if db_name in db_map:
            tasks.append(db_map[db_name](query, max_results=max_results_per_db))
            searched.append(db_name)
        else:
            logger.warning(f"Unknown database: {db_name}")

    results = await asyncio.gather(*tasks, return_exceptions=True)

    all_papers: List[PaperMeta] = []
    for r in results:
        if isinstance(r, Exception):
            logger.error(f"Database search error: {r}")
        else:
            all_papers.extend(r)

    # Deduplicate by title (fuzzy)
    seen_titles: set[str] = set()
    deduped: List[PaperMeta] = []
    for p in all_papers:
        key = p.title.lower().strip()[:80]
        if key not in seen_titles:
            seen_titles.add(key)
            deduped.append(p)

    logger.info(
        f"Multi-DB search: {len(all_papers)} raw → {len(deduped)} after dedup "
        f"across {searched}"
    )
    return deduped, searched
