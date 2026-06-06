"""
core/tools/search_tools.py
==========================
Paper search across multiple databases: ArXiv, Semantic Scholar, Crossref.
Uses standard academic APIs for discovery.
"""

from __future__ import annotations

import asyncio
import logging
import re
from typing import Any, Dict, List, Optional

import aiohttp
import time

logger = logging.getLogger(__name__)

_ARXIV_LOCK = asyncio.Lock()
_ARXIV_LAST_REQUEST = 0.0

_SEMANTIC_SCHOLAR_LOCK = asyncio.Lock()
_SEMANTIC_SCHOLAR_LAST_REQUEST = 0.0

async def _rate_limit_arxiv():
    global _ARXIV_LAST_REQUEST
    async with _ARXIV_LOCK:
        elapsed = time.time() - _ARXIV_LAST_REQUEST
        if elapsed < 3.0:
            await asyncio.sleep(3.0 - elapsed)
        _ARXIV_LAST_REQUEST = time.time()

async def _rate_limit_semantic_scholar():
    global _SEMANTIC_SCHOLAR_LAST_REQUEST
    async with _SEMANTIC_SCHOLAR_LOCK:
        elapsed = time.time() - _SEMANTIC_SCHOLAR_LAST_REQUEST
        if elapsed < 1.1:
            await asyncio.sleep(1.1 - elapsed)
        _SEMANTIC_SCHOLAR_LAST_REQUEST = time.time()


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
# ArXiv search (standard library integration)
# ---------------------------------------------------------------------------


async def search_arxiv(
    query: str,
    max_results: int = 50,
    max_retries: int = 4,
) -> List[PaperMeta]:
    """
    Search ArXiv for papers matching the query.

    Uses aiohttp and feedparser directly to adhere to the arXiv API policy,
    including a proper User-Agent and respecting rate limits.
    """
    import urllib.parse
    import feedparser

    papers: List[PaperMeta] = []
    
    # ArXiv API base URL
    base_url = "http://export.arxiv.org/api/query?"
    
    # Properly encode the search query
    encoded_query = urllib.parse.quote(query)
    
    fetch_count = min(max_results, 2000)
    
    url = f"{base_url}search_query=all:{encoded_query}&start=0&max_results={fetch_count}&sortBy=relevance&sortOrder=descending"

    headers = {
        "User-Agent": "ResearchAgent/1.0 (mailto:research@example.com)",
    }

    try:
        data = None
        async with aiohttp.ClientSession() as session:
            for attempt in range(max_retries):
                if attempt > 0:
                    # Enforce the 3-second delay requirement between requests/retries
                    await asyncio.sleep(3.0)

                await _rate_limit_arxiv()

                async with session.get(
                    url,
                    headers=headers,
                    timeout=aiohttp.ClientTimeout(total=30),
                ) as resp:
                    if resp.status in (429, 503) and attempt < max_retries - 1:
                        sleep_time = (2 ** attempt) + 3.0
                        logger.warning(f"ArXiv rate limited ({resp.status}). Retrying in {sleep_time}s...")
                        await asyncio.sleep(sleep_time)
                        continue
                    
                    if resp.status != 200:
                        logger.warning(f"ArXiv returned {resp.status}")
                        return papers

                    data = await resp.text()
                    break

        if not data:
            return papers

        # Parse the ATOM feed using feedparser
        feed = feedparser.parse(data)
        
        for entry in feed.entries:
            entry_id_full = entry.get("id", "")
            paper_id = re.sub(r"v\d+$", "", entry_id_full.split("/")[-1])
            
            title = entry.get("title", "").replace("\n", " ")
            authors = [author.get("name", "") for author in entry.get("authors", [])]
            
            year = None
            published_parsed = entry.get("published_parsed")
            if published_parsed:
                year = published_parsed.tm_year
                
            abstract = entry.get("summary", "").replace("\n", " ")
            
            pdf_url = ""
            for link in entry.get("links", []):
                if link.get("type") == "application/pdf":
                    pdf_url = link.get("href", "")
                    break
            
            source_url = pdf_url or entry_id_full

            papers.append(
                PaperMeta(
                    paper_id=paper_id,
                    title=title,
                    authors=authors,
                    year=year,
                    abstract=abstract,
                    source_url=source_url,
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
    max_retries: int = 6,
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
        "fields": "paperId,title,authors,year,abstract,url,externalIds,openAccessPdf,isOpenAccess",
    }

    try:
        data = None
        async with aiohttp.ClientSession() as session:
            for attempt in range(max_retries):
                await _rate_limit_semantic_scholar()
                
                async with session.get(
                    url, params=params, timeout=aiohttp.ClientTimeout(total=30)
                ) as resp:
                    if resp.status == 429 and attempt < max_retries - 1:
                        sleep_time = (2 ** attempt) * 2 + 1
                        logger.warning(f"Semantic Scholar rate limited (429). Retrying in {sleep_time}s...")
                        await asyncio.sleep(sleep_time)
                        continue
                    if resp.status != 200:
                        logger.warning(f"Semantic Scholar returned {resp.status}")
                        return papers

                    data = await resp.json()
                    break
        
        if not data:
            return papers

        for item in data.get("data", []):
            ext = item.get("externalIds") or {}
            oa_pdf = item.get("openAccessPdf") or {}
            source_url = oa_pdf.get("url") or item.get("url") or ""
            papers.append(
                PaperMeta(
                    paper_id=ext.get("DOI") or item.get("paperId", ""),
                    title=item.get("title", ""),
                    authors=[a.get("name", "") for a in (item.get("authors") or [])],
                    year=item.get("year"),
                    abstract=item.get("abstract") or "",
                    source_url=source_url,
                    database="semantic_scholar",
                )
            )
        logger.info(f"Semantic Scholar search for '{query}': {len(papers)} results")

    except Exception as e:
        logger.error(f"Semantic Scholar search failed: {e}")

    return papers


# ---------------------------------------------------------------------------
# Crossref search
# ---------------------------------------------------------------------------


async def search_crossref(
    query: str,
    max_results: int = 50,
    max_retries: int = 4,
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
        "select": "DOI,title,author,published-print,published-online,abstract,URL,link",
    }
    headers = {
        "User-Agent": "ResearchAgent/1.0 (mailto:research@example.com)",
    }

    try:
        data = None
        async with aiohttp.ClientSession() as session:
            for attempt in range(max_retries):
                async with session.get(
                    url,
                    params=params,
                    headers=headers,
                    timeout=aiohttp.ClientTimeout(total=30),
                ) as resp:
                    if resp.status == 429 and attempt < max_retries - 1:
                        sleep_time = 2 ** attempt
                        logger.warning(f"Crossref rate limited (429). Retrying in {sleep_time}s...")
                        await asyncio.sleep(sleep_time)
                        continue
                    if resp.status != 200:
                        logger.warning(f"Crossref returned {resp.status}")
                        return papers

                    data = await resp.json()
                    break

        if not data:
            return papers

        for item in data.get("message", {}).get("items", []):
            title_list = item.get("title", [])
            title = title_list[0] if title_list else ""

            pub = item.get("published-print") or item.get("published-online") or {}
            year = None
            if pub.get("date-parts"):
                year = pub["date-parts"][0][0] if pub["date-parts"][0] else None

            links = item.get("link") or []
            pdf_link = ""
            for link in links:
                link_url = link.get("URL", "")
                content_type = (link.get("content-type") or "").lower()
                if "pdf" in content_type or link_url.lower().endswith(".pdf"):
                    pdf_link = link_url
                    break

            source_url = pdf_link or item.get("URL") or ""

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
                    source_url=source_url,
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
