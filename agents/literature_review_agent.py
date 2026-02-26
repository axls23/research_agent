"""Literature Review Agent – discovers, retrieves, and filters academic
papers using a Hybrid AI + Snowballing strategy modelled on real
systematic review methodology (PRISMA, PICO, Title/Abstract screening).

Strategy (Option C: Hybrid AI + Snowballing):

1. **formulate_search_query** – Uses the LLM to decompose the research
   topic into PICO (Population, Intervention, Comparison, Outcome)
   concepts, then expands each with synonyms and builds boolean query
   strings.

2. **retrieve_papers** – Searches ArXiv, Semantic Scholar, and Crossref
   in parallel. Deduplicates by DOI/title. Optionally snowballs the top
   seed papers via Semantic Scholar's References & Citations endpoint.

3. **filter_papers** – Two-pass screening:
     a) Heuristic pre-filter (year range, language, metadata completeness)
     b) LLM-based Title/Abstract screen with explicit inclusion/exclusion
        and one-sentence reason (mirrors a human screener's spreadsheet).
"""

from __future__ import annotations

import asyncio
import json
import logging
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

import aiohttp
from pydantic import BaseModel, Field

from core.base_agent import ResearchAgent
from core.tools.search_tools import (
    PaperMeta,
    search_multiple_databases,
)


logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Pydantic schemas for structured LLM output
# ---------------------------------------------------------------------------


class PICODecomposition(BaseModel):
    """PICO decomposition of a research topic."""

    population: List[str] = Field(
        description="Target population / subject terms and synonyms"
    )
    intervention: List[str] = Field(
        description="Intervention / exposure terms and synonyms"
    )
    comparison: List[str] = Field(
        description="Comparison / control group terms (may be empty)"
    )
    outcome: List[str] = Field(description="Outcome / endpoint terms and synonyms")
    additional_terms: List[str] = Field(
        default_factory=list,
        description="Other relevant keywords, MeSH terms, or acronyms",
    )


class ScreeningDecision(BaseModel):
    """LLM screening verdict for a single paper."""

    paper_id: str
    decision: str = Field(description="INCLUDE or EXCLUDE")
    reason: str = Field(description="One-sentence justification")
    relevance_score: float = Field(
        description="0.0 (irrelevant) to 1.0 (highly relevant)"
    )
    confidence: float = Field(
        default=0.8,
        description="0.0-1.0: how confident the screener is in this decision",
    )


class BatchScreeningResult(BaseModel):
    """Batch of screening decisions."""

    decisions: List[ScreeningDecision]


# ---------------------------------------------------------------------------
# Agent
# ---------------------------------------------------------------------------


class LiteratureReviewAgent(ResearchAgent):
    """Agent responsible for literature search using a Hybrid AI + Snowballing
    strategy with PRISMA-compliant documentation.

    Supported actions (passed via *input_data["action"]*):
      - ``formulate_search_query`` – PICO decomposition → boolean expansion
      - ``retrieve_papers``        – multi-DB search + dedup + snowballing
      - ``filter_papers``          – heuristic pre-filter + LLM abstract screen
    """

    # Default inclusion/exclusion criteria
    DEFAULT_MIN_YEAR = 2015
    DEFAULT_MAX_SNOWBALL_SEEDS = 5
    DEFAULT_SNOWBALL_MAX_REFS = 20
    SCREENING_BATCH_SIZE = 8

    def __init__(self, llm=None):
        super().__init__(
            name="literature_review",
            description=(
                "Discovers, retrieves, and filters academic papers using "
                "PICO decomposition, multi-database search, citation "
                "snowballing, and LLM-based Title/Abstract screening"
            ),
            llm=llm,
        )

    def get_required_fields(self) -> List[str]:
        return ["action"]

    async def process(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        await super().process(input_data)
        action = input_data["action"]

        if action == "formulate_search_query":
            return await self._formulate_search_query(input_data)
        elif action == "retrieve_papers":
            return await self._retrieve_papers(input_data)
        elif action == "filter_papers":
            return await self._filter_papers(input_data)
        else:
            raise ValueError(f"Unknown action: {action}")

    # ------------------------------------------------------------------
    # 1. FORMULATE SEARCH QUERY (PICO + Boolean Expansion)
    # ------------------------------------------------------------------

    async def _formulate_search_query(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Decompose a research topic into PICO elements, expand with
        synonyms, and generate boolean search strings for multiple DBs.

        Input keys:
          - topic (str): the research question or topic
          - research_goals (list[str]): specific objectives
          - min_year (int, optional): earliest publication year

        Returns:
          - status, queries (list[str]), pico (dict), search_log (dict)
        """
        topic = data.get("topic", "")
        goals = data.get("research_goals", [])
        min_year = data.get("min_year", self.DEFAULT_MIN_YEAR)

        # ----- Step 1: PICO decomposition via LLM -----
        pico = await self._decompose_pico(topic, goals)

        # ----- Step 2: Build boolean queries -----
        queries = self._build_boolean_queries(pico, min_year)

        # ----- Step 3: Generate a plain fallback query -----
        fallback = f"{topic} {' '.join(goals)}"
        if fallback.strip() not in queries:
            queries.append(fallback.strip())

        search_log = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "pico": pico.model_dump() if isinstance(pico, BaseModel) else pico,
            "queries_generated": len(queries),
            "min_year": min_year,
        }

        self.logger.info(
            f"Formulated {len(queries)} boolean queries from PICO decomposition"
        )

        return {
            "status": "completed",
            "queries": queries,
            "pico": pico.model_dump() if isinstance(pico, BaseModel) else pico,
            "search_log": search_log,
        }

    async def _decompose_pico(self, topic: str, goals: List[str]) -> PICODecomposition:
        """Use the LLM to convert a research topic into PICO elements."""
        if not self.llm:
            # Fallback: naive split if no LLM is available
            self.logger.warning("No LLM available — using naive PICO fallback")
            return PICODecomposition(
                population=[topic.split()[0]] if topic else [""],
                intervention=goals[:1] if goals else [topic],
                comparison=[],
                outcome=goals[1:] if len(goals) > 1 else [],
                additional_terms=topic.split() if topic else [],
            )

        system_prompt = (
            "You are a systematic review methodologist. Decompose the "
            "research question into PICO elements. For each element, "
            "provide the primary term AND 2-4 synonyms or related "
            "keywords (including MeSH terms when applicable). "
            "If there is no clear comparison group, leave it empty."
        )
        prompt = (
            f"Research topic: {topic}\n"
            f"Research goals: {json.dumps(goals)}\n\n"
            "Decompose into PICO with synonyms for each element."
        )

        try:
            pico = await self.llm.generate_structured(
                prompt, PICODecomposition, system_prompt=system_prompt
            )
            return pico
        except Exception as e:
            self.logger.error(f"LLM PICO decomposition failed: {e}")
            return PICODecomposition(
                population=[topic],
                intervention=goals[:1] if goals else [topic],
                comparison=[],
                outcome=goals[1:] if len(goals) > 1 else [],
                additional_terms=[],
            )

    def _build_boolean_queries(
        self, pico: PICODecomposition, min_year: int
    ) -> List[str]:
        """Convert PICO elements into 2-3 boolean query strings."""
        queries: List[str] = []

        def _or_group(terms: List[str]) -> str:
            if not terms:
                return ""
            quoted = [f'"{t}"' if " " in t else t for t in terms]
            return f"({' OR '.join(quoted)})"

        # Query 1: Full PICO boolean
        parts = []
        if pico.population:
            parts.append(_or_group(pico.population))
        if pico.intervention:
            parts.append(_or_group(pico.intervention))
        if pico.outcome:
            parts.append(_or_group(pico.outcome))
        if parts:
            queries.append(" AND ".join(parts))

        # Query 2: Population + Intervention (broader)
        if pico.population and pico.intervention:
            queries.append(
                f"{_or_group(pico.population)} AND {_or_group(pico.intervention)}"
            )

        # Query 3: Include additional terms
        if pico.additional_terms:
            extra = _or_group(pico.additional_terms[:4])
            if pico.population:
                queries.append(f"{_or_group(pico.population)} AND {extra}")

        return queries

    # ------------------------------------------------------------------
    # 2. RETRIEVE PAPERS (Multi-DB + Dedup + Snowball)
    # ------------------------------------------------------------------

    async def _retrieve_papers(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Search multiple databases with each query, deduplicate, and
        optionally snowball the top-N seed papers via Semantic Scholar.

        Input keys:
          - queries (list[str]): search strings from formulate_search_query
          - databases (list[str], optional): DBs to search
          - max_results_per_db (int, optional): cap per DB per query
          - snowball (bool, optional): enable citation snowballing
          - snowball_seeds (int, optional): number of top papers to snowball

        Returns:
          - status, papers (list[dict]), databases_searched, retrieval_log
        """
        queries = data.get("queries", [])
        databases = data.get("databases", ["arxiv", "semantic_scholar", "crossref"])
        max_per_db = data.get("max_results_per_db", 30)
        do_snowball = data.get("snowball", True)
        max_seeds = data.get("snowball_seeds", self.DEFAULT_MAX_SNOWBALL_SEEDS)

        all_papers: List[PaperMeta] = []
        all_dbs: set = set()
        query_hit_counts: Dict[str, int] = {}

        # ----- Phase 1: Multi-query, multi-DB search -----
        for query in queries:
            papers, dbs_searched = await search_multiple_databases(
                query=query,
                databases=databases,
                max_results_per_db=max_per_db,
            )
            all_papers.extend(papers)
            all_dbs.update(dbs_searched)
            query_hit_counts[query] = len(papers)

        # ----- Phase 2: Deduplicate -----
        deduped = self._deduplicate_papers(all_papers)

        # ----- Phase 3: Snowballing (optional) -----
        snowball_papers: List[PaperMeta] = []
        snowball_log: Dict[str, Any] = {"enabled": do_snowball, "seeds": 0, "found": 0}

        if do_snowball and deduped:
            # Pick seeds: top papers by abstract length as proxy for quality
            seeds = sorted(deduped, key=lambda p: len(p.abstract), reverse=True)[
                :max_seeds
            ]
            snowball_log["seeds"] = len(seeds)

            for seed in seeds:
                refs = await self._snowball_paper(seed.paper_id)
                snowball_papers.extend(refs)

            snowball_log["found"] = len(snowball_papers)
            self.logger.info(
                f"Snowballing {len(seeds)} seeds yielded "
                f"{len(snowball_papers)} additional papers"
            )

            # Merge and re-deduplicate
            deduped = self._deduplicate_papers(deduped + snowball_papers)

        # Convert to dicts
        paper_dicts = [p.to_dict() for p in deduped]

        retrieval_log = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "queries": queries,
            "databases_searched": sorted(all_dbs),
            "query_hit_counts": query_hit_counts,
            "raw_total": len(all_papers),
            "after_dedup": len(deduped),
            "snowball": snowball_log,
        }

        self.logger.info(
            f"Retrieved {len(paper_dicts)} papers "
            f"(raw: {len(all_papers)}, snowball: {len(snowball_papers)}) "
            f"from {sorted(all_dbs)}"
        )

        return {
            "status": "completed",
            "papers": paper_dicts,
            "papers_found": len(paper_dicts),
            "databases_searched": sorted(all_dbs),
            "retrieval_log": retrieval_log,
        }

    def _deduplicate_papers(self, papers: List[PaperMeta]) -> List[PaperMeta]:
        """Deduplicate papers by DOI first, then by fuzzy title match."""
        seen_ids: set = set()
        seen_titles: set = set()
        deduped: List[PaperMeta] = []

        for p in papers:
            # Prefer DOI-based dedup if available
            doi = p.paper_id if p.paper_id.startswith("10.") else None
            if doi and doi in seen_ids:
                continue
            if doi:
                seen_ids.add(doi)

            # Fuzzy title dedup: lowercase, strip whitespace, first 80 chars
            title_key = p.title.lower().strip()[:80]
            if title_key in seen_titles:
                continue
            seen_titles.add(title_key)

            deduped.append(p)

        return deduped

    async def _snowball_paper(self, paper_id: str) -> List[PaperMeta]:
        """Fetch references and citations for a paper via Semantic Scholar."""
        papers: List[PaperMeta] = []
        base_url = "https://api.semanticscholar.org/graph/v1/paper"
        fields = "paperId,title,authors,year,abstract,url,externalIds"

        for direction in ["references", "citations"]:
            url = f"{base_url}/{paper_id}/{direction}"
            params = {
                "fields": fields,
                "limit": self.DEFAULT_SNOWBALL_MAX_REFS,
            }
            try:
                async with aiohttp.ClientSession() as session:
                    async with session.get(
                        url,
                        params=params,
                        timeout=aiohttp.ClientTimeout(total=15),
                    ) as resp:
                        if resp.status != 200:
                            self.logger.debug(
                                f"Snowball {direction} for {paper_id}: "
                                f"HTTP {resp.status}"
                            )
                            continue
                        data = await resp.json()

                for item in data.get("data", []):
                    cited = item.get("citedPaper") or item.get("citingPaper") or item
                    if not cited.get("title"):
                        continue
                    ext = cited.get("externalIds") or {}
                    papers.append(
                        PaperMeta(
                            paper_id=ext.get("DOI") or cited.get("paperId", ""),
                            title=cited.get("title", ""),
                            authors=[
                                a.get("name", "") for a in (cited.get("authors") or [])
                            ],
                            year=cited.get("year"),
                            abstract=cited.get("abstract") or "",
                            source_url=cited.get("url") or "",
                            database="semantic_scholar_snowball",
                        )
                    )
            except Exception as e:
                self.logger.error(f"Snowball {direction} failed: {e}")

        return papers

    # ------------------------------------------------------------------
    # 3. FILTER PAPERS (Heuristic + LLM Abstract Screen)
    # ------------------------------------------------------------------

    async def _filter_papers(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Two-pass Title/Abstract screening:
          1. Heuristic pre-filter (year, metadata completeness)
          2. LLM-based relevance screening with INCLUDE/EXCLUDE verdict

        Input keys:
          - papers (list[dict]): paper records from retrieve_papers
          - research_goals (list[str]): used for relevance judgement
          - topic (str): the research question
          - min_year (int, optional): earliest year to include
          - skip_llm_screen (bool, optional): skip the LLM pass

        Returns:
          - status, filtered_papers, screening_log
        """
        papers = data.get("papers", [])
        topic = data.get("topic", "")
        goals = data.get("research_goals", [])
        min_year = data.get("min_year", self.DEFAULT_MIN_YEAR)
        skip_llm = data.get("skip_llm_screen", False)

        if not papers:
            return {
                "status": "completed",
                "filtered_papers": [],
                "screening_log": {"total": 0, "after_heuristic": 0, "after_llm": 0},
            }

        # ----- Pass 1: Heuristic pre-filter -----
        heuristic_pass = []
        heuristic_excluded = []
        for p in papers:
            reason = self._heuristic_screen(p, min_year)
            if reason:
                p["included"] = False
                p["exclusion_reason"] = reason
                heuristic_excluded.append(p)
            else:
                heuristic_pass.append(p)

        self.logger.info(
            f"Heuristic screen: {len(papers)} → {len(heuristic_pass)} "
            f"({len(heuristic_excluded)} excluded)"
        )

        # ----- Pass 2: LLM-based Title/Abstract screening -----
        if skip_llm or not self.llm or not heuristic_pass:
            for p in heuristic_pass:
                p["included"] = True
            all_papers = heuristic_pass + heuristic_excluded
            return {
                "status": "completed",
                "filtered_papers": all_papers,
                "screening_log": {
                    "total": len(papers),
                    "after_heuristic": len(heuristic_pass),
                    "after_llm": len(heuristic_pass),
                    "llm_skipped": True,
                },
            }

        llm_included, llm_excluded = await self._llm_screen_batch(
            heuristic_pass, topic, goals
        )

        all_papers = llm_included + llm_excluded + heuristic_excluded

        screening_log = {
            "total": len(papers),
            "after_heuristic": len(heuristic_pass),
            "after_llm": len(llm_included),
            "heuristic_excluded": len(heuristic_excluded),
            "llm_excluded": len(llm_excluded),
            "llm_skipped": False,
            "needs_human_review": len(
                [p for p in llm_included + llm_excluded if p.get("needs_human_review")]
            ),
        }

        self.logger.info(
            f"LLM screen: {len(heuristic_pass)} → {len(llm_included)} included"
        )

        return {
            "status": "completed",
            "filtered_papers": all_papers,
            "screening_log": screening_log,
        }

    def _heuristic_screen(self, paper: Dict[str, Any], min_year: int) -> Optional[str]:
        """Apply fast heuristic exclusion criteria.
        Returns exclusion reason string, or None if paper passes."""

        # Year filter
        year = paper.get("year")
        if year is not None and year < min_year:
            return f"Publication year {year} is before minimum {min_year}"

        # Title must exist
        if not paper.get("title", "").strip():
            return "Missing title"

        # Abstract should have minimum length (very short = probably metadata-only)
        abstract = paper.get("abstract", "")
        if abstract and len(abstract.strip()) < 30:
            return "Abstract too short (likely metadata-only record)"

        return None

    async def _llm_screen_batch(
        self,
        papers: List[Dict[str, Any]],
        topic: str,
        goals: List[str],
    ) -> tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        """Screen papers in batches using the LLM.
        Returns (included_papers, excluded_papers)."""
        included = []
        excluded = []

        # Process in batches to respect context window limits
        for i in range(0, len(papers), self.SCREENING_BATCH_SIZE):
            batch = papers[i : i + self.SCREENING_BATCH_SIZE]
            decisions = await self._screen_batch(batch, topic, goals)

            for paper, decision in zip(batch, decisions):
                # Flag low-confidence decisions for human review
                if decision.confidence < 0.5:
                    paper["needs_human_review"] = True

                if decision.decision.upper() == "INCLUDE":
                    paper["included"] = True
                    paper["quality_score"] = decision.relevance_score
                    included.append(paper)
                else:
                    paper["included"] = False
                    paper["exclusion_reason"] = decision.reason
                    paper["quality_score"] = decision.relevance_score
                    excluded.append(paper)

        return included, excluded

    async def _screen_batch(
        self,
        batch: List[Dict[str, Any]],
        topic: str,
        goals: List[str],
    ) -> List[ScreeningDecision]:
        """Ask the LLM to screen a batch of papers."""
        system_prompt = (
            "You are a systematic review screener performing Title/Abstract "
            "screening. For each paper, decide INCLUDE or EXCLUDE based on "
            "relevance to the research topic and goals. Provide a one-sentence "
            "reason and a relevance score from 0.0 to 1.0.\n\n"
            "Inclusion criteria: The paper must directly address the research "
            "topic and contribute to at least one research goal.\n"
            "Exclusion criteria: Off-topic, wrong population, wrong intervention, "
            "commentary/editorial without data, duplicate study, non-English."
        )

        papers_text = ""
        for p in batch:
            papers_text += (
                f"\n---\n"
                f"ID: {p.get('paper_id', 'unknown')}\n"
                f"Title: {p.get('title', '')}\n"
                f"Year: {p.get('year', 'N/A')}\n"
                f"Abstract: {(p.get('abstract', '') or '')[:500]}\n"
            )

        prompt = (
            f"Research topic: {topic}\n"
            f"Research goals: {json.dumps(goals)}\n\n"
            f"Screen the following {len(batch)} papers:\n"
            f"{papers_text}\n\n"
            f"Return a JSON object with a 'decisions' array containing one "
            f"screening decision per paper (in order)."
        )

        try:
            result = await self.llm.generate_structured(
                prompt,
                BatchScreeningResult,
                system_prompt=system_prompt,
                temperature=0.1,
            )
            # Pad if LLM returned fewer decisions than papers
            decisions = result.decisions
            while len(decisions) < len(batch):
                decisions.append(
                    ScreeningDecision(
                        paper_id=batch[len(decisions)].get("paper_id", ""),
                        decision="INCLUDE",
                        reason="LLM did not return verdict — defaulting to include",
                        relevance_score=0.5,
                    )
                )
            return decisions[: len(batch)]

        except Exception as e:
            self.logger.error(f"LLM screening batch failed: {e}")
            # Fail-safe: include all papers if LLM is unavailable
            return [
                ScreeningDecision(
                    paper_id=p.get("paper_id", ""),
                    decision="INCLUDE",
                    reason="LLM screening unavailable — auto-included",
                    relevance_score=0.5,
                )
                for p in batch
            ]
