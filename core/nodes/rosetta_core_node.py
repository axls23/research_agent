"""core/nodes/rosetta_core_node.py
=================================
Rosetta Core node: translate silo-specific jargon into shared principles.
"""

from __future__ import annotations

import json
import logging
import re
import uuid
from collections import Counter
from typing import Any, Dict, List

from core.state import ResearchState, append_audit

logger = logging.getLogger(__name__)


def extract_candidate_terms(text: str) -> List[str]:
    """Tokenize and filter candidate jargon terms by removing common stop words."""
    words = re.findall(r"[A-Za-z][A-Za-z0-9_-]{3,}", text)
    lowered = [w.lower() for w in words]
    stop = {
        "this",
        "that",
        "with",
        "from",
        "into",
        "their",
        "there",
        "where",
        "which",
        "using",
        "model",
        "data",
        "analysis",
        "method",
        "result",
    }
    return [w for w in lowered if w not in stop]


def map_term_to_principle(term: str) -> str:
    """Evaluate candidate terms against local system mapping rules to resolve core engineering principles."""
    rules = {
        "sensor": "signal_acquisition",
        "telemetry": "signal_acquisition",
        "latency": "feedback_latency_management",
        "throughput": "flow_optimization",
        "pipeline": "flow_optimization",
        "fusion": "multi_source_synthesis",
        "genomic": "sequence_pattern_inference",
        "cad": "structural_representation",
        "simulation": "predictive_state_modeling",
        "control": "closed_loop_stabilization",
        "fault": "anomaly_detection_and_resilience",
    }
    for key, principle in rules.items():
        if key in term:
            return principle
    return "generalized_system_principle"


def translate_records_jargon(records: List[Dict[str, Any]], top_n: int = 30) -> List[Dict[str, Any]]:
    """Scan staged records and extract top-N jargon-to-principle mapping dictionaries."""
    tokens = []
    for r in records:
        text = str(r.get("abstract") or r.get("text") or "")
        tokens.extend(extract_candidate_terms(text))

    counts = Counter(tokens)
    mappings = []
    for term, freq in counts.most_common(top_n):
        mappings.append(
            {
                "term": term,
                "core_principle": map_term_to_principle(term),
                "frequency": freq,
            }
        )
    return mappings


_LLM_TRANSLATION_SYSTEM = (
    "You are the NEXUS Rosetta Core: a jargon-translation engine that maps "
    "domain-specific terminology onto domain-agnostic core principles so that "
    "structurally similar work in different fields can be linked (e.g. "
    "biology's 'angiogenesis' → 'decentralized_resource_calling'). "
    "For each input term, produce a short snake_case principle name that "
    "captures the underlying structural/mathematical mechanism, never the "
    "surface domain. Respond ONLY with a JSON object mapping each term to "
    "its principle string."
)


async def translate_terms_with_llm(
    llm: Any,
    mappings: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """Refine heuristic term→principle mappings with the deep-tier LLM.

    Falls back to the heuristic principles on any parse or transport error, so
    the node stays deterministic when no model is reachable.
    """
    terms = [m["term"] for m in mappings]
    if not terms:
        return mappings

    prompt = (
        "Translate these domain terms into core principles:\n"
        + "\n".join(f"- {t}" for t in terms)
    )
    try:
        raw = await llm.generate(
            prompt, system_prompt=_LLM_TRANSLATION_SYSTEM, temperature=0.2
        )
        first, last = raw.find("{"), raw.rfind("}")
        translated = json.loads(raw[first : last + 1]) if first != -1 and last != -1 else {}
    except Exception as e:
        logger.warning("Rosetta LLM translation failed (%s); using heuristic mapping", e)
        return mappings

    refined = []
    for m in mappings:
        principle = translated.get(m["term"])
        if isinstance(principle, str) and principle.strip():
            principle = re.sub(r"[^a-z0-9_]+", "_", principle.strip().lower()).strip("_")
        refined.append(
            {
                **m,
                "core_principle": principle or m["core_principle"],
                "translation_source": "llm" if principle else "heuristic",
            }
        )
    return refined


async def rosetta_core_node(
    state: ResearchState,
    config: Dict[str, Any] | None = None,
) -> Dict[str, Any]:
    """Create hyperedges and lightweight entities from translated jargon."""
    config = config or {}
    cfgr = config.get("configurable", {})

    records = list(state.get("papers", []))
    top_n = int(cfgr.get("rosetta_top_terms", 30))
    llm = cfgr.get("llm_deep") or cfgr.get("llm")  # prefer deep tier for abstraction

    mappings = translate_records_jargon(records, top_n=top_n)
    translation_mode = "local_heuristic"
    if llm and mappings:
        mappings = await translate_terms_with_llm(llm, mappings)
        if any(m.get("translation_source") == "llm" for m in mappings):
            translation_mode = "llm_with_heuristic_fallback"

    entities: List[Dict[str, Any]] = []
    hyperedges: List[Dict[str, Any]] = []

    for m in mappings:
        term = m.get("term", "")
        principle = m.get("core_principle", "generalized_system_principle")
        frequency = int(m.get("frequency", 1))

        entity_id = f"rc_{uuid.uuid4().hex[:10]}"
        entities.append(
            {
                "entity_id": entity_id,
                "label": "core_principle",
                "text": principle,
                "paper_ids": [r.get("paper_id", "") for r in records if r.get("paper_id")],
                "prisma_properties": {
                    "source_term": term,
                    "frequency": frequency,
                    "translation_layer": "rosetta_core",
                },
            }
        )

        hyperedges.append(
            {
                "hyperedge_id": f"he_{uuid.uuid4().hex[:10]}",
                "principle_name": principle,
                "member_entity_ids": [entity_id],
                "domain_jargon": [term],
                "weight": min(1.0, 0.25 + (frequency / 20.0)),
                "paper_ids": [r.get("paper_id", "") for r in records if r.get("paper_id")],
                "checklist_tags": ["prisma.item.core_principle_hyperedge"],
                "source_entity_labels": ["core_principle"],
            }
        )

    audit_log = append_audit(
        state,
        agent="rosetta_core_node",
        action="translate_jargon_to_core_principles",
        inputs={"record_count": len(records), "mapping_count": len(mappings)},
        output_summary=f"Mapped {len(mappings)} jargon terms into {len(hyperedges)} principles",
        provenance={"translation_mode": translation_mode},
    )

    return {
        "current_node": "rosetta_core",
        "knowledge_entities": [*state.get("knowledge_entities", []), *entities],
        "hyperedges": [*state.get("hyperedges", []), *hyperedges],
        "audit_log": audit_log,
    }
