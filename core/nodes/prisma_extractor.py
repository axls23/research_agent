"""
core/nodes/prisma_extractor.py
==============================
Two-tier PRISMA 2020 entity extraction pipeline.

Tier 1: GLiNER zero-shot NER — grounded entity span detection (local CPU).
Tier 2: LLM + Pydantic schema — structured PRISMA property extraction (Groq API).

Neither tier uses regex. GLiNER runs a bidirectional transformer for span
detection; the LLM receives GLiNER spans as grounded context and produces
validated Pydantic output.
"""

from __future__ import annotations

import difflib
import logging
import uuid
from typing import Any, Dict, List, Optional, Tuple

from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Grounding Verification
# ---------------------------------------------------------------------------

def _verify_grounding(entity_text: str, source_text: str, threshold: float = 0.6) -> bool:
    """
    Verify if an extracted entity text actually appears in the source text.
    Returns True if a fuzzy match above the threshold is found.
    """
    if not entity_text or not source_text:
        return False
        
    entity_text_lower = entity_text.lower()
    source_lower = source_text.lower()
    
    # Fast paths
    if entity_text_lower in source_lower:
        return True
    
    # If entity is very short, exact match is required
    if len(entity_text) < 10:
        return False
        
    # Fuzzy match using SequenceMatcher for longer phrases (e.g. LLM variations/omissions)
    matcher = difflib.SequenceMatcher(None, entity_text_lower, source_lower)
    
    # We don't need a full match, just a substantial overlapping block
    match = matcher.find_longest_match(0, len(entity_text_lower), 0, len(source_lower))
    match_ratio = match.size / len(entity_text_lower)
    
    return match_ratio >= threshold


# ---------------------------------------------------------------------------
# Pydantic schemas — PRISMA 2020 structured output
# ---------------------------------------------------------------------------


class PICOExtraction(BaseModel):
    """Population, Intervention, Comparator, Outcome framework."""

    population: str = Field(default="", description="Target population or participants")
    intervention: str = Field(default="", description="Intervention being studied")
    comparator: str = Field(default="", description="Comparison group or control")
    outcome: str = Field(default="", description="Primary outcome measure")


class PRISMAObjective(BaseModel):
    """PRISMA Rule Set A: Study rationale and objectives."""

    text: str = Field(
        description="Explicit statement of the main objective or question"
    )
    pico: Optional[PICOExtraction] = Field(
        default=None, description="PICO elements if identifiable"
    )


class PRISMAMethodology(BaseModel):
    """PRISMA Rule Set B: Eligibility and methodology."""

    text: str = Field(description="Summary of the methodological approach")
    inclusion_criteria: List[str] = Field(
        default_factory=list, description="Study characteristics for inclusion"
    )
    exclusion_criteria: List[str] = Field(
        default_factory=list, description="Study characteristics for exclusion"
    )
    information_sources: List[str] = Field(
        default_factory=list, description="Databases, registers, websites searched"
    )
    risk_of_bias_tool: Optional[str] = Field(
        default=None, description="Tool/version used for risk of bias assessment"
    )
    effect_measures: List[str] = Field(
        default_factory=list,
        description="Effect measures used (risk ratios, mean diffs, etc.)",
    )
    synthesis_model: Optional[str] = Field(
        default=None, description="Meta-analysis model (fixed-effect, random-effects)"
    )


class PRISMAResult(BaseModel):
    """PRISMA Rule Set C: Results and syntheses."""

    text: str = Field(description="Summary of findings")
    study_flow: Optional[Dict[str, int]] = Field(
        default=None,
        description="Flow counts: identified, screened, excluded, included",
    )
    effect_estimates: List[str] = Field(
        default_factory=list,
        description="Effect estimates with precision (e.g., OR=2.3, 95% CI 1.1-4.5)",
    )
    certainty_assessment: Optional[str] = Field(
        default=None, description="Overall certainty/confidence in evidence body"
    )


class PRISMALimitation(BaseModel):
    """PRISMA Rule Set D: Limitations."""

    text: str = Field(description="Stated limitation of the evidence or review")


class PRISMAImplication(BaseModel):
    """PRISMA Rule Set D: Implications."""

    text: str = Field(
        description="Implication for practice, policy, or future research"
    )


class PRISMATransparency(BaseModel):
    """PRISMA Rule Set D: Transparency data."""

    funding_sources: List[str] = Field(
        default_factory=list, description="Financial or non-financial support sources"
    )
    data_availability: Optional[str] = Field(
        default=None, description="Availability of data, code, materials"
    )


class CorePrinciple(BaseModel):
    """NEXUS Isomorphic Abstraction."""

    domain_jargon: str = Field(description="The original domain-specific term (e.g., 'Angiogenesis')")
    abstract_principle: str = Field(description="The domain-agnostic mathematical or structural principle (e.g., 'Decentralized Resource Calling')")
    explanation: str = Field(description="Why this jargon maps to this principle")


class HyperedgeContext(BaseModel):
    """Groups multiple nodes (Objective, Method, Result, etc.) under one unified structural principle."""

    principle_name: str = Field(description="The unifying abstract principle (must match a CorePrinciple's abstract_principle)")
    involved_entity_texts: List[str] = Field(description="List of EXACT text spans from extracted objectives/methodologies/etc. that participate in this principle")
    hyperedge_weight: float = Field(default=1.0, description="Confidence of this n-ary relationship (0.0 to 1.0)")


class PRISMAExtraction(BaseModel):
    """
    Full PRISMA 2020-aligned extraction from a paper chunk + NEXUS Extensions.

    This is the schema passed to LLM generate_structured().
    GLiNER entity spans are injected into the prompt as grounded context.
    """

    objectives: List[PRISMAObjective] = Field(default_factory=list)
    methodology: List[PRISMAMethodology] = Field(default_factory=list)
    results: List[PRISMAResult] = Field(default_factory=list)
    limitations: List[PRISMALimitation] = Field(default_factory=list)
    implications: List[PRISMAImplication] = Field(default_factory=list)
    transparency: Optional[PRISMATransparency] = Field(default=None)
    core_principles: List[CorePrinciple] = Field(default_factory=list)
    hyperedges: List[HyperedgeContext] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# PRISMA entity/relationship type constants
# ---------------------------------------------------------------------------

PRISMA_ENTITY_TYPES = {
    "paper",
    "objective",
    "methodology",
    "result",
    "limitation",
    "implication",
    "core_principle",
}

PRISMA_RELATION_MAP = {
    # (source_label, target_label) -> relationship_type
    ("paper", "objective"): "INVESTIGATES",
    ("paper", "methodology"): "UTILIZES_METHOD",
    ("paper", "result"): "REPORTS_FINDING",
    ("paper", "limitation"): "HAS_LIMITATION",
    ("result", "implication"): "SUPPORTS_IMPLICATION",
    ("paper", "core_principle"): "EXEMPLIFIES_PRINCIPLE",
}

# GLiNER label vocabulary — maps to PRISMA entity types
GLINER_LABELS = [
    "research objective",
    "population",
    "intervention",
    "comparator",
    "outcome",
    "inclusion criteria",
    "exclusion criteria",
    "information source",
    "risk of bias tool",
    "effect measure",
    "synthesis model",
    "study flow statistic",
    "effect estimate",
    "certainty assessment",
    "limitation",
    "implication",
    "funding source",
    "data availability",
]

# Map GLiNER labels → PRISMA entity types
GLINER_TO_PRISMA = {
    "research objective": "objective",
    "population": "objective",
    "intervention": "objective",
    "comparator": "objective",
    "outcome": "objective",
    "inclusion criteria": "methodology",
    "exclusion criteria": "methodology",
    "information source": "methodology",
    "risk of bias tool": "methodology",
    "effect measure": "methodology",
    "synthesis model": "methodology",
    "study flow statistic": "result",
    "effect estimate": "result",
    "certainty assessment": "result",
    "limitation": "limitation",
    "implication": "implication",
    "funding source": "paper",
    "data availability": "paper",
}


# ---------------------------------------------------------------------------
# Tier 1: GLiNER — zero-shot grounded entity extraction
# ---------------------------------------------------------------------------

_gliner_model = None  # Lazy-loaded singleton


def _get_gliner_model():
    """Lazy-load GLiNER model (downloads ~400MB on first run)."""
    global _gliner_model
    if _gliner_model is not None:
        return _gliner_model

    try:
        from gliner import GLiNER

        logger.info("Loading GLiNER model (urchade/gliner_mediumv2.1)...")
        _gliner_model = GLiNER.from_pretrained("urchade/gliner_mediumv2.1")
        logger.info("GLiNER model loaded successfully")
        return _gliner_model
    except ImportError:
        logger.warning("gliner package not installed. Install with: pip install gliner")
        return None
    except Exception as e:
        logger.error(f"Failed to load GLiNER model: {e}")
        return None


def extract_entities_gliner(
    text: str,
    paper_id: str,
    threshold: float = 0.4,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, str]]]:
    """
    Extract PRISMA-aligned entities from text using GLiNER zero-shot NER.

    Returns:
        entities: List of entity dicts with PRISMA labels & properties
        raw_spans: List of raw GLiNER span dicts for LLM grounding context
    """
    model = _get_gliner_model()
    if model is None:
        logger.warning("GLiNER unavailable, returning empty extraction")
        return [], []

    try:
        raw_predictions = model.predict_entities(
            text, GLINER_LABELS, threshold=threshold
        )
    except Exception as e:
        logger.error(f"GLiNER prediction failed: {e}")
        return [], []

    entities: List[Dict[str, Any]] = []
    raw_spans: List[Dict[str, str]] = []
    seen_texts = set()

    for pred in raw_predictions:
        span_text = pred.get("text", "").strip()
        gliner_label = pred.get("label", "")
        score = pred.get("score", 0.0)

        if not span_text or len(span_text) < 3:
            continue

        # Deduplicate by text
        if span_text.lower() in seen_texts:
            continue
        seen_texts.add(span_text.lower())

        prisma_label = GLINER_TO_PRISMA.get(gliner_label, "objective")

        entity = {
            "entity_id": str(uuid.uuid4())[:8],
            "label": prisma_label,
            "text": span_text,
            "paper_ids": [paper_id],
            "prisma_properties": {
                "gliner_label": gliner_label,
                "gliner_score": round(score, 3),
                "extraction_tier": "gliner",
            },
        }
        entities.append(entity)

        raw_spans.append(
            {
                "text": span_text,
                "label": gliner_label,
                "score": str(round(score, 3)),
            }
        )

    logger.info(
        f"GLiNER extracted {len(entities)} entities from chunk "
        f"(paper_id={paper_id})"
    )
    return entities, raw_spans


# ---------------------------------------------------------------------------
# Tier 2: LLM + Pydantic — structured PRISMA extraction
# ---------------------------------------------------------------------------

PRISMA_EXTRACTION_SYSTEM_PROMPT = """You are a systematic review data extractor following PRISMA 2020 guidelines.
You receive a text chunk from an academic paper along with pre-identified entity spans.

Extract PRISMA-aligned structured data following these rules:

RULE SET A — Objectives:
- Extract explicit statements of the main objective or research question
- Identify PICO elements (Population, Intervention, Comparator, Outcome) when present

RULE SET B — Methodology:
- Extract inclusion/exclusion criteria for study eligibility
- List information sources (databases, registers, websites)
- Identify risk of bias assessment tools and versions
- Extract effect measures (risk ratios, mean differences, etc.)
- Identify synthesis models (fixed-effect, random-effects)

RULE SET C — Results:
- Extract study flow data (identified, screened, excluded, included counts)
- Extract effect estimates with precision (confidence intervals)
- Capture certainty/confidence assessments for evidence body

RULE SET D — Discussion & Transparency:
- Extract stated limitations of the evidence or review
- Extract implications for practice, policy, or future research
- Identify funding sources and data availability statements

RULE SET E — NEXUS Isomorphic Abstraction & Hyperedges:
- Translate domain-specific jargon into universal "**Core Principles**" (abstract structural/mathematical concepts).
- Group extracted texts (Objectives, Methods, Results, Limitations) that share the same principle into **Hyperedges**. Do not just use 1-to-1 relations, group the exact text spans that manifest this pattern together.

Only extract what is explicitly stated in the text. Do not infer or fabricate data.
If a category has no relevant content in this chunk, return an empty list for that field."""


async def extract_prisma_structured(
    text: str,
    paper_id: str,
    gliner_spans: List[Dict[str, Any]],
    llm: Any,
    max_prompt_chars: int = 8000,
    strict_grounding: bool = False,
    temperature: float = 0.2,
) -> Tuple[List[Dict[str, Any]], List[Tuple[str, str, str]], List[Dict[str, Any]]]:
    """
    Use LLM with Pydantic schema to extract structured PRISMA data.

    GLiNER spans are injected as grounded context to improve precision.

    Returns:
        entities: List of PRISMA entity dicts
        relations: List of (source_text, relation_type, target_text) triples
        hyperedges: List of extracted hyperedge dictionaries
    """
    if not llm:
        return [], [], []

    # Format GLiNER spans as grounded context, ensuring they fall within our prompt text window
    valid_spans = [s for s in gliner_spans if s.get('start', 0) < max_prompt_chars]
    
    if valid_spans:
        span_context = "\n".join(
            f"  - [{s['label']}] \"{s['text']}\" (confidence: {s['score']})"
            for s in valid_spans
        )
        grounding_section = (
            f"\n\nPre-identified entity spans from NER:\n{span_context}\n\n"
            f"Use these spans as anchors for your extraction. You may refine, "
            f"extend, or correct them based on the full text context."
        )
    else:
        grounding_section = ""

    user_prompt = (
        f"Extract PRISMA 2020 structured data from this academic text chunk."
        f"{grounding_section}\n\n"
        f"--- TEXT ---\n{text[:max_prompt_chars]}\n--- END TEXT ---"
    )

    try:
        # Use generate_structured if available (Pydantic validation)
        if hasattr(llm, "generate_structured"):
            result: PRISMAExtraction = await llm.generate_structured(
                user_prompt,
                schema=PRISMAExtraction,
                system_prompt=PRISMA_EXTRACTION_SYSTEM_PROMPT,
                temperature=temperature,
            )
        else:
            # Fallback: use generate() and parse manually
            import json

            raw = await llm.generate(
                user_prompt,
                system_prompt=(
                    PRISMA_EXTRACTION_SYSTEM_PROMPT
                    + f"\n\nRespond ONLY with valid JSON matching this schema:\n"
                    f"{PRISMAExtraction.model_json_schema()}"
                ),
                temperature=temperature,
                max_tokens=2048,
            )
            cleaned = raw.strip()
            if cleaned.startswith("```"):
                cleaned = cleaned.split("\n", 1)[1]
                if cleaned.endswith("```"):
                    cleaned = cleaned[:-3]
            result = PRISMAExtraction.model_validate_json(cleaned)

    except Exception as e:
        logger.warning(f"LLM PRISMA extraction failed: {e}")
        return [], [], []

    # Convert Pydantic result into entity dicts and relation triples
    entities: List[Dict[str, Any]] = []
    relations: List[Tuple[str, str, str]] = []
    hyperedges: List[Dict[str, Any]] = []

    # Paper node (always create one per paper)
    paper_text = f"Paper:{paper_id}"

    # --- Objectives ---
    for obj in result.objectives:
        eid = str(uuid.uuid4())[:8]
        props: Dict[str, Any] = {"extraction_tier": "llm"}
        if obj.pico:
            props["pico"] = obj.pico.model_dump()
        entities.append(
            {
                "entity_id": eid,
                "label": "objective",
                "text": obj.text,
                "paper_ids": [paper_id],
                "prisma_properties": props,
            }
        )
        relations.append((paper_text, "INVESTIGATES", obj.text))

    # --- Methodology ---
    for meth in result.methodology:
        eid = str(uuid.uuid4())[:8]
        props = {
            "extraction_tier": "llm",
            "inclusion_criteria": meth.inclusion_criteria,
            "exclusion_criteria": meth.exclusion_criteria,
            "information_sources": meth.information_sources,
            "risk_of_bias_tool": meth.risk_of_bias_tool,
            "effect_measures": meth.effect_measures,
            "synthesis_model": meth.synthesis_model,
        }
        entities.append(
            {
                "entity_id": eid,
                "label": "methodology",
                "text": meth.text,
                "paper_ids": [paper_id],
                "prisma_properties": props,
            }
        )
        relations.append((paper_text, "UTILIZES_METHOD", meth.text))

    # --- Results ---
    for res in result.results:
        eid = str(uuid.uuid4())[:8]
        props = {
            "extraction_tier": "llm",
            "study_flow": res.study_flow,
            "effect_estimates": res.effect_estimates,
            "certainty_assessment": res.certainty_assessment,
        }
        entities.append(
            {
                "entity_id": eid,
                "label": "result",
                "text": res.text,
                "paper_ids": [paper_id],
                "prisma_properties": props,
            }
        )
        relations.append((paper_text, "REPORTS_FINDING", res.text))

    # --- Limitations ---
    for lim in result.limitations:
        eid = str(uuid.uuid4())[:8]
        entities.append(
            {
                "entity_id": eid,
                "label": "limitation",
                "text": lim.text,
                "paper_ids": [paper_id],
                "prisma_properties": {"extraction_tier": "llm"},
            }
        )
        relations.append((paper_text, "HAS_LIMITATION", lim.text))

    # --- Implications ---
    for imp in result.implications:
        eid = str(uuid.uuid4())[:8]
        entities.append(
            {
                "entity_id": eid,
                "label": "implication",
                "text": imp.text,
                "paper_ids": [paper_id],
                "prisma_properties": {"extraction_tier": "llm"},
            }
        )
        # Implications are linked from result nodes
        # If we have results, link from the first result
        if result.results:
            relations.append((result.results[0].text, "SUPPORTS_IMPLICATION", imp.text))

    # --- Transparency (stored as properties on paper entity) ---
    if result.transparency:
        transparency_props = {
            "extraction_tier": "llm",
            "funding_sources": result.transparency.funding_sources,
            "data_availability": result.transparency.data_availability,
        }
        entities.append(
            {
                "entity_id": str(uuid.uuid4())[:8],
                "label": "paper",
                "text": paper_text,
                "paper_ids": [paper_id],
                "prisma_properties": transparency_props,
            }
        )
        
    # --- Core Principles ---
    for cp in result.core_principles:
        eid = str(uuid.uuid4())[:8]
        entities.append(
            {
                "entity_id": eid,
                "label": "core_principle",
                "text": cp.abstract_principle,
                "paper_ids": [paper_id],
                "prisma_properties": {
                    "extraction_tier": "llm",
                    "domain_jargon": cp.domain_jargon,
                    "explanation": cp.explanation
                },
            }
        )
        relations.append((paper_text, "EXEMPLIFIES_PRINCIPLE", cp.abstract_principle))

    # --- Hyperedges ---
    for he in result.hyperedges:
        # Resolve involved entity texts to our generated entity IDs where possible
        resolved_ids = []
        for involved_text in he.involved_entity_texts:
            for ent in entities:
                # Basic string match or substring match to find the corresponding entity ID
                if involved_text.lower() in ent["text"].lower() or ent["text"].lower() in involved_text.lower():
                    resolved_ids.append(ent["entity_id"])
                    break
        
        # We also want to include the Core Principle entity itself in the hyperedge member list
        principle_ent_id = None
        for ent in entities:
            if ent["label"] == "core_principle" and ent["text"] == he.principle_name:
                principle_ent_id = ent["entity_id"]
                break
                
        if principle_ent_id and principle_ent_id not in resolved_ids:
            resolved_ids.append(principle_ent_id)

        hyperedges.append({
            "hyperedge_id": str(uuid.uuid4())[:8],
            "principle_name": he.principle_name,
            "member_entity_ids": list(set(resolved_ids)),
            "domain_jargon": [],  # Can track distinct jargons later when reducing graph
            "weight": max(0.0, min(1.0, he.hyperedge_weight)),
            "paper_ids": [paper_id]
        })

    # Final pass: check grounding and build final list
    # The source considered for grounding is the text LLM saw, up to max_prompt_chars
    source_window = text[:max_prompt_chars]
    valid_entities = []
    
    for ent in entities:
        if ent["label"] == "paper":
            valid_entities.append(ent)
            continue
            
        is_grounded = _verify_grounding(ent["text"], source_window)
        
        # If strict grounding is enabled, drop ungrounded entities
        if strict_grounding and not is_grounded:
            logger.warning(f"Dropped ungrounded {ent['label']} entity: {ent['text'][:50]}...")
            continue
            
        ent["prisma_properties"]["grounding_verified"] = is_grounded
        valid_entities.append(ent)

    # Note: Filter relations and hyperedges if entities were dropped?
    # In strict grounding mode, we ideally filter orphaned relations.
    # For now, GraphDB handles non-existent node references safely.

    logger.info(
        f"LLM extracted {len(valid_entities)} PRISMA entities, "
        f"{len(relations)} relations, and {len(hyperedges)} hyperedges (paper_id={paper_id})"
    )
    return valid_entities, relations, hyperedges


# ---------------------------------------------------------------------------
# Combined extraction — runs both tiers
# ---------------------------------------------------------------------------


async def extract_prisma_entities(
    text: str,
    paper_id: str,
    llm: Any = None,
    dual_pass: bool = False,
    rigor_level: str = "exploratory",
) -> Tuple[List[Dict[str, Any]], List[Tuple[str, str, str]], List[Dict[str, Any]]]:
    """
    Run the full 2-tier PRISMA extraction pipeline.

    1. GLiNER (Tier 1) — grounded entity spans
    2. LLM + Pydantic (Tier 2) — structured PRISMA extraction + NEXUS abstractions

    If no LLM is available, returns only GLiNER results and empty relations/hyperedges.
    If GLiNER is unavailable, falls back to LLM-only extraction.
    """
    # Tier 1: GLiNER
    gliner_entities, gliner_spans = extract_entities_gliner(text, paper_id)

    # Tier 2: LLM structured extraction (uses GLiNER spans as context)
    if llm is not None:
        strict_mode = rigor_level in ("prisma", "cochrane")
        
        llm_entities, llm_relations, llm_hyperedges = await extract_prisma_structured(
            text, paper_id, gliner_spans, llm,
            strict_grounding=strict_mode, temperature=0.2
        )
        
        if dual_pass:
            # Second pass with higher temperature for variance, then reconcile
            logger.info(f"Running dual-pass LLM extraction for {paper_id}")
            pass2_ents, pass2_rels, pass2_hyps = await extract_prisma_structured(
                text, paper_id, gliner_spans, llm,
                strict_grounding=strict_mode, temperature=0.5
            )
            
            # Reconciliation strategy: combine them, but flag unique elements for human review
            # For simplicity, we merge both sets and deduplicate by text context.
            # In a real rigorous Cochrane review, independent models/humans would extract,
            # and a third arbitrator would resolve conflicts.
            seen_texts = {e["text"].lower() for e in llm_entities if "text" in e}
            
            for p2_ent in pass2_ents:
                if p2_ent.get("text", "").lower() not in seen_texts:
                    # Mark as originating from second pass only
                    p2_ent["prisma_properties"]["dual_extraction_conflict"] = True
                    llm_entities.append(p2_ent)
                    seen_texts.add(p2_ent.get("text", "").lower())
                    
            # Combine relations and hyperedges (simply extend, graph engine usually deduplicates matching triplets)
            llm_relations.extend(pass2_rels)
            llm_hyperedges.extend(pass2_hyps)

        if llm_entities:
            # Merge: LLM entities are primary (richer), GLiNER fills gaps
            combined_entities = llm_entities
            seen_texts = {e["text"].lower() for e in llm_entities}

            for g_ent in gliner_entities:
                if g_ent["text"].lower() not in seen_texts:
                    g_ent.setdefault("prisma_properties", {})["dual_extraction"] = dual_pass
                    combined_entities.append(g_ent)
                    seen_texts.add(g_ent["text"].lower())
                    
            # Mark all LLM entities with dual_extraction status
            for e in combined_entities:
                if "prisma_properties" in e:
                    e["prisma_properties"]["dual_extraction"] = dual_pass

            return combined_entities, llm_relations, llm_hyperedges

    # Fallback: GLiNER only (no relations/hyperedges without LLM)
    return gliner_entities, [], []
