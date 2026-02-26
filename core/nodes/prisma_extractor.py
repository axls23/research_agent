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

import logging
import uuid
from typing import Any, Dict, List, Optional, Tuple

from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


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


class PRISMAExtraction(BaseModel):
    """
    Full PRISMA 2020-aligned extraction from a paper chunk.

    This is the schema passed to LLM generate_structured().
    GLiNER entity spans are injected into the prompt as grounded context.
    """

    objectives: List[PRISMAObjective] = Field(default_factory=list)
    methodology: List[PRISMAMethodology] = Field(default_factory=list)
    results: List[PRISMAResult] = Field(default_factory=list)
    limitations: List[PRISMALimitation] = Field(default_factory=list)
    implications: List[PRISMAImplication] = Field(default_factory=list)
    transparency: Optional[PRISMATransparency] = Field(default=None)


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
}

PRISMA_RELATION_MAP = {
    # (source_label, target_label) -> relationship_type
    ("paper", "objective"): "INVESTIGATES",
    ("paper", "methodology"): "UTILIZES_METHOD",
    ("paper", "result"): "REPORTS_FINDING",
    ("paper", "limitation"): "HAS_LIMITATION",
    ("result", "implication"): "SUPPORTS_IMPLICATION",
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

Only extract what is explicitly stated in the text. Do not infer or fabricate data.
If a category has no relevant content in this chunk, return an empty list for that field."""


async def extract_prisma_structured(
    text: str,
    paper_id: str,
    gliner_spans: List[Dict[str, str]],
    llm: Any,
) -> Tuple[List[Dict[str, Any]], List[Tuple[str, str, str]]]:
    """
    Use LLM with Pydantic schema to extract structured PRISMA data.

    GLiNER spans are injected as grounded context to improve precision.

    Returns:
        entities: List of PRISMA entity dicts
        relations: List of (source_text, relation_type, target_text) triples
    """
    # Format GLiNER spans as grounded context
    if gliner_spans:
        span_context = "\n".join(
            f"  - [{s['label']}] \"{s['text']}\" (confidence: {s['score']})"
            for s in gliner_spans
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
        f"--- TEXT ---\n{text[:3000]}\n--- END TEXT ---"
    )

    try:
        # Use generate_structured if available (Pydantic validation)
        if hasattr(llm, "generate_structured"):
            result: PRISMAExtraction = await llm.generate_structured(
                user_prompt,
                schema=PRISMAExtraction,
                system_prompt=PRISMA_EXTRACTION_SYSTEM_PROMPT,
                temperature=0.2,
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
                temperature=0.2,
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
        return [], []

    # Convert Pydantic result into entity dicts and relation triples
    entities: List[Dict[str, Any]] = []
    relations: List[Tuple[str, str, str]] = []

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

    logger.info(
        f"LLM extracted {len(entities)} PRISMA entities and "
        f"{len(relations)} relations (paper_id={paper_id})"
    )
    return entities, relations


# ---------------------------------------------------------------------------
# Combined extraction — runs both tiers
# ---------------------------------------------------------------------------


async def extract_prisma_entities(
    text: str,
    paper_id: str,
    llm: Any = None,
) -> Tuple[List[Dict[str, Any]], List[Tuple[str, str, str]]]:
    """
    Run the full 2-tier PRISMA extraction pipeline.

    1. GLiNER (Tier 1) — grounded entity spans
    2. LLM + Pydantic (Tier 2) — structured PRISMA extraction

    If no LLM is available, returns only GLiNER results.
    If GLiNER is unavailable, falls back to LLM-only extraction.
    """
    # Tier 1: GLiNER
    gliner_entities, gliner_spans = extract_entities_gliner(text, paper_id)

    # Tier 2: LLM structured extraction (uses GLiNER spans as context)
    if llm is not None:
        llm_entities, llm_relations = await extract_prisma_structured(
            text, paper_id, gliner_spans, llm
        )

        if llm_entities:
            # Merge: LLM entities are primary (richer), GLiNER fills gaps
            combined_entities = llm_entities
            seen_texts = {e["text"].lower() for e in llm_entities}

            for g_ent in gliner_entities:
                if g_ent["text"].lower() not in seen_texts:
                    combined_entities.append(g_ent)
                    seen_texts.add(g_ent["text"].lower())

            return combined_entities, llm_relations

    # Fallback: GLiNER only (no relations without LLM)
    return gliner_entities, []
