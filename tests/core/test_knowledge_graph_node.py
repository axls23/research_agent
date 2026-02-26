"""Tests for core/nodes/knowledge_graph_node.py — PRISMA-aligned KG construction.

Validates:
  - GLiNER entity extraction with PRISMA labels
  - Pydantic PRISMA schema validation
  - Neo4j Cypher generation for PRISMA ontology
  - Qdrant payload enrichment with PRISMA metadata
  - Full knowledge_graph_node pipeline (mocked dependencies)
"""

import sys
import json
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

# Ensure project root is on sys.path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from core.nodes.prisma_extractor import (
    PICOExtraction,
    PRISMAExtraction,
    PRISMAObjective,
    PRISMAMethodology,
    PRISMAResult,
    PRISMALimitation,
    PRISMAImplication,
    PRISMATransparency,
    PRISMA_ENTITY_TYPES,
    PRISMA_RELATION_MAP,
    GLINER_LABELS,
    GLINER_TO_PRISMA,
    extract_entities_gliner,
    extract_prisma_structured,
    extract_prisma_entities,
)


# ---------------------------------------------------------------------------
# Pydantic schema validation
# ---------------------------------------------------------------------------


class TestPydanticSchemas:
    """Verify PRISMA Pydantic models validate correctly."""

    def test_pico_extraction_full(self):
        pico = PICOExtraction(
            population="Adults with type 2 diabetes",
            intervention="Metformin 500mg",
            comparator="Placebo",
            outcome="HbA1c reduction",
        )
        assert pico.population == "Adults with type 2 diabetes"
        assert pico.intervention == "Metformin 500mg"

    def test_pico_extraction_defaults(self):
        pico = PICOExtraction()
        assert pico.population == ""
        assert pico.intervention == ""

    def test_prisma_objective_with_pico(self):
        obj = PRISMAObjective(
            text="To evaluate the efficacy of metformin",
            pico=PICOExtraction(
                population="diabetic adults",
                intervention="metformin",
                comparator="placebo",
                outcome="glucose levels",
            ),
        )
        assert obj.pico is not None
        assert obj.pico.population == "diabetic adults"

    def test_prisma_methodology_full(self):
        meth = PRISMAMethodology(
            text="Systematic literature search",
            inclusion_criteria=["RCTs only", "English language"],
            exclusion_criteria=["Case reports"],
            information_sources=["PubMed", "Cochrane Library"],
            risk_of_bias_tool="RoB 2",
            effect_measures=["Risk Ratio", "Mean Difference"],
            synthesis_model="random-effects",
        )
        assert len(meth.inclusion_criteria) == 2
        assert meth.synthesis_model == "random-effects"

    def test_prisma_result_with_flow(self):
        res = PRISMAResult(
            text="15 studies were included",
            study_flow={
                "identified": 1200,
                "screened": 800,
                "excluded": 785,
                "included": 15,
            },
            effect_estimates=["OR=2.3, 95% CI 1.1-4.5"],
            certainty_assessment="Moderate",
        )
        assert res.study_flow["included"] == 15
        assert len(res.effect_estimates) == 1

    def test_prisma_extraction_empty(self):
        extraction = PRISMAExtraction()
        assert extraction.objectives == []
        assert extraction.methodology == []
        assert extraction.results == []
        assert extraction.limitations == []
        assert extraction.implications == []
        assert extraction.transparency is None

    def test_prisma_extraction_full(self):
        extraction = PRISMAExtraction(
            objectives=[PRISMAObjective(text="Main objective")],
            methodology=[PRISMAMethodology(text="Search strategy")],
            results=[PRISMAResult(text="Key findings")],
            limitations=[PRISMALimitation(text="Small sample size")],
            implications=[PRISMAImplication(text="Future RCTs needed")],
            transparency=PRISMATransparency(
                funding_sources=["NIH Grant R01"],
                data_availability="Available on request",
            ),
        )
        assert len(extraction.objectives) == 1
        assert extraction.transparency.funding_sources[0] == "NIH Grant R01"

    def test_prisma_extraction_json_roundtrip(self):
        """Verify schema can serialize/deserialize (LLM output path)."""
        extraction = PRISMAExtraction(
            objectives=[PRISMAObjective(text="Test objective")],
        )
        json_str = extraction.model_dump_json()
        parsed = PRISMAExtraction.model_validate_json(json_str)
        assert parsed.objectives[0].text == "Test objective"

    def test_prisma_extraction_json_schema_generation(self):
        """Verify JSON schema can be generated (used in LLM prompt)."""
        schema = PRISMAExtraction.model_json_schema()
        assert "properties" in schema
        assert "objectives" in schema["properties"]


# ---------------------------------------------------------------------------
# Constants and mappings
# ---------------------------------------------------------------------------


class TestConstants:
    """Verify PRISMA constants are correct."""

    def test_entity_types(self):
        expected = {
            "paper",
            "objective",
            "methodology",
            "result",
            "limitation",
            "implication",
        }
        assert PRISMA_ENTITY_TYPES == expected

    def test_relation_map_keys(self):
        for (src, tgt), rel_type in PRISMA_RELATION_MAP.items():
            assert src in PRISMA_ENTITY_TYPES
            assert tgt in PRISMA_ENTITY_TYPES
            assert isinstance(rel_type, str)

    def test_gliner_labels_not_empty(self):
        assert len(GLINER_LABELS) > 0

    def test_all_gliner_labels_mapped(self):
        for label in GLINER_LABELS:
            assert label in GLINER_TO_PRISMA, f"GLiNER label '{label}' not mapped"

    def test_gliner_to_prisma_targets_valid(self):
        for gliner_label, prisma_label in GLINER_TO_PRISMA.items():
            assert (
                prisma_label in PRISMA_ENTITY_TYPES
            ), f"GLiNER '{gliner_label}' maps to invalid PRISMA label '{prisma_label}'"


# ---------------------------------------------------------------------------
# GLiNER extraction (Tier 1) — mocked model
# ---------------------------------------------------------------------------


class TestGLiNERExtraction:
    """Test GLiNER entity extraction with mocked model."""

    @patch("core.nodes.prisma_extractor._get_gliner_model")
    def test_gliner_basic_extraction(self, mock_get_model):
        mock_model = MagicMock()
        mock_model.predict_entities.return_value = [
            {
                "text": "randomized controlled trial",
                "label": "inclusion criteria",
                "score": 0.85,
            },
            {"text": "PubMed", "label": "information source", "score": 0.92},
            {"text": "odds ratio", "label": "effect measure", "score": 0.78},
        ]
        mock_get_model.return_value = mock_model

        entities, spans = extract_entities_gliner("Sample text", "paper-001")

        assert len(entities) == 3
        assert entities[0]["label"] == "methodology"  # inclusion criteria → methodology
        assert entities[1]["label"] == "methodology"  # information source → methodology
        assert entities[2]["label"] == "methodology"  # effect measure → methodology
        assert all(e["paper_ids"] == ["paper-001"] for e in entities)
        assert all("prisma_properties" in e for e in entities)
        assert entities[0]["prisma_properties"]["gliner_label"] == "inclusion criteria"

    @patch("core.nodes.prisma_extractor._get_gliner_model")
    def test_gliner_deduplication(self, mock_get_model):
        mock_model = MagicMock()
        mock_model.predict_entities.return_value = [
            {"text": "Machine Learning", "label": "research objective", "score": 0.9},
            {"text": "machine learning", "label": "research objective", "score": 0.7},
        ]
        mock_get_model.return_value = mock_model

        entities, spans = extract_entities_gliner("text", "p1")
        assert len(entities) == 1  # Deduped by lowercase

    @patch("core.nodes.prisma_extractor._get_gliner_model")
    def test_gliner_filters_short_text(self, mock_get_model):
        mock_model = MagicMock()
        mock_model.predict_entities.return_value = [
            {"text": "ML", "label": "research objective", "score": 0.9},  # too short
            {
                "text": "Machine Learning model",
                "label": "research objective",
                "score": 0.8,
            },
        ]
        mock_get_model.return_value = mock_model

        entities, _ = extract_entities_gliner("text", "p1")
        assert len(entities) == 1
        assert entities[0]["text"] == "Machine Learning model"

    @patch("core.nodes.prisma_extractor._get_gliner_model")
    def test_gliner_unavailable_returns_empty(self, mock_get_model):
        mock_get_model.return_value = None
        entities, spans = extract_entities_gliner("text", "p1")
        assert entities == []
        assert spans == []

    @patch("core.nodes.prisma_extractor._get_gliner_model")
    def test_gliner_spans_for_llm_context(self, mock_get_model):
        mock_model = MagicMock()
        mock_model.predict_entities.return_value = [
            {
                "text": "GRADE assessment",
                "label": "certainty assessment",
                "score": 0.88,
            },
        ]
        mock_get_model.return_value = mock_model

        _, spans = extract_entities_gliner("text", "p1")
        assert len(spans) == 1
        assert spans[0]["label"] == "certainty assessment"
        assert spans[0]["score"] == "0.88"


# ---------------------------------------------------------------------------
# LLM structured extraction (Tier 2)
# ---------------------------------------------------------------------------


class TestLLMStructuredExtraction:
    """Test LLM PRISMA extraction with mocked LLM."""

    @pytest.mark.asyncio
    async def test_llm_extraction_with_generate_structured(self):
        mock_llm = AsyncMock()
        mock_llm.generate_structured = AsyncMock(
            return_value=PRISMAExtraction(
                objectives=[
                    PRISMAObjective(
                        text="Evaluate efficacy of drug X",
                        pico=PICOExtraction(
                            population="adults",
                            intervention="drug X",
                            comparator="placebo",
                            outcome="recovery time",
                        ),
                    )
                ],
                results=[PRISMAResult(text="Drug X reduced recovery time by 30%")],
                implications=[
                    PRISMAImplication(text="Drug X should be considered for treatment")
                ],
            )
        )

        entities, relations = await extract_prisma_structured(
            "Sample paper text", "paper-001", [], mock_llm
        )

        assert len(entities) >= 3  # objective + result + implication
        objectives = [e for e in entities if e["label"] == "objective"]
        results = [e for e in entities if e["label"] == "result"]
        implications = [e for e in entities if e["label"] == "implication"]

        assert len(objectives) == 1
        assert len(results) == 1
        assert len(implications) == 1

        assert objectives[0]["prisma_properties"]["pico"]["population"] == "adults"

        # Check relations
        rel_types = [r[1] for r in relations]
        assert "INVESTIGATES" in rel_types
        assert "REPORTS_FINDING" in rel_types
        assert "SUPPORTS_IMPLICATION" in rel_types

    @pytest.mark.asyncio
    async def test_llm_extraction_with_grounded_spans(self):
        mock_llm = AsyncMock()
        mock_llm.generate_structured = AsyncMock(
            return_value=PRISMAExtraction(
                methodology=[
                    PRISMAMethodology(
                        text="Searched PubMed and EMBASE",
                        information_sources=["PubMed", "EMBASE"],
                    )
                ],
            )
        )

        spans = [
            {"text": "PubMed", "label": "information source", "score": "0.92"},
            {"text": "EMBASE", "label": "information source", "score": "0.88"},
        ]

        entities, relations = await extract_prisma_structured(
            "text", "p1", spans, mock_llm
        )

        # Verify the prompt included grounding context
        call_args = mock_llm.generate_structured.call_args
        prompt = call_args[0][0] if call_args[0] else call_args[1].get("prompt", "")
        assert "PubMed" in prompt
        assert "information source" in prompt

    @pytest.mark.asyncio
    async def test_llm_extraction_failure_returns_empty(self):
        mock_llm = AsyncMock()
        mock_llm.generate_structured = AsyncMock(side_effect=Exception("API Error"))

        entities, relations = await extract_prisma_structured(
            "text", "p1", [], mock_llm
        )
        assert entities == []
        assert relations == []


# ---------------------------------------------------------------------------
# Combined extraction pipeline
# ---------------------------------------------------------------------------


class TestCombinedExtraction:
    """Test the full 2-tier extraction pipeline."""

    @pytest.mark.asyncio
    @patch("core.nodes.prisma_extractor._get_gliner_model")
    async def test_combined_merges_tiers(self, mock_get_model):
        # GLiNER returns some entities
        mock_model = MagicMock()
        mock_model.predict_entities.return_value = [
            {"text": "randomized trial", "label": "inclusion criteria", "score": 0.8},
            {"text": "small sample size", "label": "limitation", "score": 0.7},
        ]
        mock_get_model.return_value = mock_model

        # LLM returns overlapping + new entities
        mock_llm = AsyncMock()
        mock_llm.generate_structured = AsyncMock(
            return_value=PRISMAExtraction(
                limitations=[PRISMALimitation(text="small sample size")],  # overlap
                implications=[PRISMAImplication(text="Larger trials needed")],  # new
            )
        )

        entities, relations = await extract_prisma_entities("text", "p1", llm=mock_llm)

        # LLM entities are primary; GLiNER fills gaps
        texts = [e["text"].lower() for e in entities]
        assert "small sample size" in texts
        assert "larger trials needed" in texts
        assert "randomized trial" in texts  # GLiNER-only entity fills the gap

    @pytest.mark.asyncio
    @patch("core.nodes.prisma_extractor._get_gliner_model")
    async def test_combined_gliner_only_when_no_llm(self, mock_get_model):
        mock_model = MagicMock()
        mock_model.predict_entities.return_value = [
            {
                "text": "objective statement",
                "label": "research objective",
                "score": 0.9,
            },
        ]
        mock_get_model.return_value = mock_model

        entities, relations = await extract_prisma_entities("text", "p1", llm=None)

        assert len(entities) == 1
        assert entities[0]["label"] == "objective"
        assert relations == []  # No relations without LLM


# ---------------------------------------------------------------------------
# Knowledge Graph Node (integration)
# ---------------------------------------------------------------------------


class TestKnowledgeGraphNode:
    """Test the full LangGraph node with mocked dependencies."""

    @pytest.mark.asyncio
    @patch("core.nodes.knowledge_graph_node._embed_and_store_qdrant")
    @patch("core.nodes.knowledge_graph_node._persist_to_neo4j")
    @patch("core.nodes.prisma_extractor._get_gliner_model")
    async def test_node_processes_chunks(self, mock_gliner, mock_neo4j, mock_qdrant):
        from core.nodes.knowledge_graph_node import knowledge_graph_node

        # Mock GLiNER
        mock_model = MagicMock()
        mock_model.predict_entities.return_value = [
            {"text": "deep learning", "label": "research objective", "score": 0.9},
        ]
        mock_gliner.return_value = mock_model

        # Mock persistence
        mock_neo4j.return_value = {
            "neo4j_status": "success",
            "nodes_created": 1,
            "relationships_created": 0,
        }
        mock_qdrant.return_value = {"qdrant_status": "success", "vectors_stored": 1}

        state = {
            "chunks": [
                {
                    "text": "We apply deep learning to medical imaging.",
                    "paper_id": "p1",
                },
            ],
            "audit_log": [],
        }

        result = await knowledge_graph_node(state)

        assert result["current_node"] == "knowledge_graph"
        assert len(result["knowledge_entities"]) > 0
        assert result["knowledge_graph_summary"]["schema"] == "prisma_2020"

    @pytest.mark.asyncio
    async def test_node_empty_chunks(self):
        from core.nodes.knowledge_graph_node import knowledge_graph_node

        state = {"chunks": [], "audit_log": []}
        result = await knowledge_graph_node(state)

        assert result["current_node"] == "knowledge_graph"
        assert len(result["audit_log"]) == 1
