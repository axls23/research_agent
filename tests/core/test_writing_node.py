"""Tests for core/nodes/writing_node.py.

Verifies that the final synthesis pass uses a local Ollama provider and that
the prompt is grounded in data-processing context from the pipeline state.
"""

import sys
from pathlib import Path

import pytest

# Ensure the project root is on sys.path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import core.nodes.writing_node as writing_module


class FakeOllamaProvider:
    instances = []

    def __init__(self, model: str = "qwen2.5:3b", base_url: str | None = None, **kwargs):
        self.model = model
        self.base_url = base_url
        self.calls = []
        FakeOllamaProvider.instances.append(self)

    async def generate(
        self,
        prompt: str,
        *,
        system_prompt: str | None = None,
        temperature: float = 0.7,
        max_tokens: int = 4096,
        **kwargs,
    ) -> str:
        self.calls.append(
            {
                "prompt": prompt,
                "system_prompt": system_prompt,
                "temperature": temperature,
                "max_tokens": max_tokens,
            }
        )

        if len(self.calls) == 1:
            return "LOCAL OUTLINE"
        if len(self.calls) == 2:
            return "LOCAL LITERATURE REVIEW"
        if len(self.calls) == 3:
            return "LOCAL INTRODUCTION"
        return "LOCAL FINAL PASS"


@pytest.mark.asyncio
async def test_writing_node_final_pass_uses_local_ollama_and_data_context(monkeypatch):
    monkeypatch.setenv("OLLAMA_MODEL", "qwen2.5:3b")
    monkeypatch.setenv("OLLAMA_BASE_URL", "http://localhost:11434")
    monkeypatch.setattr(writing_module, "OllamaProvider", FakeOllamaProvider)

    state = {
        "research_topic": "Local models for systematic review synthesis",
        "research_goals": ["summarise evidence", "preserve traceability"],
        "papers": [
            {
                "paper_id": "p1",
                "title": "Paper One",
                "year": 2024,
                "included": True,
                "annotations": {"method": "pdf"},
            }
        ],
        "chunks": [
            {
                "paper_id": "p1",
                "text": "Gradient token routing improves local synthesis fidelity.",
                "token_count": 12,
            }
        ],
        "knowledge_entities": [],
        "analysis_results": [
            {
                "method": "descriptive",
                "result_summary": "1 paper included, 0 entities extracted",
                "figures": [],
                "tables": [],
                "statistical_output": None,
            }
        ],
        "draft_sections": {},
        "outline": None,
        "backtrack_count": 0,
        "papers_screened": 1,
        "papers_included": 1,
        "total_tokens_extracted": 12,
        "dual_extraction_performed": True,
        "audit_log": [],
    }

    result = await writing_module.writing_node(state, config={"configurable": {}})

    assert result["draft_sections"]["final_pass"] == "LOCAL FINAL PASS"
    assert FakeOllamaProvider.instances, "expected a local Ollama provider instance"

    provider = FakeOllamaProvider.instances[0]
    assert provider.model == "qwen2.5:3b"
    assert provider.base_url == "http://localhost:11434"
    assert len(provider.calls) == 5
    assert any(
        "Gradient token routing improves local synthesis fidelity." in call["prompt"]
        for call in provider.calls
    )
    assert "local Ollama synthesis model" in provider.calls[3]["system_prompt"]