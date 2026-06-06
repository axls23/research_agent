"""
tests/core/test_orchestrator.py
===============================
Unit tests for the native LangGraph ReAct orchestrator and helper functions.
"""

import sys
from pathlib import Path
import asyncio
import pytest
from unittest.mock import AsyncMock, MagicMock, patch

# Ensure the project root is on sys.path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from core.orchestrator import (
    build_orchestrator,
    _enforce_local_model,
    _build_subagent_configs,
    run_agentic_pipeline,
)


def _run(coro):
    """Run an async coroutine synchronously for test convenience."""
    return asyncio.get_event_loop().run_until_complete(coro)


class TestLocalModelEnforcement:
    def test_enforce_local_model_blocks_cloud(self):
        # When NEXUS_LOCAL_ONLY=true, groq, openai, mistral should map to local Ollama fallback
        with patch("core.orchestrator.local_only_enabled", return_value=True):
            with patch.dict("os.environ", {"OLLAMA_MODEL": "qwen2.5:3b"}):
                assert _enforce_local_model("groq:llama-3.1") == "ollama:qwen2.5:3b"
                assert _enforce_local_model("openai:gpt-4") == "ollama:qwen2.5:3b"
                assert _enforce_local_model("mistral:mistral-small") == "ollama:qwen2.5:3b"

    def test_enforce_local_model_allows_ollama(self):
        with patch("core.orchestrator.local_only_enabled", return_value=True):
            assert _enforce_local_model("ollama:llama3") == "ollama:llama3"

    def test_enforce_local_model_allows_cloud_if_local_only_disabled(self):
        with patch("core.orchestrator.local_only_enabled", return_value=False):
            assert _enforce_local_model("groq:llama-3.1") == "groq:llama-3.1"


class TestOrchestratorBuilder:
    def test_build_subagent_configs(self):
        configs = _build_subagent_configs(global_model="ollama:qwen2.5:3b")
        assert len(configs) > 0
        names = [cfg["name"] for cfg in configs]
        assert "deep-reasoner" in names
        assert "dark-data-ingestion" in names
        assert "rosetta-core" in names
        assert "knowledge-graph" in names
        assert "analysis" in names
        assert "writing" in names

    @patch("langgraph.prebuilt.create_react_agent")
    def test_build_orchestrator_compiles_graph(self, mock_create_react_agent):
        mock_create_react_agent.return_value = MagicMock()
        orchestrator = build_orchestrator(model="ollama:qwen2.5:3b")
        assert orchestrator is not None
        assert mock_create_react_agent.called


class TestAgenticPipelineRunner:
    @patch("core.orchestrator.build_orchestrator")
    @patch("core.orchestrator.local_only_enabled", return_value=True)
    async def test_run_agentic_pipeline_invokes_orchestrator(self, mock_local_only, mock_build_orch):
        # Set up mocks
        mock_agent = MagicMock()
        mock_agent.ainvoke = AsyncMock(return_value={"messages": [{"content": "Done"}]})
        mock_build_orch.return_value = mock_agent

        # Mock the state lifecycle logs
        with patch("core.agent_tools.begin_agentic_run") as mock_begin, \
             patch("core.agent_tools.finish_agentic_run", return_value={"stages": {"dark_data_ingestion": {"validation": {"passed": True}}, "data_processing": {"validation": {"passed": True}}, "analysis": {"validation": {"passed": True}}}}) as mock_finish:
            
            result = await run_agentic_pipeline(
                project_name="Test Project",
                research_topic="Test Topic",
                research_goals=["Goal 1"],
                model="ollama:qwen2.5:3b",
                rigor_level="prisma",
            )
            
            assert result is not None
            assert mock_begin.called
            assert mock_finish.called
            mock_agent.ainvoke.assert_called_once()
