"""
tests/core/test_capabilities.py
===============================
Tests for the mosaic capability registry and the agentic contract:
subagent calls must return a structured AgentResult the supervisor can
reason over, with inline execution as the default dispatch mode.

All tests run offline — subagent runnables are stubbed.
"""

import json
import sys
from pathlib import Path

import pytest

# Ensure the project root is on sys.path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from core import capabilities
from core.capabilities import (
    AgentCapability,
    all_capabilities,
    get_capability,
    make_result,
    register_capability,
    render_catalog,
    resolve_dispatch,
    resolve_tools,
    unregister_capability,
)
from core.orchestrator import (
    ORCHESTRATOR_SYSTEM_PROMPT,
    _make_subagent_tool,
    _render_orchestrator_prompt,
)


DEFAULT_TILE_NAMES = [
    "deep-reasoner",
    "literature-search",
    "dark-data-ingestion",
    "data-processing",
    "rosetta-core",
    "knowledge-graph",
    "analysis",
    "writing",
]


class _FakeMessage:
    def __init__(self, content):
        self.content = content


class _FakeRunnable:
    """Stands in for a compiled create_react_agent runnable."""

    def __init__(self, reply="stub reply", error=None):
        self.reply = reply
        self.error = error
        self.calls = []

    def invoke(self, payload):
        self.calls.append(payload)
        if self.error is not None:
            raise self.error
        return {"messages": [_FakeMessage(self.reply)]}


def _make_cfg(name="stub-agent", dispatch=None, runnable=None):
    cap = AgentCapability(
        name=name,
        description="Test tile",
        system_prompt="You are a test tile.",
        dispatch=dispatch,
        version="9.9.9",
    )
    return {
        "name": name,
        "description": cap.description,
        "system_prompt": cap.system_prompt,
        "model": object(),
        "tools": [],
        "capability": cap,
        "runnable": runnable,
    }


def _disable_ledger(monkeypatch):
    """Make the job-queue ledger a no-op so tests never touch SQLite."""
    from core import job_queue

    monkeypatch.setattr(job_queue, "enqueue_job", lambda *a, **k: 42)
    monkeypatch.setattr(job_queue, "complete_job", lambda *a, **k: None)
    monkeypatch.setattr(job_queue, "fail_job", lambda *a, **k: None)


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------


class TestRegistry:
    def test_default_tiles_registered_in_order(self):
        names = [cap.name for cap in all_capabilities()]
        for expected in DEFAULT_TILE_NAMES:
            assert expected in names
        # Catalog order is registration order
        assert names[: len(DEFAULT_TILE_NAMES)] == DEFAULT_TILE_NAMES

    def test_get_capability_unknown_raises(self):
        with pytest.raises(KeyError):
            get_capability("no-such-tile")

    def test_register_duplicate_raises_without_replace(self):
        existing = get_capability("rosetta-core")
        with pytest.raises(ValueError):
            register_capability(existing)
        register_capability(existing, replace=True)  # idempotent with replace

    def test_register_and_unregister_new_tile(self):
        tile = AgentCapability(
            name="test-mosaic-tile",
            description="Temporary tile",
            system_prompt="prompt",
        )
        register_capability(tile)
        try:
            assert get_capability("test-mosaic-tile") is tile
            assert "test-mosaic-tile" in render_catalog()
            assert "test-mosaic-tile" in _render_orchestrator_prompt()
        finally:
            removed = unregister_capability("test-mosaic-tile")
        assert removed is tile
        assert "test-mosaic-tile" not in render_catalog()

    def test_resolve_tools_returns_callables(self):
        cap = get_capability("knowledge-graph")
        tools = resolve_tools(cap)
        assert len(tools) == len(cap.tool_names)
        assert all(callable(t) for t in tools)

    def test_resolve_tools_unknown_name_raises(self):
        bad = AgentCapability(
            name="bad-tile",
            description="d",
            system_prompt="p",
            tool_names=("definitely_not_a_tool",),
        )
        with pytest.raises(AttributeError):
            resolve_tools(bad)


# ---------------------------------------------------------------------------
# Dispatch resolution
# ---------------------------------------------------------------------------


class TestDispatchResolution:
    def test_default_is_inline(self, monkeypatch):
        monkeypatch.delenv("NEXUS_AGENT_DISPATCH", raising=False)
        cap = AgentCapability(name="t", description="d", system_prompt="p")
        assert resolve_dispatch(cap) == "inline"
        assert resolve_dispatch(None) == "inline"

    def test_env_var_overrides_default(self, monkeypatch):
        monkeypatch.setenv("NEXUS_AGENT_DISPATCH", "queue")
        cap = AgentCapability(name="t", description="d", system_prompt="p")
        assert resolve_dispatch(cap) == "queue"

    def test_tile_override_beats_env(self, monkeypatch):
        monkeypatch.setenv("NEXUS_AGENT_DISPATCH", "queue")
        cap = AgentCapability(
            name="t", description="d", system_prompt="p", dispatch="inline"
        )
        assert resolve_dispatch(cap) == "inline"

    def test_invalid_env_falls_back_to_default(self, monkeypatch):
        monkeypatch.setenv("NEXUS_AGENT_DISPATCH", "carrier-pigeon")
        assert resolve_dispatch(None) == "inline"


# ---------------------------------------------------------------------------
# Agentic contract: inline execution returns real results
# ---------------------------------------------------------------------------


class TestInlineExecution:
    def test_supervisor_receives_real_result(self, monkeypatch):
        monkeypatch.delenv("NEXUS_AGENT_DISPATCH", raising=False)
        _disable_ledger(monkeypatch)
        runnable = _FakeRunnable(reply="found 3 principles")
        cfg = _make_cfg(runnable=runnable)

        subagent_tool = _make_subagent_tool(cfg)
        result = json.loads(subagent_tool.func("translate jargon"))

        assert result["agent"] == "stub-agent"
        assert result["status"] == "ok"
        assert result["summary"] == "found 3 principles"
        assert result["error"] is None
        assert result["duration_ms"] >= 0
        assert result["capability_version"] == "9.9.9"
        # The subagent actually ran, with the query the supervisor sent.
        assert runnable.calls == [{"messages": [("user", "translate jargon")]}]

    def test_error_is_surfaced_not_swallowed(self, monkeypatch):
        monkeypatch.delenv("NEXUS_AGENT_DISPATCH", raising=False)
        _disable_ledger(monkeypatch)
        cfg = _make_cfg(runnable=_FakeRunnable(error=RuntimeError("neo4j down")))

        subagent_tool = _make_subagent_tool(cfg)
        result = json.loads(subagent_tool.func("extract entities"))

        assert result["status"] == "error"
        assert "neo4j down" in result["error"]
        assert result["agent"] == "stub-agent"

    def test_ledger_records_inline_run(self, monkeypatch):
        monkeypatch.delenv("NEXUS_AGENT_DISPATCH", raising=False)
        from core import job_queue

        recorded = {}
        monkeypatch.setattr(
            job_queue,
            "enqueue_job",
            lambda name, payload, status="PENDING": recorded.update(
                {"name": name, "payload": payload, "status": status}
            )
            or 7,
        )
        monkeypatch.setattr(
            job_queue,
            "complete_job",
            lambda job_id, result: recorded.update({"completed": job_id}),
        )

        cfg = _make_cfg(runnable=_FakeRunnable())
        result = json.loads(_make_subagent_tool(cfg).func("q"))

        assert result["job_id"] == 7
        # Inline ledger rows start IN_PROGRESS so a live worker can't steal them.
        assert recorded["status"] == "IN_PROGRESS"
        assert recorded["completed"] == 7

    def test_ledger_failure_does_not_break_run(self, monkeypatch):
        monkeypatch.delenv("NEXUS_AGENT_DISPATCH", raising=False)
        from core import job_queue

        def _boom(*a, **k):
            raise OSError("disk full")

        monkeypatch.setattr(job_queue, "enqueue_job", _boom)
        cfg = _make_cfg(runnable=_FakeRunnable(reply="still fine"))

        result = json.loads(_make_subagent_tool(cfg).func("q"))
        assert result["status"] == "ok"
        assert result["summary"] == "still fine"
        assert result["job_id"] is None


# ---------------------------------------------------------------------------
# Queue dispatch (external worker mode)
# ---------------------------------------------------------------------------


class TestQueueDispatch:
    def test_queue_mode_returns_queued_envelope(self, monkeypatch):
        from core import job_queue

        captured = {}
        monkeypatch.setattr(
            job_queue,
            "enqueue_job",
            lambda name, payload, status="PENDING": captured.update(
                {"name": name, "status": status}
            )
            or 11,
        )

        cfg = _make_cfg(dispatch="queue")
        result = json.loads(_make_subagent_tool(cfg).func("q"))

        assert result["status"] == "queued"
        assert result["job_id"] == 11
        # Queue-mode jobs stay PENDING so the external worker picks them up.
        assert captured["status"] == "PENDING"


# ---------------------------------------------------------------------------
# Prompt and config integration
# ---------------------------------------------------------------------------


class TestOrchestratorIntegration:
    def test_prompt_lists_all_default_tiles(self):
        for name in DEFAULT_TILE_NAMES:
            assert name in ORCHESTRATOR_SYSTEM_PROMPT

    def test_prompt_keeps_contract_keywords(self):
        assert "PRISMA" in ORCHESTRATOR_SYSTEM_PROMPT
        assert "neo4j_vector_search" in ORCHESTRATOR_SYSTEM_PROMPT
        assert '"queued"' in ORCHESTRATOR_SYSTEM_PROMPT

    def test_subagent_configs_carry_capability(self):
        from core.orchestrator import _build_subagent_configs

        configs = _build_subagent_configs(global_model="ollama:qwen2.5:3b")
        assert [c["name"] for c in configs] == [
            cap.name for cap in all_capabilities()
        ]
        for cfg in configs:
            assert isinstance(cfg["capability"], AgentCapability)
            assert len(cfg["tools"]) == len(cfg["capability"].tool_names)


class TestMakeResult:
    def test_all_keys_always_present(self):
        result = make_result(agent="a", status="ok")
        assert set(result) == {
            "agent",
            "status",
            "summary",
            "error",
            "job_id",
            "duration_ms",
            "capability_version",
        }
