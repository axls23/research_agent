# NEXUS Agentic Contract — Mosaic Capability Tiles

The agentic pipeline is composed of **capability tiles**: self-contained,
declarative subagent definitions registered in `core/capabilities.py`.
The ReAct supervisor (`core/orchestrator.py`) composes whatever tiles are
registered — it hard-codes nothing about individual agents. Growing the
system means adding a tile, like adding a piece to a mosaic; the supervisor
prompt catalog, tool wiring, and result handling all follow automatically.

## The tile: `AgentCapability`

| Field | Purpose |
|---|---|
| `name` | Stable kebab-case identity, e.g. `rosetta-core`. |
| `description` | Supervisor-facing summary used for tool selection. |
| `system_prompt` | The tile's own system prompt. |
| `tool_names` | Names resolved against `core.agent_tools` at build time. |
| `model_tier` | `"fast"` (retrieve/ground) or `"deep"` (reason/generate). |
| `dispatch` | Optional per-tile override: `"inline"` or `"queue"`. |
| `catalog_note` | Extra guidance shown only in the supervisor catalog (e.g. the air-gap warning on `literature-search`). |
| `version` | Bump when the tile's contract changes; echoed in every result. |

Tiles are **frozen dataclasses** — a tile's contract cannot be mutated at
runtime, only replaced (`register_capability(cap, replace=True)`).

## The result: `AgentResult`

Every subagent call returns a JSON envelope the supervisor can reason over.
All keys are always present:

```json
{
  "agent": "rosetta-core",
  "status": "ok",
  "summary": "Translated 14 silo terms into 5 shared principles ...",
  "error": null,
  "job_id": 42,
  "duration_ms": 1830,
  "capability_version": "1.0.0"
}
```

- `status`: `"ok"` | `"error"` | `"queued"`.
- `error`: populated on failure — errors are surfaced to the supervisor,
  never swallowed, so it can retry with a refined query or route around
  the failure.
- `job_id`: the ledger row in the SQLite job queue (`outputs/job_queue.db`),
  `null` if the ledger was unavailable (ledger failures never break a run).

## Dispatch modes

Resolution order: **tile override → `NEXUS_AGENT_DISPATCH` env var → default (`inline`)**.

- **`inline`** (default): the subagent executes synchronously in-process and
  the supervisor receives its real output. A ledger row is written to the
  job queue with status `IN_PROGRESS` → `COMPLETED`/`FAILED` for
  observability. Rows are inserted as `IN_PROGRESS` so a concurrently
  running external worker can never pick them up and double-execute.
- **`queue`**: the legacy external-worker mode. The job is enqueued as
  `PENDING` and the supervisor receives a `"queued"` envelope; a separate
  `python nexus.py worker --agent <name>` process must drain the queue.
  Use this only when a deployment intentionally runs workers out-of-process.

## Adding a new tile

1. Implement the tile's tools as plain functions in `core/agent_tools.py`
   (docstrings become the tool schemas).
2. Register the capability — either in the `_DEFAULT_CAPABILITIES` tuple in
   `core/capabilities.py`, or at runtime from a plugin:

   ```python
   from core.capabilities import AgentCapability, register_capability

   register_capability(AgentCapability(
       name="simulation-sandbox",
       description="Run hypothesis simulations against extracted models.",
       system_prompt="You are a simulation specialist ...",
       tool_names=("run_simulation", "validate_quality"),
       model_tier="deep",
   ))
   ```

3. That's it. The supervisor prompt catalog, tool wiring, dispatch, ledger,
   and result envelope are derived from the tile. `build_orchestrator()`
   re-renders its prompt from the live registry, so tiles registered before
   building the orchestrator appear automatically.

Scoping a tile's capability up or down later means editing only that tile:
its tools, prompt, tier, or dispatch — nothing in the supervisor.

## Compatibility notes

- `core.orchestrator._build_subagent_configs()` still returns the legacy
  dict shape (`name` / `description` / `system_prompt` / `model` / `tools`)
  consumed by the `nexus.py` worker and integration tests, plus a
  `capability` key carrying the tile.
- `tests/core/test_capabilities.py` covers the registry, dispatch
  resolution, inline/queue execution, error surfacing, and ledger behavior —
  all offline with stubbed runnables.
