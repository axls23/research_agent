# NEXUS Research Agent — Repository Audit

**Report date:** 2026-07-11
**Auditor:** Automated code review (Claude)
**Repository:** `research_agent` (branch `main`, HEAD `da068aa7` — a `wip:` commit)
**Scope:** Full-repository architecture, correctness, efficiency, security, docs, and testing review.

> **Method & caveat.** This audit is based on a close reading of the source. The
> project's Python dependencies are **not installed** in the audit environment
> (system Python is 3.14; the project targets 3.10), so the test suite and
> pipeline could **not** be executed at runtime. Findings marked *Confirmed by
> code reading* are high-confidence from static analysis but were not
> runtime-verified. Where that distinction matters, it is called out.

---

## 1. Executive Summary

NEXUS is an ambitious **local-first, multi-agent systematic-review engine**. It
ingests papers and "dark data," extracts PRISMA-2020-aligned entities into a
Neo4j knowledge graph, detects cross-domain "isomorphic" principles, runs
validation gates with human-in-the-loop, and drafts research output — all served
through a FastAPI backend and a polished Next.js frontend.

**The design is genuinely strong. The wiring between the parts is not.** Several
individual modules (the PRISMA extractor, the multi-database search client, the
state/audit model, the validation gates) are well-built and could stand on their
own. But the **agentic serving path that the API and CLI actually call is broken
end-to-end**: the orchestrator dispatches work into a fire-and-forget SQLite
queue, never collects results, and returns a canned fallback string to the user.
Meanwhile the fully-wired *deterministic* pipeline that the architecture doc
describes in detail is **dead code**, exercised only by unit tests.

The net effect: **as currently wired, a user request produces no real research
output.** The frontend will stream stage updates and then render
*"I have completed analyzing the research related to your query."* — the literal
fallback in [api.py:152](api.py#L152).

### Overall maturity

| Dimension | Rating (1–5) | One-line verdict |
|---|:---:|---|
| Vision / architecture design | ★★★★☆ | Coherent, well-thought-out, appropriately scoped |
| Individual module quality | ★★★★☆ | PRISMA extractor, search, state, validation are solid |
| **End-to-end integration** | **★☆☆☆☆** | **Serving path does not return real results** |
| Correctness / reliability | ★★☆☆☆ | Multiple integration-seam bugs + latent import errors |
| Computing efficiency | ★★☆☆☆ | Model/driver reloading, no batching, sequential |
| Security | ★★☆☆☆ | `exec()` of LLM output; arbitrary Cypher; template policy |
| Testing / CI | ★★☆☆☆ | Mocked, dead-path coverage, no CI, collection hazard |
| Docs / repo hygiene | ★★☆☆☆ | README & status docs describe a system that no longer exists |
| Frontend | ★★★★☆ | Clean Next.js 16 app, correctly wired to the API |

**Bottom line:** this is a **high-potential prototype at roughly 55–65% of a
working v1**, held back by a small number of high-impact integration defects.
The distance from "impressive demo architecture" to "actually produces a
systematic review" is smaller than it looks — perhaps **2–4 focused weeks** — but
it requires fixing the orchestration seam first, before any feature work.

---

## 2. What the Project Is (and where it actually is)

### Intended architecture (per `architecture.md`)
Two execution modes:
1. **Deterministic mode** — a LangGraph `StateGraph`: ingest → process → rosetta →
   knowledge-graph → analysis → writing → audit, with validation + human gates
   inserted for PRISMA/Cochrane rigor.
2. **Agentic mode** — a ReAct supervisor that dynamically delegates to subagents.

### Actual state of each
| Component | Status | Notes |
|---|---|---|
| `core/state.py` (ResearchState, audit) | ✅ **Solid** | Well-typed TypedDicts, append-only SHA-256 audit trail, PRISMA counts. |
| `core/nodes/prisma_extractor.py` | ✅ **Strong** | GLiNER (Tier 1) + LLM/Pydantic (Tier 2) + grounding verification. Best module in the repo. |
| `core/tools/search_tools.py` | ✅ **Good** | arXiv/Semantic Scholar/Crossref, real rate-limiting, retries, dedup. |
| `core/tools/validation_tools.py` + `core/workflows/*.yaml` | ✅ **Good** | Real, config-driven PRISMA/Cochrane gates. |
| `core/graph.py` `EagerGraphRunner` (deterministic mode) | ⚠️ **Dead code** | Only called from `tests/core/test_graph.py`; `run_research_pipeline` never uses it (see §4.1). |
| `core/orchestrator.py` (agentic mode) | ❌ **Broken seam** | Dispatches to a queue and never reads results (see §4.1). |
| `core/job_queue.py` + `nexus.py worker` | ⚠️ **Orphaned** | Worker exists but runs out-of-band; results never rejoin the run. |
| `core/reasoning.py` `NativeReasoningLoop` | ⚠️ **Unsafe** | `exec()` of model-generated code, mislabeled "sandbox" (see §7). |
| `api.py` (FastAPI + SSE) | ⚠️ **Wired to broken path** | Streaming/UX is nice; consumes a result that isn't populated. |
| `frontend-next/` (Next.js 16) | ✅ **Good** | Clean, correctly points at `:8000`, SSE chat, backend monitor. |
| Docs (`README.md`, `SYSTEM_STATUS.md`) | ❌ **Stale** | Describe a pre-pivot `agents/` + Groq system that no longer exists. |

The project has clearly undergone a **pivot** (from a Groq-backed "research
agent" with an `agents/` package to a local-first, Ollama/vLLM "NEXUS" with
`core/nodes/`). Much of the confusion in the repo is pivot residue that was never
cleaned up.

---

## 3. Strengths (what to preserve)

1. **PRISMA extraction pipeline** — `prisma_extractor.py` is genuinely good:
   two-tier (GLiNER zero-shot NER → LLM structured Pydantic extraction), with
   `_verify_grounding()` to drop hallucinated spans in strict mode, dual-pass for
   Cochrane, and checklist-tag provenance. This is real, defensible methodology.
2. **Audit/compliance model** — `state.py` `append_audit()` hashes inputs
   (SHA-256), timestamps in UTC, and keeps an append-only log. The compliance
   story (reproducibility, PRISMA counts, validation reports, human decisions)
   is a strong differentiator for the enterprise/regulated use case.
3. **Search client** — `search_tools.py` respects arXiv's 3-second policy, sends a
   proper User-Agent, backs off on 429/503, and dedupes across three databases
   concurrently via `asyncio.gather`.
4. **Config-driven rigor** — validation criteria live in
   `core/workflows/{exploratory,prisma,cochrane}.yaml`, not hardcoded. Good design.
5. **Frontend** — modern Next.js 16 / React 19, Tailwind v4, SSE streaming chat,
   a live backend monitor page, environment-based API URL. Coherent and clean.
6. **Local-first intent** — the air-gap policy (`NEXUS_LOCAL_ONLY`) and the
   Ollama/vLLM abstraction are the right call for the "sovereign / dark-data"
   positioning.

---

## 4. Critical Findings (block real-world use)

### 4.1 — The agentic pipeline dispatches into a void  🔴 *Critical* — *Confirmed by code reading*

This is the headline issue. Trace the serving path:

1. `api.py` and `nexus.py run` and `cli.py` all call
   `run_research_pipeline()` ([api.py:687](api.py#L687), [cli.py:182](cli.py#L182)).
2. `run_research_pipeline()` **ignores its `mode` argument** and always delegates
   to `run_agentic_pipeline()` — [core/graph.py:379-392](core/graph.py#L379-L392).
3. `run_agentic_pipeline()` builds a ReAct supervisor whose tools are the seven
   subagents ([core/orchestrator.py:336-340](core/orchestrator.py#L336-L340)).
4. Each subagent tool, when the LLM calls it, does **not run the subagent**. It
   enqueues a SQLite row and returns a string telling the model *not to wait*:

   ```python
   # core/orchestrator.py:275-284
   def call_subagent(query: str) -> str:
       from core.job_queue import enqueue_job
       job_id = enqueue_job(name, {"query": query})
       return f"Task asynchronously dispatched to {name} subagent. Job ID: {job_id}. Do not wait for the result."
   ```

   The compiled `subagent_runnable` created just above it
   ([orchestrator.py:269](core/orchestrator.py#L269)) is **never invoked** — dead code.
5. Nothing in the API/CLI path starts a worker. The only consumer is
   `nexus.py worker --agent <name>` ([nexus.py:444](nexus.py#L444)), a **separate
   manual process per agent**. Even when run, its results are written to the
   SQLite `jobs.result` column and to that process's **in-memory**
   `_AGENTIC_STATE` — they never return to the orchestrator (which was told "do
   not wait") nor to the API process.
6. `run_agentic_pipeline()` therefore returns the orchestrator's raw chat
   messages, with an empty `stage_summary` — [orchestrator.py:475-477](core/orchestrator.py#L475-L477).
7. `_build_chat_payload()` looks for `draft_sections` / `analysis_results` /
   `papers` / `audit_log` in that result, finds none, and returns the canned
   fallback — [api.py:138-152](api.py#L138-L152).

**Consequence:** the user gets *"I have completed analyzing the research related
to your query."* with no papers, no citations, no analysis. The multi-agent
"orchestration" is theater: a supervisor narrating delegations that never
execute and whose outputs it never sees.

**Root cause:** `_AGENTIC_STATE` ([agent_tools.py:45-52](core/agent_tools.py#L45-L52))
is a **module-level global** used as the cross-agent blackboard. That can never
work across the process boundary the job queue introduces. The system has two
incompatible designs (in-process shared state *and* a cross-process queue)
half-merged together.

**Fix direction (choose one, don't keep both):**
- **Simplest / recommended:** delete the queue indirection. Make each subagent
  tool `await subagent_runnable.ainvoke(...)` and **return the result to the
  supervisor**, threading state through LangGraph's real state channel instead of
  a global. This is the intended "agents-as-tools" pattern.
- **Or:** commit to the distributed model — API enqueues, workers run, API polls
  `job_status`/`jobs.result` and reassembles state from a **shared** store
  (SQLite/Redis), not a Python global. More infrastructure; only worth it if you
  need horizontal scale.

### 4.2 — The deterministic pipeline (the one that works) is never used  🔴 *Critical*

The `EagerGraphRunner` in `core/graph.py` is a complete, linear, gated pipeline
that actually calls the nodes in order and threads `ResearchState` correctly.
It is the system `architecture.md` describes. But `run_research_pipeline()`
short-circuits to agentic mode for **all** rigor levels
([graph.py:379-392](core/graph.py#L379-L392)), so the eager runner only ever runs
inside `tests/core/test_graph.py`. You have a working engine sitting unused next
to a broken one that ships.

**Fix direction:** route `mode="default"/"langgraph"` (and arguably PRISMA/Cochrane
by default) back through `build_research_graph()` + `EagerGraphRunner.ainvoke()`.
This alone could give you a **functioning product today** while §4.1 is redesigned.

### 4.3 — `literature_search` ↔ `search_multiple_databases` contract mismatch  🔴 *High* — *Confirmed by code reading*

Three compounding bugs at one seam:

- `search_multiple_databases(query: str, ...)` takes a **single string** and
  returns a **tuple** `(papers, databases_searched)` — [search_tools.py:364-368](core/tools/search_tools.py#L364-L368).
- The caller passes a **list** and treats the return as a flat records list:
  ```python
  # core/agent_tools.py:187-192
  records = await search_multiple_databases(topics, max_results_per_db=...)
  _AGENTIC_STATE["papers"].extend(records)   # records is a 2-tuple
  ```
  Passing a list where a `str` is expected makes `urllib.parse.quote(query)`
  raise inside each DB call; `asyncio.gather(return_exceptions=True)` swallows it,
  yielding zero papers. `.extend((papers, searched))` then appends the two
  *lists* as if they were paper records, and `len(records)` is always `2`.
- Even on the happy path, search returns `PaperMeta` **objects**, but downstream
  code assumes **dicts** (`p.get("paper_id")`), and `PaperMeta.to_dict()` is never
  called.

**Fix:** normalize the contract — iterate topics, call with strings, unpack the
tuple, and map `PaperMeta.to_dict()` before storing.

### 4.4 — `MistralProvider` import will crash if OCR is enabled  🟠 *High* — *Confirmed by code reading*

`extract_with_mistral()` does `from core.llm_provider import MistralProvider`
([core/tools/extraction_tools.py:40](core/tools/extraction_tools.py#L40)), but
`core/llm_provider.py` defines **no** `MistralProvider` (only Ollama and
FastRLM). `data_processing_node` defaults `use_mistral_ocr=True`
([data_processing_node.py:37](core/nodes/data_processing_node.py#L37)). So the
moment `MISTRAL_API_KEY` is set, extraction raises `ImportError` instead of doing
OCR. (It's masked today only because the key is usually unset, taking the PyPDF2
fallback.) This is also an **air-gap contradiction** — see §7.

### 4.5 — Chunk embeddings are computed only in a degenerate branch  🟠 *Medium* — *Confirmed by code reading*

In `knowledge_graph_node`, chunk embedding
(`embed_model.encode(chunk_texts)`) sits **inside** `if all_entities and not
all_relations:` — [knowledge_graph_node.py:398-419](core/nodes/knowledge_graph_node.py#L398-L419).
In the normal case (relations present), chunk embeddings are never produced, so
`c["embedding"]` is missing and `_persist_to_neo4j` skips writing chunk vectors
([kg_node:117](core/nodes/knowledge_graph_node.py#L117)). The `chunk_embeddings`
vector index is created but effectively never populated — GraphRAG chunk
retrieval is silently degraded.

---

## 5. Correctness & Reliability (medium severity)

| # | Finding | Location |
|---|---|---|
| 5.1 | `run_research_pipeline`'s `interactive`, `allow_auto_override`, `config_path`, `mode` params are accepted but **ignored** (always agentic). Misleading API. | [graph.py:350-392](core/graph.py#L350-L392) |
| 5.2 | `_parse_structured_with_retry` final fallback builds a `ValidationError` via a ternary that, on success, **raises the constructed-but-empty error** rather than a descriptive one; the `ValueError` branch is unreachable on modern Pydantic. | [llm_provider.py:122-127](core/llm_provider.py#L122-L127) |
| 5.3 | `NativeReasoningLoop._generate` uses `asyncio.get_event_loop().run_until_complete` inside a sync method — raises inside a running loop (i.e., under FastAPI/async). | [reasoning.py:159-161](core/reasoning.py#L159-L161) |
| 5.4 | `config.yaml` still declares `knowledge_graph.backend: networkx` and an `agents.enabled` list from the old architecture; the code hardcodes Neo4j and ignores these keys. Config no longer matches code. | [config/config.yaml:94-99](config/config.yaml#L94) |
| 5.5 | Tiering is nominal: both `fast` and `deep` tiers are `qwen2.5:3b`; `architecture.md` claims "8B fast / 70B deep." No functional tiering exists. | [config/config.yaml:41-51](config/config.yaml#L41) |
| 5.6 | `knowledge_graph_node` re-creates a fresh Neo4j driver **per paper** and re-issues `CREATE VECTOR INDEX … IF NOT EXISTS` on every write. | [kg_node:441-453](core/nodes/knowledge_graph_node.py#L441-L453) |
| 5.7 | Backtrack loop (`writing → literature_review`) is only reachable in the dead eager runner; in agentic mode the `needs_more_papers` signal is never acted on. | [graph.py:90-98](core/graph.py#L90-L98) |

---

## 6. Computing Efficiency

The system is **functionally sequential and reloads heavy resources repeatedly.**
On modest local hardware (the stated deployment target), a single PRISMA run will
be slow and memory-spiky. Concrete issues, roughly in impact order:

1. **Embedding model reloaded on every call.**
   `SentenceTransformer("allenai/specter2_base")` (~400 MB) is loaded fresh
   *inside* `knowledge_graph_node` on each invocation
   ([kg_node:390](core/nodes/knowledge_graph_node.py#L390)) **and** inside
   `neo4j_vector_search` on **every search** ([agent_tools:533](core/agent_tools.py#L533)).
   Each load is multi-second + hundreds of MB. → Make it a module-level lazy
   singleton (as GLiNER already is at [prisma_extractor:273-294](core/nodes/prisma_extractor.py#L273-L294)).
   *This is the single biggest, cheapest win.*
2. **Neo4j driver churn + no batching.** A driver is opened/closed per paper
   (§5.6), and each entity/relation/chunk/hyperedge-member is a separate
   `session.run()` round-trip ([kg_node:126-243](core/nodes/knowledge_graph_node.py#L126-L243)).
   For a review with hundreds of entities this is hundreds of network
   round-trips. → One long-lived driver; batch with `UNWIND $rows`.
3. **No concurrency across chunks.** Up to 50 chunks are processed in a serial
   `for` loop, each doing 1 GLiNER pass + 1–2 LLM calls
   ([kg_node:363-369](core/nodes/knowledge_graph_node.py#L363-L369)). With a 3B
   model this is minutes of wall-clock. → Bounded `asyncio.gather` (respecting
   local model concurrency limits).
4. **`max_parallel_agents: 1`** ([config.yaml:113](config/config.yaml#L113)) — a
   deliberate choice for sequential AirLLM layer loading, but it means the whole
   pipeline is single-threaded by config.
5. **Redundant DDL** — the two `CREATE VECTOR INDEX` statements run on every
   persistence call rather than once at startup.
6. **Whole-model self-critique per gate** — non-exploratory runs add an extra LLM
   round-trip at each validation gate ([quality_validator_node:111-112](core/nodes/quality_validator_node.py#L111)).
   Reasonable, but budget for it.

**Efficiency verdict:** correctness-of-speed is achievable with ~4 targeted
changes (singletons, driver reuse, batched Cypher, bounded concurrency). None
require architectural change; all are localized.

---

## 7. Security

| Severity | Finding |
|---|---|
| 🔴 High | **Arbitrary code execution via `exec()`.** `NativeReasoningLoop` runs LLM-generated Python with `exec(combined_code, self.globals)` and no isolation ([reasoning.py:121-131](core/reasoning.py#L121-L131)), despite the docstring calling it a "Python sandbox." A document ingested into context could prompt-inject code that runs in-process (file access, network, etc.). If the deep-reasoner is ever wired to real input, this is RCE. → Use a real sandbox (subprocess with seccomp/nsjail, `RestrictedPython`, or a container), or remove the code-exec capability. |
| 🟠 Med | **Unrestricted Cypher from the model.** The `neo4j_query` tool executes any Cypher the LLM emits ([agent_tools.py:583-625](core/agent_tools.py#L583)). A hallucinated/injected `MATCH (n) DETACH DELETE n` would wipe the graph. → Enforce read-only sessions for model-driven queries; allowlist procedures. |
| 🟠 Med | **Air-gap policy is contradicted in code.** `llm_provider.py` advertises "only local backends… strict air-gap," yet `extract_with_mistral` calls the **cloud** Mistral Document AI when a key is present ([extraction_tools.py:23-68](core/tools/extraction_tools.py#L23)), and `requirements-phase1.txt` pins `anthropic`, `openai`, `groq`, `google-cloud-*`. The guarantee is aspirational, not enforced at the dependency/egress layer. |
| 🟡 Low | **`SECURITY.md` is the unedited GitHub template** (placeholder "5.1.x / 4.0.x" table, "Tell them where to go…"). |
| ✅ Good | No secrets are committed; `.env` is gitignored; keys are read from env. |

---

## 8. Documentation & Repository Hygiene

- **`README.md` is wrong.** It documents an `agents/` package, a `pip install -e
  .` `orchestrator.start_research_project(...)` API, and a directory tree that
  **do not exist** in the current code. A new contributor following it will fail
  immediately.
- **`SYSTEM_STATUS.md` is stale** — claims Groq is "configured and working,"
  `.venv` active, and a frontend on `:8080/index.html`. The real frontend is
  Next.js on `:3000` talking to `:8000`.
- **`requirements-phase1.txt` is UTF-16-encoded** (hence the byte-spaced diff
  output) and contains an **editable install pointing at a leaked Windows dev
  path**: `-e c:\users\sxhil_25660\documents\github\research_agent`. It is
  non-portable and pins a large, mostly-unused cloud dependency surface. `pip
  install -r` will not reproduce a clean env.
- **`pyproject.toml` drift** — author is "Your Name / your.email@example.com";
  deps disagree with `requirements-phase1.txt`; `langchain-ollama>=1.1.0` is not
  a real published version (current is ~0.2/0.3); `turbovec[langchain]` is
  unusual and unpinned.
- **Three overlapping entry points** — `nexus.py`, `cli.py`, and
  `run_research_agent.py` all launch the pipeline; plus `run_frontend.{py,bat,ps1}`
  and `presentation_demo.bat`. Pick one CLI (`nexus.py` is the most polished) and
  delete the rest.
- **Data artifacts are committed** despite `.gitignore` — `chunks/*.jsonl` and
  `2311.16101.pdf` are tracked (they predate the ignore rules; `.gitignore`
  doesn't retroactively untrack). Repo carries binary/data weight.
- **Probe/scratch scripts committed** — `tmp/neo4j_*_probe.py`,
  `tmp_read_docx.py`, `test_proxy_8001.py`, `test_airllm_pipeline.py`,
  `test_groq_integration.py` (tests a *blocked* provider).
- **No CI.** No `.github/workflows/`. Origin shows Dependabot PRs and many
  bot-driven `*/audit-technical-debts`, `*/hunt-logical-errors-architecture`
  branches — i.e., the debt is known and has been repeatedly poked at, but
  nothing gates `main`.

---

## 9. Testing

- ~10 files under `tests/` — mostly **unit tests, heavily mocked**. E.g.
  `test_orchestrator.py` mocks `build_orchestrator` entirely and asserts
  `ainvoke` was called once; it **cannot** catch the §4.1 dispatch bug because it
  never exercises the real tool.
- `test_graph.py` covers only the **dead** `EagerGraphRunner` path.
- **`tests/integration/test_deep_agents_e2e.py` is not a pytest module** — it runs
  logic at import time and calls `sys.exit(1)` on failure
  ([test_deep_agents_e2e.py:210-211](tests/integration/test_deep_agents_e2e.py#L210)).
  Once deps are installed, this will **abort `pytest` collection** for the whole
  suite.
- **Async tests likely don't run.** `pyproject.toml`'s pytest config sets no
  `asyncio_mode`, and `test_run_agentic_pipeline_invokes_orchestrator` is an
  `async def` without an await harness — pytest will warn "coroutine never
  awaited" and effectively no-op it (false green).
- **No test covers the actual serving seam** (`api → run_agentic_pipeline →
  payload shape`), which is exactly where the product is broken.
- Root-level `test_*.py` scripts target **removed/blocked** integrations (Groq,
  vLLM proxy, AirLLM).

---

## 10. Prioritized Roadmap

### P0 — Make it actually produce a result (days, not weeks)
1. **Unbreak orchestration (§4.1).** Either (a) route PRISMA/Cochrane through the
   working `EagerGraphRunner` now (§4.2) for an immediate functioning product,
   and/or (b) rewrite subagent tools to invoke `subagent_runnable` and return
   results, threading state through LangGraph instead of a global.
2. **Fix the search seam (§4.3)** — string-per-topic, unpack the tuple,
   `PaperMeta.to_dict()`.
3. **Fix `_build_chat_payload`** to read whatever shape the (now-real) result
   returns, so the API stops emitting the canned fallback.
4. **Add one true end-to-end test** (mock only the LLM/network) that asserts
   papers → chunks → entities → draft actually flow through.

### P1 — Correctness, safety, speed (1–2 weeks)
5. Singleton the Specter2 embedder; reuse one Neo4j driver; batch Cypher with
   `UNWIND` (§6.1–6.2). Fix the chunk-embedding branch (§4.5).
6. Remove/replace the `exec()` reasoning sandbox, or gate it behind a real
   isolation boundary (§7). Make model-driven Cypher read-only.
7. Resolve the air-gap contradiction: either drop cloud extraction + trim cloud
   deps, or make "local-only" an *enforced* mode with clear egress boundaries.
   Fix the `MistralProvider` import (§4.4).
8. Delete the dead path you don't choose (eager runner *or* the queue), so the
   codebase has **one** execution model.

### P2 — Hygiene & trust (ongoing)
9. Rewrite `README.md` to the real architecture; delete `SYSTEM_STATUS.md` or
   regenerate it; fill in `SECURITY.md`.
10. Regenerate `requirements.txt` (UTF-8, no editable local path, pinned,
    trimmed); fix `pyproject.toml` metadata.
11. Add a minimal CI (`ruff` + `mypy` + `pytest`) gating `main`. Make
    `test_deep_agents_e2e.py` a real pytest module; enable `asyncio_mode = auto`.
12. Untrack `chunks/` and `*.pdf`; consolidate to a single CLI; remove scratch/
    probe scripts.

---

## 11. Feature Opportunities (once the base works)

These are worth building **after** P0/P1 — they're where the product's stated
vision (cross-domain "isomorphic" insight) becomes a real differentiator:

- **Isomorphic-cluster detection** — `IsomorphicCluster` is modeled in state but
  no node actually computes cross-domain matches from hyperedge embeddings. This
  is the headline "NEXUS" value prop and is currently unimplemented.
- **PRISMA flow diagram export** — `prisma_flow_diagram` / `audit_export_path`
  fields exist; auto-generate the standard PRISMA 2020 flow figure (identified →
  screened → included) from the audit counts you already track.
- **Real GraphRAG retrieval** — once chunk embeddings are populated (§4.5), wire
  `analysis`/`writing` to retrieve grounded context (vector → Cypher expand)
  instead of passing raw entity lists.
- **Incremental / resumable runs** — you already hash inputs; use it to cache
  extraction and skip re-processing unchanged papers.
- **Screening with confidence + human queue** — `PaperRecord.needs_human_review`
  exists; add an actual title/abstract screening step that routes low-confidence
  papers to the human-in-the-loop UI.
- **Evaluation harness** — a small gold set + metrics (extraction precision/recall
  vs. annotated papers) to make "PRISMA-aligned" a measurable claim, not a label.
- **Observability** — the frontend already has a monitor page; add token/latency/
  cost counters (`ReasoningUsage` is defined but unused) and per-stage timings.

---

## 12. Scorecard

| Area | Grade | Trend |
|---|:---:|---|
| Architecture & vision | B+ | Strong foundation, needs consolidation to one model |
| Core module quality | B+ | Extractor/search/state/validation are real assets |
| End-to-end functionality | D− | Serving path returns no real output today |
| Correctness | C− | Fixable integration-seam bugs, a few latent crashes |
| Efficiency | C | Localized wins available; no deep refactor needed |
| Security | C− | `exec()` + open Cypher + aspirational air-gap |
| Testing & CI | C− | Mocked, dead-path, no gate, collection hazard |
| Docs & hygiene | C− | Docs describe a system that no longer exists |
| Frontend | B+ | Clean and correctly integrated |

**Overall: C / "promising prototype."** The gap between the current state and a
credible v1 is **narrow but load-bearing**: fix the orchestration seam (§4.1–4.3)
and this becomes a demonstrably working, differentiated systematic-review engine.
Leave it unfixed and the impressive surface area (7 agents, PRISMA gates, Neo4j,
SSE UI) is a facade over a pipeline that doesn't deliver a result.

---

*Prepared 2026-07-11. Findings are static-analysis-based; runtime verification was
not possible because project dependencies are not installed in the audit
environment. Re-run the suite and an end-to-end smoke test after P0 fixes to
confirm.*
