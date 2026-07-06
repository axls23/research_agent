# Research Agent System Status

Last verified: 2026-07-06 (test suite run + manual import checks — see below for method).

## Test Suite

- `pytest tests/ --ignore=tests/agents/test_agents.py --ignore=tests/integration/test_deep_agents_e2e.py`
  → **133 passed, 1 skipped** (full run, ~8 minutes; GLiNER/model-adjacent tests are the slow part).
- `tests/agents/test_agents.py` fails to collect: imports a top-level `agents/` package that
  no longer exists (pre-dates the LangGraph multi-agent rewrite). Dead test file.
- `tests/integration/test_deep_agents_e2e.py` is a manual smoke script, not a pytest suite
  (it calls `sys.exit(1)` directly). Its subagent-count check is stale — `core/capabilities.py`
  now registers 8 tiles by design.

## Environment

- **No `.venv` in this repo.** Whatever venv earlier notes reference does not currently exist.
  Confirm which interpreter you're using before running anything
  (`python -c "import sys; print(sys.executable)"`).
- **No `.env` file** and no `GROQ_API_KEY` / `NEO4J_PASSWORD` / `OLLAMA_*` set in the shell as
  of this writing. A live run against Groq or Neo4j will fail auth/connection until these are
  set. `config/config.yaml` defaults to the local `llamacpp` provider (`gemma-4-E2B-it` on
  `:8001`), which doesn't need `GROQ_API_KEY`.
- `api.py` and `core/graph.py` import cleanly with the packages currently installed.

## Servers (manual — start these yourself, nothing runs persistently)

### Backend API (FastAPI)
- Start: `python api.py`
- URL: http://localhost:8000, health check at `/health`
- CORS: allows `http://localhost:3000` only

### Frontend (Next.js)
- Start: `python run_frontend.py` (or `cd frontend-next && npm run dev`)
- URL: http://localhost:3000
- Main route `/` redirects to `/workspace` (NEXUS Workspace — semantic-bridge graph +
  live discovery-run panel wired to `/health`, `/api/backend/monitor`, `/api/chat/stream`)

## Known gaps

- `core/researcher_agent.py` is dead legacy code (old single-agent architecture, pre-dates
  the LangGraph rewrite) with three unimplemented `# TODO` methods. Nothing imports it.
- `frontend-next/lib/nexusEngine.ts`'s graph data (silos/principles/bridges) is hardcoded
  demo content, not live pipeline output.
- `config/config.yaml`'s `knowledge_graph.backend: networkx | neo4j` flag is unused —
  nothing in `core/` reads it or imports networkx. Neo4j is the only backend actually wired.
