"""
tests/integration/test_deep_agents_e2e.py
==========================================
End-to-end integration tests for the native LangGraph ReAct orchestrator loop.
Checks node helper functions, tool connectivity, and orchestrator instantiation.
"""

import asyncio
import os
import sys
import traceback
from dotenv import load_dotenv

load_dotenv()

# Ensure project root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

# ── helpers ──────────────────────────────────────────────────────────────────

GREEN = "\033[92m"
RED = "\033[91m"
YELLOW = "\033[93m"
CYAN = "\033[96m"
RESET = "\033[0m"
BOLD = "\033[1m"

passed = 0
failed = 0
skipped = 0


def _run(coro):
    return asyncio.get_event_loop().run_until_complete(coro)


def report(name, success, detail="", skip=False):
    global passed, failed, skipped
    if skip:
        skipped += 1
        print(f"  {YELLOW}SKIP{RESET}  {name}: {detail}")
    elif success:
        passed += 1
        print(f"  {GREEN}PASS{RESET}  {name}")
    else:
        failed += 1
        print(f"  {RED}FAIL{RESET}  {name}: {detail}")


# ==============================================================================
# TIER 1: Node & Utility Functions (offline, local-first logic)
# ==============================================================================

print(f"\n{BOLD}{CYAN}=== TIER 1: Refactored Node & Utility Functions ==={RESET}\n")

# ── Ingestion Helpers ────────────────────────────────────────────────────────
try:
    from core.nodes.dark_data_ingestion_node import ingest_local_sources, normalize_record

    # Test local scanning without errors
    records = ingest_local_sources(source_patterns=["config/*.yaml"], max_files=5)
    report(
        "Ingestion: ingest_local_sources",
        isinstance(records, list),
        f"staged {len(records)} files"
    )

    # Test normalization
    norm = normalize_record({"paper_id": "test_doc", "text": "This is test jargon text."})
    report(
        "Ingestion: normalize_record",
        norm["paper_id"] == "test_doc" and norm["abstract"] == "This is test jargon text." and norm["included"] is True
    )

except Exception as e:
    report("Ingestion Helpers", False, str(e))

# ── Rosetta Translation Helpers ──────────────────────────────────────────────
try:
    from core.nodes.rosetta_core_node import extract_candidate_terms, map_term_to_principle, translate_records_jargon

    # Candidate extraction
    terms = extract_candidate_terms("This is a simulation telemetry control system.")
    report(
        "Rosetta: extract_candidate_terms",
        "telemetry" in terms and "control" in terms and "this" not in terms
    )

    # Principle mapping
    report("Rosetta: map_term_to_principle telemetry", map_term_to_principle("telemetry") == "signal_acquisition")
    report("Rosetta: map_term_to_principle custom", map_term_to_principle("custom_jargon") == "generalized_system_principle")

except Exception as e:
    report("Rosetta Helpers", False, str(e))


# ==============================================================================
# TIER 2: Live Services Tool Connectivity
# ==============================================================================

print(f"\n{BOLD}{CYAN}=== TIER 2: Tool Connection (live services) ==={RESET}\n")

# ── Neo4j Vector Search ───────────────────────────────────────────────────────
try:
    from core.agent_tools import neo4j_vector_search

    neo4j_pw = os.environ.get("NEO4J_PASSWORD")
    if not neo4j_pw:
        report("Neo4j Vector: search", False, skip=True, detail="NEO4J_PASSWORD not set")
    else:
        results = neo4j_vector_search("machine learning healthcare", limit=5)
        report(
            "Neo4j Vector: search",
            isinstance(results, list),
            f"returned {len(results)} results",
        )

        results_filtered = neo4j_vector_search(
            "methodology", prisma_label="methodology", limit=3
        )
        report(
            "Neo4j Vector: search (PRISMA filter)",
            isinstance(results_filtered, list),
            f"returned {len(results_filtered)} methodology results",
        )

except Exception as e:
    report("Neo4j Vector tools", False, f"{e}\n{traceback.format_exc()}")

# ── Neo4j Query ──────────────────────────────────────────────────────────────
try:
    from core.agent_tools import neo4j_query

    neo4j_pw = os.environ.get("NEO4J_PASSWORD")
    if not neo4j_pw:
        report("Neo4j: query", False, skip=True, detail="NEO4J_PASSWORD not set")
    else:
        records = neo4j_query("RETURN 1 AS ping")
        report(
            "Neo4j: ping",
            isinstance(records, list) and len(records) > 0,
            f"returned {len(records)} records",
        )

except Exception as e:
    report("Neo4j tools", False, f"{e}\n{traceback.format_exc()}")


# ==============================================================================
# TIER 3: Orchestrator Build (native LangGraph)
# ==============================================================================

print(f"\n{BOLD}{CYAN}=== TIER 3: Orchestrator (native LangGraph build) ==={RESET}\n")

try:
    from core.orchestrator import build_orchestrator, ORCHESTRATOR_SYSTEM_PROMPT

    has_tools = "neo4j_vector_search" in ORCHESTRATOR_SYSTEM_PROMPT
    has_prisma = "PRISMA" in ORCHESTRATOR_SYSTEM_PROMPT
    report(
        "Orchestrator: system prompt validation",
        has_tools and has_prisma,
        f"tools={has_tools}, prisma={has_prisma}",
    )

    from core.orchestrator import _build_subagent_configs

    configs = _build_subagent_configs()
    report(
        "Orchestrator: subagent configs count",
        len(configs) == 7,  # including deep-reasoner
        f"built {len(configs)} subagent configs: {[c['name'] for c in configs]}",
    )

    # Verify building compiled orchestrator
    orchestrator = build_orchestrator(model="ollama:qwen2.5:3b")
    report("Orchestrator: native build success", orchestrator is not None)

except Exception as e:
    report("Orchestrator config", False, f"{e}\n{traceback.format_exc()}")


# ==============================================================================
# TIER 4: Graph Runner Mode Check
# ==============================================================================

print(f"\n{BOLD}{CYAN}=== TIER 4: Graph Runner Mode Check ==={RESET}\n")

try:
    from core.graph import run_research_pipeline
    import inspect

    sig = inspect.signature(run_research_pipeline)
    has_mode = "mode" in sig.parameters
    report("graph.py: mode parameter", has_mode)
except Exception as e:
    report("Graph runner", False, str(e))


# ==============================================================================
# SUMMARY
# ==============================================================================

print(f"\n{BOLD}{'═' * 50}{RESET}")
print(
    f"  {GREEN}PASSED: {passed}{RESET}  |  {RED}FAILED: {failed}{RESET}  |  {YELLOW}SKIPPED: {skipped}{RESET}"
)
print(f"{BOLD}{'═' * 50}{RESET}\n")

if failed > 0:
    sys.exit(1)
