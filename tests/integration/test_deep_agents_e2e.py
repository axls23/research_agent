"""
tests/integration/test_deep_agents_e2e.py
==========================================
End-to-end integration tests for the ReAct agent loop.

Tier 1: Agent-level tests (no LLM required)
Tier 2: Tool-level tests (Qdrant/Neo4j connectivity)
Tier 3: Orchestrator test (requires GROQ_API_KEY)
"""

import asyncio
import os
import sys
import traceback
from dotenv import load_dotenv

load_dotenv()

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
# TIER 1: Agent Actions (no LLM, no external services)
# ==============================================================================

print(f"\n{BOLD}{CYAN}=== TIER 1: Agent Actions (offline) ==={RESET}\n")

# ── KnowledgeGraphAgent ──────────────────────────────────────────────────────
try:
    from agents.knowledge_graph_agent import KnowledgeGraphAgent

    kg = KnowledgeGraphAgent()

    # initialize_graph
    r = _run(kg.process({"action": "initialize_graph", "project_name": "Test Project"}))
    report("KG: initialize_graph", r["status"] == "completed" and "graph_id" in r)

    # check_coverage (will return 0s because Qdrant won't have data for this topic)
    r = _run(kg.process({"action": "check_coverage", "topic": "test_topic_xyz"}))
    report(
        "KG: check_coverage",
        r["status"] == "completed" and "coverage" in r,
        f"coverage={r.get('coverage', {})}",
    )

except Exception as e:
    report("KG agent", False, str(e))

# ── AnalysisAgent ────────────────────────────────────────────────────────────
try:
    from agents.analysis_agent import AnalysisAgent

    aa = AnalysisAgent()

    # explore_data
    entities = [
        {"label": "result", "text": "p<0.05"},
        {"label": "methodology", "text": "RCT"},
    ]
    r = _run(aa.process({"action": "explore_data", "entities": entities}))
    report(
        "Analysis: explore_data",
        r["status"] == "completed" and r["entity_count"] == 2,
        f"dist={r.get('label_distribution', {})}",
    )

    # run_statistical_tests (legacy)
    r = _run(aa.process({"action": "run_statistical_tests"}))
    report("Analysis: run_statistical_tests", r["status"] == "completed")

    # create_visualizations (legacy)
    r = _run(aa.process({"action": "create_visualizations"}))
    report("Analysis: create_visualizations", r["status"] == "completed")

except Exception as e:
    report("Analysis agent", False, str(e))

# ── DataProcessingAgent ──────────────────────────────────────────────────────
try:
    from agents.data_processing_agent import DataProcessingAgent

    dp = DataProcessingAgent()

    # prepare_data with string inputs
    r = _run(
        dp.process(
            {
                "action": "prepare_data",
                "documents": ["This is a research abstract about AI in healthcare."],
            }
        )
    )
    report(
        "DataProc: prepare_data (strings)",
        r["status"] == "completed" and r["chunk_count"] == 1,
    )

    # prepare_data with dict inputs
    r = _run(
        dp.process(
            {
                "action": "prepare_data",
                "papers": [
                    {"abstract": "Neural networks for diagnosis.", "paper_id": "p1"}
                ],
            }
        )
    )
    report(
        "DataProc: prepare_data (dicts)",
        r["status"] == "completed" and r["chunk_count"] == 1,
    )

    # extract_text (non-existent file — should still return completed)
    r = _run(dp.process({"action": "extract_text", "file_path": "nonexistent.pdf"}))
    report("DataProc: extract_text (missing file)", r["status"] == "completed")

except Exception as e:
    report("DataProc agent", False, str(e))

# ── WritingAssistantAgent ────────────────────────────────────────────────────
try:
    from agents.writing_assistant_agent import WritingAssistantAgent

    wa = WritingAssistantAgent()

    # synthesize_literature (no LLM — uses fallback)
    r = _run(
        wa.process(
            {
                "action": "synthesize_literature",
                "topic": "AI healthcare",
                "entities": [{"text": "deep learning"}, {"text": "CNN"}],
                "papers": [{"title": "Paper 1"}],
            }
        )
    )
    report(
        "Writing: synthesize_literature",
        r["status"] == "completed" and r.get("paper_count") == 1,
    )

    # detect_gaps
    r = _run(wa.process({"action": "detect_gaps", "topic": "AI healthcare"}))
    report(
        "Writing: detect_gaps",
        r["status"] == "completed" and "domain_coverage" in r,
        f"gaps={len(r.get('gaps', []))}",
    )

    # summarize_results (legacy)
    r = _run(wa.process({"action": "summarize_results"}))
    report("Writing: summarize_results", r["status"] == "completed")

    # generate_outline (legacy)
    r = _run(wa.process({"action": "generate_outline", "topic": "AI"}))
    report("Writing: generate_outline", r["status"] == "completed")

except Exception as e:
    report("Writing agent", False, str(e))


# ==============================================================================
# TIER 2: Tool Wrappers (Qdrant / Neo4j connectivity)
# ==============================================================================

print(f"\n{BOLD}{CYAN}=== TIER 2: Tool Wrappers (live services) ==={RESET}\n")

# ── Qdrant Search ────────────────────────────────────────────────────────────
try:
    from core.agent_tools import qdrant_search

    qdrant_url = os.environ.get("QDRANT_URL")
    if not qdrant_url:
        report("Qdrant: search", False, skip=True, detail="QDRANT_URL not set")
    else:
        results = qdrant_search("machine learning healthcare", limit=5)
        report(
            "Qdrant: search",
            isinstance(results, list),
            f"returned {len(results)} results",
        )

        # With PRISMA label filter
        results_filtered = qdrant_search(
            "methodology", prisma_label="methodology", limit=3
        )
        report(
            "Qdrant: search (PRISMA filter)",
            isinstance(results_filtered, list),
            f"returned {len(results_filtered)} methodology results",
        )

except Exception as e:
    report("Qdrant tools", False, f"{e}\n{traceback.format_exc()}")

# ── Neo4j Query ──────────────────────────────────────────────────────────────
try:
    from core.agent_tools import neo4j_query

    neo4j_pw = os.environ.get("NEO4J_PASSWORD")
    if not neo4j_pw:
        report("Neo4j: query", False, skip=True, detail="NEO4J_PASSWORD not set")
    else:
        # Simple metadata query
        records = neo4j_query("RETURN 1 AS ping")
        report(
            "Neo4j: ping",
            isinstance(records, list) and len(records) > 0,
            f"returned {len(records)} records",
        )

        # PRISMA graph query
        records = neo4j_query(
            "MATCH (n) RETURN labels(n) AS labels, count(n) AS count LIMIT 10"
        )
        report(
            "Neo4j: node count",
            isinstance(records, list),
            f"returned {len(records)} label groups",
        )

except Exception as e:
    report("Neo4j tools", False, f"{e}\n{traceback.format_exc()}")


# ==============================================================================
# TIER 3: Orchestrator Build (requires deepagents + LLM key)
# ==============================================================================

print(f"\n{BOLD}{CYAN}=== TIER 3: Orchestrator (deepagents SDK) ==={RESET}\n")

# ── Deep Agents import check ─────────────────────────────────────────────────
try:
    from deepagents import create_deep_agent

    report("deepagents: import", True)
except ImportError:
    report("deepagents: import", False, "pip install deepagents")

# ── Build orchestrator ───────────────────────────────────────────────────────
try:
    from core.orchestrator import build_orchestrator, ORCHESTRATOR_SYSTEM_PROMPT

    # Verify system prompt contains key elements
    has_subagents = "literature-search" in ORCHESTRATOR_SYSTEM_PROMPT
    has_tools = "qdrant_search" in ORCHESTRATOR_SYSTEM_PROMPT
    has_prisma = "PRISMA" in ORCHESTRATOR_SYSTEM_PROMPT
    report(
        "Orchestrator: system prompt",
        has_subagents and has_tools and has_prisma,
        f"subagents={has_subagents}, tools={has_tools}, prisma={has_prisma}",
    )

    # Check subagent configs
    from core.orchestrator import _build_subagent_configs

    configs = _build_subagent_configs()
    report(
        "Orchestrator: subagent configs",
        len(configs) == 5,
        f"built {len(configs)} subagent configs: {[c['name'] for c in configs]}",
    )

    # Verify each subagent has tools
    all_have_tools = all(len(c.get("tools", [])) > 0 for c in configs)
    report(
        "Orchestrator: subagent tools",
        all_have_tools,
        f"all subagents have tools assigned",
    )

except Exception as e:
    report("Orchestrator config", False, f"{e}\n{traceback.format_exc()}")

# ── Build & invoke (requires API key) ────────────────────────────────────────
groq_key = os.environ.get("GROQ_API_KEY")
if not groq_key:
    report(
        "Orchestrator: build",
        False,
        skip=True,
        detail="GROQ_API_KEY not set — set it in .env to run full E2E",
    )
else:
    try:
        orchestrator = build_orchestrator(model="groq:llama-3.3-70b-versatile")
        report("Orchestrator: build", orchestrator is not None)

        # Quick invoke with a simple query
        result = _run(
            orchestrator.ainvoke(
                {
                    "messages": [
                        {
                            "role": "user",
                            "content": (
                                "Search for 3 papers on 'transformer architectures in medical imaging'. "
                                "Extract PRISMA entities and check coverage. Report findings."
                            ),
                        }
                    ]
                }
            )
        )
        report(
            "Orchestrator: invoke",
            result is not None and "messages" in result,
            f"got {len(result.get('messages', []))} messages back",
        )

    except Exception as e:
        report("Orchestrator: build/invoke", False, f"{e}\n{traceback.format_exc()}")


# ==============================================================================
# TIER 4: Graph Runner (dual mode)
# ==============================================================================

print(f"\n{BOLD}{CYAN}=== TIER 4: Graph Runner Mode Check ==={RESET}\n")

try:
    from core.graph import run_research_pipeline
    import inspect

    sig = inspect.signature(run_research_pipeline)
    has_mode = "mode" in sig.parameters
    report("graph.py: mode parameter", has_mode)

    if has_mode:
        mode_param = sig.parameters["mode"]
        report(
            "graph.py: mode default",
            str(mode_param.default) == "deterministic",
            f"default={mode_param.default}",
        )
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
