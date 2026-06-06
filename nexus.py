#!/usr/bin/env python3
"""
nexus.py — Master CLI Entry Point
===================================
Unified terminal interface for the NEXUS Research Agent harness.

Usage:
    python nexus.py run --topic "Your research topic"
    python nexus.py run --topic "Your topic" --rigor cochrane --model qwen2.5:7b
    python nexus.py check
    python nexus.py status

Environment Variables (override via .env or shell):
    OLLAMA_BASE_URL     Ollama API endpoint   (default: http://localhost:11434)
    VLLM_BASE_URL       vLLM OpenAI-compat URL (default: http://localhost:8000/v1)
    VLLM_MODEL          Default vLLM model     (default: vllm:google/gemma-4-26B-A4B-it)
    NEO4J_URI           Neo4j bolt URI
    NEO4J_USER          Neo4j username
    NEO4J_PASSWORD      Neo4j password
    RESEARCH_AGENT_RIGOR  Default rigor level  (default: prisma)
    NEXUS_LOCAL_ONLY    Enforce air-gap policy (default: true)
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import re
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from urllib import error, request

# ---------------------------------------------------------------------------
# Bootstrap: ensure project root is on sys.path and .env is loaded
# ---------------------------------------------------------------------------

_PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(_PROJECT_ROOT))

# Fix Windows terminal encoding — force UTF-8 output
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[attr-defined]
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[attr-defined]
    except (AttributeError, OSError):
        pass

try:
    from dotenv import load_dotenv

    load_dotenv(_PROJECT_ROOT / ".env")
except ImportError:
    pass  # dotenv optional in minimal deployments

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

_LOG_FORMAT = "%(asctime)s │ %(levelname)-7s │ %(name)s │ %(message)s"
_LOG_DATE = "%H:%M:%S"


def _configure_logging(verbosity: int = 0) -> None:
    level = logging.WARNING
    if verbosity == 1:
        level = logging.INFO
    elif verbosity >= 2:
        level = logging.DEBUG

    logging.basicConfig(level=level, format=_LOG_FORMAT, datefmt=_LOG_DATE)
    # Quiet noisy libs unless explicitly requested
    if verbosity < 2:
        for name in ("httpx", "httpcore", "urllib3", "neo4j"):
            logging.getLogger(name).setLevel(logging.WARNING)


logger = logging.getLogger("nexus")


# ---------------------------------------------------------------------------
# Console styling (lightweight, no external deps beyond stdlib)
# ---------------------------------------------------------------------------


class _Style:
    """ANSI escape helpers for styled terminal output."""

    RESET = "\033[0m"
    BOLD = "\033[1m"
    DIM = "\033[2m"
    GREEN = "\033[92m"
    YELLOW = "\033[93m"
    RED = "\033[91m"
    CYAN = "\033[96m"
    MAGENTA = "\033[95m"
    BLUE = "\033[94m"
    WHITE = "\033[97m"

    @staticmethod
    def supports_color() -> bool:
        if os.getenv("NO_COLOR"):
            return False
        if sys.platform == "win32":
            return os.getenv("TERM") or os.getenv("WT_SESSION") or True
        return hasattr(sys.stdout, "isatty") and sys.stdout.isatty()

    @classmethod
    def c(cls, text: str, *codes: str) -> str:
        if not cls.supports_color():
            return text
        prefix = "".join(codes)
        return f"{prefix}{text}{cls.RESET}"


S = _Style


def _banner() -> str:
    return S.c(
        r"""
    ╔══════════════════════════════════════════════════════╗
    ║           N  E  X  U  S     E  N  G  I  N  E        ║
    ║       Sovereign Research Agentic Harness v2.0        ║
    ╚══════════════════════════════════════════════════════╝
""",
        S.CYAN,
        S.BOLD,
    )


def _section(title: str) -> str:
    return S.c(f"\n  ── {title} ", S.BOLD, S.MAGENTA) + S.c("─" * (48 - len(title)), S.DIM)


# ---------------------------------------------------------------------------
# Pre-flight checks
# ---------------------------------------------------------------------------


def _check_ollama() -> Tuple[bool, str, List[str]]:
    """Probe the Ollama API for reachability and list available models."""
    base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
    if base_url.endswith("/v1"):
        base_url = base_url[:-3]

    try:
        req = request.Request(f"{base_url}/api/tags", method="GET")
        with request.urlopen(req, timeout=5) as resp:
            payload = json.loads(resp.read().decode("utf-8", errors="replace"))
        models = [
            m.get("name", "")
            for m in payload.get("models", [])
            if isinstance(m, dict) and m.get("name")
        ]
        return True, base_url, models
    except (error.URLError, OSError) as e:
        return False, base_url, []
    except Exception as e:
        return False, base_url, []


def _check_neo4j() -> Tuple[bool, str]:
    """Test Neo4j connectivity."""
    uri = os.getenv("NEO4J_URI", "bolt://localhost:7687")
    password = os.getenv("NEO4J_PASSWORD", "")
    if not password:
        return False, f"{uri} (no password configured)"
    try:
        from neo4j import GraphDatabase

        driver = GraphDatabase.driver(uri, auth=(os.getenv("NEO4J_USER", "neo4j"), password))
        driver.verify_connectivity()
        driver.close()
        return True, uri
    except Exception as e:
        return False, f"{uri} ({e})"


def _check_vllm() -> Tuple[bool, str]:
    """Test vLLM endpoint reachability."""
    base_url = os.getenv("VLLM_BASE_URL", "http://localhost:8000/v1")
    try:
        req = request.Request(f"{base_url}/models", method="GET")
        with request.urlopen(req, timeout=3) as resp:
            _ = resp.read()
        return True, base_url
    except Exception:
        return False, base_url


def cmd_check(_args: argparse.Namespace) -> None:
    """Run pre-flight infrastructure checks."""
    print(_banner())
    print(_section("Infrastructure Checks"))

    # Ollama
    ok, url, models = _check_ollama()
    status = S.c("✓ ONLINE", S.GREEN) if ok else S.c("✗ OFFLINE", S.RED)
    print(f"  Ollama      {status}  {S.c(url, S.DIM)}")
    if ok and models:
        print(f"              Models: {S.c(', '.join(models[:8]), S.CYAN)}")

    # Neo4j
    ok, detail = _check_neo4j()
    status = S.c("✓ ONLINE", S.GREEN) if ok else S.c("✗ OFFLINE", S.RED)
    print(f"  Neo4j       {status}  {S.c(detail, S.DIM)}")

    # vLLM
    ok, detail = _check_vllm()
    status = S.c("✓ ONLINE", S.GREEN) if ok else S.c("○ OFFLINE", S.YELLOW)
    print(f"  vLLM        {status}  {S.c(detail, S.DIM)}")

    # Air-gap policy
    from core.llm_provider import local_only_enabled

    policy = S.c("ENFORCED", S.GREEN) if local_only_enabled() else S.c("DISABLED", S.YELLOW)
    print(f"  Air-Gap     {policy}")
    print()


# ---------------------------------------------------------------------------
# Topic / input normalisation
# ---------------------------------------------------------------------------


def _normalize_topic(raw: str) -> str:
    text = " ".join((raw or "").strip().split())
    if not text:
        return ""
    explicit = re.search(r"(?:research\s+topic|topic)\s*[:=-]\s*(.+)", text, re.IGNORECASE)
    if explicit:
        text = explicit.group(1).strip()
    text = re.sub(
        r"^(can\s+you|could\s+you|please|i\s+want\s+to|i\s+need\s+to|help\s+me)\s+",
        "",
        text,
        flags=re.IGNORECASE,
    )
    return re.sub(r"[?.!]+$", "", text).strip()


def _normalize_model(model: str) -> str:
    """Resolve raw model input into an qualified reference."""
    value = (model or "").strip()
    default = os.getenv("VLLM_MODEL", "vllm:google/gemma-4-26B-A4B-it")

    if not value:
        return default
    if value.lower().startswith("ollama:") or value.lower().startswith("vllm:"):
        return value
    if ":" in value:
        prefix = value.split(":", 1)[0].lower()
        blocked = {"groq", "openai", "anthropic", "mistral", "airllm"}
        if prefix in blocked:
            logger.warning("Provider '%s' blocked by air-gap policy; using ollama:%s", prefix, default)
            return f"ollama:{default}"
        if prefix in {"fast_rlm", "fast-rlm"}:
            return value
    return f"ollama:{value}"


# ---------------------------------------------------------------------------
# Run command
# ---------------------------------------------------------------------------


def cmd_run(args: argparse.Namespace) -> None:
    """Execute the full research pipeline."""
    print(_banner())

    # Resolve inputs
    topic = _normalize_topic(args.topic)
    if not topic:
        topic = "Quantum Machine Learning algorithms for simulating molecular dynamics"
        print(S.c("  ⚡ No topic provided — using demo topic.", S.YELLOW))

    model = _normalize_model(args.model)
    rigor = args.rigor
    project = args.project_name or "Research Analysis"

    goals: List[str] = [g.strip() for g in (args.goal or []) if g and g.strip()]
    if not goals:
        goals = [
            f"Summarize key methods and evidence for: {topic}",
            f"Identify major limitations and open challenges for: {topic}",
        ]

    # Set environment
    os.environ["RESEARCH_AGENT_MODE"] = "agentic"
    os.environ["RESEARCH_AGENT_RIGOR"] = rigor
    os.environ["AGENTIC_MODEL"] = model
    os.makedirs("outputs", exist_ok=True)
    os.makedirs("logs", exist_ok=True)

    # Display config
    print(_section("Configuration"))
    print(f"  Project     {S.c(project, S.WHITE, S.BOLD)}")
    print(f"  Topic       {S.c(topic, S.WHITE, S.BOLD)}")
    print(f"  Rigor       {S.c(rigor.upper(), S.CYAN)}")
    print(f"  Model       {S.c(model, S.CYAN)}")
    print(f"  Goals       {S.c(str(len(goals)), S.CYAN)} configured")
    for i, g in enumerate(goals, 1):
        print(f"              {S.c(f'{i}.', S.DIM)} {g}")
    print()

    # Pre-flight
    if not args.skip_checks:
        print(_section("Pre-flight"))
        if model.startswith("vllm:"):
            ok, url = _check_vllm()
            if not ok:
                print(S.c(f"  ⚠ Cannot reach vLLM server at {url}. Is it running?", S.YELLOW))
            else:
                print(S.c(f"  ✓ vLLM server online at {url}.", S.GREEN))
        else:
            ok, url, models = _check_ollama()
            if not ok:
                print(S.c(f"  ✗ Cannot reach Ollama at {url}. Start Ollama and retry.", S.RED))
                sys.exit(1)
            model_name = model.split(":", 1)[1] if ":" in model else model
            if model_name and model_name not in models:
                available = ", ".join(models[:6]) or "<none>"
                print(S.c(f"  ⚠ Model '{model_name}' not found. Available: {available}", S.YELLOW))
                print(S.c(f"    Pull it with: ollama pull {model_name}", S.DIM))
            else:
                print(S.c(f"  ✓ Ollama online, model '{model_name}' available.", S.GREEN))
        print()

    # Execute
    print(_section("Pipeline Execution"))
    print(S.c("  Launching agentic pipeline...\n", S.CYAN))
    t0 = time.monotonic()

    async def _execute() -> Dict[str, Any]:
        from core.graph import run_research_pipeline

        return await run_research_pipeline(
            project_name=project,
            research_topic=topic,
            research_goals=goals,
            rigor_level=rigor,
            interactive=not args.non_interactive,
            mode="agentic",
            agentic_model=model,
        )

    try:
        result = asyncio.run(_execute())
        elapsed = time.monotonic() - t0
        print()
        print(_section("Results"))
        print(S.c(f"  ✓ Pipeline completed in {elapsed:.1f}s", S.GREEN, S.BOLD))

        if isinstance(result, dict):
            if "papers_included" in result:
                print(f"    Papers included:    {result.get('papers_included', 'N/A')}")
            messages = result.get("messages", [])
            if messages:
                last = messages[-1] if messages else None
                if hasattr(last, "content"):
                    content = last.content
                elif isinstance(last, dict):
                    content = last.get("content", "")
                else:
                    content = str(last) if last else ""
                if content:
                    # Truncate for terminal display
                    preview = content[:500] + ("..." if len(content) > 500 else "")
                    print(f"\n  {S.c('Final Output:', S.BOLD)}")
                    print(f"  {preview}")

            if "audit_export_path" in result:
                print(f"\n    Audit log: {S.c(result['audit_export_path'], S.CYAN)}")

            agentic_val = result.get("agentic_validation", {})
            stages = agentic_val.get("stages", {})
            if stages:
                print(f"\n  {S.c('Stage Summary:', S.BOLD)}")
                for name, info in stages.items():
                    v = info.get("validation", {})
                    passed = v.get("passed", True) if v else True
                    icon = S.c("✓", S.GREEN) if passed else S.c("✗", S.RED)
                    print(f"    {icon} {name}")

        print()

    except KeyboardInterrupt:
        print(S.c("\n  Pipeline interrupted by user.", S.YELLOW))
        sys.exit(130)
    except RuntimeError as e:
        elapsed = time.monotonic() - t0
        print(S.c(f"\n  ✗ Pipeline failed after {elapsed:.1f}s: {e}", S.RED))
        sys.exit(1)
    except Exception as e:
        elapsed = time.monotonic() - t0
        logger.exception("Pipeline crashed")
        print(S.c(f"\n  ✗ Unexpected error after {elapsed:.1f}s: {e}", S.RED))
        sys.exit(1)


# ---------------------------------------------------------------------------
# Status command
# ---------------------------------------------------------------------------


def cmd_status(_args: argparse.Namespace) -> None:
    """Show system status and configuration."""
    print(_banner())
    print(_section("Environment"))
    env_keys = [
        ("OLLAMA_BASE_URL", "http://localhost:11434"),
        ("OLLAMA_MODEL", "qwen2.5:3b"),
        ("VLLM_BASE_URL", "http://localhost:8000/v1"),
        ("NEO4J_URI", "bolt://localhost:7687"),
        ("NEXUS_LOCAL_ONLY", "true"),
        ("RESEARCH_AGENT_RIGOR", "prisma"),
    ]
    for key, default in env_keys:
        val = os.getenv(key, "")
        if val:
            print(f"  {key:<25s} = {S.c(val, S.CYAN)}")
        else:
            print(f"  {key:<25s} = {S.c(f'(default: {default})', S.DIM)}")

    print(_section("Project Structure"))
    dirs = ["core", "core/nodes", "config", "outputs", "logs", "tests"]
    for d in dirs:
        p = _PROJECT_ROOT / d
        exists = p.exists()
        icon = S.c("✓", S.GREEN) if exists else S.c("○", S.DIM)
        print(f"  {icon} {d}/")
    print()


# ---------------------------------------------------------------------------
# Worker command
# ---------------------------------------------------------------------------

def cmd_worker(args: argparse.Namespace) -> None:
    """Run a background worker for a specific subagent."""
    from core.job_queue import get_next_job, complete_job, fail_job
    import time
    
    agent_name = args.agent
    print(_banner())
    print(_section(f"Worker: {agent_name}"))
    print(S.c(f"  Polling for '{agent_name}' jobs...", S.CYAN))
    
    while True:
        job = get_next_job(agent_name)
        if not job:
            if not getattr(args, 'daemon', False):
                print(S.c("  No pending jobs. Exiting.", S.DIM))
                break
            time.sleep(5)
            continue
            
        job_id, payload = job
        print(S.c(f"\n  [Job {job_id}] Started execution", S.GREEN))
        try:
            from core.orchestrator import _build_subagent_configs
            
            configs = _build_subagent_configs()
            cfg = next((c for c in configs if c["name"] == agent_name), None)
            if not cfg:
                raise ValueError(f"Unknown agent: {agent_name}")
                
            from langgraph.prebuilt import create_react_agent
            runnable = create_react_agent(cfg["model"], cfg["tools"], prompt=cfg["system_prompt"])
            
            query = payload.get("query", "")
            response = runnable.invoke({"messages": [("user", query)]})
            messages = response.get("messages", [])
            content = messages[-1].content if messages else "No response"
            
            complete_job(job_id, {"result": content})
            print(S.c(f"  [Job {job_id}] Completed successfully", S.GREEN))
        except Exception as e:
            logger.exception(f"Job {job_id} failed")
            fail_job(job_id, str(e))
            print(S.c(f"  [Job {job_id}] Failed: {e}", S.RED))
            
        if not getattr(args, 'daemon', False):
            break

# ---------------------------------------------------------------------------
# Argument parser
# ---------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="nexus",
        description="NEXUS Research Agent — Sovereign Agentic Harness CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python nexus.py run --topic "CRISPR delivery mechanisms" --rigor prisma
  python nexus.py run --topic "Quantum ML" --model qwen2.5:7b -vv
  python nexus.py check
  python nexus.py status
        """,
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="count",
        default=0,
        help="Increase log verbosity (-v info, -vv debug).",
    )

    sub = parser.add_subparsers(dest="command", help="Available commands")

    # --- run ---
    run_p = sub.add_parser("run", help="Execute the full research pipeline.")
    run_p.add_argument(
        "--topic",
        type=str,
        default=os.getenv("RESEARCH_TOPIC", ""),
        help="Research topic (natural language). Falls back to demo topic if omitted.",
    )
    run_p.add_argument(
        "--rigor",
        type=str,
        choices=["exploratory", "prisma", "cochrane"],
        default=os.getenv("RESEARCH_AGENT_RIGOR", "prisma"),
        help="Methodological rigor level (default: prisma).",
    )
    run_p.add_argument(
        "--model",
        type=str,
        default=os.getenv("VLLM_MODEL", "vllm:google/gemma-4-26B-A4B-it"),
        help="vLLM or Ollama model name (default: vllm:google/gemma-4-26B-A4B-it).",
    )
    run_p.add_argument(
        "--project-name",
        type=str,
        default=os.getenv("RESEARCH_PROJECT_NAME", "Research Analysis"),
        help="Project name for audit trail.",
    )
    run_p.add_argument(
        "--goal",
        action="append",
        default=None,
        help="Research goal. Repeat --goal for multiple goals.",
    )
    run_p.add_argument(
        "--skip-checks",
        action="store_true",
        help="Skip pre-flight infrastructure checks.",
    )
    run_p.add_argument(
        "--non-interactive",
        action="store_true",
        help="Disable interactive human-in-the-loop prompts.",
    )
    run_p.set_defaults(func=cmd_run)

    # --- check ---
    check_p = sub.add_parser("check", help="Run pre-flight infrastructure checks.")
    check_p.set_defaults(func=cmd_check)

    # --- status ---
    status_p = sub.add_parser("status", help="Show environment and project status.")
    status_p.set_defaults(func=cmd_status)

    # --- worker ---
    worker_p = sub.add_parser("worker", help="Run a background subagent worker.")
    worker_p.add_argument("--agent", type=str, required=True, help="Subagent name (e.g. data-processing)")
    worker_p.add_argument("--daemon", action="store_true", help="Run continuously")
    worker_p.set_defaults(func=cmd_worker)

    return parser


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    _configure_logging(args.verbose)

    if not args.command:
        parser.print_help()
        sys.exit(0)

    args.func(args)


if __name__ == "__main__":
    main()
