"""
Research Agent Startup Script
Phase 1: Core Infrastructure Implementation
Updated: October 29, 2025

Main entry point to run the integrated research agent system.
"""

import os
import sys
import asyncio
import logging
import argparse
import json
import re
from pathlib import Path
from urllib import error, request
from typing import List, Tuple

# Add project to Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from dotenv import load_dotenv

load_dotenv()

# Setup basic logging locally since research_agent/utils/logger was removed
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger("run_research_agent")

from core.graph import run_research_pipeline


_DEFAULT_TOPIC = "Quantum Machine Learning algorithms for simulating molecular dynamics"
_DEFAULT_GOALS = [
    "Analyze the computational speedup of quantum algorithms over classical counterparts",
    "Identify leading noise-mitigation strategies in near-term quantum hardware (NISQ)",
]


def _normalize_ollama_model(model: str) -> str:
    """Convert raw model input into an Ollama-qualified model reference."""
    value = (model or "").strip()
    default_model = os.getenv("OLLAMA_MODEL", "qwen2.5:3b")
    if default_model.lower().startswith("ollama:"):
        default_model = default_model.split(":", 1)[1]

    if not value:
        return f"ollama:{default_model}"
    if value.lower().startswith("ollama:"):
        return value
    if ":" in value:
        prefix = value.split(":", 1)[0].lower()
        known_non_ollama = {"groq", "fast_rlm", "fast-rlm", "openai", "anthropic", "airllm"}
        if prefix in known_non_ollama:
            logger.warning(
                "Non-ollama model '%s' requested; forcing default Ollama model 'ollama:%s'.",
                value,
                default_model,
            )
            return f"ollama:{default_model}"
    return f"ollama:{value}"


def check_requirements(selected_model: str):
    """Check if all requirements are met"""
    base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
    if base_url.endswith("/v1"):
        base_url = base_url[:-3]

    try:
        req = request.Request(f"{base_url}/api/tags", method="GET")
        with request.urlopen(req, timeout=3) as resp:
            payload = json.loads(resp.read().decode("utf-8", errors="replace"))
        model_names = [m.get("name", "") for m in payload.get("models", []) if isinstance(m, dict)]
        requested_model_name = selected_model.split(":", 1)[1] if ":" in selected_model else selected_model
        if requested_model_name and requested_model_name not in model_names:
            logger.warning(
                "Requested model '%s' not found in Ollama tags at %s. Available: %s",
                requested_model_name,
                base_url,
                ", ".join(model_names[:10]) or "<none>",
            )
        logger.info("Ollama endpoint reachable at %s", base_url)
        return True
    except error.URLError as e:
        logger.error("Cannot reach Ollama at %s: %s", base_url, e)
        return False
    except Exception as e:
        logger.error("Ollama readiness check failed: %s", e)
        return False


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
    text = re.sub(r"[?.!]+$", "", text).strip()
    return text


def _resolve_research_inputs(args: argparse.Namespace) -> Tuple[str, str, List[str]]:
    env_topic = os.getenv("RESEARCH_TOPIC", "")
    topic_candidate = args.topic or env_topic
    topic = _normalize_topic(topic_candidate)

    goals: List[str] = [g.strip() for g in (args.goal or []) if g and g.strip()]

    if not topic:
        return "Quantum ML Analysis", _DEFAULT_TOPIC, list(_DEFAULT_GOALS)

    project_name = args.project_name.strip() if (args.project_name or "").strip() else "Research Analysis"

    if not goals:
        goals = [
            f"Summarize key methods and evidence for: {topic}",
            f"Identify major limitations and open challenges for: {topic}",
        ]

    return project_name, topic, goals


async def main_async(args):
    """Async execution of the pipeline"""
    requested_mode = (args.mode or "agentic").lower()
    if requested_mode != "agentic":
        logger.warning("Mode '%s' requested; routing to agentic mode.", requested_mode)

    resolved_model = _normalize_ollama_model(args.model or os.getenv("OLLAMA_MODEL", "qwen2.5:3b"))

    logger.info("Using Mode: agentic")
    logger.info(f"Using Rigor Level: {args.rigor}")
    logger.info("Using Model: %s", resolved_model)

    project_name, topic, goals = _resolve_research_inputs(args)

    logger.info(f"Running research pipeline on: '{topic}'")

    try:
        result_state = await run_research_pipeline(
            project_name=project_name,
            research_topic=topic,
            research_goals=goals,
            rigor_level=args.rigor,
            interactive=False,  # Set to False so it doesn't wait indefinitely in tests
            mode="agentic",
            agentic_model=resolved_model,
        )

        logger.info("[SUCCESS] Research Pipeline Completed!")

        if "audit_export_path" in result_state:
            logger.info(f"Audit exported to: {result_state['audit_export_path']}")

    except Exception as e:
        logger.error(f"Pipeline failed: {e}", exc_info=True)
        sys.exit(1)


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description="Run the Research Agent System")
    parser.add_argument(
        "--mode",
        type=str,
        choices=["default", "langgraph", "agentic"],
        default="agentic",
        help="Execution mode alias. All values route to agentic LangGraph deep-agent mode.",
    )
    parser.add_argument(
        "--rigor",
        type=str,
        choices=["exploratory", "prisma", "cochrane"],
        default="prisma",
        help="Methodological rigor (exploratory, prisma, cochrane).",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=os.getenv("OLLAMA_MODEL", "qwen2.5:3b"),
        help="Ollama model name or fully qualified ollama:<model> (default: qwen2.5:3b).",
    )
    parser.add_argument(
        "--project-name",
        type=str,
        default=os.getenv("RESEARCH_PROJECT_NAME", "Research Analysis"),
        help="Project name used in audit artifacts.",
    )
    parser.add_argument(
        "--topic",
        type=str,
        default=os.getenv("RESEARCH_TOPIC", ""),
        help=(
            "Research topic. If omitted, falls back to demo topic. "
            "You can also pass natural chat text and the topic will be normalized."
        ),
    )
    parser.add_argument(
        "--goal",
        action="append",
        default=[],
        help="Research goal. Repeat --goal for multiple goals.",
    )
    args = parser.parse_args()

    resolved_model = _normalize_ollama_model(args.model)
    args.model = resolved_model

    # Expose to other components via environment variables
    os.environ["RESEARCH_AGENT_MODE"] = "agentic"
    os.environ["RESEARCH_AGENT_RIGOR"] = args.rigor
    os.environ["AGENTIC_MODEL"] = resolved_model

    print(
        f"""
    =======================================
    |       Research Agent System         |
    |   AI-Powered Research Assistant     |
    =======================================
    Mode: agentic | Rigor: {args.rigor} | Model: {resolved_model}
    """
    )

    # Check requirements
    if not check_requirements(resolved_model):
        sys.exit(1)

    # Create necessary directories
    os.makedirs("outputs", exist_ok=True)
    os.makedirs("logs", exist_ok=True)

    asyncio.run(main_async(args))


if __name__ == "__main__":
    main()
