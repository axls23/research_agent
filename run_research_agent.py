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
from pathlib import Path

# Add project to Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from dotenv import load_dotenv
load_dotenv()

# Setup basic logging locally since research_agent/utils/logger was removed
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("run_research_agent")

from core.graph import run_research_pipeline

def check_requirements():
    """Check if all requirements are met"""
    # Just check if at least one LLM key is set, though Groq is primary
    if not os.getenv("GROQ_API_KEY") and not os.getenv("MISTRAL_API_KEY") and not os.getenv("OPENAI_API_KEY"):
        logger.warning("No API keys found (GROQ_API_KEY, MISTRAL_API_KEY, OPENAI_API_KEY). Operations requiring an LLM may fail.")
    return True

async def main_async(args):
    """Async execution of the pipeline"""
    logger.info(f"Using Mode: {args.mode}")
    logger.info(f"Using Rigor Level: {args.rigor}")
    
    # In a real use case, these would be passed via CLI args as well, 
    # but for a simple demo test, we can use placeholder topics.
    project_name = "LangGraph Integration Test"
    topic = "The impact of Multi-Agent Architectures on Academic Research"
    goals = [
        "Identify core architectures used in literature",
        "Evaluate the reduction in information overload"
    ]
    
    logger.info(f"Running research pipeline on: '{topic}'")
    
    try:
        result_state = await run_research_pipeline(
            project_name=project_name,
            research_topic=topic,
            research_goals=goals,
            rigor_level=args.rigor,
            interactive=False  # Set to False so it doesn't wait indefinitely in tests
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
    parser.add_argument("--mode", type=str, choices=["default", "langgraph"], default="langgraph",
                        help="Execution mode (default or langgraph).")
    parser.add_argument("--rigor", type=str, choices=["exploratory", "prisma", "cochrane"], default="prisma",
                        help="Methodological rigor (exploratory, prisma, cochrane).")
    args = parser.parse_args()
    
    # Expose to other components via environment variables
    os.environ["RESEARCH_AGENT_MODE"] = args.mode
    os.environ["RESEARCH_AGENT_RIGOR"] = args.rigor

    print(f"""
    =======================================
    |       Research Agent System         |
    |   AI-Powered Research Assistant     |
    =======================================
    Mode: {args.mode} | Rigor: {args.rigor}
    """)
    
    # Check requirements
    if not check_requirements():
        sys.exit(1)
    
    # Create necessary directories
    os.makedirs("outputs", exist_ok=True)
    os.makedirs("logs", exist_ok=True)
    
    if args.mode == "langgraph":
        asyncio.run(main_async(args))
    else:
        logger.info("Default mode selected, but API server is not available. Try --mode langgraph")

if __name__ == "__main__":
    main()
