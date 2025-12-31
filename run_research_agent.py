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
from pathlib import Path
import webbrowser
import uvicorn
from multiprocessing import Process
import time

# Add project to Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from research_agent.utils.logger import setup_logging
from research_agent.utils.config import config
from research_agent.api.server import app

# Setup logging
logger = setup_logging()


def check_requirements():
    """Check if all requirements are met"""
    required_env_vars = [
        "OPENAI_API_KEY",  # For paperconstructor
    ]
    
    missing = []
    for var in required_env_vars:
        if not os.getenv(var):
            missing.append(var)
    
    if missing:
        logger.error(f"Missing required environment variables: {missing}")
        logger.info("Please set the following environment variables:")
        for var in missing:
            logger.info(f"  - {var}")
        return False
    
    return True


def start_api_server():
    """Start the FastAPI server"""
    logger.info("Starting Research Agent API server...")
    
    # Configure server
    host = config.get("api_host", "0.0.0.0")
    port = config.get("api_port", 8000)
    
    # Run server
    uvicorn.run(
        "research_agent.api.server:app",
        host=host,
        port=port,
        reload=True,
        log_level="info"
    )


def open_frontend():
    """Open the frontend in default browser"""
    time.sleep(3)  # Wait for server to start
    
    frontend_path = project_root / "frontend" / "index.html"
    if frontend_path.exists():
        logger.info(f"Opening frontend: {frontend_path}")
        webbrowser.open(f"file:///{frontend_path}")
    else:
        logger.error(f"Frontend not found at {frontend_path}")


def main():
    """Main entry point"""
    print("""
    ╔═══════════════════════════════════════╗
    ║       Research Agent System           ║
    ║   AI-Powered Research Assistant       ║
    ╚═══════════════════════════════════════╝
    """)
    
    # Check requirements
    if not check_requirements():
        sys.exit(1)
    
    # Create necessary directories
    os.makedirs("outputs", exist_ok=True)
    os.makedirs("logs", exist_ok=True)
    
    try:
        # Start API server in separate process
        api_process = Process(target=start_api_server)
        api_process.start()
        
        # Open frontend
        open_frontend()
        
        logger.info("Research Agent system is running!")
        logger.info("API Server: http://localhost:8000")
        logger.info("API Docs: http://localhost:8000/docs")
        logger.info("\nPress Ctrl+C to stop the server\n")
        
        # Keep main process running
        api_process.join()
        
    except KeyboardInterrupt:
        logger.info("\nShutting down Research Agent system...")
        if api_process.is_alive():
            api_process.terminate()
            api_process.join()
        logger.info("Goodbye!")
    except Exception as e:
        logger.error(f"Error running Research Agent: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
