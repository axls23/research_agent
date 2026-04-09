import asyncio
import os
import sys
from pathlib import Path

# Add project to path
sys.path.insert(0, str(Path(__file__).parent))

from core.orchestrator import run_agentic_pipeline

async def main():
    print("=========================================")
    print("Testing Agentic Pipeline with AirLLM")
    print("=========================================")
    
    try:
        # Note: AirLLM needs the model to be downloaded locally.
        # This might take some time on the first run.
        result = await run_agentic_pipeline(
            project_name="AirLLM Local Test",
            research_topic="Impact of fine-tuning local LLMs",
            research_goals=["Evaluate memory usage", "Assess reasoning performance"],
            model="airllm:Qwen/Qwen2.5-7B-Instruct" # using a smaller model to test if 70b isn't downloaded yet
        )
        
        print("\n[SUCCESS] Pipeline completed successfully!")
        print(f"Result keys: {result.keys() if isinstance(result, dict) else type(result)}")
        
    except Exception as e:
        print(f"\n[ERROR] Pipeline failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(main())
