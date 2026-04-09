import asyncio
import os
import logging
from core.llm_provider import ChatFastRLM

# Set up logging to see what's happening
logging.basicConfig(level=logging.INFO)

async def test():
    # Mock some env vars if needed
    os.environ["RLM_PRIMARY_MODEL"] = "Qwen/Qwen2.5-1.5B-Instruct"
    os.environ["RLM_MODEL_BASE_URL"] = "http://172.30.177.136:8000/v1"
    
    chat = ChatFastRLM(temperature=0)
    
    prompt = """Context: The user is testing a native reasoning loop.
Query: What is 123 * 456? Calculate it using Python."""
    
    print("Sending query to Native Reasoning Loop...")
    from langchain_core.messages import HumanMessage
    response = await chat.ainvoke([HumanMessage(content=prompt)])
    
    print("\n--- RESPONSE ---")
    print(response.content)
    print("----------------")

if __name__ == "__main__":
    asyncio.run(test())
