import os
import sys
import json
import logging
import asyncio
from typing import Dict, Any, List
from dotenv import load_dotenv

# Load environment variables from .env
load_dotenv()

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import uvicorn

import fast_rlm

# Global environment overrides for Fast-RLM to use local vLLM serving from WSL
os.environ["RLM_MODEL_API_KEY"] = "dummy-local-key"
os.environ["RLM_MODEL_BASE_URL"] = "http://172.30.177.136:8000/v1"

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("fast_rlm_proxy")

app = FastAPI(title="Fast-RLM OpenAI Proxy")

class Message(BaseModel):
    role: str
    content: str

class ChatCompletionRequest(BaseModel):
    model: str
    messages: List[Message]
    temperature: float = 0.7
    max_tokens: int = 4096

@app.post("/v1/chat/completions")
async def chat_completions(req: ChatCompletionRequest):
    """
    OpenAI-compatible endpoint for LiteLLM to use Fast-RLM backend.
    """
    # Convert OpenAI messages to a single prompt string for fast-rlm
    full_prompt = ""
    system_prompt = ""
    
    for msg in req.messages:
        if msg.role == "system":
            system_prompt += f"{msg.content}\n\n"
        else:
            full_prompt += f"{msg.role}: {msg.content}\n"
            
    if system_prompt:
        full_prompt = f"{system_prompt}\n{full_prompt}"
        
    config = {
        "primary_agent": os.getenv("RLM_PRIMARY_MODEL", "Qwen/Qwen2.5-0.5B"),
        "temperature": req.temperature,
    }
    
    logger.info(f"Routing to Fast-RLM. Prompt length: {len(full_prompt)}")
    
    # Run fast-rlm in an executor so we don't block the FastAPI async loop
    loop = asyncio.get_event_loop()
    
    def _run_fast():
        return fast_rlm.run(query=full_prompt, config=config, verbose=False)
        
    try:
        response = await loop.run_in_executor(None, _run_fast)
        
        # Fast-RLM returns a dict with 'results'
        text_output = ""
        if isinstance(response, dict) and "results" in response:
            text_output = str(response["results"])
        else:
            text_output = str(response)
            
        # Format the exact JSON response expected by OpenAI clients (like LiteLLM)
        return JSONResponse({
            "id": "chatcmpl-fastrlm",
            "object": "chat.completion",
            "created": 1234567890,
            "model": req.model,
            "choices": [
                {
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": text_output,
                    },
                    "finish_reason": "stop"
                }
            ],
            "usage": {
                "prompt_tokens": 0,
                "completion_tokens": 0,
                "total_tokens": 0
            }
        })
        
    except Exception as e:
        logger.error(f"Fast-RLM error: {e}")
        return JSONResponse(
            status_code=500,
            content={"error": {"message": str(e), "type": "fast_rlm_error"}}
        )

if __name__ == "__main__":
    logger.info("Starting Fast-RLM proxy on http://127.0.0.1:8001/v1")
    logger.info(f"Fast-RLM Engine Base URL: {os.environ.get('RLM_MODEL_BASE_URL')}")
    uvicorn.run(app, host="127.0.0.1", port=8001)
