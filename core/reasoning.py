import os
import json
import logging
import asyncio
import re
import sys
from typing import Any, Dict, List, Optional, Union
from dataclasses import dataclass, asdict

# LangChain Imports
from langchain_core.messages import BaseMessage, AIMessage, HumanMessage, SystemMessage
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.outputs import ChatResult, ChatGeneration
from pydantic import Field

logger = logging.getLogger(__name__)

@dataclass
class ReasoningUsage:
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    cost: Optional[float] = None

class NativeReasoningLoop:
    """
    A native Python implementation of the Recursive LLM / Reasoning loop.
    Replaces the dependency on the Deno-based fast-rlm engine.
    """
    
    SYSTEM_PROMPT = """You are a deep-reasoning agent with access to a Python REPL.
Your task is to answer user queries by thinking step-by-step and running code when needed.

REPL Environment:
1. You have a `context` variable with the input data.
2. Use `print()` to see outputs.
3. You can use `await llm_query(prompt)` to call a sub-LLM for complex sub-tasks.
4. When finished, call `FINAL(answer)` where answer is your final result (primitive or structure).

How to interact:
Wrap your Python code in ```python blocks. You can write reasoning outside these blocks.
Example:
Some reasoning...
```python
# Check context
print(len(context))
FINAL("The answer is X")
```

Rules:
- This is a multi-turn environment. If the first code run doesn't solve it, wait for output and try again.
- Be concise. Minimize token usage.
- Use `asyncio.gather` for parallel sub-queries via `llm_query`.
"""

    def __init__(
        self,
        llm: BaseChatModel,
        max_depth: int = 3,
        max_calls: int = 15,
        model_name: Optional[str] = None
    ):
        self.llm = llm
        self.max_depth = max_depth
        self.max_calls = max_calls
        self.model_name = model_name or getattr(llm, "model_name", "primary")
        self.usage = ReasoningUsage()
        self.globals = {} # Sandbox globals

    async def run(self, query: str, context_data: Any = None, depth: int = 0) -> Any:
        """Execute the reasoning loop."""
        if depth > self.max_depth:
            return "Error: Maximum reasoning depth reached."

        messages = [
            SystemMessage(content=self.SYSTEM_PROMPT),
            HumanMessage(content=f"Context: {context_data}\n\nQuery: {query}")
        ]

        # Initialize sandbox
        self.globals = {
            "context": context_data,
            "llm_query": lambda p: self.run(p, context_data=context_data, depth=depth + 1),
            "FINAL": self._set_final_result,
            "asyncio": asyncio,
            "__final_result__": None,
            "__final_result_set__": False
        }

        for i in range(self.max_calls):
            # 1. Call LLM
            response = await self.llm.ainvoke(messages)
            content = response.content
            messages.append(AIMessage(content=content))
            
            # Extract code blocks
            code_blocks = re.findall(r"```python\n(.*?)\n```", content, re.DOTALL)
            if not code_blocks:
                code_blocks = re.findall(r"```repl\n(.*?)\n```", content, re.DOTALL)

            if not code_blocks:
                # LLM didn't use code. Check if it just replied.
                if "FINAL(" in content:
                    # Try to extract the arg from FINAL(...) manually if not in a block
                    match = re.search(r"FINAL\((.*?)\)", content)
                    if match:
                        val_str = match.group(1).strip("'\"")
                        return val_str
                return content

            # 2. Execute Code
            stdout_buffer = []
            
            def buffered_print(*args, **kwargs):
                stdout_buffer.append(" ".join(map(str, args)))

            self.globals["print"] = buffered_print
            
            combined_code = "\n".join(code_blocks)
            
            try:
                # Need to handle async code if LLM used await
                if "await" in combined_code:
                    # Wrap in a coroutine to run
                    wrapped_code = f"async def __run_snippet__():\n" + "\n".join(["    " + l for l in combined_code.split("\n")])
                    exec(wrapped_code, self.globals)
                    await self.globals["__run_snippet__"]()
                else:
                    exec(combined_code, self.globals)
            except Exception as e:
                buffered_print(f"Error executing code: {e}")

            # 3. Check for FINAL result
            if self.globals.get("__final_result_set__"):
                return self.globals.get("__final_result__")

            # 4. Feed output back
            output = "\n".join(stdout_buffer)
            messages.append(HumanMessage(content=f"Output:\n{output if output else '[Empty Output]'}\n\nWhat next?"))
            
        return "Error: Maximum calls reached without FINAL result."

    def _set_final_result(self, value):
        self.globals["__final_result__"] = value
        self.globals["__final_result_set__"] = True

class ChatNativeReasoning(BaseChatModel):
    """
    A custom LangChain chat model that implements the reasoning loop natively.
    """
    model: str = Field(default="primary")
    temperature: float = 0.7
    max_depth: int = 3
    
    @property
    def _llm_type(self) -> str:
        return "native-reasoning"

    def _generate(self, messages: List[BaseMessage], stop: Optional[List[str]] = None, **kwargs: Any) -> ChatResult:
        # Fallback to sync run if needed, but we prefer async
        return asyncio.run(self._agenerate(messages, stop, **kwargs))

    async def _agenerate(self, messages: List[BaseMessage], stop: Optional[List[str]] = None, **kwargs: Any) -> ChatResult:
        # Extract the user query and context
        # In this implementation, we take the last human message as the query
        query = messages[-1].content
        
        # Determine which LLM to use for the loop
        from langchain_openai import ChatOpenAI
        base_llm = ChatOpenAI(
            model=os.getenv("RLM_PRIMARY_MODEL", "Qwen/Qwen2.5-1.5B-Instruct"),
            base_url=os.getenv("OPENAI_API_BASE", "http://172.30.177.136:8000/v1"),
            api_key="dummy",
            temperature=self.temperature
        )
        
        loop = NativeReasoningLoop(llm=base_llm, max_depth=self.max_depth)
        result = await loop.run(query)
        
        message = AIMessage(content=str(result))
        return ChatResult(generations=[ChatGeneration(message=message)])
