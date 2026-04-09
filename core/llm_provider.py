"""
core/llm_provider.py
====================
LLM "brain" abstraction — every agent calls this for reasoning.

Supports:
- Groq (primary, via langchain_groq)
- Mistral (for Document AI annotations)
- Ollama (local fallback)
- Configurable from config.yaml
"""

from __future__ import annotations
import json
import logging
import os
import yaml
from pathlib import Path
from typing import Any, Dict, List, Optional, Type, Union
from pydantic import BaseModel, Field

from langchain_core.messages import BaseMessage, AIMessage, HumanMessage, SystemMessage
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.outputs import ChatResult, ChatGeneration

from core.reasoning import NativeReasoningLoop

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Abstract interface
# ---------------------------------------------------------------------------


class LLMProvider:
    """
    Abstract base for LLM providers.

    Every agent receives an ``LLMProvider`` instance so it can call
    ``generate()`` or ``generate_structured()`` without caring about
    which backend is active.
    """

    async def generate(
        self,
        prompt: str,
        *,
        system_prompt: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: int = 4096,
        **kwargs: Any,
    ) -> str:
        """Return plain-text completion."""
        raise NotImplementedError

    async def generate_structured(
        self,
        prompt: str,
        schema: Type[BaseModel],
        *,
        system_prompt: Optional[str] = None,
        temperature: float = 0.3,
        **kwargs: Any,
    ) -> BaseModel:
        """Return a parsed Pydantic model from the LLM output."""
        raise NotImplementedError

    async def embed(self, texts: List[str]) -> List[List[float]]:
        """Return embeddings for a batch of texts."""
        raise NotImplementedError


# ---------------------------------------------------------------------------
# Groq Provider (primary)
# ---------------------------------------------------------------------------


class GroqProvider(LLMProvider):
    """
    Groq LPU inference — ultra-low latency for open-source models.

    Models are configured in ``config/config.yaml`` under ``llm.groq_models``.
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = "llama-3.1-8b-instant",
        **kwargs: Any,
    ):
        self.api_key = api_key or os.getenv("GROQ_API_KEY", "")
        self.model = model
        self._kwargs = kwargs

        if not self.api_key:
            logger.warning(
                "GROQ_API_KEY not set — GroqProvider will fail on first call"
            )

    def _get_chat_model(self):  # noqa: ANN202
        from langchain_groq import ChatGroq

        return ChatGroq(
            api_key=self.api_key,
            model=self.model,
            **self._kwargs,
        )

    async def generate(
        self,
        prompt: str,
        *,
        system_prompt: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: int = 4096,
        **kwargs: Any,
    ) -> str:
        from langchain_core.messages import HumanMessage, SystemMessage

        chat = self._get_chat_model()
        messages = []
        if system_prompt:
            messages.append(SystemMessage(content=system_prompt))
        messages.append(HumanMessage(content=prompt))

        result = await chat.ainvoke(
            messages,
            temperature=temperature,
            max_tokens=max_tokens,
            **kwargs,
        )
        return result.content

    async def generate_structured(
        self,
        prompt: str,
        schema: Type[BaseModel],
        *,
        system_prompt: Optional[str] = None,
        temperature: float = 0.3,
        **kwargs: Any,
    ) -> BaseModel:
        raw = await self.generate(
            prompt,
            system_prompt=(
                (system_prompt or "")
                + f"\n\nRespond ONLY with valid JSON matching this schema:\n"
                f"{json.dumps(schema.model_json_schema(), indent=2)}"
            ),
            temperature=temperature,
            **kwargs,
        )
        # Strip markdown fences if the model wraps output
        cleaned = raw.strip()
        if cleaned.startswith("```"):
            cleaned = cleaned.split("\n", 1)[1]
            if cleaned.endswith("```"):
                cleaned = cleaned[:-3]
        return schema.model_validate_json(cleaned)


# ---------------------------------------------------------------------------
# Mistral Provider (Document AI annotations)
# ---------------------------------------------------------------------------


class MistralProvider(LLMProvider):
    """
    Mistral AI — used primarily for Document AI OCR/annotation
    via the ``mistralai`` SDK.

    Ref: https://docs.mistral.ai/capabilities/document_ai/annotations
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = "mistral-small-latest",
        **kwargs: Any,
    ):
        self.api_key = api_key or os.getenv("MISTRAL_API_KEY", "")
        self.model = model
        self._kwargs = kwargs

    def _get_client(self):  # noqa: ANN202
        from mistralai import Mistral

        return Mistral(api_key=self.api_key)

    async def generate(
        self,
        prompt: str,
        *,
        system_prompt: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: int = 4096,
        **kwargs: Any,
    ) -> str:
        client = self._get_client()
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        response = await client.chat.complete_async(
            model=self.model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        return response.choices[0].message.content

    async def annotate_document(
        self,
        file_path: str,
        query: str = "Extract all text, tables, figures, and equations with their positions.",
    ) -> Dict[str, Any]:
        """
        Use Mistral Document AI to annotate a PDF.

        Returns structured annotations including:
        - Full text extraction
        - Table positions and content
        - Figure captions
        - Equation LaTeX
        """
        import base64

        client = self._get_client()

        # Read and encode PDF
        with open(file_path, "rb") as f:
            pdf_data = base64.b64encode(f.read()).decode("utf-8")

        # Upload file via Mistral's document understanding
        response = await client.chat.complete_async(
            model="mistral-small-latest",
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image_url",
                            "image_url": f"data:application/pdf;base64,{pdf_data}",
                        },
                        {
                            "type": "text",
                            "text": query,
                        },
                    ],
                }
            ],
        )

        text_output = response.choices[0].message.content

        return {
            "full_text": text_output,
            "model_used": "mistral-small-latest",
            "file_path": file_path,
            "annotations": {
                "raw_response": text_output,
            },
        }


# ---------------------------------------------------------------------------
# AirLLM Provider (local fallback for large models)
# ---------------------------------------------------------------------------


class AirLLMProvider(LLMProvider):
    """Local LLM via AirLLM — allows running 70B models on 8GB VRAM."""

    def __init__(
        self,
        model: str = "meta-llama/Meta-Llama-3.1-70B-Instruct",
        **kwargs: Any,
    ):
        self.model = model
        self._kwargs = kwargs
        self._model_instance = None
        self._tokenizer = None

    def _get_model(self):
        if self._model_instance is None:
            from airllm import AutoModel
            import transformers
            # For Qwen and newer models, AirLLM requires setting profiling and max length explicitly
            kwargs = self._kwargs.copy()
            if "max_seq_len" not in kwargs:
                kwargs["max_seq_len"] = 512
            if "compression" not in kwargs:
                kwargs["compression"] = "4bit" # Huge speedup for disk reads
            
            self._model_instance = AutoModel.from_pretrained(
                self.model,
                **kwargs
            )
            self._tokenizer = transformers.AutoTokenizer.from_pretrained(
                self.model, 
                trust_remote_code=True
            )
        return self._model_instance, self._tokenizer

    async def generate(
        self,
        prompt: str,
        *,
        system_prompt: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: int = 4096,
        **kwargs: Any,
    ) -> str:
        # Note: AirLLM is memory optimized but slow and runs synchronously.
        # Run in a separate thread if non-blocking is required.
        model, tokenizer = self._get_model()
        
        # Manually assemble prompt to avoid chat_template mismatching positions
        full_prompt = ""
        if system_prompt:
            full_prompt += f"<|im_start|>system\n{system_prompt}<|im_end|>\n"
        full_prompt += f"<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n"
        
        # Max seq len handled explicitly to prevent rotary embedding dimension mismatches
        max_seq_len = self._kwargs.get("max_seq_len", 512)
        input_tokens = tokenizer(
            full_prompt, 
            return_tensors="pt", 
            return_attention_mask=False, 
            truncation=True, 
            max_length=max_seq_len - 100 # leave space for generation
        )
        
        import torch
        # Move to CUDA if available, required by AirLLM tensor sharding
        input_ids = input_tokens['input_ids'].cuda() if torch.cuda.is_available() else input_tokens['input_ids']
        
        # AirLLM generate
        generation_output = model.generate(
            input_ids,
            max_new_tokens=min(max_tokens, 100), # Cap for safety in airllm loop
            use_cache=True,
            return_dict_in_generate=True
        )
        
        # Decode only the newly generated tokens
        output = tokenizer.decode(generation_output.sequences[0][input_tokens['input_ids'].shape[1]:], skip_special_tokens=True)
        return output

    async def generate_structured(
        self,
        prompt: str,
        schema: Type[BaseModel],
        *,
        system_prompt: Optional[str] = None,
        temperature: float = 0.3,
        **kwargs: Any,
    ) -> BaseModel:
        raw = await self.generate(
            prompt,
            system_prompt=(
                (system_prompt or "")
                + f"\n\nRespond ONLY with valid JSON matching this schema:\n"
                f"{json.dumps(schema.model_json_schema(), indent=2)}"
            ),
            temperature=temperature,
            **kwargs,
        )
        # Strip markdown fences
        cleaned = raw.strip()
        if cleaned.startswith("```"):
            cleaned = cleaned.split("\n", 1)[1]
            if cleaned.endswith("```"):
                cleaned = cleaned[:-3]
        return schema.model_validate_json(cleaned)

# ---------------------------------------------------------------------------
# FastRLM Provider (local models via fast-rlm)
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Native LangChain Provider for Fast-RLM
# ---------------------------------------------------------------------------

class ChatFastRLM(BaseChatModel):
    """
    A custom LangChain chat model that implements a native reasoning loop.
    Replaces the external Deno-based fast-rlm dependency.
    """
    model_name: str = Field(default="primary")
    temperature: float = 0.7
    max_depth: int = 3
    
    @property
    def _llm_type(self) -> str:
        return "fast-rlm-native-v2"

    def _convert_messages_to_prompt(self, messages: List[BaseMessage]) -> str:
        prompt = ""
        for m in messages:
            if isinstance(m, SystemMessage):
                prompt += f"System: {m.content}\n\n"
            elif isinstance(m, HumanMessage):
                prompt += f"User: {m.content}\n"
            elif isinstance(m, AIMessage):
                prompt += f"Assistant: {m.content}\n"
            else:
                prompt += f"{m.type}: {m.content}\n"
        return prompt

    def _generate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[Any] = None,
        **kwargs: Any,
    ) -> ChatResult:
        import asyncio
        import nest_asyncio
        try:
            loop = asyncio.get_event_loop()
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            
        if loop.is_running():
            nest_asyncio.apply()
            
        return loop.run_until_complete(self._agenerate(messages, stop, run_manager, **kwargs))

    async def _agenerate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[Any] = None,
        **kwargs: Any,
    ) -> ChatResult:
        
        prompt = self._convert_messages_to_prompt(messages)
        
        # Determine the base address for vLLM
        vllm_base = os.getenv("RLM_MODEL_BASE_URL", "http://172.30.177.136:8000/v1")
        primary_model = os.getenv("RLM_PRIMARY_MODEL", "Qwen/Qwen2.5-1.5B-Instruct")
        
        # Instantiate the base LLM for the reasoning loop (directly to vLLM)
        base_llm = ChatOpenAI(
            model=primary_model,
            base_url=vllm_base,
            api_key="dummy",
            temperature=self.temperature
        )
        
        # We split the prompt into 'Context' and 'Query' if possible to help the loop
        context_data = ""
        query = prompt
        if "Context:" in prompt and "Query:" in prompt:
            parts = prompt.split("Query:", 1)
            context_data = parts[0].replace("Context:", "").strip()
            query = parts[1].strip()
        elif "Context:" in prompt:
            parts = prompt.split("Context:", 1)
            # Find the next newline or separator?
            # For simplicity, we'll just treat the whole thing as query if unsure
            pass

        reasoning_loop = NativeReasoningLoop(
            llm=base_llm, 
            max_depth=self.max_depth,
            model_name=self.model_name
        )
        
        result = await reasoning_loop.run(query=query, context_data=context_data)
        
        message = AIMessage(content=str(result))
        return ChatResult(generations=[ChatGeneration(message=message)])


class FastRLMProvider(LLMProvider):
    """
    Wrapper for ChatFastRLM to satisfy the internal LLMProvider interface.
    Used for RAG nodes and legacy logic that expects .generate().
    """
    def __init__(self, model: str = "primary", **kwargs: Any):
        self.model = ChatFastRLM(model_name=model, **kwargs)

    async def generate(self, prompt: str, system_prompt: Optional[str] = None, **kwargs: Any) -> str:
        messages = []
        if system_prompt:
            messages.append(SystemMessage(content=system_prompt))
        messages.append(HumanMessage(content=prompt))
        
        result = await self.model.ainvoke(messages, **kwargs)
        return result.content

    async def generate_structured(self, prompt: str, schema: Type[BaseModel], system_prompt: Optional[str] = None, **kwargs: Any) -> BaseModel:
        raw = await self.generate(
            prompt,
            system_prompt=(
                (system_prompt or "")
                + f"\n\nRespond ONLY with valid JSON matching this schema:\n"
                f"{json.dumps(schema.model_json_schema(), indent=2)}"
            ),
            **kwargs
        )
        cleaned = raw.strip()
        if cleaned.startswith("```"):
            cleaned = cleaned.split("\n", 1)[1]
            if cleaned.endswith("```"):
                cleaned = cleaned[:-3]
        return schema.model_validate_json(cleaned)


# ---------------------------------------------------------------------------
# Factory — create provider from config
# ---------------------------------------------------------------------------


def create_llm_provider(
    provider: str = "groq",
    model: Optional[str] = None,
    **kwargs: Any,
) -> LLMProvider:
    """
    Factory function that creates the correct LLMProvider from a
    provider name string.  Called by graph.py at startup.

    Examples::

        llm = create_llm_provider("groq", model="llama-3.1-8b-instant")
        llm = create_llm_provider("mistral")
        llm = create_llm_provider("airllm", model="meta-llama/Meta-Llama-3.1-70B-Instruct")
        llm = create_llm_provider("fast_rlm", model="primary")
    """
    provider = provider.lower()

    if provider == "groq":
        return GroqProvider(model=model or "llama-3.1-8b-instant", **kwargs)
    elif provider == "mistral":
        return MistralProvider(model=model or "mistral-small-latest", **kwargs)
    elif provider == "airllm":
        return AirLLMProvider(model=model or "meta-llama/Meta-Llama-3.1-70B-Instruct", **kwargs)
    elif provider == "fast_rlm":
        return FastRLMProvider(model=model or "primary", **kwargs)
    elif provider == "openai":
        from langchain_openai import ChatOpenAI
        
        # Point to local vLLM if OPENAI_API_BASE is set, default to vLLM's local port
        base_url = os.environ.get("OPENAI_API_BASE", "http://localhost:8000/v1")
        # Langchain requires ChatOpenAI to be wrapped in an interface that matches LLMProvider
        # We can hijack GroqProvider's struct since it uses standard Langchain Chat Models
        class OpenAIProvider(GroqProvider):
            def _get_chat_model(self):
                return ChatOpenAI(
                    model=self.model,
                    api_key=os.environ.get("OPENAI_API_KEY", "vllm-dummy-key"),
                    base_url=base_url,
                    **self._kwargs
                )
        return OpenAIProvider(model=model or "Qwen/Qwen2.5-7B-Instruct", **kwargs)
    else:
        raise ValueError(
            f"Unknown LLM provider: {provider!r}. " f"Supported: groq, mistral, airllm, fast_rlm, openai"
        )


def create_llm_from_config(config_path: str = "config/config.yaml") -> LLMProvider:
    """Load LLM provider settings from the project config file."""
    import yaml

    path = Path(config_path)
    if not path.exists():
        logger.warning(f"Config not found at {config_path}, using Groq defaults")
        return GroqProvider()

    with open(path) as f:
        cfg = yaml.safe_load(f)

    llm_cfg = cfg.get("llm", {})
    provider = llm_cfg.get("provider", "groq")
    model = llm_cfg.get("model") or llm_cfg.get("groq_default_model")

    return create_llm_provider(
        provider=provider,
        model=model,
        temperature=llm_cfg.get("temperature", 0.7),
    )


# ---------------------------------------------------------------------------
# Tiered providers — Agentic RAG model routing
# ---------------------------------------------------------------------------


def create_tiered_providers(
    config_path: str = "config/config.yaml",
) -> Dict[str, Any]:
    """
    Create tier-specific LLM providers from config.

    Returns a dict with::

        {
            "fast": LLMProvider,         # 8B — screening, queries
            "deep": LLMProvider,         # 70B — synthesis, analysis
            "agent_tiers": {             # node name → tier name
                "literature_review": "fast",
                "writing": "deep",
                ...
            },
        }

    Falls back to a single GroqProvider for all tiers if config
    is missing or the tiers section is absent.
    """
    import yaml

    path = Path(config_path)
    if not path.exists():
        logger.warning(f"Config not found at {config_path}, using single Groq default")
        default = GroqProvider()
        return {"fast": default, "deep": default, "agent_tiers": {}}

    with open(path) as f:
        cfg = yaml.safe_load(f) or {}

    llm_cfg = cfg.get("llm", {})
    tiers_cfg = llm_cfg.get("tiers", {})
    agent_tiers = llm_cfg.get("agent_tiers", {})

    providers: Dict[str, LLMProvider] = {}

    for tier_name, tier_def in tiers_cfg.items():
        if not isinstance(tier_def, dict):
            continue
        providers[tier_name] = create_llm_provider(
            provider=tier_def.get("provider", "groq"),
            model=tier_def.get("model"),
            temperature=tier_def.get("temperature", 0.7),
        )
        logger.info(
            f"Tier '{tier_name}' → {tier_def.get('provider')}/"
            f"{tier_def.get('model')}"
        )

    # Ensure at least fast + deep exist (fallback)
    if "fast" not in providers:
        providers["fast"] = GroqProvider(model="llama-3.1-8b-instant")
    if "deep" not in providers:
        providers["deep"] = providers["fast"]

    providers["agent_tiers"] = agent_tiers  # type: ignore[assignment]
    return providers
