"""
core/llm_provider.py
====================
LLM "brain" abstraction — every agent calls this for reasoning.
Only supports local inference backends (Ollama, vLLM) to enforce strict air-gap compliance.
"""

from __future__ import annotations
import json
import logging
import os
import re
import yaml
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Type, Union
from pydantic import BaseModel, Field, ValidationError

from langchain_core.messages import BaseMessage, AIMessage, HumanMessage, SystemMessage
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.outputs import ChatResult, ChatGeneration

from core.reasoning import NativeReasoningLoop

logger = logging.getLogger(__name__)


def local_only_enabled() -> bool:
    """Return True when runtime must avoid public-cloud model providers."""
    raw = (os.getenv("NEXUS_LOCAL_ONLY", "true") or "true").strip().lower()
    return raw not in {"0", "false", "no", "off"}


# ---------------------------------------------------------------------------
# Shared JSON cleaning & retry utilities
# ---------------------------------------------------------------------------


def _clean_llm_json(raw: str) -> str:
    """Strip markdown fences, preamble prose, and trailing junk from LLM JSON output."""
    cleaned = raw.strip()

    # Remove leading prose before the first { or [
    first_brace = -1
    for i, ch in enumerate(cleaned):
        if ch in ('{', '['):
            first_brace = i
            break
    if first_brace > 0:
        cleaned = cleaned[first_brace:]

    # Remove trailing prose after the last } or ]
    last_brace = -1
    for i in range(len(cleaned) - 1, -1, -1):
        if cleaned[i] in ('}', ']'):
            last_brace = i
            break
    if last_brace >= 0 and last_brace < len(cleaned) - 1:
        cleaned = cleaned[: last_brace + 1]

    # Handle markdown fences (```json ... ```, ``` ... ```, etc.)
    fence_pattern = re.compile(r'^```(?:json|JSON)?\s*\n?', re.MULTILINE)
    cleaned = fence_pattern.sub('', cleaned)
    cleaned = cleaned.replace('```', '').strip()

    return cleaned


async def _parse_structured_with_retry(
    generate_fn: Callable,
    prompt: str,
    schema: Type[BaseModel],
    system_prompt: Optional[str] = None,
    temperature: float = 0.3,
    max_retries: int = 2,
    **kwargs: Any,
) -> BaseModel:
    """
    Parse LLM output into a Pydantic schema with retry-on-failure.

    On ValidationError, feeds the error back to the LLM for self-correction
    (up to max_retries). This prevents silent data loss from transient
    formatting errors.
    """
    schema_json = json.dumps(schema.model_json_schema(), indent=2)
    base_system = (
        (system_prompt or "")
        + f"\n\nRespond ONLY with valid JSON matching this schema:\n{schema_json}"
    )

    last_error = None
    for attempt in range(1 + max_retries):
        effective_prompt = prompt
        effective_system = base_system

        if attempt > 0 and last_error:
            # Feed the error back for self-correction
            effective_prompt = (
                f"Your previous JSON response was invalid.\n"
                f"Error: {last_error}\n\n"
                f"Please fix the JSON and respond ONLY with valid JSON.\n\n"
                f"Original request:\n{prompt}"
            )

        raw = await generate_fn(
            effective_prompt,
            system_prompt=effective_system,
            temperature=temperature,
            **kwargs,
        )

        cleaned = _clean_llm_json(raw)
        try:
            return schema.model_validate_json(cleaned)
        except (ValidationError, json.JSONDecodeError, ValueError) as e:
            last_error = str(e)[:500]  # Cap error length for prompt budget
            logger.warning(
                "Structured parse attempt %d/%d failed: %s",
                attempt + 1, 1 + max_retries, last_error[:120],
            )

    # Final fallback: raise so callers can handle
    raise ValidationError.from_exception_data(
        title=schema.__name__,
        line_errors=[],
    ) if hasattr(ValidationError, 'from_exception_data') else ValueError(
        f"Failed to parse {schema.__name__} after {1 + max_retries} attempts: {last_error}"
    )


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
# Ollama Provider (local)
# ---------------------------------------------------------------------------


class OllamaProvider(LLMProvider):
    """Local LLM via native Ollama chat API."""

    def __init__(
        self,
        model: str = "qwen2.5:3b",
        base_url: Optional[str] = None,
        **kwargs: Any,
    ):
        self.model = model
        raw_base = base_url or os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
        # ChatOllama expects the Ollama host, not OpenAI-style /v1 path.
        self.base_url = raw_base[:-3] if raw_base.endswith("/v1") else raw_base
        self._kwargs = kwargs

    def _get_chat_model(
        self,
        *,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
    ):  # noqa: ANN202
        from langchain_ollama import ChatOllama

        model_kwargs = dict(self._kwargs)
        if temperature is not None:
            model_kwargs["temperature"] = temperature
        # ChatOllama uses num_predict as the output token budget.
        if max_tokens is not None:
            model_kwargs["num_predict"] = max_tokens

        return ChatOllama(
            model=self.model,
            base_url=self.base_url,
            **model_kwargs,
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
        chat = self._get_chat_model(temperature=temperature, max_tokens=max_tokens)
        messages = []
        if system_prompt:
            messages.append(SystemMessage(content=system_prompt))
        messages.append(HumanMessage(content=prompt))

        result = await chat.ainvoke(messages, **kwargs)
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
        return await _parse_structured_with_retry(
            self.generate, prompt, schema,
            system_prompt=system_prompt, temperature=temperature, **kwargs,
        )


# ---------------------------------------------------------------------------
# llama.cpp Provider (local, OpenAI-compatible)
# ---------------------------------------------------------------------------


class LlamaCppProvider(LLMProvider):
    """Local LLM served by llama.cpp's OpenAI-compatible ``/v1`` endpoint.

    llama-server (llama.cpp) exposes a Chat Completions API, so we drive it
    through ``langchain_openai.ChatOpenAI`` with a dummy key. Stays within the
    air-gap policy: the base URL points at localhost, never a public cloud.
    """

    def __init__(
        self,
        model: Optional[str] = None,
        base_url: Optional[str] = None,
        **kwargs: Any,
    ):
        # llama-server ignores the model name for single-model mounts, but
        # ChatOpenAI requires a non-empty string.
        self.model = model or os.getenv("LLAMACPP_MODEL", "local-model")
        raw_base = base_url or os.getenv("LLAMACPP_BASE_URL", "http://127.0.0.1:8001/v1")
        self.base_url = raw_base if raw_base.endswith("/v1") else raw_base.rstrip("/") + "/v1"
        self.api_key = os.getenv("LLAMACPP_API_KEY", "sk-no-key-required")
        self._kwargs = kwargs

    def _get_chat_model(
        self,
        *,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
    ):  # noqa: ANN202
        from langchain_openai import ChatOpenAI

        model_kwargs = dict(self._kwargs)
        if temperature is not None:
            model_kwargs["temperature"] = temperature
        if max_tokens is not None:
            model_kwargs["max_tokens"] = max_tokens

        return ChatOpenAI(
            model=self.model,
            base_url=self.base_url,
            api_key=self.api_key,
            **model_kwargs,
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
        chat = self._get_chat_model(temperature=temperature, max_tokens=max_tokens)
        messages = []
        if system_prompt:
            messages.append(SystemMessage(content=system_prompt))
        messages.append(HumanMessage(content=prompt))

        result = await chat.ainvoke(messages, **kwargs)
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
        return await _parse_structured_with_retry(
            self.generate, prompt, schema,
            system_prompt=system_prompt, temperature=temperature, **kwargs,
        )


# ---------------------------------------------------------------------------
# Native LangChain Provider for Fast-RLM (vLLM)
# ---------------------------------------------------------------------------

class ChatFastRLM(BaseChatModel):
    """
    A custom LangChain chat model that implements a native reasoning loop.
    Communicates directly with a local vLLM instance.
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
        try:
            loop = asyncio.get_event_loop()
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            
        return loop.run_until_complete(self._agenerate(messages, stop, run_manager, **kwargs))

    async def _agenerate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[Any] = None,
        **kwargs: Any,
    ) -> ChatResult:
        from langchain_openai import ChatOpenAI

        prompt = self._convert_messages_to_prompt(messages)

        # Dynamic local vLLM configuration (no hardcoded WSL NAT IPs)
        vllm_base = os.getenv("VLLM_BASE_URL", "http://localhost:8000/v1")
        primary_model = os.getenv("RLM_PRIMARY_MODEL", "Qwen/Qwen2.5-1.5B-Instruct")

        base_llm = ChatOpenAI(
            model=primary_model,
            base_url=vllm_base,
            api_key="dummy",
            temperature=self.temperature
        )

        context_data = ""
        query = prompt
        if "Context:" in prompt and "Query:" in prompt:
            parts = prompt.split("Query:", 1)
            context_data = parts[0].replace("Context:", "").strip()
            query = parts[1].strip()

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
        return await _parse_structured_with_retry(
            self.generate, prompt, schema,
            system_prompt=system_prompt, **kwargs,
        )


# ---------------------------------------------------------------------------
# Factory — create provider from config
# ---------------------------------------------------------------------------


def create_llm_provider(
    provider: str = "ollama",
    model: Optional[str] = None,
    **kwargs: Any,
) -> LLMProvider:
    """
    Factory function that creates the correct LLMProvider.
    Ensures local-only execution and throws errors on cloud-specific providers.
    """
    provider = provider.lower()

    if provider in {"groq", "mistral", "openai"}:
        fallback_model = os.getenv("OLLAMA_MODEL", "qwen2.5:3b")
        logger.warning(
            "Provider '%s' blocked/removed by local air-gap policy; falling back to local ollama:%s",
            provider,
            fallback_model,
        )
        provider = "ollama"
        model = fallback_model

    if provider == "ollama":
        return OllamaProvider(model=model or os.getenv("OLLAMA_MODEL", "qwen2.5:3b"), **kwargs)
    elif provider in {"llamacpp", "llama_cpp", "llama-cpp", "vllm"}:
        # vllm and llama.cpp both speak the OpenAI-compatible /v1 API locally.
        return LlamaCppProvider(model=model, **kwargs)
    elif provider in {"fast_rlm", "fast-rlm"}:
        return FastRLMProvider(model=model or "primary", **kwargs)
    else:
        raise ValueError(
            f"Unknown or unsupported local LLM provider: {provider!r}. "
            f"Supported: ollama, llamacpp, fast_rlm"
        )


def create_llm_from_config(config_path: str = "config/config.yaml") -> LLMProvider:
    """Load LLM provider settings from the project config file."""
    path = Path(config_path)
    if not path.exists():
        logger.warning(f"Config not found at {config_path}, using Ollama defaults")
        return OllamaProvider(model=os.getenv("OLLAMA_MODEL", "qwen2.5:3b"))

    with open(path) as f:
        cfg = yaml.safe_load(f)

    llm_cfg = cfg.get("llm", {})
    provider = llm_cfg.get("provider", "ollama")
    model = llm_cfg.get("model")

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
    """
    path = Path(config_path)
    if not path.exists():
        logger.warning(f"Config not found at {config_path}, using single Ollama default")
        default = OllamaProvider(model=os.getenv("OLLAMA_MODEL", "qwen2.5:3b"))
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
            provider=tier_def.get("provider", "ollama"),
            model=tier_def.get("model"),
            temperature=tier_def.get("temperature", 0.7),
        )
        logger.info(
            f"Tier '{tier_name}' → {tier_def.get('provider')}/"
            f"{tier_def.get('model')}"
        )

    # Ensure at least fast + deep exist (fallback)
    if "fast" not in providers:
        providers["fast"] = OllamaProvider(model=os.getenv("OLLAMA_MODEL_FAST", os.getenv("OLLAMA_MODEL", "qwen2.5:3b")))
    if "deep" not in providers:
        providers["deep"] = providers["fast"]

    providers["agent_tiers"] = agent_tiers  # type: ignore[assignment]
    return providers
