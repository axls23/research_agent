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
from pathlib import Path
from typing import Any, Dict, List, Optional, Type, Union

from pydantic import BaseModel

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
            lines = cleaned.split("\n", 1)
            cleaned = lines[1] if len(lines) > 1 else cleaned[3:]
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
# Ollama Provider (local fallback)
# ---------------------------------------------------------------------------

class OllamaProvider(LLMProvider):
    """Local LLM via Ollama — works offline, no API keys."""

    def __init__(
        self,
        model: str = "llama3.1:8b",
        base_url: str = "http://localhost:11434",
        **kwargs: Any,
    ):
        self.model = model
        self.base_url = base_url
        self._kwargs = kwargs

    async def generate(
        self,
        prompt: str,
        *,
        system_prompt: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: int = 4096,
        **kwargs: Any,
    ) -> str:
        from langchain_ollama import ChatOllama
        from langchain_core.messages import HumanMessage, SystemMessage

        chat = ChatOllama(
            model=self.model,
            base_url=self.base_url,
            temperature=temperature,
            num_predict=max_tokens,
        )
        messages = []
        if system_prompt:
            messages.append(SystemMessage(content=system_prompt))
        messages.append(HumanMessage(content=prompt))

        result = await chat.ainvoke(messages)
        return result.content


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
        llm = create_llm_provider("ollama", model="llama3.1:8b")
    """
    provider = provider.lower()

    if provider == "groq":
        return GroqProvider(model=model or "llama-3.1-8b-instant", **kwargs)
    elif provider == "mistral":
        return MistralProvider(model=model or "mistral-small-latest", **kwargs)
    elif provider == "ollama":
        return OllamaProvider(model=model or "llama3.1:8b", **kwargs)
    else:
        raise ValueError(
            f"Unknown LLM provider: {provider!r}. "
            f"Supported: groq, mistral, ollama"
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
