"""LLM generator abstraction for the RAG pipeline.

Providers:
- `mock` (deterministic, offline, no credentials) - default for tests/smoke
- `openai_compatible` (OpenAI, Ollama, vLLM, LM Studio, Together, ...) via
  the standard /chat/completions HTTP contract
- `gemini` (Google Generative Language API) via raw HTTP

The harness depends only on the `LLMProvider` interface so the model can be
swapped through configuration without touching pipeline code.
"""

from __future__ import annotations

import os
from abc import ABC, abstractmethod
from typing import Any

import httpx


class LLMProvider(ABC):
    """Chat-style generator abstraction."""

    name: str = "abstract"

    @abstractmethod
    def generate(
        self,
        messages: list[dict[str, str]],
        *,
        max_tokens: int,
        temperature: float,
        timeout: float,
    ) -> str:
        """Return the assistant text for a list of {'role', 'content'} messages."""
        raise NotImplementedError

    def describe(self) -> dict[str, Any]:
        return {"provider": self.name}


class MockLLMProvider(LLMProvider):
    """Deterministic generator used for offline tests and credential-free smoke.

    `behavior` selects the output strategy:
    - "grounded":   answers with the first sentence of the top context passage
                    (verbatim, so the lexical grounding validator passes)
    - "ungrounded": returns a made-up sentence unrelated to the context
    - "refuse":     returns the refusal marker
    - "empty":      returns an empty string (triggers output validation)
    """

    name = "mock"

    def __init__(self, behavior: str = "grounded", model_name: str = "mock-rag-generator"):
        self._behavior = behavior
        self._model_name = model_name

    @property
    def behavior(self) -> str:
        return self._behavior

    def _context_from_messages(self, messages: list[dict[str, str]]) -> str:
        for message in reversed(messages):
            if message.get("role") != "user":
                continue
            content = message.get("content", "")
            if "BEGIN RETRIEVED CONTEXT" in content:
                return content
        return ""

    def generate(
        self,
        messages: list[dict[str, str]],
        *,
        max_tokens: int = 512,
        temperature: float = 0.0,
        timeout: float = 30.0,
    ) -> str:
        if self._behavior == "refuse":
            return "REFUSED: The retrieved context does not contain enough information."
        if self._behavior == "ungrounded":
            return "The Eiffel Tower is in Paris and Mount Everest is the tallest mountain."
        if self._behavior == "empty":
            return ""
        context = self._context_from_messages(messages)
        # First sentence of the first context passage (verbatim) is grounded.
        segment = context.split("BEGIN RETRIEVED CONTEXT", 1)[-1]
        segment = segment.split("END RETRIEVED CONTEXT", 1)[0]
        lines = [line.strip() for line in segment.split("\n") if line.strip()]
        body = [line for line in lines if not line.startswith("[Source")]
        if not body:
            return "REFUSED: no retrieved context available."
        first_sentence = body[0].split(". ", 1)[0]
        if not first_sentence.endswith("."):
            first_sentence += "."
        return first_sentence[: max(1, max_tokens)].rstrip()

    def describe(self) -> dict[str, Any]:
        return {"provider": self.name, "model": self._model_name, "behavior": self._behavior}


class OpenAICompatibleProvider(LLMProvider):
    """Chat-completions provider over the standard OpenAI HTTP contract.

    Works with OpenAI, Together, Groq, Ollama (http://localhost:11434/v1),
    vLLM, LM Studio, etc. The API key is read from the env var named by
    `api_key_env` (optional for local endpoints that need none).
    """

    name = "openai_compatible"

    def __init__(
        self,
        model_name: str,
        base_url: str | None = None,
        api_key_env: str | None = None,
        default_headers: dict[str, str] | None = None,
    ):
        if not model_name:
            raise ValueError("model_name is required for openai_compatible provider.")
        self._model_name = model_name
        self._base_url = (base_url or "https://api.openai.com/v1").rstrip("/")
        self._api_key_env = api_key_env
        self._default_headers = default_headers or {}

    def _headers(self) -> dict[str, str]:
        headers = {"Content-Type": "application/json", **self._default_headers}
        if self._api_key_env:
            key = os.environ.get(self._api_key_env)
            if not key:
                raise RuntimeError(
                    f"API key env var '{self._api_key_env}' is not set. "
                    "Set it or switch llm.provider to 'mock'."
                )
            headers["Authorization"] = f"Bearer {key}"
        return headers

    def generate(
        self,
        messages: list[dict[str, str]],
        *,
        max_tokens: int,
        temperature: float,
        timeout: float,
    ) -> str:
        payload = {
            "model": self._model_name,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        with httpx.Client(timeout=timeout) as client:
            response = client.post(
                f"{self._base_url}/chat/completions",
                headers=self._headers(),
                json=payload,
            )
        if response.status_code != 200:
            raise RuntimeError(f"LLM API error {response.status_code}: {response.text[:200]}")
        data = response.json()
        try:
            return str(data["choices"][0]["message"]["content"])
        except (KeyError, IndexError, TypeError) as exc:
            raise RuntimeError(f"Unexpected LLM API response shape: {data}") from exc

    def describe(self) -> dict[str, Any]:
        return {"provider": self.name, "model": self._model_name, "base_url": self._base_url}


class GeminiProvider(LLMProvider):
    """Google Gemini via the Generative Language REST API.

    Strong Indic language support and a generous free tier make it a practical
    prototype choice for Hindi/Indic RAG. Uses raw HTTP (no SDK dependency).
    """

    name = "gemini"

    def __init__(self, model_name: str = "gemini-2.0-flash", api_key_env: str = "GEMINI_API_KEY"):
        self._model_name = model_name
        self._api_key_env = api_key_env

    def generate(
        self,
        messages: list[dict[str, str]],
        *,
        max_tokens: int,
        temperature: float,
        timeout: float,
    ) -> str:
        key = os.environ.get(self._api_key_env)
        if not key:
            raise RuntimeError(
                f"API key env var '{self._api_key_env}' is not set. "
                "Set it or switch llm.provider to 'mock'."
            )
        # Convert chat messages to the Gemini contents format.
        contents: list[dict[str, Any]] = []
        for message in messages:
            role = "model" if message["role"] == "assistant" else "user"
            if role == "user" and contents and contents[-1]["role"] == "user":
                contents[-1]["parts"].append({"text": message["content"]})
            else:
                contents.append({"role": role, "parts": [{"text": message["content"]}]})
        payload = {
            "contents": contents,
            "generationConfig": {
                "temperature": temperature,
                "maxOutputTokens": max_tokens,
            },
        }
        url = (
            "https://generativelanguage.googleapis.com/v1beta/models/"
            f"{self._model_name}:generateContent?key={key}"
        )
        with httpx.Client(timeout=timeout) as client:
            response = client.post(url, json=payload)
        if response.status_code != 200:
            raise RuntimeError(f"Gemini API error {response.status_code}: {response.text[:200]}")
        data = response.json()
        try:
            parts = data["candidates"][0]["content"]["parts"]
            return "".join(part.get("text", "") for part in parts)
        except (KeyError, IndexError, TypeError) as exc:
            raise RuntimeError(f"Unexpected Gemini response shape: {data}") from exc

    def describe(self) -> dict[str, Any]:
        return {"provider": self.name, "model": self._model_name}


def build_llm_provider(
    provider: str,
    model_name: str | None = None,
    base_url: str | None = None,
    api_key_env: str | None = None,
) -> LLMProvider:
    normalized = provider.lower().replace("-", "_")
    if normalized == "mock":
        return MockLLMProvider(model_name=model_name or "mock-rag-generator")
    if normalized in {"openai_compatible", "openai"}:
        return OpenAICompatibleProvider(
            model_name=model_name or "gpt-4o-mini",
            base_url=base_url,
            api_key_env=api_key_env or "OPENAI_API_KEY",
        )
    if normalized == "gemini":
        return GeminiProvider(
            model_name=model_name or "gemini-2.0-flash",
            api_key_env=api_key_env or "GEMINI_API_KEY",
        )
    raise ValueError(f"Unsupported LLM provider: {provider}")
