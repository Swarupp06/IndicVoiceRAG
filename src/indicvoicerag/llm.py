"""LLM generator abstraction for the RAG pipeline.

All providers return the same normalized `LLMResponse`, so no provider-specific
code leaks into the harness:

    {"text": ..., "provider": ..., "model": ..., "latency_ms": ..., "usage": {...}}

Providers:
- `mock`        deterministic, offline, no credentials - tests only
- `ollama`      local inference through the Ollama HTTP API (no key, no cloud)
- `gemini`      Google Generative Language API      (GEMINI_API_KEY)
- `groq`        GroqCloud OpenAI-compatible API     (GROQ_API_KEY)
- `openrouter`  OpenRouter OpenAI-compatible API    (OPENROUTER_API_KEY)
- `openai_compatible`  any other /chat/completions endpoint

API keys are read from environment variables only; nothing is hardcoded.
"""

from __future__ import annotations

import os
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

import httpx


class LLMError(RuntimeError):
    """Base class for provider failures."""


class LLMConfigurationError(LLMError):
    """Provider cannot run (e.g. missing API key)."""


class LLMTimeoutError(LLMError):
    """Provider did not answer within the timeout."""


class LLMResponseError(LLMError):
    """Provider answered with an error status or a malformed body."""


@dataclass(slots=True)
class LLMResponse:
    """Normalized generation result shared by every provider."""

    text: str
    provider: str
    model: str
    latency_ms: float = 0.0
    usage: dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return {
            "text": self.text,
            "provider": self.provider,
            "model": self.model,
            "latency_ms": round(self.latency_ms, 2),
            "usage": self.usage,
        }


class LLMProvider(ABC):
    """Chat-style generator abstraction."""

    name: str = "abstract"
    _model_name: str = "unknown"

    @abstractmethod
    def generate(
        self,
        messages: list[dict[str, str]],
        *,
        max_tokens: int = 512,
        temperature: float = 0.0,
        timeout: float = 30.0,
    ) -> LLMResponse:
        """Return a normalized response for a list of {'role', 'content'} messages."""
        raise NotImplementedError

    @property
    def model_name(self) -> str:
        return self._model_name

    def available(self) -> bool:
        """True when the provider can be used right now (credentials/reachability)."""
        return True

    def describe(self) -> dict[str, Any]:
        return {"provider": self.name, "model": self.model_name}


def _require_key(env_var: str, provider: str) -> str:
    key = os.environ.get(env_var, "").strip()
    if not key:
        raise LLMConfigurationError(
            f"API key env var '{env_var}' is not set for provider '{provider}'. "
            f"Export {env_var} or select a different llm.provider."
        )
    return key


def _post_json(
    url: str,
    payload: dict[str, Any],
    headers: dict[str, str],
    timeout: float,
    provider: str,
) -> dict[str, Any]:
    try:
        with httpx.Client(timeout=timeout) as client:
            response = client.post(url, json=payload, headers=headers)
    except httpx.TimeoutException as exc:
        raise LLMTimeoutError(f"{provider} request timed out after {timeout}s") from exc
    except httpx.HTTPError as exc:
        raise LLMResponseError(f"{provider} transport error: {type(exc).__name__}: {exc}") from exc
    if response.status_code != 200:
        raise LLMResponseError(f"{provider} API error {response.status_code}: {response.text[:200]}")
    try:
        data = response.json()
    except ValueError as exc:
        raise LLMResponseError(f"{provider} returned a non-JSON body: {response.text[:200]}") from exc
    if not isinstance(data, dict):
        raise LLMResponseError(f"{provider} returned an unexpected JSON body: {str(data)[:200]}")
    return data


class MockLLMProvider(LLMProvider):
    """Deterministic generator used for offline tests only.

    `behavior` selects the output strategy:
    - "grounded":   first sentence of the top context passage (verbatim)
    - "ungrounded": a made-up sentence unrelated to the context
    - "refuse":     the refusal marker
    - "empty":      an empty string (triggers output validation)
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

    def _text(self, messages: list[dict[str, str]], max_tokens: int) -> str:
        if self._behavior == "refuse":
            return "REFUSED: The retrieved context does not contain enough information."
        if self._behavior == "ungrounded":
            return "The Eiffel Tower is in Paris and Mount Everest is the tallest mountain."
        if self._behavior == "empty":
            return ""
        context = self._context_from_messages(messages)
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

    def generate(
        self,
        messages: list[dict[str, str]],
        *,
        max_tokens: int = 512,
        temperature: float = 0.0,
        timeout: float = 30.0,
    ) -> LLMResponse:
        started = time.perf_counter()
        text = self._text(messages, max_tokens)
        return LLMResponse(
            text=text,
            provider=self.name,
            model=self._model_name,
            latency_ms=(time.perf_counter() - started) * 1000.0,
            usage={},
        )

    def describe(self) -> dict[str, Any]:
        return {"provider": self.name, "model": self._model_name, "behavior": self._behavior}


class OpenAICompatibleProvider(LLMProvider):
    """Chat-completions provider over the standard OpenAI HTTP contract.

    Base class for Groq, OpenRouter and any self-hosted OpenAI-compatible
    endpoint. The API key is read from the env var named by `api_key_env`
    (optional for local endpoints that need none).
    """

    name = "openai_compatible"
    default_base_url = "https://api.openai.com/v1"
    default_model = "gpt-4o-mini"
    default_api_key_env: str | None = "OPENAI_API_KEY"
    api_key_required = True

    def __init__(
        self,
        model_name: str | None = None,
        base_url: str | None = None,
        api_key_env: str | None = None,
        default_headers: dict[str, str] | None = None,
    ):
        self._model_name = model_name or self.default_model
        if not self._model_name:
            raise LLMConfigurationError(f"model_name is required for provider '{self.name}'.")
        self._base_url = (base_url or self.default_base_url).rstrip("/")
        self._api_key_env = api_key_env if api_key_env is not None else self.default_api_key_env
        self._default_headers = default_headers or {}

    @property
    def base_url(self) -> str:
        return self._base_url

    def available(self) -> bool:
        if not self.api_key_required or not self._api_key_env:
            return True
        return bool(os.environ.get(self._api_key_env, "").strip())

    def _headers(self) -> dict[str, str]:
        headers = {"Content-Type": "application/json", **self._default_headers}
        if self._api_key_env and self.api_key_required:
            headers["Authorization"] = f"Bearer {_require_key(self._api_key_env, self.name)}"
        elif self._api_key_env:
            key = os.environ.get(self._api_key_env, "").strip()
            if key:
                headers["Authorization"] = f"Bearer {key}"
        return headers

    def generate(
        self,
        messages: list[dict[str, str]],
        *,
        max_tokens: int = 512,
        temperature: float = 0.0,
        timeout: float = 30.0,
    ) -> LLMResponse:
        payload = {
            "model": self._model_name,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        headers = self._headers()
        started = time.perf_counter()
        data = _post_json(
            f"{self._base_url}/chat/completions", payload, headers, timeout, self.name
        )
        latency_ms = (time.perf_counter() - started) * 1000.0
        try:
            text = data["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError) as exc:
            raise LLMResponseError(
                f"Malformed {self.name} response: {str(data)[:200]}"
            ) from exc
        if text is None:
            raise LLMResponseError(f"Malformed {self.name} response: null content")
        usage = data.get("usage") or {}
        return LLMResponse(
            text=str(text),
            provider=self.name,
            model=str(data.get("model") or self._model_name),
            latency_ms=latency_ms,
            usage={
                "prompt_tokens": usage.get("prompt_tokens"),
                "completion_tokens": usage.get("completion_tokens"),
                "total_tokens": usage.get("total_tokens"),
            },
        )

    def describe(self) -> dict[str, Any]:
        return {"provider": self.name, "model": self._model_name, "base_url": self._base_url}


class GroqProvider(OpenAICompatibleProvider):
    """GroqCloud (LPU inference), OpenAI-compatible.

    Free plan: no credit card, per-model RPM/RPD caps (see README cost table).
    """

    name = "groq"
    default_base_url = "https://api.groq.com/openai/v1"
    default_model = "llama-3.1-8b-instant"
    default_api_key_env = "GROQ_API_KEY"


class OpenRouterProvider(OpenAICompatibleProvider):
    """OpenRouter, OpenAI-compatible. Only `:free` model variants cost ₹0."""

    name = "openrouter"
    default_base_url = "https://openrouter.ai/api/v1"
    default_model = "google/gemma-4-31b-it:free"
    default_api_key_env = "OPENROUTER_API_KEY"

    def generate(
        self,
        messages: list[dict[str, str]],
        *,
        max_tokens: int = 512,
        temperature: float = 0.0,
        timeout: float = 30.0,
    ) -> LLMResponse:
        if not self._model_name.endswith(":free"):
            raise LLMConfigurationError(
                f"OpenRouter model '{self._model_name}' is not a ':free' variant; "
                "paid variants are rejected to keep the project at zero cost."
            )
        return super().generate(
            messages, max_tokens=max_tokens, temperature=temperature, timeout=timeout
        )


class OllamaProvider(LLMProvider):
    """Local inference via the Ollama HTTP API - no API key, no network cost."""

    name = "ollama"
    default_base_url = "http://localhost:11434"
    default_model = "qwen2.5:1.5b-instruct"

    def __init__(self, model_name: str | None = None, base_url: str | None = None):
        self._model_name = model_name or self.default_model
        self._base_url = (base_url or os.environ.get("OLLAMA_HOST") or self.default_base_url).rstrip("/")
        if self._base_url.endswith("/v1"):  # tolerate an OpenAI-style base url
            self._base_url = self._base_url[: -len("/v1")]

    @property
    def base_url(self) -> str:
        return self._base_url

    def available(self) -> bool:
        try:
            with httpx.Client(timeout=2.0) as client:
                response = client.get(f"{self._base_url}/api/tags")
            if response.status_code != 200:
                return False
            models = [m.get("name", "") for m in response.json().get("models", [])]
        except (httpx.HTTPError, ValueError):
            return False
        return any(m == self._model_name or m.startswith(f"{self._model_name}:") for m in models)

    def generate(
        self,
        messages: list[dict[str, str]],
        *,
        max_tokens: int = 512,
        temperature: float = 0.0,
        timeout: float = 30.0,
    ) -> LLMResponse:
        payload = {
            "model": self._model_name,
            "messages": messages,
            "stream": False,
            "options": {"temperature": temperature, "num_predict": max_tokens},
        }
        started = time.perf_counter()
        data = _post_json(
            f"{self._base_url}/api/chat", payload, {"Content-Type": "application/json"}, timeout, self.name
        )
        latency_ms = (time.perf_counter() - started) * 1000.0
        try:
            text = data["message"]["content"]
        except (KeyError, TypeError) as exc:
            raise LLMResponseError(f"Malformed ollama response: {str(data)[:200]}") from exc
        if text is None:
            raise LLMResponseError("Malformed ollama response: null content")
        return LLMResponse(
            text=str(text),
            provider=self.name,
            model=str(data.get("model") or self._model_name),
            latency_ms=latency_ms,
            usage={
                "prompt_tokens": data.get("prompt_eval_count"),
                "completion_tokens": data.get("eval_count"),
                "total_tokens": (data.get("prompt_eval_count") or 0) + (data.get("eval_count") or 0),
                "eval_duration_ms": round((data.get("eval_duration") or 0) / 1e6, 2),
            },
        )

    def describe(self) -> dict[str, Any]:
        return {"provider": self.name, "model": self._model_name, "base_url": self._base_url}


class GeminiProvider(LLMProvider):
    """Google Gemini via the Generative Language REST API (free tier, API key only)."""

    name = "gemini"
    default_model = "gemini-2.5-flash-lite"
    default_api_key_env = "GEMINI_API_KEY"
    base_url = "https://generativelanguage.googleapis.com/v1beta"

    def __init__(self, model_name: str | None = None, api_key_env: str | None = None):
        self._model_name = model_name or self.default_model
        self._api_key_env = api_key_env or self.default_api_key_env

    def available(self) -> bool:
        return bool(os.environ.get(self._api_key_env, "").strip())

    @staticmethod
    def _to_contents(messages: list[dict[str, str]]) -> tuple[list[dict[str, Any]], str | None]:
        contents: list[dict[str, Any]] = []
        system_instruction: str | None = None
        for message in messages:
            role = message.get("role")
            content = message.get("content", "")
            if role == "system":
                system_instruction = content if system_instruction is None else f"{system_instruction}\n\n{content}"
                continue
            gemini_role = "model" if role == "assistant" else "user"
            if contents and contents[-1]["role"] == gemini_role:
                contents[-1]["parts"].append({"text": content})
            else:
                contents.append({"role": gemini_role, "parts": [{"text": content}]})
        return contents, system_instruction

    def generate(
        self,
        messages: list[dict[str, str]],
        *,
        max_tokens: int = 512,
        temperature: float = 0.0,
        timeout: float = 30.0,
    ) -> LLMResponse:
        key = _require_key(self._api_key_env, self.name)
        contents, system_instruction = self._to_contents(messages)
        payload: dict[str, Any] = {
            "contents": contents,
            "generationConfig": {"temperature": temperature, "maxOutputTokens": max_tokens},
        }
        if system_instruction:
            payload["systemInstruction"] = {"parts": [{"text": system_instruction}]}
        url = f"{self.base_url}/models/{self._model_name}:generateContent"
        started = time.perf_counter()
        data = _post_json(
            url,
            payload,
            {"Content-Type": "application/json", "x-goog-api-key": key},
            timeout,
            self.name,
        )
        latency_ms = (time.perf_counter() - started) * 1000.0
        try:
            parts = data["candidates"][0]["content"]["parts"]
            text = "".join(part.get("text", "") for part in parts)
        except (KeyError, IndexError, TypeError) as exc:
            raise LLMResponseError(f"Malformed gemini response: {str(data)[:200]}") from exc
        usage = data.get("usageMetadata") or {}
        return LLMResponse(
            text=text,
            provider=self.name,
            model=str(data.get("modelVersion") or self._model_name),
            latency_ms=latency_ms,
            usage={
                "prompt_tokens": usage.get("promptTokenCount"),
                "completion_tokens": usage.get("candidatesTokenCount"),
                "total_tokens": usage.get("totalTokenCount"),
            },
        )

    def describe(self) -> dict[str, Any]:
        return {"provider": self.name, "model": self._model_name}


class FallbackLLMProvider(LLMProvider):
    """Try providers in order: free cloud provider -> ollama -> (mock, tests only).

    A provider is skipped when `available()` is False; a provider that raises is
    recorded and the next one is tried. Mock is only ever part of the chain when
    it is configured explicitly (`allow_mock_fallback`).
    """

    name = "fallback"

    def __init__(self, providers: list[LLMProvider]):
        if not providers:
            raise LLMConfigurationError("Fallback chain needs at least one provider.")
        self._providers = providers
        self._last_used: LLMProvider = providers[0]

    @property
    def providers(self) -> list[LLMProvider]:
        return list(self._providers)

    @property
    def last_used(self) -> LLMProvider:
        return self._last_used

    @property
    def model_name(self) -> str:
        return self._providers[0].model_name

    def available(self) -> bool:
        return any(provider.available() for provider in self._providers)

    def generate(
        self,
        messages: list[dict[str, str]],
        *,
        max_tokens: int = 512,
        temperature: float = 0.0,
        timeout: float = 30.0,
    ) -> LLMResponse:
        errors: list[str] = []
        for provider in self._providers:
            if not provider.available():
                errors.append(f"{provider.name}: unavailable")
                continue
            try:
                response = provider.generate(
                    messages, max_tokens=max_tokens, temperature=temperature, timeout=timeout
                )
            except LLMError as exc:
                errors.append(f"{provider.name}: {type(exc).__name__}: {exc}")
                continue
            self._last_used = provider
            return response
        raise LLMError("All providers failed -> " + " | ".join(errors))

    def describe(self) -> dict[str, Any]:
        return {
            "provider": self.name,
            "model": self.model_name,
            "chain": [p.describe() for p in self._providers],
            "last_used": self._last_used.name,
        }


PROVIDER_ALIASES = {
    "openai": "openai_compatible",
    "open_router": "openrouter",
    "google": "gemini",
    "local": "ollama",
}


def normalize_provider_name(provider: str) -> str:
    normalized = provider.strip().lower().replace("-", "_")
    return PROVIDER_ALIASES.get(normalized, normalized)


def build_llm_provider(
    provider: str,
    model_name: str | None = None,
    base_url: str | None = None,
    api_key_env: str | None = None,
) -> LLMProvider:
    """Build a single provider by name (no fallback chain)."""
    normalized = normalize_provider_name(provider)
    if normalized == "mock":
        return MockLLMProvider(model_name=model_name or "mock-rag-generator")
    if normalized == "ollama":
        return OllamaProvider(model_name=model_name, base_url=base_url)
    if normalized == "gemini":
        return GeminiProvider(model_name=model_name, api_key_env=api_key_env)
    if normalized == "groq":
        return GroqProvider(model_name=model_name, base_url=base_url, api_key_env=api_key_env)
    if normalized == "openrouter":
        return OpenRouterProvider(model_name=model_name, base_url=base_url, api_key_env=api_key_env)
    if normalized == "openai_compatible":
        return OpenAICompatibleProvider(
            model_name=model_name, base_url=base_url, api_key_env=api_key_env
        )
    raise ValueError(f"Unsupported LLM provider: {provider}")


def build_provider_chain(
    provider: str,
    model_name: str | None = None,
    base_url: str | None = None,
    api_key_env: str | None = None,
    fallback_providers: list[str] | None = None,
    fallback_models: dict[str, str] | None = None,
    allow_mock_fallback: bool = False,
) -> LLMProvider:
    """Build the primary provider plus its configured fallbacks.

    `allow_mock_fallback` must be set explicitly; production/demo runs never
    silently degrade to the mock generator.
    """
    primary = build_llm_provider(provider, model_name, base_url, api_key_env)
    chain: list[LLMProvider] = [primary]
    seen = {normalize_provider_name(provider)}
    models = fallback_models or {}
    for fallback in fallback_providers or []:
        normalized = normalize_provider_name(fallback)
        if normalized in seen:
            continue
        if normalized == "mock" and not allow_mock_fallback:
            continue
        seen.add(normalized)
        chain.append(build_llm_provider(normalized, models.get(normalized)))
    if len(chain) == 1:
        return primary
    return FallbackLLMProvider(chain)
