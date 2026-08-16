"""Provider-level tests for the Phase 2.5 LLM abstraction.

No test performs a real API call: every HTTP interaction is faked by patching
`httpx.Client` inside `indicvoicerag.llm`.
"""

from __future__ import annotations

import json
from typing import Any

import httpx
import pytest

from indicvoicerag import llm as llm_module
from indicvoicerag.config import AppConfig, LLMConfig
from indicvoicerag.llm import (
    FallbackLLMProvider,
    GeminiProvider,
    GroqProvider,
    LLMConfigurationError,
    LLMError,
    LLMResponse,
    LLMResponseError,
    LLMTimeoutError,
    MockLLMProvider,
    OllamaProvider,
    OpenAICompatibleProvider,
    OpenRouterProvider,
    build_llm_provider,
    build_provider_chain,
    normalize_provider_name,
)
from indicvoicerag.pipeline import build_llm_from_config

MESSAGES = [
    {"role": "system", "content": "You are grounded."},
    {"role": "user", "content": "BEGIN RETRIEVED CONTEXT\nParis is the capital of France.\nEND RETRIEVED CONTEXT"},
]


class _FakeResponse:
    def __init__(self, status_code: int = 200, payload: Any = None, text: str | None = None):
        self.status_code = status_code
        self._payload = payload
        self.text = text if text is not None else json.dumps(payload)

    def json(self) -> Any:
        if self._payload is None:
            raise ValueError("not json")
        return self._payload


class _FakeClient:
    """Stands in for httpx.Client; records the last request it received."""

    last_url: str | None = None
    last_payload: dict[str, Any] | None = None
    last_headers: dict[str, str] | None = None

    def __init__(self, response: Any = None, error: Exception | None = None, **_: Any):
        self._response = response
        self._error = error

    def __enter__(self) -> "_FakeClient":
        return self

    def __exit__(self, *_: Any) -> bool:
        return False

    def post(self, url: str, json: dict[str, Any] | None = None, headers: dict[str, str] | None = None) -> Any:
        type(self).last_url = url
        type(self).last_payload = json
        type(self).last_headers = headers
        if self._error is not None:
            raise self._error
        return self._response

    def get(self, url: str) -> Any:
        type(self).last_url = url
        if self._error is not None:
            raise self._error
        return self._response


def _patch_client(monkeypatch: pytest.MonkeyPatch, response: Any = None, error: Exception | None = None) -> None:
    monkeypatch.setattr(
        llm_module.httpx, "Client", lambda **kwargs: _FakeClient(response=response, error=error, **kwargs)
    )


def _openai_payload(text: str = "Paris is the capital of France.") -> dict[str, Any]:
    return {
        "model": "llama-3.1-8b-instant",
        "choices": [{"message": {"role": "assistant", "content": text}}],
        "usage": {"prompt_tokens": 10, "completion_tokens": 7, "total_tokens": 17},
    }


# --- provider selection ---
def test_build_llm_provider_selects_each_provider() -> None:
    assert isinstance(build_llm_provider("mock"), MockLLMProvider)
    assert isinstance(build_llm_provider("ollama"), OllamaProvider)
    assert isinstance(build_llm_provider("gemini"), GeminiProvider)
    assert isinstance(build_llm_provider("groq"), GroqProvider)
    assert isinstance(build_llm_provider("openrouter"), OpenRouterProvider)
    assert isinstance(build_llm_provider("openai_compatible"), OpenAICompatibleProvider)


def test_provider_name_aliases_and_unknown_provider() -> None:
    assert normalize_provider_name("Open-Router") == "openrouter"
    assert normalize_provider_name("GOOGLE") == "gemini"
    assert normalize_provider_name("local") == "ollama"
    with pytest.raises(ValueError):
        build_llm_provider("definitely-not-a-provider")


def test_provider_defaults_are_free_models() -> None:
    assert build_llm_provider("groq").model_name == "llama-3.1-8b-instant"
    assert build_llm_provider("gemini").model_name == "gemini-2.5-flash-lite"
    assert build_llm_provider("openrouter").model_name.endswith(":free")
    assert build_llm_provider("ollama").model_name == "qwen2.5:1.5b-instruct"


def test_build_llm_from_config_drops_stale_mock_model() -> None:
    config = AppConfig(llm=LLMConfig(provider="groq", model_name="mock-rag-generator"))
    provider = build_llm_from_config(config)
    assert provider.model_name == "llama-3.1-8b-instant"


# --- missing API key ---
def test_missing_api_key_makes_provider_unavailable(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    provider = GroqProvider()
    assert provider.available() is False
    with pytest.raises(LLMConfigurationError) as excinfo:
        provider.generate(MESSAGES)
    assert "GROQ_API_KEY" in str(excinfo.value)


def test_missing_gemini_key_raises_configuration_error(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    provider = GeminiProvider()
    assert provider.available() is False
    with pytest.raises(LLMConfigurationError):
        provider.generate(MESSAGES)


def test_api_key_is_read_from_environment_only(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("GROQ_API_KEY", "test-key-not-real")
    _patch_client(monkeypatch, response=_FakeResponse(payload=_openai_payload()))
    GroqProvider().generate(MESSAGES)
    assert _FakeClient.last_headers is not None
    assert _FakeClient.last_headers["Authorization"] == "Bearer test-key-not-real"


# --- normalized output ---
def test_openai_compatible_response_is_normalized(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("GROQ_API_KEY", "test-key-not-real")
    _patch_client(monkeypatch, response=_FakeResponse(payload=_openai_payload()))
    result = GroqProvider().generate(MESSAGES, max_tokens=64, temperature=0.2, timeout=5.0)
    assert isinstance(result, LLMResponse)
    assert result.text == "Paris is the capital of France."
    assert result.provider == "groq"
    assert result.model == "llama-3.1-8b-instant"
    assert result.latency_ms >= 0.0
    assert result.usage["total_tokens"] == 17
    assert set(result.as_dict()) == {"text", "provider", "model", "latency_ms", "usage"}
    assert _FakeClient.last_payload == {
        "model": "llama-3.1-8b-instant",
        "messages": MESSAGES,
        "temperature": 0.2,
        "max_tokens": 64,
    }
    assert _FakeClient.last_url == "https://api.groq.com/openai/v1/chat/completions"


def test_gemini_response_is_normalized(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("GEMINI_API_KEY", "test-key-not-real")
    payload = {
        "candidates": [{"content": {"parts": [{"text": "Paris."}]}}],
        "modelVersion": "gemini-2.5-flash-lite",
        "usageMetadata": {"promptTokenCount": 11, "candidatesTokenCount": 2, "totalTokenCount": 13},
    }
    _patch_client(monkeypatch, response=_FakeResponse(payload=payload))
    result = GeminiProvider().generate(MESSAGES)
    assert result.text == "Paris."
    assert result.provider == "gemini"
    assert result.usage["total_tokens"] == 13
    assert _FakeClient.last_headers is not None
    assert _FakeClient.last_headers["x-goog-api-key"] == "test-key-not-real"
    assert _FakeClient.last_payload is not None
    assert _FakeClient.last_payload["systemInstruction"]["parts"][0]["text"] == "You are grounded."
    assert _FakeClient.last_payload["contents"][0]["role"] == "user"


def test_ollama_response_is_normalized(monkeypatch: pytest.MonkeyPatch) -> None:
    payload = {
        "model": "qwen2.5:1.5b-instruct",
        "message": {"role": "assistant", "content": "पेरिस।"},
        "prompt_eval_count": 20,
        "eval_count": 4,
        "eval_duration": 2_000_000,
    }
    _patch_client(monkeypatch, response=_FakeResponse(payload=payload))
    result = OllamaProvider().generate(MESSAGES, max_tokens=32)
    assert result.text == "पेरिस।"
    assert result.provider == "ollama"
    assert result.usage["total_tokens"] == 24
    assert result.usage["eval_duration_ms"] == 2.0
    assert _FakeClient.last_url == "http://localhost:11434/api/chat"
    assert _FakeClient.last_payload is not None
    assert _FakeClient.last_payload["options"] == {"temperature": 0.0, "num_predict": 32}
    assert _FakeClient.last_payload["stream"] is False


def test_ollama_needs_no_api_key(monkeypatch: pytest.MonkeyPatch) -> None:
    payload = {"models": [{"name": "qwen2.5:1.5b-instruct"}]}
    _patch_client(monkeypatch, response=_FakeResponse(payload=payload))
    assert OllamaProvider().available() is True
    assert OllamaProvider(model_name="not-pulled").available() is False


def test_ollama_unreachable_is_unavailable(monkeypatch: pytest.MonkeyPatch) -> None:
    _patch_client(monkeypatch, error=httpx.ConnectError("connection refused"))
    assert OllamaProvider().available() is False


# --- zero-cost enforcement ---
def test_openrouter_rejects_paid_model(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key-not-real")
    _patch_client(monkeypatch, response=_FakeResponse(payload=_openai_payload()))
    with pytest.raises(LLMConfigurationError) as excinfo:
        OpenRouterProvider(model_name="openai/gpt-4o").generate(MESSAGES)
    assert "free" in str(excinfo.value).lower()


def test_openrouter_accepts_free_model(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key-not-real")
    _patch_client(monkeypatch, response=_FakeResponse(payload=_openai_payload("free answer")))
    result = OpenRouterProvider(model_name="google/gemma-4-31b-it:free").generate(MESSAGES)
    assert result.text == "free answer"
    assert result.provider == "openrouter"


# --- timeout / failure / malformed ---
def test_provider_timeout_raises_llm_timeout_error(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("GROQ_API_KEY", "test-key-not-real")
    _patch_client(monkeypatch, error=httpx.ReadTimeout("too slow"))
    with pytest.raises(LLMTimeoutError):
        GroqProvider().generate(MESSAGES, timeout=0.01)


def test_provider_http_error_status_raises_response_error(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("GROQ_API_KEY", "test-key-not-real")
    _patch_client(monkeypatch, response=_FakeResponse(status_code=429, payload={"error": "rate limited"}))
    with pytest.raises(LLMResponseError) as excinfo:
        GroqProvider().generate(MESSAGES)
    assert "429" in str(excinfo.value)


def test_provider_transport_error_raises_response_error(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("GROQ_API_KEY", "test-key-not-real")
    _patch_client(monkeypatch, error=httpx.ConnectError("no route"))
    with pytest.raises(LLMResponseError):
        GroqProvider().generate(MESSAGES)


@pytest.mark.parametrize(
    "payload",
    [
        {"choices": []},
        {"choices": [{"message": {}}]},
        {"choices": [{"message": {"content": None}}]},
        {"unexpected": True},
    ],
)
def test_malformed_openai_response(monkeypatch: pytest.MonkeyPatch, payload: dict[str, Any]) -> None:
    monkeypatch.setenv("GROQ_API_KEY", "test-key-not-real")
    _patch_client(monkeypatch, response=_FakeResponse(payload=payload))
    with pytest.raises(LLMResponseError):
        GroqProvider().generate(MESSAGES)


def test_non_json_body_raises_response_error(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("GROQ_API_KEY", "test-key-not-real")
    _patch_client(monkeypatch, response=_FakeResponse(payload=None, text="<html>gateway</html>"))
    with pytest.raises(LLMResponseError):
        GroqProvider().generate(MESSAGES)


def test_malformed_ollama_and_gemini_responses(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("GEMINI_API_KEY", "test-key-not-real")
    _patch_client(monkeypatch, response=_FakeResponse(payload={"message": {}}))
    with pytest.raises(LLMResponseError):
        OllamaProvider().generate(MESSAGES)
    _patch_client(monkeypatch, response=_FakeResponse(payload={"candidates": []}))
    with pytest.raises(LLMResponseError):
        GeminiProvider().generate(MESSAGES)


# --- mock provider ---
def test_mock_provider_behaviors() -> None:
    grounded = MockLLMProvider().generate(MESSAGES)
    assert grounded.provider == "mock"
    assert grounded.model == "mock-rag-generator"
    assert "Paris" in grounded.text
    assert MockLLMProvider("refuse").generate(MESSAGES).text.startswith("REFUSED")
    assert MockLLMProvider("empty").generate(MESSAGES).text == ""
    assert "Eiffel" in MockLLMProvider("ungrounded").generate(MESSAGES).text


# --- fallback chain ---
class _StubProvider(llm_module.LLMProvider):
    def __init__(self, name: str, *, is_available: bool = True, error: Exception | None = None):
        self.name = name
        self._model_name = f"{name}-model"
        self._is_available = is_available
        self._error = error
        self.calls = 0

    def available(self) -> bool:
        return self._is_available

    def generate(
        self,
        messages: list[dict[str, str]],
        *,
        max_tokens: int = 512,
        temperature: float = 0.0,
        timeout: float = 30.0,
    ) -> LLMResponse:
        self.calls += 1
        if self._error is not None:
            raise self._error
        return LLMResponse(text=f"answer from {self.name}", provider=self.name, model=self._model_name)


def test_fallback_skips_unavailable_and_failing_providers() -> None:
    cloud = _StubProvider("groq", is_available=False)
    broken = _StubProvider("gemini", error=LLMResponseError("boom"))
    local = _StubProvider("ollama")
    chain = FallbackLLMProvider([cloud, broken, local])
    result = chain.generate(MESSAGES)
    assert result.provider == "ollama"
    assert cloud.calls == 0 and broken.calls == 1 and local.calls == 1
    assert chain.last_used is local


def test_fallback_raises_when_every_provider_fails() -> None:
    chain = FallbackLLMProvider([_StubProvider("groq", error=LLMTimeoutError("slow"))])
    with pytest.raises(LLMError) as excinfo:
        chain.generate(MESSAGES)
    assert "groq" in str(excinfo.value)


def test_mock_is_never_a_silent_fallback() -> None:
    chain = build_provider_chain("groq", fallback_providers=["ollama", "mock"])
    assert isinstance(chain, FallbackLLMProvider)
    assert [p.name for p in chain.providers] == ["groq", "ollama"]

    explicit = build_provider_chain("groq", fallback_providers=["mock"], allow_mock_fallback=True)
    assert isinstance(explicit, FallbackLLMProvider)
    assert [p.name for p in explicit.providers] == ["groq", "mock"]


def test_single_provider_chain_returns_provider_directly() -> None:
    provider = build_provider_chain("ollama", fallback_providers=["ollama"])
    assert isinstance(provider, OllamaProvider)
