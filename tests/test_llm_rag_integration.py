"""RAG-harness integration tests for the provider layer (no real API calls)."""

from __future__ import annotations

from typing import Any

import pytest

from indicvoicerag.benchmark import BenchmarkQuery, load_benchmark_queries, percentile, run_benchmark, summarize
from indicvoicerag.config import AppConfig, GuardrailConfig, LLMConfig, RetrievalConfig
from indicvoicerag.harness import HarnessComponents, RAGHarness
from indicvoicerag.llm import (
    LLMResponse,
    LLMProvider,
    LLMResponseError,
    LLMTimeoutError,
    MockLLMProvider,
)
from indicvoicerag.vector_store import RetrievalHit

_PASSAGE = (
    "The Ganges is the longest river in India and flows for about 2525 kilometres "
    "from the Himalayas to the Bay of Bengal."
)


def _hit(score: float = 0.9) -> RetrievalHit:
    return RetrievalHit(
        chunk_id="d1:0",
        document_id="d1",
        score=score,
        text=_PASSAGE,
        metadata={"query_id": "q1", "language": "eng", "relevance": 1.0},
    )


class _RecordingProvider(LLMProvider):
    """Captures the generate() kwargs the harness passes through."""

    name = "recording"

    def __init__(self, text: str = _PASSAGE, error: Exception | None = None):
        self._model_name = "recording-model"
        self._text = text
        self._error = error
        self.calls: list[dict[str, Any]] = []

    def generate(
        self,
        messages: list[dict[str, str]],
        *,
        max_tokens: int = 512,
        temperature: float = 0.0,
        timeout: float = 30.0,
    ) -> LLMResponse:
        self.calls.append(
            {
                "messages": messages,
                "max_tokens": max_tokens,
                "temperature": temperature,
                "timeout": timeout,
            }
        )
        if self._error is not None:
            raise self._error
        return LLMResponse(
            text=self._text,
            provider=self.name,
            model=self._model_name,
            latency_ms=12.5,
            usage={"total_tokens": 42},
        )


def _harness(llm: LLMProvider, hits: list[RetrievalHit] | None = None) -> RAGHarness:
    resolved = hits if hits is not None else [_hit()]
    components = HarnessComponents(
        retrieve_fn=lambda query, k: resolved,
        llm=llm,
        retrieval_config=RetrievalConfig(top_k=3),
        llm_config=LLMConfig(provider="recording", model_name="recording-model", max_retries=1, max_tokens=128),
        guardrails=GuardrailConfig(),
    )
    return RAGHarness(components)


def test_harness_passes_generation_parameters_through() -> None:
    provider = _RecordingProvider()
    response = _harness(provider).answer("How long is the Ganges?")
    call = provider.calls[0]
    assert call["max_tokens"] == 128
    assert call["temperature"] == 0.0
    assert call["timeout"] == 30.0
    assert [m["role"] for m in call["messages"]] == ["system", "user"]
    assert response.grounded is True


def test_structured_response_carries_provider_metadata() -> None:
    response = _harness(_RecordingProvider()).answer("How long is the Ganges?", debug=True)
    assert response.llm == {
        "provider": "recording",
        "model": "recording-model",
        "latency_ms": 12.5,
        "usage": {"total_tokens": 42},
    }
    payload = response.as_dict(include_metrics=True)
    assert payload["llm"]["provider"] == "recording"
    assert set(payload["metrics_ms"]) >= {"retrieval", "context", "generation", "total"}


@pytest.mark.parametrize("error", [LLMTimeoutError("timed out"), LLMResponseError("boom"), RuntimeError("x")])
def test_provider_failure_is_contained_by_guardrails(error: Exception) -> None:
    provider = _RecordingProvider(error=error)
    response = _harness(provider).answer("How long is the Ganges?")
    assert response.guardrail == "generation_failed"
    assert response.grounded is False
    assert type(error).__name__ in (response.reason or "")
    assert len(provider.calls) == 2  # initial attempt + one configured retry


def test_mock_provider_runs_end_to_end_offline() -> None:
    response = _harness(MockLLMProvider()).answer("How long is the Ganges?")
    assert response.llm is not None
    assert response.llm["provider"] == "mock"
    assert response.grounded is True


# --- benchmark harness ---
def test_percentile_nearest_rank() -> None:
    values = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
    assert percentile(values, 50) == 50
    assert percentile(values, 70) == 70
    assert percentile(values, 100) == 100
    assert percentile([], 50) == 0.0
    assert percentile([7.5], 50) == 7.5


def test_run_benchmark_records_stages_and_summary() -> None:
    harness = _harness(_RecordingProvider())
    queries = [
        BenchmarkQuery(id="q1", category="english_supported", language="en", query="How long is the Ganges?"),
        BenchmarkQuery(id="q2", category="indic_supported", language="hi", query="गंगा कितनी लंबी है?"),
    ]
    records = run_benchmark(harness, queries, provider_label="recording", model_label="recording-model")
    assert len(records) == 2
    first = records[0]
    assert first.provider == "recording"
    assert first.model == "recording-model"
    assert first.total_latency_ms >= first.retrieval_latency_ms
    summary = summarize(records)
    assert summary["queries"] == 2
    assert summary["errors"] == 0
    assert set(summary["latency_ms"]) == {"retrieval", "context", "generation", "grounding", "total"}
    assert set(summary["latency_ms"]["total"]) == {"p50", "p70", "p100", "avg", "n"}
    assert summary["by_category"]["english_supported"]["total"] == 1


def test_run_benchmark_records_provider_errors_without_raising() -> None:
    class _Exploding(LLMProvider):
        name = "exploding"

        def generate(self, messages: list[dict[str, str]], **kwargs: Any) -> LLMResponse:
            raise AssertionError("provider blew up")

    harness = _harness(_Exploding())
    harness._components.retrieve_fn = lambda query, k: (_ for _ in ()).throw(RuntimeError("retriever down"))  # noqa: SLF001
    records = run_benchmark(
        harness,
        [BenchmarkQuery(id="q1", category="unsupported", language="en", query="anything")],
        provider_label="exploding",
        model_label="none",
    )
    assert records[0].guardrail == "benchmark_error"
    assert "RuntimeError" in (records[0].error or "")
    assert summarize(records)["errors"] == 1


def test_benchmark_query_set_is_fixed_and_covers_every_category() -> None:
    queries = load_benchmark_queries()
    assert 15 <= len(queries) <= 30
    assert len({q.id for q in queries}) == len(queries)
    categories = {q.category for q in queries}
    assert categories == {
        "english_supported",
        "indic_supported",
        "evidence_required",
        "low_relevance",
        "unsupported",
    }
    assert any(q.language == "hi" for q in queries)


def test_app_config_defaults_never_enable_silent_mock_fallback() -> None:
    config = AppConfig()
    assert config.llm.allow_mock_fallback is False
    assert config.llm.fallback_providers == []
