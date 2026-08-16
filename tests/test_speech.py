"""Offline integration tests: audio -> STT -> text -> existing RAGHarness."""

from __future__ import annotations

import wave
from pathlib import Path

import pytest

from indicvoicerag.config import (
    GuardrailConfig,
    LLMConfig,
    RetrievalConfig,
)
from indicvoicerag.harness import HarnessComponents, RAGHarness
from indicvoicerag.llm import MockLLMProvider
from indicvoicerag.speech import answer_from_audio, transcribe_audio
from indicvoicerag.stt import MockSTT
from indicvoicerag.vector_store import RetrievalHit

_CONTEXT = (
    "The capital of India is New Delhi. "
    "It has many historical monuments and a growing population."
)


def _write_wav(path: Path, seconds: float = 0.5, rate: int = 16000) -> Path:
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(rate)
        handle.writeframes(b"\x00\x00" * int(seconds * rate))
    return path


def _harness() -> RAGHarness:
    def retrieve_fn(query: str, top_k: int) -> list[RetrievalHit]:
        return [
            RetrievalHit(
                chunk_id="d1:0",
                document_id="d1",
                score=0.92,
                text=_CONTEXT,
                metadata={
                    "query_id": "q1",
                    "query_text": query,
                    "language": "en",
                    "relevance": 1.0,
                },
            )
        ]

    components = HarnessComponents(
        retrieve_fn=retrieve_fn,
        llm=MockLLMProvider("grounded"),
        retrieval_config=RetrievalConfig(top_k=5),
        llm_config=LLMConfig(provider="mock", max_retries=0),
        guardrails=GuardrailConfig(),
    )
    return RAGHarness(components)


def test_transcribe_audio_returns_structured_result(tmp_path: Path) -> None:
    audio = _write_wav(tmp_path / "speech.wav")
    stt = MockSTT(text="What is the capital of India?")
    result = transcribe_audio(stt, str(audio))
    assert result.text == "What is the capital of India?"
    assert result.duration_seconds == pytest.approx(0.5)


def test_audio_to_stt_to_rag_end_to_end(tmp_path: Path) -> None:
    audio = _write_wav(tmp_path / "question.wav")
    stt = MockSTT(text="What is the capital of India?")
    harness = _harness()
    transcription, response = answer_from_audio(stt, harness, str(audio))
    assert transcription.text == "What is the capital of India?"
    assert response.query == "What is the capital of India?"
    assert response.answer
    assert response.grounded


def test_text_rag_path_unchanged() -> None:
    """The existing text path must keep working independently of STT."""
    harness = _harness()
    response = harness.answer("What is the capital of India?", debug=True)
    assert response.grounded
    assert "New Delhi" in response.answer
    assert "retrieval" in (response.metrics or {})


def test_audio_pipeline_returns_debug_metrics(tmp_path: Path) -> None:
    audio = _write_wav(tmp_path / "debug.wav")
    stt = MockSTT(text="What is the capital of India?")
    harness = _harness()
    transcription, response = answer_from_audio(stt, harness, str(audio), debug=True)
    assert response.metrics is not None
    assert "total" in response.metrics
