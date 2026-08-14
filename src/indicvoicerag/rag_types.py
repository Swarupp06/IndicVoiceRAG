"""Structured input/output types for the Phase 2 RAG pipeline."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(slots=True)
class SourceInfo:
    """One retrieved passage cited as evidence for an answer."""

    document_id: str
    chunk_id: str
    score: float
    query_id: str | None = None
    language: str | None = None
    relevance: float | None = None
    excerpt: str = ""

    def as_dict(self) -> dict[str, Any]:
        return {
            "document_id": self.document_id,
            "chunk_id": self.chunk_id,
            "score": round(self.score, 4),
            "query_id": self.query_id,
            "language": self.language,
            "relevance": self.relevance,
            "excerpt": self.excerpt[:300],
        }


@dataclass(slots=True)
class RAGResponse:
    """Structured output of the RAG harness.

    `guardrail` is None on the normal answered path and carries the guardrail
    name (e.g. 'invalid_input', 'unsafe_input', 'off_topic', 'low_relevance',
    'no_evidence', 'generation_failed', 'ungrounded') on any other path.
    """

    query: str
    answer: str
    grounded: bool
    confidence: float
    sources: list[SourceInfo] = field(default_factory=list)
    reason: str | None = None
    guardrail: str | None = None
    metrics: dict[str, float] | None = None

    def as_dict(self, include_metrics: bool = False) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "query": self.query,
            "answer": self.answer,
            "grounded": self.grounded,
            "confidence": round(self.confidence, 4),
            "sources": [source.as_dict() for source in self.sources],
            "reason": self.reason,
            "guardrail": self.guardrail,
        }
        if include_metrics and self.metrics is not None:
            payload["metrics_ms"] = {k: round(v, 1) for k, v in self.metrics.items()}
        return payload
