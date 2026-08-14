"""Guardrails for the RAG pipeline.

- input validation (empty / too long / malformed)
- lightweight safety check (unsafe / inappropriate input)
- retrieval quality check (off-topic / low relevance / no evidence)

Kept intentionally lightweight; this is not a production safety classifier.
"""

from __future__ import annotations

from dataclasses import dataclass
import re

from .vector_store import RetrievalHit


@dataclass(slots=True)
class InputValidationResult:
    valid: bool
    reason: str | None = None


@dataclass(slots=True)
class SafetyResult:
    safe: bool
    categories: list[str]
    reason: str | None = None


@dataclass(slots=True)
class RetrievalQualityResult:
    ok: bool
    guardrail: str  # "off_topic" | "low_relevance" | "no_evidence"
    reason: str
    best_score: float


class InputValidator:
    def __init__(self, max_query_chars: int = 1000):
        self._max_query_chars = max_query_chars

    def validate(self, query: str) -> InputValidationResult:
        if query is None:
            return InputValidationResult(valid=False, reason="query is None")
        text = query.strip()
        if not text:
            return InputValidationResult(valid=False, reason="query is empty")
        if len(text) > self._max_query_chars:
            return InputValidationResult(
                valid=False,
                reason=f"query exceeds {self._max_query_chars} characters",
            )
        return InputValidationResult(valid=True)


# Lightweight pattern list. Deliberately conservative and small.
_UNSAFE_PATTERNS: list[tuple[str, str]] = [
    ("hate", r"\b(?:hate|kill|die|die|rape|murder)\b"),
    ("self_harm", r"\b(?:suicide|kill myself|hurt myself|self-harm)\b"),
    ("violence", r"\b(?:bomb|explosive|terrorist attack|how to make a weapon)\b"),
    ("explicit", r"\b(?:porn|sexual content)\b"),
    ("unsafe_hindi", r"(?:मार डाल|बम|आत्महत्या|आपको मार|बलात्कार|किल कर)\b"),
]


class SafetyChecker:
    """Lightweight regex safety gate. Not a substitute for a real classifier."""

    def check(self, text: str) -> SafetyResult:
        if not text:
            return SafetyResult(safe=True, categories=[])
        lowered = text.lower()
        matched: list[str] = []
        for category, pattern in _UNSAFE_PATTERNS:
            if re.search(pattern, lowered):
                matched.append(category)
        if matched:
            return SafetyResult(
                safe=False,
                categories=matched,
                reason=f"input matched unsafe patterns: {', '.join(matched)}",
            )
        return SafetyResult(safe=True, categories=[])


class RetrievalQualityChecker:
    """Decides whether retrieval evidence is strong enough to answer."""

    def __init__(self, min_score: float = 0.35, min_hits: int = 1):
        self._min_score = min_score
        self._min_hits = min_hits

    def check(self, hits: list[RetrievalHit]) -> RetrievalQualityResult:
        if not hits or len(hits) < self._min_hits:
            return RetrievalQualityResult(
                ok=False,
                guardrail="no_evidence",
                reason="no retrieved passages above the evidence threshold",
                best_score=0.0,
            )
        best = hits[0].score
        if best < self._min_score:
            return RetrievalQualityResult(
                ok=False,
                guardrail="low_relevance",
                reason=(
                    f"best retrieval score {best:.4f} is below the evidence "
                    f"threshold {self._min_score:.4f}; the query may be "
                    "off-topic or out of domain"
                ),
                best_score=best,
            )
        return RetrievalQualityResult(
            ok=True,
            guardrail="off_topic",
            reason=None,
            best_score=best,
        )
