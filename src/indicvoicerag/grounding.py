"""Grounding validation for generated answers.

The abstraction allows swapping in stronger validators later (embedding
similarity, NLI models, LLM-as-judge). The current implementation is a
lexical containment check: each answer sentence must overlap the retrieved
context on content tokens.

LIMITATIONS of the lexical method (documented, see README):
- It is NOT a hallucination detector. Paraphrases with low token overlap are
  flagged even when semantically supported.
- Numeric facts, dates and named entities are not checked individually.
- Indic morphology (inflection) can reduce token overlap for valid answers.
  A semantic (embedding-based or NLI) validator is the planned upgrade.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
import re
from typing import Iterable

from .prompts import is_refusal


@dataclass(slots=True)
class GroundingResult:
    grounded: bool
    score: float
    unsupported_claims: list[str] = field(default_factory=list)
    reason: str | None = None

    def as_dict(self) -> dict[str, object]:
        return {
            "grounded": self.grounded,
            "score": round(self.score, 4),
            "unsupported_claims": self.unsupported_claims,
            "reason": self.reason,
        }


class GroundingValidator(ABC):
    @abstractmethod
    def validate(self, answer: str, context_text: str, query: str) -> GroundingResult:
        raise NotImplementedError


_STOPWORDS = {
    "a", "an", "the", "and", "or", "but", "of", "in", "on", "at", "to", "for",
    "with", "is", "are", "was", "were", "be", "been", "being", "it", "its",
    "this", "that", "these", "those", "from", "as", "by", "has", "have", "had",
    "do", "does", "did", "not", "no", "yes", "i", "you", "he", "she", "we",
    "they", "है", "हैं", "और", "में", "की", "का", "के", "को", "से", "पर", "नहीं",
    "एक", "था", "थे", "हो", "हैं", "यह", "वह", "कर", "किया", "की", "ने",
}


def _split_sentences(text: str) -> list[str]:
    parts = re.split(r"(?<=[.!?।])\s+", text.strip())
    return [part.strip() for part in parts if part and part.strip()]


def _content_tokens(text: str) -> list[str]:
    tokens: list[str] = []
    for token in re.split(r"[\s\W_]+", text.lower()):
        if not token:
            continue
        if token in _STOPWORDS:
            continue
        if len(token) < 2:
            continue
        tokens.append(token)
    return tokens


def _overlap_ratio(claim_tokens: Iterable[str], context_tokens: set[str]) -> float:
    unique = list(set(claim_tokens))
    if not unique:
        return 1.0
    matched = sum(1 for token in unique if token in context_tokens)
    return matched / len(unique)


class LexicalGroundingValidator(GroundingValidator):
    """Per-sentence lexical containment of the answer within the context."""

    def __init__(self, threshold: float = 0.35):
        self._threshold = threshold

    def validate(self, answer: str, context_text: str, query: str) -> GroundingResult:
        if not answer or not answer.strip():
            return GroundingResult(
                grounded=False,
                score=0.0,
                unsupported_claims=[],
                reason="generated answer was empty",
            )
        if is_refusal(answer):
            return GroundingResult(
                grounded=True,
                score=1.0,
                unsupported_claims=[],
                reason="generator refused; no claims to ground",
            )

        context_tokens = set(_content_tokens(context_text))
        claims = _split_sentences(answer)
        unsupported: list[str] = []
        claim_scores: list[float] = []
        for claim in claims:
            claim_tokens = _content_tokens(claim)
            ratio = _overlap_ratio(claim_tokens, context_tokens)
            claim_scores.append(ratio)
            if ratio < self._threshold:
                unsupported.append(claim)

        if claims:
            score = sum(claim_scores) / len(claim_scores)
        else:
            score = 0.0

        if unsupported:
            reason = (
                f"{len(unsupported)} of {len(claims)} claim(s) could not be "
                f"lexically grounded (threshold={self._threshold})."
            )
        else:
            reason = None

        return GroundingResult(
            grounded=not unsupported,
            score=score,
            unsupported_claims=unsupported,
            reason=reason,
        )


def build_grounding_validator(threshold: float) -> GroundingValidator:
    return LexicalGroundingValidator(threshold=threshold)
