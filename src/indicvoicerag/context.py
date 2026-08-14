"""Context engineering for the RAG pipeline.

Responsibilities:
- order retrieved passages by relevance (already rank-ordered by the store)
- remove duplicate passages (by normalized text)
- cap the number of sources and the total context token budget
- emit the context with clear per-source boundaries and metadata
"""

from __future__ import annotations

from dataclasses import dataclass, field
import re

from .config import GuardrailConfig
from .vector_store import RetrievalHit


CONTEXT_BEGIN = "BEGIN RETRIEVED CONTEXT"
CONTEXT_END = "END RETRIEVED CONTEXT"


def _normalize(text: str) -> str:
    return re.sub(r"\s+", " ", text.strip().lower())


def _estimate_tokens(text: str) -> int:
    return len(text.split())


@dataclass(slots=True)
class ContextPassage:
    """One delimited evidence passage selected for the LLM context."""

    source_index: int
    chunk_id: str
    document_id: str
    score: float
    text: str
    metadata: dict[str, object] = field(default_factory=dict)

    def header(self) -> str:
        return f"[Source {self.source_index}] (document_id={self.document_id}, score={self.score:.4f})"


class ContextBuilder:
    """Build a deduplicated, bounded, delimited context from retrieval hits."""

    def __init__(self, config: GuardrailConfig):
        self._config = config

    def build(self, hits: list[RetrievalHit]) -> list[ContextPassage]:
        """Select and order passages, enforcing dedup / doc count / token limits."""
        budget = max(1, int(self._config.context_max_tokens))
        max_docs = max(1, int(self._config.max_context_docs))

        seen_text: set[str] = set()
        passages: list[ContextPassage] = []
        total_tokens = 0

        for hit in hits:
            if len(passages) >= max_docs:
                break
            norm = _normalize(hit.text)
            if not norm or norm in seen_text:
                continue
            seen_text.add(norm)

            tokens = _estimate_tokens(hit.text)
            if total_tokens + tokens > budget:
                remaining = budget - total_tokens
                if remaining <= 0:
                    break
                trimmed = " ".join(hit.text.split()[:remaining])
                tokens = remaining
                hit_text = trimmed
            else:
                hit_text = hit.text

            passages.append(
                ContextPassage(
                    source_index=len(passages) + 1,
                    chunk_id=hit.chunk_id,
                    document_id=hit.document_id,
                    score=hit.score,
                    text=hit_text,
                    metadata=hit.metadata,
                )
            )
            total_tokens += tokens
            if total_tokens >= budget:
                break

        return passages

    def render(self, passages: list[ContextPassage]) -> str:
        """Render the context with explicit delimiters and source boundaries."""
        if not passages:
            return f"{CONTEXT_BEGIN}\n(no retrieved context)\n{CONTEXT_END}"
        blocks = [CONTEXT_BEGIN]
        for passage in passages:
            blocks.append(passage.header())
            blocks.append(passage.text)
        blocks.append(CONTEXT_END)
        return "\n\n".join(blocks)
