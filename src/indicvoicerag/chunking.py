from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
import math
import re
from typing import Iterable

from .config import ChunkingConfig
from .schemas import DocumentChunk, NormalizedDocument


def _tokenize(text: str) -> list[str]:
    return [token for token in re.split(r"\s+", text.strip()) if token]


def _chunk_id(document_id: str, index: int) -> str:
    return f"{document_id}:{index}"


def _base_metadata(document: NormalizedDocument) -> dict[str, object]:
    metadata: dict[str, object] = {
        "query_id": document.query_id,
        "query_text": document.query_text,
        "query_type": document.query_type,
        "language": document.language,
        "source_language": document.source_language,
        "target_language": document.target_language,
        "relevance": document.relevance,
    }
    metadata.update(document.metadata)
    return metadata


class BaseChunker(ABC):
    def __init__(self, config: ChunkingConfig):
        self.config = config

    @abstractmethod
    def chunk(self, document: NormalizedDocument) -> list[DocumentChunk]:
        raise NotImplementedError


class FixedTokenChunker(BaseChunker):
    def chunk(self, document: NormalizedDocument) -> list[DocumentChunk]:
        tokens = _tokenize(document.passage_text)
        if not tokens:
            return []

        window = self.config.chunk_size
        overlap = self.config.overlap
        if window <= 0:
            raise ValueError("chunk_size must be positive.")
        if overlap >= window:
            raise ValueError("overlap must be smaller than chunk_size.")
        stride = window - overlap

        chunks: list[DocumentChunk] = []
        idx = 0
        chunk_index = 0
        while idx < len(tokens):
            segment = tokens[idx : idx + window]
            if not segment:
                break
            text = " ".join(segment)
            chunks.append(
                DocumentChunk(
                    chunk_id=_chunk_id(document.document_id, chunk_index),
                    document_id=document.document_id,
                    text=text,
                    chunk_index=chunk_index,
                    metadata={
                        **_base_metadata(document),
                        "strategy": "fixed",
                        "start_token": idx,
                        "end_token": idx + len(segment),
                    },
                )
            )
            chunk_index += 1
            idx += stride
        return chunks


class SentenceAwareChunker(BaseChunker):
    def chunk(self, document: NormalizedDocument) -> list[DocumentChunk]:
        raw_sentences = re.split(self.config.sentence_separator_regex, document.passage_text.strip())
        sentences = [s.strip() for s in raw_sentences if s and s.strip()]
        if not sentences:
            return []

        max_tokens = self.config.chunk_size
        if max_tokens <= 0:
            raise ValueError("chunk_size must be positive.")

        chunks: list[DocumentChunk] = []
        buffer: list[str] = []
        buffer_tokens = 0
        chunk_index = 0

        for sentence in sentences:
            sentence_tokens = _tokenize(sentence)
            if not sentence_tokens:
                continue
            if buffer and buffer_tokens + len(sentence_tokens) > max_tokens:
                chunks.append(
                    DocumentChunk(
                        chunk_id=_chunk_id(document.document_id, chunk_index),
                        document_id=document.document_id,
                        text=" ".join(buffer),
                        chunk_index=chunk_index,
                        metadata={
                            **_base_metadata(document),
                            "strategy": "sentence_aware",
                            "sentence_count": len(buffer),
                        },
                    )
                )
                chunk_index += 1
                buffer = []
                buffer_tokens = 0

            buffer.append(sentence)
            buffer_tokens += len(sentence_tokens)

        if buffer:
            chunks.append(
                DocumentChunk(
                    chunk_id=_chunk_id(document.document_id, chunk_index),
                    document_id=document.document_id,
                    text=" ".join(buffer),
                    chunk_index=chunk_index,
                    metadata={
                        **_base_metadata(document),
                        "strategy": "sentence_aware",
                        "sentence_count": len(buffer),
                    },
                )
            )
        return chunks


def _token_frequency_vector(tokens: Iterable[str]) -> dict[str, float]:
    counts: dict[str, float] = {}
    for token in tokens:
        counts[token] = counts.get(token, 0.0) + 1.0
    return counts


def _cosine_similarity(left: dict[str, float], right: dict[str, float]) -> float:
    if not left or not right:
        return 0.0
    dot = 0.0
    for token, l_count in left.items():
        dot += l_count * right.get(token, 0.0)
    left_norm = math.sqrt(sum(v * v for v in left.values()))
    right_norm = math.sqrt(sum(v * v for v in right.values()))
    if left_norm == 0.0 or right_norm == 0.0:
        return 0.0
    return dot / (left_norm * right_norm)


class SemanticChunker(BaseChunker):
    def chunk(self, document: NormalizedDocument) -> list[DocumentChunk]:
        raw_sentences = re.split(self.config.sentence_separator_regex, document.passage_text.strip())
        sentences = [s.strip() for s in raw_sentences if s and s.strip()]
        if not sentences:
            return []

        threshold = self.config.semantic_threshold
        max_tokens = self.config.chunk_size
        if max_tokens <= 0:
            raise ValueError("chunk_size must be positive.")

        chunks: list[DocumentChunk] = []
        current: list[str] = [sentences[0]]
        current_tokens = len(_tokenize(sentences[0]))
        current_signature = _token_frequency_vector(_tokenize(sentences[0]))
        chunk_index = 0

        for sentence in sentences[1:]:
            sentence_tokens = _tokenize(sentence)
            if not sentence_tokens:
                continue
            sentence_signature = _token_frequency_vector(sentence_tokens)
            similarity = _cosine_similarity(current_signature, sentence_signature)
            would_overflow = current_tokens + len(sentence_tokens) > max_tokens
            should_break = similarity < threshold or would_overflow

            if should_break:
                chunks.append(
                    DocumentChunk(
                        chunk_id=_chunk_id(document.document_id, chunk_index),
                        document_id=document.document_id,
                        text=" ".join(current),
                        chunk_index=chunk_index,
                        metadata={
                            **_base_metadata(document),
                            "strategy": "semantic",
                            "semantic_threshold": threshold,
                        },
                    )
                )
                chunk_index += 1
                current = [sentence]
                current_tokens = len(sentence_tokens)
                current_signature = sentence_signature
            else:
                current.append(sentence)
                current_tokens += len(sentence_tokens)
                for token, value in sentence_signature.items():
                    current_signature[token] = current_signature.get(token, 0.0) + value

        if current:
            chunks.append(
                DocumentChunk(
                    chunk_id=_chunk_id(document.document_id, chunk_index),
                    document_id=document.document_id,
                    text=" ".join(current),
                    chunk_index=chunk_index,
                    metadata={
                        **_base_metadata(document),
                        "strategy": "semantic",
                        "semantic_threshold": threshold,
                    },
                )
            )
        return chunks


def build_chunker(config: ChunkingConfig) -> BaseChunker:
    strategy = config.strategy.lower()
    if strategy == "fixed":
        return FixedTokenChunker(config)
    if strategy in {"sentence", "sentence_aware"}:
        return SentenceAwareChunker(config)
    if strategy == "semantic":
        return SemanticChunker(config)
    raise ValueError(f"Unknown chunking strategy: {config.strategy}")
