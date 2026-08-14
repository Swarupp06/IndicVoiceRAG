from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, asdict
import json
from pathlib import Path
from typing import Any
import warnings

import numpy as np

from .config import VectorConfig
from .schemas import DocumentChunk


@dataclass(slots=True)
class RetrievalHit:
    chunk_id: str
    document_id: str
    score: float
    text: str
    metadata: dict[str, Any]


class VectorStore(ABC):
    @abstractmethod
    def add(self, embeddings: np.ndarray, chunks: list[DocumentChunk]) -> None:
        raise NotImplementedError

    @abstractmethod
    def search(self, query_embedding: np.ndarray, top_k: int) -> list[RetrievalHit]:
        raise NotImplementedError

    @abstractmethod
    def save(self) -> None:
        raise NotImplementedError

    @abstractmethod
    def load(self) -> None:
        raise NotImplementedError


class NumpyVectorStore(VectorStore):
    def __init__(self, config: VectorConfig):
        self._config = config
        self._embeddings: np.ndarray | None = None
        self._chunks: list[DocumentChunk] = []

    def add(self, embeddings: np.ndarray, chunks: list[DocumentChunk]) -> None:
        if len(embeddings) != len(chunks):
            raise ValueError("Embeddings and chunks count mismatch.")
        if len(chunks) == 0:
            return
        if self._embeddings is None:
            self._embeddings = embeddings.astype(np.float32)
        else:
            self._embeddings = np.vstack([self._embeddings, embeddings.astype(np.float32)])
        self._chunks.extend(chunks)

    def search(self, query_embedding: np.ndarray, top_k: int) -> list[RetrievalHit]:
        if self._embeddings is None or len(self._chunks) == 0:
            return []
        query = query_embedding.astype(np.float32)
        scores = self._embeddings @ query
        ranked_indices = np.argsort(scores)[::-1][:top_k]
        hits: list[RetrievalHit] = []
        for idx in ranked_indices:
            chunk = self._chunks[int(idx)]
            hits.append(
                RetrievalHit(
                    chunk_id=chunk.chunk_id,
                    document_id=chunk.document_id,
                    score=float(scores[int(idx)]),
                    text=chunk.text,
                    metadata=chunk.metadata,
                )
            )
        return hits

    def save(self) -> None:
        if self._embeddings is None:
            return
        index_path = Path(self._config.index_path)
        metadata_path = Path(self._config.metadata_path)
        index_path.parent.mkdir(parents=True, exist_ok=True)
        metadata_path.parent.mkdir(parents=True, exist_ok=True)
        np.save(index_path.with_suffix(".npy"), self._embeddings)
        self._write_metadata(metadata_path)

    def load(self) -> None:
        index_path = Path(self._config.index_path)
        metadata_path = Path(self._config.metadata_path)
        npy_path = index_path.with_suffix(".npy")
        if not npy_path.exists() or not metadata_path.exists():
            raise FileNotFoundError(f"Index or metadata missing: {npy_path}, {metadata_path}")
        self._embeddings = np.load(npy_path)
        self._chunks = self._read_metadata(metadata_path)

    def _write_metadata(self, metadata_path: Path) -> None:
        with metadata_path.open("w", encoding="utf-8") as handle:
            for chunk in self._chunks:
                handle.write(json.dumps(asdict(chunk), ensure_ascii=False) + "\n")

    @staticmethod
    def _read_metadata(metadata_path: Path) -> list[DocumentChunk]:
        chunks: list[DocumentChunk] = []
        with metadata_path.open("r", encoding="utf-8") as handle:
            for line in handle:
                text = line.strip()
                if not text:
                    continue
                payload = json.loads(text)
                chunks.append(DocumentChunk(**payload))
        return chunks


class FaissVectorStore(VectorStore):
    def __init__(self, config: VectorConfig):
        try:
            import faiss
        except ImportError as exc:
            raise ImportError(
                "faiss-cpu is not installed. Install optional dependency or use provider='numpy'."
            ) from exc
        self._faiss = faiss
        self._config = config
        self._index: Any | None = None
        self._chunks: list[DocumentChunk] = []

    def add(self, embeddings: np.ndarray, chunks: list[DocumentChunk]) -> None:
        if len(embeddings) != len(chunks):
            raise ValueError("Embeddings and chunks count mismatch.")
        if len(chunks) == 0:
            return
        matrix = embeddings.astype(np.float32)
        if self._index is None:
            dimension = matrix.shape[1]
            self._index = self._faiss.IndexFlatIP(dimension)
        self._index.add(matrix)
        self._chunks.extend(chunks)

    def search(self, query_embedding: np.ndarray, top_k: int) -> list[RetrievalHit]:
        if self._index is None:
            return []
        query = np.expand_dims(query_embedding.astype(np.float32), axis=0)
        scores, indices = self._index.search(query, top_k)
        hits: list[RetrievalHit] = []
        for score, idx in zip(scores[0], indices[0], strict=True):
            if idx < 0 or idx >= len(self._chunks):
                continue
            chunk = self._chunks[int(idx)]
            hits.append(
                RetrievalHit(
                    chunk_id=chunk.chunk_id,
                    document_id=chunk.document_id,
                    score=float(score),
                    text=chunk.text,
                    metadata=chunk.metadata,
                )
            )
        return hits

    def save(self) -> None:
        if self._index is None:
            return
        index_path = Path(self._config.index_path)
        metadata_path = Path(self._config.metadata_path)
        index_path.parent.mkdir(parents=True, exist_ok=True)
        metadata_path.parent.mkdir(parents=True, exist_ok=True)
        self._faiss.write_index(self._index, str(index_path))
        with metadata_path.open("w", encoding="utf-8") as handle:
            for chunk in self._chunks:
                handle.write(json.dumps(asdict(chunk), ensure_ascii=False) + "\n")

    def load(self) -> None:
        index_path = Path(self._config.index_path)
        metadata_path = Path(self._config.metadata_path)
        if not index_path.exists() or not metadata_path.exists():
            raise FileNotFoundError(f"Index or metadata missing: {index_path}, {metadata_path}")
        self._index = self._faiss.read_index(str(index_path))
        self._chunks = NumpyVectorStore._read_metadata(metadata_path)


def build_vector_store(config: VectorConfig) -> VectorStore:
    provider = config.provider.lower()
    if provider == "numpy":
        return NumpyVectorStore(config)
    if provider == "faiss":
        try:
            return FaissVectorStore(config)
        except ImportError:
            warnings.warn(
                "faiss-cpu is not installed; falling back to the NumPy vector store.",
                stacklevel=2,
            )
            return NumpyVectorStore(config)
    raise ValueError(f"Unsupported vector provider: {config.provider}")
