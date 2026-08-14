from __future__ import annotations

from abc import ABC, abstractmethod
import hashlib
import re
from typing import Any, Iterable

import numpy as np

from .config import EmbeddingConfig


def _tokenize(text: str) -> list[str]:
    return [tok for tok in re.split(r"\s+", text.lower().strip()) if tok]


class EmbeddingProvider(ABC):
    @property
    @abstractmethod
    def dimension(self) -> int:
        raise NotImplementedError

    @abstractmethod
    def embed_texts(self, texts: Iterable[str], purpose: str = "document") -> np.ndarray:
        """Embed texts. `purpose` is 'query' or 'document' (drives model prefixes)."""
        raise NotImplementedError

    def embed_documents(self, texts: Iterable[str]) -> np.ndarray:
        return self.embed_texts(texts, purpose="document")

    def embed_query(self, text: str) -> np.ndarray:
        vectors = self.embed_texts([text], purpose="query")
        return vectors[0]

    def describe(self) -> dict[str, Any]:
        return {"provider": self.__class__.__name__, "dimension": self.dimension}


class HashEmbeddingProvider(EmbeddingProvider):
    """Deterministic hashed bag-of-words vectorizer.

    Intended for offline tests and dimensionality checks only. It is NOT a
    semantic embedding and must not be used as the production default.
    """

    def __init__(self, dimension: int = 256, normalize: bool = True):
        if dimension <= 0:
            raise ValueError("dimension must be positive.")
        self._dimension = dimension
        self._normalize = normalize

    @property
    def dimension(self) -> int:
        return self._dimension

    def embed_texts(self, texts: Iterable[str], purpose: str = "document") -> np.ndarray:
        rows: list[np.ndarray] = []
        for text in texts:
            vec = np.zeros(self._dimension, dtype=np.float32)
            tokens = _tokenize(text)
            for token in tokens:
                digest = hashlib.sha1(token.encode("utf-8")).hexdigest()
                slot = int(digest[:8], 16) % self._dimension
                sign = 1.0 if int(digest[8:10], 16) % 2 == 0 else -1.0
                vec[slot] += sign
            if self._normalize:
                norm = float(np.linalg.norm(vec))
                if norm > 0:
                    vec /= norm
            rows.append(vec)
        if not rows:
            return np.empty((0, self._dimension), dtype=np.float32)
        return np.vstack(rows)

    def describe(self) -> dict[str, Any]:
        return {"provider": "hash", "model": "hash-256", "dimension": self._dimension}


def _detect_prefix_style(model_name: str) -> str:
    lowered = model_name.lower()
    if "e5" in lowered or "instructor" in lowered:
        return "e5"
    return "none"


class SentenceTransformerProvider(EmbeddingProvider):
    """Real multilingual embeddings via sentence-transformers (PyTorch).

    Recommended models (all support Hindi + 15+ other Indic scripts):
      - intfloat/multilingual-e5-small (384-dim, ~118M params, fast on CPU)
      - intfloat/multilingual-e5-base  (768-dim, ~278M params)
    e5 models require 'query: ' / 'passage: ' prefixes; they are applied
    automatically based on the encode purpose.
    """

    def __init__(self, model_name: str, normalize: bool = True, prefix_style: str | None = None):
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:
            raise ImportError(
                "sentence-transformers (and torch) is not installed. Install it before using "
                "provider='sentence_transformers'."
            ) from exc
        self._model = SentenceTransformer(model_name)
        self._normalize = normalize
        try:
            self._dimension = int(self._model.get_embedding_dimension())
        except AttributeError:
            self._dimension = int(self._model.get_sentence_embedding_dimension())
        self._model_name = model_name
        self._prefix_style = prefix_style or _detect_prefix_style(model_name)

    @property
    def dimension(self) -> int:
        return self._dimension

    def _prefixed(self, texts: list[str], purpose: str) -> list[str]:
        if self._prefix_style != "e5":
            return texts
        prefix = "query: " if purpose == "query" else "passage: "
        return [prefix + text for text in texts]

    def embed_texts(self, texts: Iterable[str], purpose: str = "document") -> np.ndarray:
        payload = self._prefixed(list(texts), purpose)
        vectors = self._model.encode(
            payload,
            convert_to_numpy=True,
            normalize_embeddings=self._normalize,
        )
        return np.asarray(vectors, dtype=np.float32)

    def describe(self) -> dict[str, Any]:
        return {
            "provider": "sentence_transformers",
            "model": self._model_name,
            "dimension": self._dimension,
            "runtime": "pytorch-cpu",
            "prefix_style": self._prefix_style,
        }


class FastEmbedProvider(EmbeddingProvider):
    """Real multilingual embeddings via fastembed (ONNX runtime, CPU).

    A lightweight alternative to PyTorch with faster CPU inference and a much
    smaller install. Model must be in fastembed's supported list, e.g.
    sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2 (384-dim).
    """

    def __init__(self, model_name: str = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2", normalize: bool = True):
        try:
            from fastembed import TextEmbedding
        except ImportError as exc:
            raise ImportError(
                "fastembed is not installed. Install it before using provider='fastembed'."
            ) from exc
        self._model = TextEmbedding(model_name)
        self._normalize = normalize
        size_getter = getattr(self._model, "get_embedding_size", None)
        if size_getter is not None:
            self._dimension = int(size_getter(self._model.model_name))
        else:
            self._dimension = int(getattr(self._model, "embedding_size"))
        self._model_name = model_name

    @property
    def dimension(self) -> int:
        return self._dimension

    def embed_texts(self, texts: Iterable[str], purpose: str = "document") -> np.ndarray:
        payload = list(texts)
        if not payload:
            return np.empty((0, self._dimension), dtype=np.float32)
        if purpose == "query":
            vectors = [list(self._model.query_embed(text))[0] for text in payload]
        else:
            vectors = [v.tolist() for v in self._model.embed(payload)]
        matrix = np.asarray(vectors, dtype=np.float32)
        if self._normalize:
            norms = np.linalg.norm(matrix, axis=1, keepdims=True)
            norms[norms == 0] = 1.0
            matrix = matrix / norms
        return matrix

    def describe(self) -> dict[str, Any]:
        return {
            "provider": "fastembed",
            "model": self._model_name,
            "dimension": self._dimension,
            "runtime": "onnx-cpu",
        }


def build_embedding_provider(config: EmbeddingConfig) -> EmbeddingProvider:
    provider = config.provider.lower().replace("-", "_")
    if provider == "hash":
        return HashEmbeddingProvider(dimension=config.dimension, normalize=config.normalize)
    if provider == "sentence_transformers":
        return SentenceTransformerProvider(model_name=config.model_name, normalize=config.normalize)
    if provider == "fastembed":
        return FastEmbedProvider(model_name=config.model_name, normalize=config.normalize)
    raise ValueError(f"Unsupported embedding provider: {config.provider}")
