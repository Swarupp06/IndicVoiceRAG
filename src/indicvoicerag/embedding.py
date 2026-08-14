from __future__ import annotations

from abc import ABC, abstractmethod
import hashlib
import math
import re
from typing import Iterable

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
    def embed_texts(self, texts: Iterable[str]) -> np.ndarray:
        raise NotImplementedError

    def embed_query(self, text: str) -> np.ndarray:
        vectors = self.embed_texts([text])
        return vectors[0]


class HashEmbeddingProvider(EmbeddingProvider):
    def __init__(self, dimension: int = 256, normalize: bool = True):
        if dimension <= 0:
            raise ValueError("dimension must be positive.")
        self._dimension = dimension
        self._normalize = normalize

    @property
    def dimension(self) -> int:
        return self._dimension

    def embed_texts(self, texts: Iterable[str]) -> np.ndarray:
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


class SentenceTransformerProvider(EmbeddingProvider):
    def __init__(self, model_name: str, normalize: bool = True):
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:
            raise ImportError(
                "sentence-transformers is not installed. Install it before using provider='sentence_transformers'."
            ) from exc
        self._model = SentenceTransformer(model_name)
        self._normalize = normalize
        self._dimension = int(self._model.get_sentence_embedding_dimension())

    @property
    def dimension(self) -> int:
        return self._dimension

    def embed_texts(self, texts: Iterable[str]) -> np.ndarray:
        vectors = self._model.encode(list(texts), convert_to_numpy=True, normalize_embeddings=self._normalize)
        return np.asarray(vectors, dtype=np.float32)


def build_embedding_provider(config: EmbeddingConfig) -> EmbeddingProvider:
    provider = config.provider.lower()
    if provider == "hash":
        return HashEmbeddingProvider(dimension=config.dimension, normalize=config.normalize)
    if provider in {"sentence-transformers", "sentence_transformers"}:
        return SentenceTransformerProvider(model_name=config.model_name, normalize=config.normalize)
    raise ValueError(f"Unsupported embedding provider: {config.provider}")
