import numpy as np

from indicvoicerag.config import EmbeddingConfig
from indicvoicerag.embedding import HashEmbeddingProvider, build_embedding_provider


def test_hash_embedding_provider_shape_and_normalization() -> None:
    provider = HashEmbeddingProvider(dimension=64, normalize=True)
    vectors = provider.embed_texts(["hello world", "hello goa"])
    assert vectors.shape == (2, 64)
    norms = np.linalg.norm(vectors, axis=1)
    assert np.allclose(norms[norms > 0], 1.0)


def test_embedding_provider_factory_hash() -> None:
    provider = build_embedding_provider(EmbeddingConfig(provider="hash", dimension=32))
    assert provider.dimension == 32
