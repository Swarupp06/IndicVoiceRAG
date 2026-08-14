"""Wiring: assemble a working RAG pipeline from configuration.

Reuses the Phase 1B retrieval stack (real dataset sample -> normalization ->
chunking -> real embeddings -> FAISS) and adds the Phase 2 harness on top.
"""

from __future__ import annotations

from .chunking import build_chunker
from .config import AppConfig
from .dataset import DatasetInspector
from .embedding import build_embedding_provider
from .harness import HarnessComponents, RAGHarness
from .llm import build_llm_provider
from .retrieval import RetrievalEngine
from .vector_store import build_vector_store


def build_retrieval_engine(config: AppConfig) -> RetrievalEngine:
    chunker = build_chunker(config.chunking)
    embedder = build_embedding_provider(config.embedding)
    store = build_vector_store(config.vector)
    return RetrievalEngine(chunker=chunker, embedder=embedder, store=store)


def build_indexed_pipeline(
    config: AppConfig,
    size: int | None = None,
    load_persisted: bool = False,
) -> tuple[RAGHarness, DatasetInspector]:
    """Build a harness whose retriever is backed by a real-data index.

    Either indexes a fresh sample (real MSMARCO-XI access, real embeddings,
    FAISS) or loads a persisted index + metadata from `vector.index_path`.
    """
    inspector = DatasetInspector(config.dataset)
    engine = build_retrieval_engine(config)

    if load_persisted:
        engine.store.load()
    else:
        docs = inspector.normalized_sample(size)
        engine.index_documents(docs)

    harness = build_harness(config, engine.query)
    return harness, inspector


def build_harness(config: AppConfig, retrieve_fn) -> RAGHarness:
    """Build a harness around an arbitrary retrieve callable."""
    llm = build_llm_provider(
        provider=config.llm.provider,
        model_name=config.llm.model_name,
        base_url=config.llm.base_url,
        api_key_env=config.llm.api_key_env,
    )
    components = HarnessComponents(
        retrieve_fn=retrieve_fn,
        llm=llm,
        retrieval_config=config.retrieval,
        llm_config=config.llm,
        guardrails=config.guardrails,
    )
    return RAGHarness(components)
