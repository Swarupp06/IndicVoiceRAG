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
from .llm import LLMProvider, build_provider_chain, normalize_provider_name
from .retrieval import RetrievalEngine
from .stt import STTProvider, build_stt_provider
from .tts import TTSProvider, build_tts_provider
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


_MOCK_MODEL_NAMES = {"", "mock-rag-generator"}


def build_llm_from_config(config: AppConfig) -> LLMProvider:
    """Build the configured provider chain, ignoring a stale mock model name."""
    model_name: str | None = config.llm.model_name
    if normalize_provider_name(config.llm.provider) != "mock" and model_name in _MOCK_MODEL_NAMES:
        model_name = None
    return build_provider_chain(
        provider=config.llm.provider,
        model_name=model_name,
        base_url=config.llm.base_url or None,
        api_key_env=config.llm.api_key_env or None,
        fallback_providers=config.llm.fallback_providers,
        fallback_models=config.llm.fallback_models,
        allow_mock_fallback=config.llm.allow_mock_fallback,
    )


def build_harness(config: AppConfig, retrieve_fn) -> RAGHarness:
    """Build a harness around an arbitrary retrieve callable."""
    llm = build_llm_from_config(config)
    components = HarnessComponents(
        retrieve_fn=retrieve_fn,
        llm=llm,
        retrieval_config=config.retrieval,
        llm_config=config.llm,
        guardrails=config.guardrails,
    )
    return RAGHarness(components)


def build_stt_from_config(config: AppConfig) -> STTProvider:
    """Build the configured STT provider (mock | faster_whisper)."""
    return build_stt_provider(
        provider=config.stt.provider,
        model_name=config.stt.model_name,
        device=config.stt.device,
        compute_type=config.stt.compute_type,
        language=config.stt.language,
        beam_size=config.stt.beam_size,
        vad_filter=config.stt.vad_filter,
        download_root=config.stt.download_root,
    )


def build_tts_from_config(config: AppConfig) -> TTSProvider:
    """Build the configured TTS provider (mock | mms)."""
    return build_tts_provider(
        provider=config.tts.provider,
        language=config.tts.language,
        cache_dir=config.tts.cache_dir,
    )
