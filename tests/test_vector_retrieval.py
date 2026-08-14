from pathlib import Path

from indicvoicerag.chunking import build_chunker
from indicvoicerag.config import ChunkingConfig, EmbeddingConfig, VectorConfig
from indicvoicerag.embedding import build_embedding_provider
from indicvoicerag.retrieval import RetrievalEngine
from indicvoicerag.schemas import NormalizedDocument
from indicvoicerag.vector_store import build_vector_store


def _docs() -> list[NormalizedDocument]:
    return [
        NormalizedDocument(
            document_id="d1",
            query_id=None,
            query_text=None,
            passage_text="Goa has many beaches and coastal tourism.",
            language="en",
            source_language=None,
            target_language=None,
            query_type=None,
            relevance=None,
            metadata={},
        ),
        NormalizedDocument(
            document_id="d2",
            query_id=None,
            query_text=None,
            passage_text="Delhi is the capital of India.",
            language="en",
            source_language=None,
            target_language=None,
            query_type=None,
            relevance=None,
            metadata={},
        ),
    ]


def test_retrieval_engine_returns_top_hit() -> None:
    engine = RetrievalEngine(
        chunker=build_chunker(ChunkingConfig(strategy="fixed", chunk_size=16, overlap=2)),
        embedder=build_embedding_provider(EmbeddingConfig(provider="hash", dimension=128)),
        store=build_vector_store(VectorConfig(provider="numpy")),
    )
    engine.index_documents(_docs())
    hits = engine.query("Which place has beaches in goa?", top_k=1)
    assert len(hits) == 1
    assert hits[0].chunk_id.startswith("d1:")
    assert hits[0].document_id == "d1"


def test_vector_store_persistence_roundtrip(tmp_path: Path) -> None:
    config = VectorConfig(
        provider="numpy",
        index_path=str(tmp_path / "idx.bin"),
        metadata_path=str(tmp_path / "meta.jsonl"),
    )
    store = build_vector_store(config)
    engine = RetrievalEngine(
        chunker=build_chunker(ChunkingConfig(strategy="fixed", chunk_size=16, overlap=2)),
        embedder=build_embedding_provider(EmbeddingConfig(provider="hash", dimension=64)),
        store=store,
    )
    engine.index_documents(_docs())
    store.save()

    reloaded = build_vector_store(config)
    reloaded.load()
    hits = reloaded.search(engine.embedder.embed_query("Which place has beaches in goa?"), top_k=1)
    assert len(hits) == 1
    assert hits[0].document_id == "d1"
    assert hits[0].metadata is not None
