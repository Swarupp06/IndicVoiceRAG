from indicvoicerag.chunking import build_chunker
from indicvoicerag.config import ChunkingConfig
from indicvoicerag.schemas import NormalizedDocument


def _doc() -> NormalizedDocument:
    return NormalizedDocument(
        document_id="d1",
        query_id="q1",
        query_text="What is Goa?",
        passage_text=(
            "Goa is a state in India. "
            "It is known for beaches and tourism. "
            "Panaji is the capital city. "
            "Many people visit during winter."
        ),
        language="en",
        source_language="en",
        target_language="en",
        query_type="factoid",
        relevance=1.0,
        metadata={"source": "unit-test"},
    )


def test_fixed_chunker_produces_overlapping_chunks() -> None:
    chunker = build_chunker(ChunkingConfig(strategy="fixed", chunk_size=6, overlap=2))
    chunks = chunker.chunk(_doc())
    assert len(chunks) >= 2
    assert chunks[0].metadata["strategy"] == "fixed"
    assert chunks[0].metadata["source"] == "unit-test"


def test_sentence_aware_chunker_splits_by_sentence() -> None:
    chunker = build_chunker(ChunkingConfig(strategy="sentence_aware", chunk_size=8, overlap=0))
    chunks = chunker.chunk(_doc())
    assert len(chunks) >= 2
    assert all(chunk.metadata["strategy"] == "sentence_aware" for chunk in chunks)


def test_semantic_chunker_available() -> None:
    chunker = build_chunker(ChunkingConfig(strategy="semantic", chunk_size=16, overlap=0, semantic_threshold=0.2))
    chunks = chunker.chunk(_doc())
    assert len(chunks) >= 1
    assert all(chunk.metadata["strategy"] == "semantic" for chunk in chunks)
