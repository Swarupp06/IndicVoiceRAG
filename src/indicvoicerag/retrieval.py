from __future__ import annotations

from dataclasses import dataclass

from .chunking import BaseChunker
from .embedding import EmbeddingProvider
from .schemas import DocumentChunk, NormalizedDocument
from .vector_store import RetrievalHit, VectorStore


@dataclass(slots=True)
class RetrievalEngine:
    chunker: BaseChunker
    embedder: EmbeddingProvider
    store: VectorStore

    def index_documents(self, documents: list[NormalizedDocument]) -> list[DocumentChunk]:
        chunks: list[DocumentChunk] = []
        for doc in documents:
            chunks.extend(self.chunker.chunk(doc))

        vectors = self.embedder.embed_documents(chunk.text for chunk in chunks)
        self.store.add(vectors, chunks)
        return chunks

    def query(self, query_text: str, top_k: int) -> list[RetrievalHit]:
        query_embedding = self.embedder.embed_query(query_text)
        return self.store.search(query_embedding=query_embedding, top_k=top_k)
