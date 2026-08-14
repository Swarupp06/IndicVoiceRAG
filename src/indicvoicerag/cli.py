from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path
from typing import Any

from .chunking import build_chunker
from .config import AppConfig, load_config
from .dataset import DatasetInspector
from .embedding import build_embedding_provider
from .retrieval import RetrievalEngine
from .vector_store import build_vector_store


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="IndicVoiceRAG Phase 1 toolkit")
    parser.add_argument("--config", help="Path to TOML config", default=None)
    sub = parser.add_subparsers(dest="command", required=True)

    inspect = sub.add_parser("inspect-dataset", help="Inspect dataset repo/config/files/schema")
    inspect.add_argument("--rows", type=int, default=5)

    sample = sub.add_parser("sample-docs", help="Print normalized sample documents")
    sample.add_argument("--size", type=int, default=None)

    build = sub.add_parser("build-index", help="Build local vector index from sample")
    build.add_argument("--size", type=int, default=None)
    build.add_argument("--save", action="store_true")

    retrieve = sub.add_parser("retrieve", help="Run retrieval against sample index")
    retrieve.add_argument("--query", required=True)
    retrieve.add_argument("--size", type=int, default=None)
    retrieve.add_argument("--top-k", type=int, default=None)

    smoke = sub.add_parser("smoke-test", help="Run tiny end-to-end indexing+retrieval smoke test")
    smoke.add_argument("--query", default="What is the passage about?")
    smoke.add_argument("--size", type=int, default=32)
    return parser


def _print_json(payload: dict[str, Any]) -> None:
    print(json.dumps(payload, indent=2, ensure_ascii=False))


def _load_app_config(path: str | None) -> AppConfig:
    config = load_config(path)
    return config


def inspect_dataset(config: AppConfig, rows: int) -> None:
    inspector = DatasetInspector(config.dataset)
    files_info = inspector.inspect_hub_files()
    schema_info = inspector.inspect_sample_schema(rows=rows)
    _print_json({"hub": files_info, "sample_schema": schema_info})


def sample_docs(config: AppConfig, size: int | None) -> None:
    inspector = DatasetInspector(config.dataset)
    docs = inspector.normalized_sample(size)
    payload = [asdict(doc) for doc in docs]
    _print_json({"count": len(payload), "documents": payload})


def _build_engine(config: AppConfig) -> RetrievalEngine:
    chunker = build_chunker(config.chunking)
    embedder = build_embedding_provider(config.embedding)
    store = build_vector_store(config.vector)
    return RetrievalEngine(chunker=chunker, embedder=embedder, store=store)


def build_index(config: AppConfig, size: int | None, save: bool) -> None:
    inspector = DatasetInspector(config.dataset)
    docs = inspector.normalized_sample(size)
    engine = _build_engine(config)
    chunks = engine.index_documents(docs)
    if save:
        engine.store.save()
    _print_json(
        {
            "documents_indexed": len(docs),
            "chunks_indexed": len(chunks),
            "index_saved": save,
            "index_path": str(Path(config.vector.index_path).resolve()),
        }
    )


def retrieve(config: AppConfig, query: str, size: int | None, top_k: int | None) -> None:
    inspector = DatasetInspector(config.dataset)
    docs = inspector.normalized_sample(size)
    engine = _build_engine(config)
    engine.index_documents(docs)
    hits = engine.query(query_text=query, top_k=top_k or config.retrieval.top_k)
    _print_json(
        {
            "query": query,
            "sample_documents": len(docs),
            "hits": [asdict(hit) for hit in hits],
        }
    )


def smoke_test(config: AppConfig, query: str, size: int) -> None:
    retrieve(config=config, query=query, size=size, top_k=config.retrieval.top_k)


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()
    config = _load_app_config(args.config)

    if args.command == "inspect-dataset":
        inspect_dataset(config, rows=args.rows)
    elif args.command == "sample-docs":
        sample_docs(config, size=args.size)
    elif args.command == "build-index":
        build_index(config, size=args.size, save=args.save)
    elif args.command == "retrieve":
        retrieve(config, query=args.query, size=args.size, top_k=args.top_k)
    elif args.command == "smoke-test":
        smoke_test(config, query=args.query, size=args.size)
    else:
        raise ValueError(f"Unsupported command: {args.command}")


if __name__ == "__main__":
    main()
