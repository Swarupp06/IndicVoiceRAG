from __future__ import annotations

import argparse
from dataclasses import asdict
import json
import sys
import time
from pathlib import Path
from typing import Any

from .chunking import ChunkingConfig, build_chunker
from .config import AppConfig, load_config
from .dataset import DatasetInspector
from .embedding import EmbeddingConfig, build_embedding_provider
from .evaluation import evaluate_retrieval
from .retrieval import RetrievalEngine
from .vector_store import build_vector_store


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="IndicVoiceRAG Phase 1 toolkit")
    parser.add_argument("--config", help="Path to TOML config", default=None)
    sub = parser.add_subparsers(dest="command", required=True)

    inspect = sub.add_parser("inspect-dataset", help="Inspect dataset repo/config/files/remote schema")
    inspect.add_argument("--rows", type=int, default=5)

    sample = sub.add_parser("sample-docs", help="Print normalized sample documents")
    sample.add_argument("--size", type=int, default=None)

    build = sub.add_parser("build-index", help="Build local vector index from real sample")
    build.add_argument("--size", type=int, default=None)
    build.add_argument("--save", action="store_true")

    retrieve = sub.add_parser("retrieve", help="Run retrieval against real-data index")
    retrieve.add_argument("--query", required=True)
    retrieve.add_argument("--size", type=int, default=None)
    retrieve.add_argument("--top-k", type=int, default=None)

    chunk_stats = sub.add_parser("chunk-stats", help="Run all chunking strategies on a real sample")
    chunk_stats.add_argument("--size", type=int, default=None)

    evaluate = sub.add_parser("evaluate-retrieval", help="Hit@K / Recall@K / MRR on a real sample")
    evaluate.add_argument("--size", type=int, default=None)
    evaluate.add_argument("--top-k", type=int, default=10)

    smoke = sub.add_parser("smoke-test", help="End-to-end real-data + real-embedding smoke test")
    smoke.add_argument("--query", default="What is the passage about?")
    smoke.add_argument("--size", type=int, default=32)
    return parser


def _print_json(payload: dict[str, Any]) -> None:
    print(json.dumps(payload, indent=2, ensure_ascii=False))


def _load_app_config(path: str | None) -> AppConfig:
    return load_config(path)


def _build_engine(config: AppConfig) -> RetrievalEngine:
    chunker = build_chunker(config.chunking)
    embedder = build_embedding_provider(config.embedding)
    store = build_vector_store(config.vector)
    return RetrievalEngine(chunker=chunker, embedder=embedder, store=store)


def _data_provenance(inspector: DatasetInspector) -> dict[str, Any]:
    meta = inspector.sample_cache_meta()
    if meta:
        return {
            "source": meta.get("source", "real_data_cache"),
            "repo_id": meta.get("repo_id"),
            "split": meta.get("split"),
            "language": meta.get("language"),
            "parquet_file": meta.get("parquet_file"),
            "url": meta.get("url"),
            "cached_at": meta.get("cached_at"),
            "record_count": meta.get("record_count"),
        }
    selected = inspector.select_parquet_files()
    return {
        "source": "not_fetched",
        "repo_id": inspector.config.repo_id,
        "split": inspector.config.split,
        "language": inspector.config.language,
        "parquet_file": selected[0] if selected else None,
    }


def inspect_dataset(config: AppConfig, rows: int) -> None:
    inspector = DatasetInspector(config.dataset)
    files_info = inspector.inspect_hub_files()
    remote_schema: dict[str, Any] | None = None
    try:
        remote_schema = inspector.inspect_remote_schema()
    except Exception as exc:  # noqa: BLE001
        remote_schema = {"error": f"{type(exc).__name__}: {exc}"}
    schema_info = inspector.inspect_sample_schema(rows=rows)
    _print_json({"hub": files_info, "remote_schema": remote_schema, "sample_schema": schema_info})


def sample_docs(config: AppConfig, size: int | None) -> None:
    inspector = DatasetInspector(config.dataset)
    docs = inspector.normalized_sample(size)
    payload = [asdict(doc) for doc in docs]
    _print_json({"count": len(payload), "provenance": _data_provenance(inspector), "documents": payload})


def build_index(config: AppConfig, size: int | None, save: bool) -> None:
    inspector = DatasetInspector(config.dataset)
    docs = inspector.normalized_sample(size)
    engine = _build_engine(config)
    start = time.time()
    chunks = engine.index_documents(docs)
    elapsed = time.time() - start
    if save:
        engine.store.save()
    _print_json(
        {
            "provenance": _data_provenance(inspector),
            "embedder": engine.embedder.describe(),
            "chunking_strategy": config.chunking.strategy,
            "documents_indexed": len(docs),
            "chunks_indexed": len(chunks),
            "index_seconds": round(elapsed, 2),
            "index_saved": save,
            "index_path": str(Path(config.vector.index_path).resolve()),
        }
    )


def retrieve(config: AppConfig, query: str, size: int | None, top_k: int | None) -> None:
    inspector = DatasetInspector(config.dataset)
    docs = inspector.normalized_sample(size)
    engine = _build_engine(config)
    start = time.time()
    engine.index_documents(docs)
    hits = engine.query(query_text=query, top_k=top_k or config.retrieval.top_k)
    elapsed = time.time() - start
    _print_json(
        {
            "query": query,
            "provenance": _data_provenance(inspector),
            "embedder": engine.embedder.describe(),
            "sample_documents": len(docs),
            "retrieval_seconds": round(elapsed, 2),
            "hits": [asdict(hit) for hit in hits],
        }
    )


def chunk_stats(config: AppConfig, size: int | None) -> None:
    inspector = DatasetInspector(config.dataset)
    docs = inspector.normalized_sample(size)
    report: dict[str, Any] = {
        "provenance": _data_provenance(inspector),
        "documents": len(docs),
        "strategies": {},
    }
    for strategy in ("fixed", "sentence_aware", "semantic"):
        cfg = ChunkingConfig(
            strategy=strategy,
            chunk_size=config.chunking.chunk_size,
            overlap=config.chunking.overlap,
            semantic_threshold=config.chunking.semantic_threshold,
            sentence_separator_regex=config.chunking.sentence_separator_regex,
        )
        chunker = build_chunker(cfg)
        chunks = [chunk for doc in docs for chunk in chunker.chunk(doc)]
        lengths = [len(chunk.text.split()) for chunk in chunks]
        report["strategies"][strategy] = {
            "chunks": len(chunks),
            "avg_tokens_per_chunk": round(sum(lengths) / len(lengths), 2) if lengths else 0,
            "min_tokens": min(lengths) if lengths else 0,
            "max_tokens": max(lengths) if lengths else 0,
            "total_tokens": sum(lengths),
        }
    _print_json(report)


def run_retrieval_evaluation(config: AppConfig, size: int | None, top_k: int) -> None:
    inspector = DatasetInspector(config.dataset)
    docs = inspector.normalized_sample(size)
    engine = _build_engine(config)
    start = time.time()
    engine.index_documents(docs)
    index_seconds = time.time() - start
    eval_start = time.time()
    metrics = evaluate_retrieval(docs, engine.query, top_k=top_k)
    eval_seconds = time.time() - eval_start
    _print_json(
        {
            "provenance": _data_provenance(inspector),
            "embedder": engine.embedder.describe(),
            "chunking_strategy": config.chunking.strategy,
            "documents_indexed": len(docs),
            "index_seconds": round(index_seconds, 2),
            "eval_seconds": round(eval_seconds, 2),
            "metrics": metrics.as_dict(),
        }
    )


def smoke_test(config: AppConfig, query: str, size: int) -> None:
    inspector = DatasetInspector(config.dataset)
    docs = inspector.normalized_sample(size)
    engine = _build_engine(config)
    start = time.time()
    engine.index_documents(docs)
    hits = engine.query(query_text=query, top_k=config.retrieval.top_k)
    elapsed = time.time() - start
    provenance = _data_provenance(inspector)
    embedder = engine.embedder.describe()
    real_data = bool(provenance.get("url")) or provenance.get("source") in {"huggingface_hub_live_fetch", "real_data_cache"}
    real_model = embedder.get("provider") in {"sentence_transformers", "fastembed"}
    _print_json(
        {
            "smoke_test": True,
            "passed": real_data and real_model and len(hits) > 0,
            "real_msmarco_xi_data_used": real_data,
            "real_embedding_model_used": real_model,
            "query": query,
            "provenance": provenance,
            "embedder": embedder,
            "pipeline": "REAL MSMARCO-XI -> NORMALIZATION -> CHUNKING -> REAL EMBEDDING -> VECTOR INDEX -> QUERY EMBEDDING -> TOP-K RETRIEVAL",
            "documents_indexed": len(docs),
            "elapsed_seconds": round(elapsed, 2),
            "hits": [asdict(hit) for hit in hits],
        }
    )


def main() -> None:
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except AttributeError:
        pass
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
    elif args.command == "chunk-stats":
        chunk_stats(config, size=args.size)
    elif args.command == "evaluate-retrieval":
        run_retrieval_evaluation(config, size=args.size, top_k=args.top_k)
    elif args.command == "smoke-test":
        smoke_test(config, query=args.query, size=args.size)
    else:
        raise ValueError(f"Unsupported command: {args.command}")


if __name__ == "__main__":
    main()
