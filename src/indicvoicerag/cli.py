from __future__ import annotations

import argparse
from dataclasses import asdict, replace
import json
import sys
import time
from pathlib import Path
from typing import Any

from .benchmark import (
    benchmark_report,
    load_benchmark_queries,
    records_as_dicts,
    run_benchmark,
    summarize,
)
from .chunking import ChunkingConfig, build_chunker
from .config import AppConfig, load_config
from .cost import PROVIDER_COSTS
from .dataset import DatasetInspector
from .embedding import build_embedding_provider
from .evaluation import evaluate_retrieval
from .llm import build_llm_provider, normalize_provider_name
from .pipeline import build_harness, build_llm_from_config, build_retrieval_engine
from .rag_evaluate import build_sample_cases, evaluate_rag
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

    rag = sub.add_parser("rag", help="Run the full RAG pipeline (retrieval -> LLM -> guardrails)")
    rag.add_argument("--query", required=True)
    rag.add_argument("--size", type=int, default=None)
    rag.add_argument("--top-k", type=int, default=None)
    rag.add_argument("--debug", action="store_true", help="Include per-stage latency metrics")
    rag.add_argument(
        "--provider",
        default=None,
        help="Override llm.provider (ollama|gemini|groq|openrouter|openai_compatible|mock)",
    )
    rag.add_argument("--model", default=None, help="Override llm.model_name")
    rag.add_argument("--load-index", action="store_true", help="Load persisted vector index instead of re-indexing")

    rag_eval = sub.add_parser("rag-evaluate", help="RAG evaluation harness on real data")
    rag_eval.add_argument("--size", type=int, default=None)
    rag_eval.add_argument("--queries", type=int, default=20)
    rag_eval.add_argument("--top-k", type=int, default=None)
    rag_eval.add_argument("--debug", action="store_true")

    providers = sub.add_parser("providers", help="List LLM providers, availability and verified cost status")
    providers.add_argument("--json", action="store_true", help="Print raw JSON instead of a table")

    bench = sub.add_parser(
        "benchmark-providers",
        help="Run the fixed benchmark query set through the full RAG pipeline per provider",
    )
    bench.add_argument(
        "--providers",
        default="ollama",
        help="Comma-separated providers to benchmark (only ₹0-verified providers should be used)",
    )
    bench.add_argument("--models", default=None, help="Comma-separated provider=model overrides")
    bench.add_argument("--queries-file", default=None, help="Benchmark query set (default benchmarks/queries.json)")
    bench.add_argument("--size", type=int, default=None, help="Sample size when building the index")
    bench.add_argument("--top-k", type=int, default=None)
    bench.add_argument("--load-index", action="store_true", help="Load the persisted index instead of re-indexing")
    bench.add_argument("--out", default=None, help="Write the full JSON report to this path")
    bench.add_argument("--rows-out", default=None, help="Write per-query rows as CSV to this path")
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
    if inspector.config.local_sample_path:
        return {
            "source": "local_fixture",
            "repo_id": inspector.config.repo_id,
            "split": inspector.config.split,
            "language": inspector.config.language,
            "parquet_file": None,
            "url": None,
        }
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


def run_rag(
    config: AppConfig,
    query: str,
    size: int | None,
    top_k: int | None,
    debug: bool,
    provider: str | None,
    model: str | None,
    load_index: bool,
) -> None:
    if provider or model:
        config.llm = replace(
            config.llm,
            provider=provider or config.llm.provider,
            model_name=model or config.llm.model_name,
        )

    if normalize_provider_name(config.llm.provider) == "mock":
        print(
            "WARNING: llm.provider='mock' is a deterministic test stub, not a real generator. "
            "Use --provider ollama|gemini|groq|openrouter for a real answer.",
            file=sys.stderr,
        )

    inspector = DatasetInspector(config.dataset)
    engine = build_retrieval_engine(config)
    if load_index:
        engine.store.load()
        documents = len(engine.store._chunks)  # noqa: SLF001 - internal count for report
    else:
        docs = inspector.normalized_sample(size)
        engine.index_documents(docs)
        documents = len(docs)

    harness = build_harness(config, engine.query)
    response = harness.answer(query=query, top_k=top_k, debug=debug)

    _print_json(
        {
            "rag_response": response.as_dict(include_metrics=debug),
            "pipeline": "TEXT QUERY -> VALIDATION -> RETRIEVAL -> CONTEXT -> LLM -> GROUNDING -> GUARDRAILS -> STRUCTURED RESPONSE",
            "provenance": _data_provenance(inspector),
            "embedder": engine.embedder.describe(),
            "llm": harness.llm.describe(),
            "documents_indexed": documents,
            "top_k": top_k or config.retrieval.top_k,
            "guardrails": {
                "min_retrieval_score": config.guardrails.min_retrieval_score,
                "grounding_threshold": config.guardrails.grounding_threshold,
                "context_max_tokens": config.guardrails.context_max_tokens,
            },
        }
    )


def run_rag_evaluate(config: AppConfig, size: int | None, queries: int, top_k: int | None, debug: bool) -> None:
    inspector = DatasetInspector(config.dataset)
    docs = inspector.normalized_sample(size)
    engine = build_retrieval_engine(config)
    engine.index_documents(docs)

    harness = build_harness(config, engine.query)

    seen: set[str] = set()
    real_queries: list[tuple[str, str | None]] = []
    for doc in docs:
        if doc.query_id is None or not doc.query_text:
            continue
        if doc.query_id in seen:
            continue
        seen.add(doc.query_id)
        real_queries.append((doc.query_text, doc.query_id))
        if len(real_queries) >= queries:
            break

    unanswerable = [
        "बैंकॉक में कल का मौसम कैसा था?",
        "What is the score of yesterday's cricket match?",
        "क्या आज शेयर बाजार बंद होगा?",
    ]
    cases = build_sample_cases(real_queries, extra_unanswerable=unanswerable)
    report = evaluate_rag(harness, cases, top_k=top_k, debug=debug)

    _print_json(
        {
            "rag_evaluation": report.as_dict(),
            "provenance": _data_provenance(inspector),
            "embedder": engine.embedder.describe(),
            "llm": harness.llm.describe(),
            "note": (
                "No human/LLM answer-quality label set exists yet; this measures "
                "pipeline behavior (answered/refused/grounded + latency), not answer quality."
            ),
            "unanswerable_probes": unanswerable,
        }
    )


def list_providers(config: AppConfig, as_json: bool) -> None:
    rows: list[dict[str, Any]] = []
    for name, cost in PROVIDER_COSTS.items():
        try:
            provider = build_llm_provider(name, model_name=None)
            available = provider.available()
            model = provider.model_name
        except Exception as exc:  # noqa: BLE001 - reported, not raised
            available = False
            model = cost.model
            rows.append({**cost.as_dict(), "available_now": available, "default_model": model, "error": str(exc)})
            continue
        rows.append({**cost.as_dict(), "available_now": available, "default_model": model})
    if as_json:
        _print_json({"configured_provider": config.llm.provider, "providers": rows})
        return
    header = f"{'provider':<18}{'default model':<32}{'available':<11}{'cost status'}"
    print(header)
    print("-" * len(header))
    for row in rows:
        print(
            f"{row['provider']:<18}{row['default_model']:<32}"
            f"{'yes' if row['available_now'] else 'no':<11}{row['status']}"
        )
    print(f"\nconfigured provider: {config.llm.provider}")


def _parse_model_overrides(raw: str | None) -> dict[str, str]:
    overrides: dict[str, str] = {}
    for item in (raw or "").split(","):
        item = item.strip()
        if not item:
            continue
        if "=" not in item:
            raise ValueError(f"--models expects provider=model pairs, got '{item}'")
        provider, model = item.split("=", 1)
        overrides[normalize_provider_name(provider)] = model.strip()
    return overrides


def _write_rows_csv(path: str, rows: list[dict[str, Any]]) -> None:
    import csv

    if not rows:
        return
    fields = [key for key in rows[0] if key != "usage"]
    with open(path, "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row[key] for key in fields})


def run_provider_benchmark(
    config: AppConfig,
    providers: str,
    models: str | None,
    queries_file: str | None,
    size: int | None,
    top_k: int | None,
    load_index: bool,
    out: str | None,
    rows_out: str | None,
) -> None:
    queries = load_benchmark_queries(queries_file)
    overrides = _parse_model_overrides(models)
    requested = [normalize_provider_name(p) for p in providers.split(",") if p.strip()]

    inspector = DatasetInspector(config.dataset)
    engine = build_retrieval_engine(config)
    if load_index:
        engine.store.load()
        documents = len(engine.store._chunks)  # noqa: SLF001 - internal count for report
    else:
        docs = inspector.normalized_sample(size)
        engine.index_documents(docs)
        documents = len(docs)

    runs: dict[str, Any] = {}
    all_rows: list[dict[str, Any]] = []
    for name in requested:
        cost = PROVIDER_COSTS.get(name)
        if cost is None:
            runs[name] = {"skipped": f"unknown provider '{name}'"}
            continue
        if not cost.eligible_zero_cost:
            runs[name] = {"skipped": "provider is NOT ELIGIBLE for the ₹0 project", "cost": cost.as_dict()}
            continue
        model = overrides.get(name)
        keeps_endpoint = name == normalize_provider_name(config.llm.provider)
        provider_config = replace(
            config.llm,
            provider=name,
            model_name=model or "",
            base_url=config.llm.base_url if keeps_endpoint else None,
            api_key_env=config.llm.api_key_env if keeps_endpoint else None,
            fallback_providers=[],
        )
        llm = build_llm_from_config(replace(config, llm=provider_config))
        if not llm.available():
            runs[name] = {
                "skipped": "provider not available (missing API key or unreachable endpoint)",
                "cost": {**cost.as_dict(), "model": llm.model_name},
                "model": llm.model_name,
            }
            continue
        harness = build_harness(replace(config, llm=provider_config), engine.query)
        records = run_benchmark(harness, queries, provider_label=name, model_label=llm.model_name)
        rows = records_as_dicts(records)
        all_rows.extend(rows)
        runs[name] = {
            "model": llm.model_name,
            "cost": {**cost.as_dict(), "model": llm.model_name},
            "summary": summarize(records),
            "records": rows,
        }

    report = benchmark_report(
        runs=runs,
        query_count=len(queries),
        dataset=_data_provenance(inspector),
        embedder=engine.embedder.describe(),
    )
    report["documents_indexed"] = documents
    report["top_k"] = top_k or config.retrieval.top_k
    if out:
        Path(out).parent.mkdir(parents=True, exist_ok=True)
        Path(out).write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    if rows_out:
        Path(rows_out).parent.mkdir(parents=True, exist_ok=True)
        _write_rows_csv(rows_out, all_rows)

    summary_only = {
        name: {k: v for k, v in run.items() if k != "records"} for name, run in runs.items()
    }
    _print_json({**report, "runs": summary_only, "report_path": out, "rows_path": rows_out})


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
    elif args.command == "rag":
        run_rag(
            config,
            query=args.query,
            size=args.size,
            top_k=args.top_k,
            debug=args.debug,
            provider=args.provider,
            model=args.model,
            load_index=args.load_index,
        )
    elif args.command == "rag-evaluate":
        run_rag_evaluate(
            config,
            size=args.size,
            queries=args.queries,
            top_k=args.top_k,
            debug=args.debug,
        )
    elif args.command == "providers":
        list_providers(config, as_json=args.json)
    elif args.command == "benchmark-providers":
        run_provider_benchmark(
            config,
            providers=args.providers,
            models=args.models,
            queries_file=args.queries_file,
            size=args.size,
            top_k=args.top_k,
            load_index=args.load_index,
            out=args.out,
            rows_out=args.rows_out,
        )
    else:
        raise ValueError(f"Unsupported command: {args.command}")


if __name__ == "__main__":
    main()
