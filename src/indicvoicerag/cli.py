from __future__ import annotations

import argparse
from dataclasses import asdict, replace
import json
import os
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

from .audio_capture import MicrophoneError, MicrophoneRecorder, write_pcm_wav
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
from .pipeline import build_harness, build_llm_from_config, build_retrieval_engine, build_stt_from_config
from .rag_evaluate import build_sample_cases, evaluate_rag
from .retrieval import RetrievalEngine
from .speech import answer_from_audio
from .stt import STTError, build_stt_provider, normalize_stt_provider_name
from .stt_benchmark import STTSample, load_samples, run_stt_benchmark
from .tts import TTSProvider, TTSError, build_tts_provider, normalize_tts_provider_name
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
    rag.add_argument(
        "--tts-output",
        default=None,
        help="Synthesize the RAG answer to this WAV via the configured local TTS (Phase 3D)",
    )

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

    transcribe = sub.add_parser("transcribe", help="Transcribe a local audio file (Phase 3A STT)")
    transcribe.add_argument("--audio", required=True, help="Path to a local audio file (WAV preferred)")
    transcribe.add_argument("--language", default=None, help="Hint language code (e.g. en, hi); default auto-detect")
    transcribe.add_argument("--json", action="store_true", help="Print raw JSON instead of a table")

    transcribe_rag = sub.add_parser("transcribe-rag", help="Audio -> STT -> text -> existing RAGHarness")
    transcribe_rag.add_argument("--audio", required=True, help="Path to a local audio file (WAV preferred)")
    transcribe_rag.add_argument("--language", default=None, help="Hint language code (e.g. en, hi)")
    transcribe_rag.add_argument("--size", type=int, default=None)
    transcribe_rag.add_argument("--top-k", type=int, default=None)
    transcribe_rag.add_argument("--debug", action="store_true", help="Include per-stage latency metrics")
    transcribe_rag.add_argument("--load-index", action="store_true", help="Load persisted vector index instead of re-indexing")

    benchmark_stt = sub.add_parser(
        "benchmark-stt",
        help="Reproducible local STT benchmark: duration / load / RTF / language / WER",
    )
    benchmark_stt.add_argument("--audio", action="append", default=None, help="Audio file (repeatable); default: auto-discover samples/*.wav")
    benchmark_stt.add_argument("--samples-dir", default="samples", help="Directory with .wav + .txt ground-truth sidecars (used when --audio is omitted)")
    benchmark_stt.add_argument("--label", action="append", default=None, help="Per-case label (repeatable)")
    benchmark_stt.add_argument("--ground-truth", action="append", default=None, help="Per-case reference transcript (repeatable)")
    benchmark_stt.add_argument("--language", action="append", default=None, help="Per-case expected language (repeatable)")
    benchmark_stt.add_argument("--no-warmup", action="store_true", help="Do not pre-load the model before timing")
    benchmark_stt.add_argument("--repeat", type=int, default=1, help="Per-sample runs; the fastest is reported")
    benchmark_stt.add_argument("--out", default=None, help="Write the full JSON report to this path")
    benchmark_stt.add_argument("--json", action="store_true", help="Print raw JSON instead of a table")

    listen = sub.add_parser(
        "listen",
        help="Record from the microphone, transcribe locally, answer via the existing RAGHarness (Phase 3C)",
    )
    listen.add_argument("--duration", type=float, default=None, help="Recording duration in seconds (default: config.audio.duration_seconds = 5.0)")
    listen.add_argument("--sample-rate", type=int, default=None, help="Capture sample rate (default: config.audio.sample_rate = 16000)")
    listen.add_argument("--channels", type=int, default=None, help="Capture channels (default: config.audio.channels = 1)")
    listen.add_argument("--language", default=None, help="STT language hint (e.g. hi, en); default from config, empty = auto-detect")
    listen.add_argument("--model", default=None, help="Override stt.model_name (e.g. small); does not change the config")
    listen.add_argument("--device", type=int, default=None, help="sounddevice input device index (default: system default)")
    listen.add_argument("--top-k", type=int, default=None, help="RAG top-k (default: config.retrieval.top_k)")
    listen.add_argument("--size", type=int, default=None, help="Sample size when building a fresh RAG index (ignored with --load-index)")
    listen.add_argument("--load-index", action="store_true", help="Load the persisted vector index instead of re-indexing")
    listen.add_argument("--no-rag", action="store_true", help="Transcribe only; skip the RAG answer")
    listen.add_argument("--debug", action="store_true", help="Keep the temporary WAV and print per-stage latency")
    listen.add_argument("--json", action="store_true", help="Print raw JSON instead of a table")

    synth = sub.add_parser(
        "synthesize",
        help="text -> local TTS -> WAV output (Phase 3D)",
    )
    synth.add_argument("--text", required=True, help="Text to synthesize (Hindi by default)")
    synth.add_argument("--language", default=None, help="Language code (e.g. hi, bn, ta); default from config.tts.language")
    synth.add_argument("--output", default=None, help="Output WAV path (default: tmp/<slug>.wav)")
    synth.add_argument("--json", action="store_true", help="Print raw JSON instead of a table")
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
    tts_output: str | None = None,
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

    tts_audio = None
    if tts_output:
        tts_audio = synthesize_answer_audio(config, response.answer, tts_output)

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
            "tts_audio": tts_audio.as_dict() if tts_audio is not None else None,
        }
    )


def synthesize_answer_audio(config: AppConfig, answer: str, output: str) -> Any:
    """Phase 3D output layer: RAG answer text -> local TTS -> WAV.

    Pure output glue; the RAG response stays structured and text-based.
    """
    tts = build_tts_provider(
        provider=config.tts.provider,
        language=config.tts.language,
        cache_dir=config.tts.cache_dir,
    )
    if normalize_tts_provider_name(config.tts.provider) == "mock":
        print(
            "WARNING: tts.provider='mock' is a deterministic test stub. "
            "Use tts.provider='mms' for real local speech synthesis.",
            file=sys.stderr,
        )
    return tts.synthesize(answer, language=config.tts.language, output_path=output)


def run_synthesize(config: AppConfig, text: str, language: str | None, output: str | None, as_json: bool) -> None:
    tts = build_tts_provider(
        provider=config.tts.provider,
        language=config.tts.language,
        cache_dir=config.tts.cache_dir,
    )
    if normalize_tts_provider_name(config.tts.provider) == "mock":
        print(
            "WARNING: tts.provider='mock' is a deterministic test stub. "
            "Use tts.provider='mms' for real local speech synthesis.",
            file=sys.stderr,
        )
    requested = language or config.tts.language
    out_path = output
    if out_path is None:
        out_path = _default_tts_output(text, requested)
    try:
        result = tts.synthesize(text, language=requested, output_path=out_path)
    except TTSError as exc:
        print(f"ERROR: text-to-speech failed: {exc}", file=sys.stderr)
        sys.exit(1)

    if as_json:
        _print_json(result.as_dict())
        return
    print(f"language         : {result.language}")
    print(f"model            : {result.model} ({tts.name})")
    print(f"text length      : {len(text)} chars")
    print(f"audio duration   : {result.duration_seconds:.3f} s")
    print(f"sample rate      : {result.sample_rate} Hz (mono 16-bit PCM WAV)")
    print(f"model load       : {result.load_time_ms or 0.0:.1f} ms")
    print(f"synthesis        : {result.synthesis_time_ms:.1f} ms")
    print(f"RTF              : {result.rtf:.3f}")
    print(f"output           : {result.output_path}")


def _default_tts_output(text: str, language: str) -> str:
    slug = "".join(c for c in text.strip().lower().split()[0] if c.isalnum())[:20] if text.split() else "speech"
    return f"tmp/{slug}-{language or 'auto'}.wav"


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


def run_transcribe(config: AppConfig, audio: str, language: str | None, as_json: bool) -> None:
    stt = build_stt_from_config(config)
    if normalize_stt_provider_name(config.stt.provider) == "mock":
        print(
            "WARNING: stt.provider='mock' is a deterministic test stub. "
            "Use --config with stt.provider='faster_whisper' for a real transcription.",
            file=sys.stderr,
        )
    result = stt.transcribe(audio, language=language)
    if as_json:
        _print_json(result.as_dict())
        return
    print(f"audio file       : {result.audio_path or audio}")
    print(f"model            : {result.model} ({stt.name})")
    print(f"detected language: {result.language or 'N/A'} (p={result.language_probability:.3f})")
    print(f"duration         : {result.duration_seconds:.3f} s")
    print(f"model load       : {result.load_time_ms or 0.0:.1f} ms")
    print(f"processing       : {result.processing_time_ms:.1f} ms")
    print(f"RTF              : {result.rtf:.3f}")
    print(f"transcription    : {result.text}")


def run_transcribe_rag(
    config: AppConfig,
    audio: str,
    language: str | None,
    size: int | None,
    top_k: int | None,
    debug: bool,
    load_index: bool,
) -> None:
    stt = build_stt_from_config(config)
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

    transcription, response = answer_from_audio(
        stt,
        harness,
        audio,
        language=language,
        top_k=top_k,
        debug=debug,
    )
    _print_json(
        {
            "transcription": transcription.as_dict(),
            "rag_response": response.as_dict(include_metrics=debug),
            "pipeline": "AUDIO FILE -> LOCAL STT -> TEXT -> RETRIEVAL -> CONTEXT -> LLM -> GROUNDING -> GUARDRAILS -> STRUCTURED RESPONSE",
            "provenance": _data_provenance(inspector),
            "embedder": engine.embedder.describe(),
            "documents_indexed": documents,
            "top_k": top_k or config.retrieval.top_k,
        }
    )


def run_stt_benchmark_cli(
    config: AppConfig,
    audio_paths: list[str] | None,
    samples_dir: str,
    labels: list[str] | None,
    ground_truths: list[str] | None,
    languages: list[str] | None,
    warmup: bool,
    repeat: int,
    out: str | None,
    as_json: bool,
) -> None:
    stt = build_stt_from_config(config)
    if audio_paths:
        samples: list[STTSample] = []
        for index, path in enumerate(audio_paths):
            samples.append(
                STTSample(
                    path=path,
                    label=(labels or [""] * len(audio_paths))[index] if labels else path,
                    language_expected=(languages or [None] * len(audio_paths))[index] if languages else None,
                    reference=(ground_truths or [None] * len(audio_paths))[index] if ground_truths else None,
                )
            )
    else:
        samples = load_samples(samples_dir)
        if not samples:
            raise FileNotFoundError(f"no *.wav files found in {samples_dir!r}; use --audio to pass files explicitly")

    report = run_stt_benchmark(stt, samples, repeat=repeat, warmup=warmup)
    if out:
        Path(out).parent.mkdir(parents=True, exist_ok=True)
        Path(out).write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    if as_json:
        _print_json(report)
        return

    provider = report["provider"]
    print(f"STT benchmark: {provider.get('provider')} / {provider.get('model')} "
          f"(device={provider.get('device') or 'cpu'}, compute={provider.get('compute_type') or 'n/a'})")
    summary = report["summary"]
    print(f"samples={summary['samples']} with_reference={summary['with_reference']} "
          f"wer_measured={summary['wer_measured']} errors={summary['errors']} "
          f"avg_wer={summary['avg_wer'] if summary['avg_wer'] is not None else 'n/a'}")
    header = f"{'label':<16}{'dur(s)':>8}{'load(ms)':>10}{'proc(ms)':>10}{'RTF':>8}{'lang':>6}{'WER':>8}"
    print(header)
    print("-" * len(header))
    for case in report["samples"]:
        if "error" in case:
            print(f"{case['label']:<16}{'--':>8}{'--':>10}{'--':>10}{'--':>8}{'--':>6}{'--':>8}  {case['error']}")
            continue
        wer = "n/a" if case.get("wer") is None else f"{case['wer']:.3f}"
        print(
            f"{case['label']:<16}{case['audio_duration_seconds']:>8.2f}{case['model_load_ms']:>10.1f}"
            f"{case['transcription_time_ms']:>10.1f}{case['rtf']:>8.2f}{(case['detected_language'] or '?'):>6}"
            f"{wer:>8}"
        )
    for case in report["samples"]:
        if "error" in case:
            continue
        print(f"\n[{case['label']}] {case['text']}")

    if summary["wer_measured"] < summary["with_reference"]:
        print("\nWER is only reported when a ground-truth reference exists (samples/<name>.txt sidecar or --ground-truth).")


def _build_stt_for_listen(config: AppConfig, model: str | None) -> Any:
    """Build the configured STT provider, honoring an optional model override."""
    if model:
        return build_stt_provider(
            provider=config.stt.provider,
            model_name=model,
            device=config.stt.device,
            compute_type=config.stt.compute_type,
            language=config.stt.language or None,
            beam_size=config.stt.beam_size,
            vad_filter=config.stt.vad_filter,
            download_root=config.stt.download_root,
        )
    return build_stt_from_config(config)


def run_listen(
    config: AppConfig,
    *,
    duration: float | None = None,
    sample_rate: int | None = None,
    channels: int | None = None,
    language: str | None = None,
    model: str | None = None,
    top_k: int | None = None,
    size: int | None = None,
    load_index: bool = False,
    device: int | None = None,
    debug: bool = False,
    as_json: bool = False,
    no_rag: bool = False,
    recorder: MicrophoneRecorder | None = None,
    harness: Any | None = None,
) -> int:
    """MICROPHONE -> WAV -> local STT -> text -> existing RAGHarness.

    Returns a process exit code (0 ok, 1 controlled failure) so normal CLI use
    never prints an opaque traceback. Recording time is reported but never
    counted as system processing latency.
    """
    stt = _build_stt_for_listen(config, model)
    if normalize_stt_provider_name(config.stt.provider) == "mock":
        print(
            "WARNING: stt.provider='mock' is a deterministic test stub. "
            "Use stt.provider='faster_whisper' for a real transcription.",
            file=sys.stderr,
        )

    rec = recorder or MicrophoneRecorder(
        sample_rate=sample_rate or config.audio.sample_rate,
        channels=channels or config.audio.channels,
        duration_seconds=duration or config.audio.duration_seconds,
        device=device,
    )

    try:
        print("Press Enter to start recording...", end="", flush=True, file=sys.stderr)
        input()
        print(f"Recording for {rec.duration_seconds:.1f} s - speak now.", file=sys.stderr)
    except KeyboardInterrupt:
        print("\nRecording cancelled.", file=sys.stderr)
        return 1
    except EOFError:
        # non-interactive stdin (e.g. piped input): start recording immediately
        print(f"\nRecording for {rec.duration_seconds:.1f} s - speak now.", file=sys.stderr)

    try:
        capture = rec.record()
    except MicrophoneError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        print("\nRecording cancelled.", file=sys.stderr)
        return 1
    recording_duration = capture.duration_seconds

    t0 = time.perf_counter()
    fd, tmp_name = tempfile.mkstemp(suffix=".wav", prefix="ivr_listen_")
    os.close(fd)
    tmp_path = Path(tmp_name)
    write_pcm_wav(tmp_path, capture.samples, capture.sample_rate)
    prep_ms = (time.perf_counter() - t0) * 1000.0

    try:
        if no_rag:
            transcription = stt.transcribe(str(tmp_path), language=language)
            rag_response = None
        else:
            active_harness = harness
            if active_harness is None:
                inspector = DatasetInspector(config.dataset)
                engine = build_retrieval_engine(config)
                if load_index:
                    engine.store.load()
                else:
                    docs = inspector.normalized_sample(size)
                    engine.index_documents(docs)
                active_harness = build_harness(config, engine.query)
            transcription, rag_response = answer_from_audio(
                stt, active_harness, str(tmp_path), language=language, top_k=top_k, debug=True
            )
    except STTError as exc:
        if not debug:
            tmp_path.unlink(missing_ok=True)
        print(f"ERROR: speech-to-text failed: {exc}", file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        if not debug:
            tmp_path.unlink(missing_ok=True)
        print("\nInterrupted.", file=sys.stderr)
        return 1
    except Exception as exc:  # noqa: BLE001 - controlled CLI errors, no tracebacks
        if not debug:
            tmp_path.unlink(missing_ok=True)
        print(f"ERROR: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1

    rag_metrics = (rag_response.metrics if rag_response is not None else None) or {}
    rag_total_ms = float(rag_metrics.get("total", 0.0) or 0.0)
    stt_ms = float(transcription.processing_time_ms or 0.0)
    total_post_ms = prep_ms + stt_ms + rag_total_ms

    if as_json:
        payload: dict[str, Any] = {
            "language": transcription.language,
            "language_probability": round(transcription.language_probability, 4),
            "transcription": transcription.text,
            "recording_duration_seconds": round(recording_duration, 3),
            "stt": {
                "model": transcription.model,
                "processing_ms": round(stt_ms, 1),
                "rtf": round(transcription.rtf, 4),
                "model_load_ms": round(transcription.load_time_ms or 0.0, 1),
            },
            "audio_prep_ms": round(prep_ms, 1),
            "rag": rag_response.as_dict(include_metrics=True) if rag_response is not None else None,
            "rag_total_ms": round(rag_total_ms, 1),
            "total_post_recording_ms": round(total_post_ms, 1),
            "temp_wav": str(tmp_path) if debug else None,
        }
        _print_json(payload)
    else:
        print(f"Detected language: {transcription.language or 'N/A'} (p={transcription.language_probability:.3f})")
        print(f"Transcription: {transcription.text}")
        if rag_response is not None:
            print(f"RAG answer: {rag_response.answer}")
            print(f"Grounded: {rag_response.grounded} (confidence={rag_response.confidence:.4f})")
            if rag_response.guardrail:
                print(f"Guardrail: {rag_response.guardrail}")
            if rag_response.sources:
                print("Sources:")
                for source in rag_response.sources:
                    excerpt = source.excerpt.strip().replace("\n", " ")
                    print(f"  - {source.document_id} (score {source.score:.4f}) {excerpt[:100]}")
        print("Latency:")
        print(f"  recording duration      : {recording_duration:.3f} s   (speaking time, NOT system processing)")
        print(f"  audio prep (write WAV)  : {prep_ms:.1f} ms")
        print(f"  STT processing          : {stt_ms:.1f} ms")
        print(f"  STT RTF                 : {transcription.rtf:.3f}")
        print(f"  STT model load          : {transcription.load_time_ms or 0.0:.1f} ms   (one-time)")
        if rag_response is not None:
            print(f"  RAG retrieval           : {rag_metrics.get('retrieval', 0.0):.1f} ms")
            print(f"  RAG context             : {rag_metrics.get('context', 0.0):.1f} ms")
            print(f"  RAG generation          : {rag_metrics.get('generation', 0.0):.1f} ms")
            print(f"  RAG grounding           : {rag_metrics.get('grounding', 0.0):.1f} ms")
        print(f"  total post-recording    : {total_post_ms:.1f} ms")

    if debug:
        print(f"temp wav kept: {tmp_path}", file=sys.stderr)
    else:
        try:
            tmp_path.unlink(missing_ok=True)
        except OSError:
            pass
    return 0


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
            tts_output=args.tts_output,
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
    elif args.command == "transcribe":
        run_transcribe(config, audio=args.audio, language=args.language, as_json=args.json)
    elif args.command == "transcribe-rag":
        run_transcribe_rag(
            config,
            audio=args.audio,
            language=args.language,
            size=args.size,
            top_k=args.top_k,
            debug=args.debug,
            load_index=args.load_index,
        )
    elif args.command == "benchmark-stt":
        run_stt_benchmark_cli(
            config,
            audio_paths=args.audio,
            samples_dir=args.samples_dir,
            labels=args.label,
            ground_truths=args.ground_truth,
            languages=args.language,
            warmup=not args.no_warmup,
            repeat=args.repeat,
            out=args.out,
            as_json=args.json,
        )
    elif args.command == "listen":
        sys.exit(
            run_listen(
                config,
                duration=args.duration,
                sample_rate=args.sample_rate,
                channels=args.channels,
                language=args.language,
                model=args.model,
                top_k=args.top_k,
                size=args.size,
                load_index=args.load_index,
                device=args.device,
                debug=args.debug,
                as_json=args.json,
                no_rag=args.no_rag,
            )
        )
    elif args.command == "synthesize":
        run_synthesize(
            config,
            text=args.text,
            language=args.language,
            output=args.output,
            as_json=args.json,
        )
    else:
        raise ValueError(f"Unsupported command: {args.command}")


if __name__ == "__main__":
    main()
