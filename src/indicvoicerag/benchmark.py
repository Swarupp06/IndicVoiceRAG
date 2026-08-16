"""Provider benchmark harness (Phase 2.5).

Runs the SAME fixed query set through the full RAG pipeline for every provider
and records per-stage latency, grounding and guardrail outcomes. Latency
percentiles use the nearest-rank method, so P100 is the observed maximum.

No quality score is invented: the benchmark records the raw answer plus the
grounding/guardrail decision, and quality is reported qualitatively.
"""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterable

from .harness import RAGHarness

DEFAULT_QUERY_SET = Path(__file__).resolve().parents[2] / "benchmarks" / "queries.json"

CATEGORIES = (
    "english_supported",
    "indic_supported",
    "evidence_required",
    "low_relevance",
    "unsupported",
)


@dataclass(slots=True)
class BenchmarkQuery:
    id: str
    category: str
    language: str
    query: str
    note: str | None = None
    expected_answerable: bool | None = None


@dataclass(slots=True)
class BenchmarkRecord:
    provider: str
    model: str
    query_id: str
    category: str
    language: str
    query: str
    retrieval_latency_ms: float
    context_latency_ms: float
    generation_latency_ms: float
    grounding_latency_ms: float
    total_latency_ms: float
    grounded: bool
    guardrail: str | None
    answer: str
    confidence: float
    error: str | None = None
    usage: dict[str, Any] = field(default_factory=dict)


def load_benchmark_queries(path: str | Path | None = None) -> list[BenchmarkQuery]:
    query_path = Path(path) if path else DEFAULT_QUERY_SET
    payload = json.loads(query_path.read_text(encoding="utf-8"))
    return [BenchmarkQuery(**item) for item in payload["queries"]]


def percentile(values: Iterable[float], pct: float) -> float:
    """Nearest-rank percentile (P100 == max)."""
    ordered = sorted(float(v) for v in values)
    if not ordered:
        return 0.0
    rank = max(1, math.ceil(pct / 100.0 * len(ordered)))
    return round(ordered[min(rank, len(ordered)) - 1], 2)


def _stage(metrics: dict[str, float] | None, key: str) -> float:
    if not metrics:
        return 0.0
    return round(float(metrics.get(key, 0.0)), 2)


def run_benchmark(
    harness: RAGHarness,
    queries: Iterable[BenchmarkQuery],
    provider_label: str,
    model_label: str,
    top_k: int | None = None,
) -> list[BenchmarkRecord]:
    records: list[BenchmarkRecord] = []
    for query in queries:
        error: str | None = None
        try:
            response = harness.answer(query.query, top_k=top_k, debug=True)
        except Exception as exc:  # noqa: BLE001 - a failing provider is a result too
            records.append(
                BenchmarkRecord(
                    provider=provider_label,
                    model=model_label,
                    query_id=query.id,
                    category=query.category,
                    language=query.language,
                    query=query.query,
                    retrieval_latency_ms=0.0,
                    context_latency_ms=0.0,
                    generation_latency_ms=0.0,
                    grounding_latency_ms=0.0,
                    total_latency_ms=0.0,
                    grounded=False,
                    guardrail="benchmark_error",
                    answer="",
                    confidence=0.0,
                    error=f"{type(exc).__name__}: {exc}",
                )
            )
            continue

        llm_info = response.llm or {}
        records.append(
            BenchmarkRecord(
                provider=str(llm_info.get("provider") or provider_label),
                model=str(llm_info.get("model") or model_label),
                query_id=query.id,
                category=query.category,
                language=query.language,
                query=query.query,
                retrieval_latency_ms=_stage(response.metrics, "retrieval"),
                context_latency_ms=_stage(response.metrics, "context"),
                generation_latency_ms=_stage(response.metrics, "generation"),
                grounding_latency_ms=_stage(response.metrics, "grounding"),
                total_latency_ms=_stage(response.metrics, "total"),
                grounded=response.grounded,
                guardrail=response.guardrail,
                answer=response.answer,
                confidence=response.confidence,
                error=error,
                usage=dict(llm_info.get("usage") or {}),
            )
        )
    return records


def summarize(records: list[BenchmarkRecord]) -> dict[str, Any]:
    """Latency percentiles + behavior counts for one provider run."""
    generated = [r for r in records if r.generation_latency_ms > 0.0]
    stages = {
        "retrieval": [r.retrieval_latency_ms for r in records],
        "context": [r.context_latency_ms for r in records],
        "generation": [r.generation_latency_ms for r in generated],
        "grounding": [r.grounding_latency_ms for r in records],
        "total": [r.total_latency_ms for r in records],
    }
    latency = {
        stage: {
            "p50": percentile(values, 50),
            "p70": percentile(values, 70),
            "p100": percentile(values, 100),
            "avg": round(sum(values) / len(values), 2) if values else 0.0,
            "n": len(values),
        }
        for stage, values in stages.items()
    }
    guardrails: dict[str, int] = {}
    for record in records:
        key = record.guardrail or "answered"
        guardrails[key] = guardrails.get(key, 0) + 1
    by_category: dict[str, dict[str, int]] = {}
    for record in records:
        bucket = by_category.setdefault(
            record.category, {"total": 0, "answered": 0, "refused": 0, "grounded": 0}
        )
        bucket["total"] += 1
        if record.guardrail in (None, "ungrounded"):
            bucket["answered"] += 1
        else:
            bucket["refused"] += 1
        if record.grounded:
            bucket["grounded"] += 1
    return {
        "queries": len(records),
        "errors": sum(1 for r in records if r.error),
        "latency_ms": latency,
        "avg_generation_ms": latency["generation"]["avg"],
        "guardrails": guardrails,
        "by_category": by_category,
        "sub_200ms_total": latency["total"]["p50"] < 200.0,
    }


def benchmark_report(
    runs: dict[str, dict[str, Any]],
    query_count: int,
    dataset: dict[str, Any],
    embedder: dict[str, Any],
) -> dict[str, Any]:
    return {
        "benchmark": "indicvoicerag-phase2.5-llm-providers",
        "query_count": query_count,
        "dataset": dataset,
        "embedder": embedder,
        "runs": runs,
    }


def records_as_dicts(records: list[BenchmarkRecord]) -> list[dict[str, Any]]:
    return [asdict(record) for record in records]
