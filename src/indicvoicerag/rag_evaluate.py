"""Small RAG evaluation harness.

Measures pipeline behavior on real queries without a human/LLM answer-quality
label set (which we do not have yet - documented in the README):

- answerable vs refused outcomes
- grounded vs ungrounded answers
- refusal behavior (off-topic / low relevance / no evidence)
- generation and total latency
- retrieval metrics can be layered on top of the Phase 1B evaluation
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
import time
from typing import Callable, Iterable

from .harness import RAGHarness
from .rag_types import RAGResponse


@dataclass(slots=True)
class RAGEvalCase:
    query: str
    query_id: str | None = None
    expected_answerable: bool | None = None  # None = unknown (no label set yet)


@dataclass(slots=True)
class RAGEvalReport:
    total: int = 0
    answered: int = 0
    refused: int = 0
    grounded_answers: int = 0
    ungrounded_answers: int = 0
    breakdown: dict[str, int] = field(default_factory=dict)
    generation_ms: list[float] = field(default_factory=list)
    total_ms: list[float] = field(default_factory=list)

    @property
    def refusal_rate(self) -> float:
        return self.refused / self.total if self.total else 0.0

    @property
    def grounded_ratio(self) -> float:
        return self.grounded_answers / self.answered if self.answered else 0.0

    @property
    def avg_generation_ms(self) -> float:
        return sum(self.generation_ms) / len(self.generation_ms) if self.generation_ms else 0.0

    @property
    def avg_total_ms(self) -> float:
        return sum(self.total_ms) / len(self.total_ms) if self.total_ms else 0.0

    def as_dict(self) -> dict[str, object]:
        payload = asdict(self)
        payload["refusal_rate"] = round(self.refusal_rate, 4)
        payload["grounded_ratio"] = round(self.grounded_ratio, 4)
        payload["avg_generation_ms"] = round(self.avg_generation_ms, 1)
        payload["avg_total_ms"] = round(self.avg_total_ms, 1)
        return payload


_REFUSAL_GUARDRAILS = {
    "invalid_input",
    "unsafe_input",
    "off_topic",
    "low_relevance",
    "no_evidence",
    "generation_failed",
}


def _outcome(response: RAGResponse) -> str:
    if response.guardrail in _REFUSAL_GUARDRAILS:
        return response.guardrail
    if response.guardrail == "ungrounded":
        return "ungrounded"
    return "grounded" if response.grounded else "ungrounded"


def evaluate_rag(
    harness: RAGHarness,
    cases: Iterable[RAGEvalCase],
    top_k: int | None = None,
    debug: bool = False,
) -> RAGEvalReport:
    report = RAGEvalReport()
    for case in cases:
        started = time.perf_counter()
        response = harness.answer(case.query, top_k=top_k, debug=debug)
        total_ms = (time.perf_counter() - started) * 1000.0
        outcome = _outcome(response)

        report.total += 1
        report.breakdown[outcome] = report.breakdown.get(outcome, 0) + 1
        if outcome in {"grounded", "ungrounded"}:
            report.answered += 1
            if outcome == "grounded":
                report.grounded_answers += 1
            else:
                report.ungrounded_answers += 1
        else:
            report.refused += 1

        if response.metrics and "generation" in response.metrics:
            report.generation_ms.append(response.metrics["generation"])
        else:
            report.generation_ms.append(total_ms)
        report.total_ms.append(total_ms)
    return report


def build_sample_cases(
    queries: Iterable[tuple[str, str | None]],
    extra_unanswerable: Iterable[str] = (),
) -> list[RAGEvalCase]:
    """Build eval cases from real (query, query_id) pairs plus synthetic
    out-of-domain queries to exercise refusal behavior."""
    cases = [RAGEvalCase(query=q, query_id=qid) for q, qid in queries]
    cases.extend(RAGEvalCase(query=q, expected_answerable=False) for q in extra_unanswerable)
    return cases
