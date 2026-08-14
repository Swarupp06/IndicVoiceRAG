from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

from .schemas import NormalizedDocument


@dataclass(slots=True)
class QueryEvaluationCase:
    query_id: str
    relevant_document_ids: set[str]
    ranked_document_ids: list[str]


def hit_at_k(relevant_ids: set[str], ranked_ids: list[str], k: int) -> float:
    top = ranked_ids[:k]
    return 1.0 if any(doc_id in relevant_ids for doc_id in top) else 0.0


def recall_at_k(relevant_ids: set[str], ranked_ids: list[str], k: int) -> float:
    if not relevant_ids:
        return 0.0
    top = set(ranked_ids[:k])
    return len(relevant_ids.intersection(top)) / len(relevant_ids)


def reciprocal_rank(relevant_ids: set[str], ranked_ids: list[str]) -> float:
    for rank, doc_id in enumerate(ranked_ids, start=1):
        if doc_id in relevant_ids:
            return 1.0 / rank
    return 0.0


def mean_reciprocal_rank(cases: Iterable[QueryEvaluationCase]) -> float:
    values = [reciprocal_rank(case.relevant_document_ids, case.ranked_document_ids) for case in cases]
    if not values:
        return 0.0
    return sum(values) / len(values)


def build_query_relevance_map(documents: Iterable[NormalizedDocument]) -> dict[str, set[str]]:
    relevance_by_query: dict[str, set[str]] = {}
    for doc in documents:
        if doc.query_id is None or doc.relevance is None:
            continue
        if doc.relevance <= 0:
            continue
        relevance_by_query.setdefault(doc.query_id, set()).add(doc.document_id)
    return relevance_by_query
