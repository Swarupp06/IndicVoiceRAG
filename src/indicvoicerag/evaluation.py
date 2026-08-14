from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Iterable

from .schemas import NormalizedDocument
from .vector_store import RetrievalHit


@dataclass(slots=True)
class QueryEvaluationCase:
    query_id: str
    relevant_document_ids: set[str]
    ranked_document_ids: list[str]


@dataclass(slots=True)
class RetrievalMetrics:
    queries: int
    hit_at_1: float
    hit_at_3: float
    hit_at_5: float
    hit_at_10: float
    recall_at_5: float
    recall_at_10: float
    mrr: float

    def as_dict(self) -> dict[str, float | int]:
        return {
            "queries": self.queries,
            "hit@1": round(self.hit_at_1, 4),
            "hit@3": round(self.hit_at_3, 4),
            "hit@5": round(self.hit_at_5, 4),
            "hit@10": round(self.hit_at_10, 4),
            "recall@5": round(self.recall_at_5, 4),
            "recall@10": round(self.recall_at_10, 4),
            "mrr": round(self.mrr, 4),
        }


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


def ranked_document_ids_from_hits(hits: Iterable[RetrievalHit], top_k: int) -> list[str]:
    """Aggregate chunk-level hits into a de-duplicated document-level ranking."""
    seen: set[str] = set()
    ranked: list[str] = []
    for hit in hits:
        if hit.document_id in seen:
            continue
        seen.add(hit.document_id)
        ranked.append(hit.document_id)
        if len(ranked) >= top_k:
            break
    return ranked


def build_evaluation_cases(
    documents: Iterable[NormalizedDocument],
    ranked_by_query: dict[str, list[str]],
    top_k: int,
) -> list[QueryEvaluationCase]:
    relevance_map = build_query_relevance_map(documents)
    cases: list[QueryEvaluationCase] = []
    for query_id, relevant in sorted(relevance_map.items()):
        ranked = ranked_by_query.get(query_id, [])
        cases.append(
            QueryEvaluationCase(
                query_id=query_id,
                relevant_document_ids=relevant,
                ranked_document_ids=ranked[:top_k],
            )
        )
    return cases


def summarize_metrics(cases: list[QueryEvaluationCase]) -> RetrievalMetrics:
    if not cases:
        return RetrievalMetrics(0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    n = float(len(cases))
    return RetrievalMetrics(
        queries=len(cases),
        hit_at_1=sum(hit_at_k(c.relevant_document_ids, c.ranked_document_ids, 1) for c in cases) / n,
        hit_at_3=sum(hit_at_k(c.relevant_document_ids, c.ranked_document_ids, 3) for c in cases) / n,
        hit_at_5=sum(hit_at_k(c.relevant_document_ids, c.ranked_document_ids, 5) for c in cases) / n,
        hit_at_10=sum(hit_at_k(c.relevant_document_ids, c.ranked_document_ids, 10) for c in cases) / n,
        recall_at_5=sum(recall_at_k(c.relevant_document_ids, c.ranked_document_ids, 5) for c in cases) / n,
        recall_at_10=sum(recall_at_k(c.relevant_document_ids, c.ranked_document_ids, 10) for c in cases) / n,
        mrr=mean_reciprocal_rank(cases),
    )


def evaluate_retrieval(
    documents: list[NormalizedDocument],
    retrieve_fn: Callable[[str, int], list[RetrievalHit]],
    top_k: int = 10,
) -> RetrievalMetrics:
    """Run MSMARCO-style evaluation over documents that carry query ids.

    Each document must expose `query_id` and `relevance` (from the real
    dataset's `is_selected` labels). The query text for a query is taken from
    the first relevant document's `query_text`.
    """
    relevance_map = build_query_relevance_map(documents)
    query_text_by_id: dict[str, str] = {}
    for doc in documents:
        if doc.query_id and doc.query_text and doc.query_id not in query_text_by_id:
            query_text_by_id[doc.query_id] = doc.query_text

    ranked_by_query: dict[str, list[str]] = {}
    for query_id in relevance_map:
        query_text = query_text_by_id.get(query_id)
        if not query_text:
            continue
        hits = retrieve_fn(query_text, top_k * 3)
        ranked_by_query[query_id] = ranked_document_ids_from_hits(hits, top_k)

    cases = build_evaluation_cases(documents, ranked_by_query, top_k)
    return summarize_metrics(cases)
