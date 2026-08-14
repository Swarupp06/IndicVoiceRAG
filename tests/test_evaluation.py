from indicvoicerag.evaluation import (
    evaluate_retrieval,
    hit_at_k,
    mean_reciprocal_rank,
    ranked_document_ids_from_hits,
    recall_at_k,
)
from indicvoicerag.schemas import NormalizedDocument
from indicvoicerag.vector_store import RetrievalHit


def test_hit_recall_and_mrr() -> None:
    relevant = {"d2", "d3"}
    ranked = ["d1", "d2", "d4"]
    assert hit_at_k(relevant, ranked, k=1) == 0.0
    assert hit_at_k(relevant, ranked, k=2) == 1.0
    assert recall_at_k(relevant, ranked, k=2) == 0.5
    assert mean_reciprocal_rank([]) == 0.0


def _hit(chunk_id: str, doc_id: str) -> RetrievalHit:
    return RetrievalHit(chunk_id=chunk_id, document_id=doc_id, score=0.5, text="", metadata={})


def test_ranked_document_ids_aggregates_chunks_to_documents() -> None:
    hits = [
        _hit("d1:0", "d1"),
        _hit("d2:0", "d2"),
        _hit("d1:1", "d1"),
        _hit("d3:0", "d3"),
        _hit("d4:0", "d4"),
    ]
    ranked = ranked_document_ids_from_hits(hits, top_k=3)
    assert ranked == ["d1", "d2", "d3"]


def _doc(query_id: str, doc_id: str, relevant: bool) -> NormalizedDocument:
    return NormalizedDocument(
        document_id=doc_id,
        query_id=query_id,
        query_text=f"query for {query_id}",
        passage_text="passage body",
        language="hin",
        source_language="en",
        target_language="hin",
        query_type="DESCRIPTION",
        relevance=1.0 if relevant else 0.0,
        metadata={},
    )


def test_evaluate_retrieval_computes_hit_recall_mrr() -> None:
    docs = [
        _doc("q1", "q1:0", False),
        _doc("q1", "q1:1", True),
        _doc("q1", "q1:2", False),
        _doc("q2", "q2:0", True),
        _doc("q2", "q2:1", False),
    ]

    def retrieve_fn(query_text: str, top_k: int) -> list[RetrievalHit]:
        if query_text == "query for q1":
            order = ["q1:0", "q1:1", "q1:2"]
        else:
            order = ["q2:1", "q2:0"]
        return [_hit(f"{doc}:0", doc) for doc in order[:top_k]]

    metrics = evaluate_retrieval(docs, retrieve_fn, top_k=5)
    payload = metrics.as_dict()
    assert payload["queries"] == 2
    # q1 ranked [q1:0, q1:1] -> hit@1=0, rr=1/2; q2 ranked [q2:1, q2:0] -> hit@1=0, rr=1/2
    assert payload["hit@1"] == 0.0
    assert payload["hit@3"] == 1.0
    assert payload["mrr"] == 0.5
