from indicvoicerag.config import GuardrailConfig, LLMConfig, RetrievalConfig
from indicvoicerag.harness import HarnessComponents, RAGHarness
from indicvoicerag.llm import MockLLMProvider
from indicvoicerag.rag_evaluate import (
    RAGEvalCase,
    build_sample_cases,
    evaluate_rag,
)
from indicvoicerag.vector_store import RetrievalHit


def _hit(doc_id: str, score: float, text: str) -> RetrievalHit:
    return RetrievalHit(
        chunk_id=f"{doc_id}:0",
        document_id=doc_id,
        score=score,
        text=text,
        metadata={"query_id": "q1", "language": "en", "relevance": 1.0, "strategy": "fixed"},
    )


def _harness(retrieve_fn, behavior: str = "grounded") -> RAGHarness:
    components = HarnessComponents(
        retrieve_fn=retrieve_fn,
        llm=MockLLMProvider(behavior),
        retrieval_config=RetrievalConfig(top_k=5),
        llm_config=LLMConfig(provider="mock", max_retries=0),
        guardrails=GuardrailConfig(),
    )
    return RAGHarness(components)


def test_evaluate_rag_reports_breakdown_and_latency() -> None:
    hits = [_hit("d1", 0.9, "The capital of India is New Delhi. It is a big city.")]

    def retrieve_fn(query: str, top_k: int):  # noqa: ARG001
        return hits

    harness = _harness(retrieve_fn)
    cases = build_sample_cases(
        [("What is the capital of India?", "q1")],
        extra_unanswerable=["What is the weather in Bangkok tomorrow?"],
    )
    # Force the extra probe to be refused by the quality check.
    def retrieve_conditional(query: str, top_k: int):
        if "Bangkok" in query:
            return [_hit("d9", 0.05, "unrelated")]
        return hits

    harness = _harness(retrieve_conditional)
    report = evaluate_rag(harness, cases, debug=True)
    assert report.total == 2
    assert report.answered == 1
    assert report.refused == 1
    assert report.grounded_answers == 1
    assert report.ungrounded_answers == 0
    assert report.breakdown.get("low_relevance") == 1
    assert report.refusal_rate == 0.5
    assert report.grounded_ratio == 1.0
    assert report.avg_total_ms > 0
    assert "avg_generation_ms" in report.as_dict()


def test_evaluate_rag_ungrounded_answer_path() -> None:
    hits = [_hit("d1", 0.9, "The capital of India is New Delhi.")]

    def retrieve_fn(query: str, top_k: int):  # noqa: ARG001
        return hits

    harness = _harness(retrieve_fn, behavior="ungrounded")
    report = evaluate_rag(harness, [RAGEvalCase(query="q", query_id="q1")], debug=False)
    assert report.answered == 1
    assert report.ungrounded_answers == 1
    assert report.breakdown.get("ungrounded") == 1
    assert report.grounded_ratio == 0.0
