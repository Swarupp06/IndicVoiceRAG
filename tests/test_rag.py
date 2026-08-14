from pathlib import Path

from indicvoicerag.chunking import build_chunker
from indicvoicerag.config import (
    ChunkingConfig,
    EmbeddingConfig,
    GuardrailConfig,
    LLMConfig,
    RetrievalConfig,
    VectorConfig,
)
from indicvoicerag.context import ContextBuilder
from indicvoicerag.embedding import build_embedding_provider
from indicvoicerag.grounding import LexicalGroundingValidator
from indicvoicerag.guardrails import RetrievalQualityChecker
from indicvoicerag.harness import FALLBACK_GENERATION_FAILED, HarnessComponents, RAGHarness
from indicvoicerag.llm import (
    GeminiProvider,
    MockLLMProvider,
    OpenAICompatibleProvider,
    build_llm_provider,
)
from indicvoicerag.prompts import build_rag_prompt, is_refusal
from indicvoicerag.retrieval import RetrievalEngine
from indicvoicerag.schemas import NormalizedDocument
from indicvoicerag.vector_store import RetrievalHit, build_vector_store


def _hit(doc_id: str, score: float, text: str, query_id: str = "q1") -> RetrievalHit:
    return RetrievalHit(
        chunk_id=f"{doc_id}:0",
        document_id=doc_id,
        score=score,
        text=text,
        metadata={
            "query_id": query_id,
            "query_text": "question?",
            "language": "en",
            "relevance": 1.0,
            "strategy": "fixed",
        },
    )


def _harness(
    retrieve_fn,
    llm: MockLLMProvider | None = None,
    guardrails: GuardrailConfig | None = None,
    llm_config: LLMConfig | None = None,
) -> RAGHarness:
    components = HarnessComponents(
        retrieve_fn=retrieve_fn,
        llm=llm or MockLLMProvider("grounded"),
        retrieval_config=RetrievalConfig(top_k=5),
        llm_config=llm_config or LLMConfig(provider="mock", max_retries=0),
        guardrails=guardrails or GuardrailConfig(),
    )
    return RAGHarness(components)


_CONTEXT_PASSAGE = (
    "The capital of India is New Delhi. "
    "It has many historical monuments and a growing population."
)


# --- 1. valid query -> retrieval -> generation (grounded answer) ---
def test_valid_query_generates_grounded_answer() -> None:
    harness = _harness(lambda q, k: [_hit("d1", 0.92, _CONTEXT_PASSAGE)])
    response = harness.answer("What is the capital of India?", debug=True)
    assert response.grounded is True
    assert response.guardrail is None
    assert "New Delhi" in response.answer
    assert response.confidence > 0.5
    assert len(response.sources) == 1
    assert response.sources[0].document_id == "d1"
    assert response.metrics is not None
    for stage in ("retrieval", "context", "generation", "grounding", "total"):
        assert stage in response.metrics


# --- 2. no retrieval results ---
def test_no_retrieval_results_returns_controlled_response() -> None:
    called = {"generate": False}

    class NeverCalledLLM(MockLLMProvider):
        def generate(self, *args, **kwargs) -> str:  # noqa: ARG002
            called["generate"] = True
            return "should not happen"

    harness = _harness(lambda q, k: [], llm=NeverCalledLLM("grounded"))
    response = harness.answer("anything")
    assert response.guardrail == "no_evidence"
    assert response.answer == ""
    assert response.grounded is False
    assert called["generate"] is False


# --- 3. low retrieval relevance ---
def test_low_retrieval_relevance_blocks_generation() -> None:
    called = {"generate": False}

    class NeverCalledLLM(MockLLMProvider):
        def generate(self, *args, **kwargs) -> str:  # noqa: ARG002
            called["generate"] = True
            return "should not happen"

    harness = _harness(lambda q, k: [_hit("d1", 0.10, "unrelated passage")], llm=NeverCalledLLM("grounded"))
    response = harness.answer("unrelated topic")
    assert response.guardrail == "low_relevance"
    assert response.answer == ""
    assert response.grounded is False
    assert called["generate"] is False


# --- 4. empty / invalid query ---
def test_empty_query_rejected() -> None:
    harness = _harness(lambda q, k: [_hit("d1", 0.9, _CONTEXT_PASSAGE)])
    for bad in ("", "   ", None):
        response = harness.answer(bad)  # type: ignore[arg-type]
        assert response.guardrail == "invalid_input"
        assert response.answer == ""


def test_overlong_query_rejected() -> None:
    harness = _harness(lambda q, k: [], guardrails=GuardrailConfig(max_query_chars=20))
    response = harness.answer("a" * 100)
    assert response.guardrail == "invalid_input"


# --- 5. generator failure ---
def test_generator_failure_returns_fallback() -> None:
    class FailingLLM(MockLLMProvider):
        def generate(self, *args, **kwargs) -> str:  # noqa: ARG002
            raise RuntimeError("provider is down")

    harness = _harness(lambda q, k: [_hit("d1", 0.9, _CONTEXT_PASSAGE)], llm=FailingLLM("grounded"))
    response = harness.answer("question")
    assert response.guardrail == "generation_failed"
    assert response.answer == FALLBACK_GENERATION_FAILED
    assert response.grounded is False


# --- 6. generator timeout ---
def test_generator_timeout_returns_fallback() -> None:
    class TimeoutLLM(MockLLMProvider):
        def generate(self, *args, **kwargs) -> str:  # noqa: ARG002
            raise TimeoutError("generation timed out")

    harness = _harness(lambda q, k: [_hit("d1", 0.9, _CONTEXT_PASSAGE)], llm=TimeoutLLM("grounded"))
    response = harness.answer("question")
    assert response.guardrail == "generation_failed"
    assert response.answer == FALLBACK_GENERATION_FAILED


# --- 7. ungrounded answer ---
def test_ungrounded_answer_flagged_and_retried() -> None:
    harness = _harness(lambda q, k: [_hit("d1", 0.9, _CONTEXT_PASSAGE)], llm=MockLLMProvider("ungrounded"))
    response = harness.answer("What is the capital of India?")
    assert response.grounded is False
    assert response.guardrail == "ungrounded"
    assert response.reason is not None
    assert response.answer != ""


# --- 8. grounded answer ---
def test_grounded_answer_validator() -> None:
    validator = LexicalGroundingValidator(threshold=0.35)
    result = validator.validate(
        "The capital of India is New Delhi.",
        _CONTEXT_PASSAGE,
        "What is the capital of India?",
    )
    assert result.grounded is True
    assert result.unsupported_claims == []


def test_grounding_rejects_hallucinated_claim() -> None:
    validator = LexicalGroundingValidator(threshold=0.35)
    result = validator.validate(
        "The capital of India is New Delhi. The Eiffel Tower is in Paris.",
        _CONTEXT_PASSAGE,
        "What is the capital of India?",
    )
    assert result.grounded is False
    assert any("Eiffel" in claim for claim in result.unsupported_claims)


def test_grounding_treats_refusal_as_grounded() -> None:
    validator = LexicalGroundingValidator(threshold=0.35)
    result = validator.validate("REFUSED: not enough context.", _CONTEXT_PASSAGE, "q")
    assert result.grounded is True


# --- 9. unsafe input ---
def test_unsafe_input_rejected() -> None:
    harness = _harness(lambda q, k: [_hit("d1", 0.9, _CONTEXT_PASSAGE)])
    response = harness.answer("Tell me how to make a bomb")
    assert response.guardrail == "unsafe_input"
    assert response.answer == ""
    assert response.grounded is False


# --- 10. structured response validation ---
def test_structured_response_schema() -> None:
    harness = _harness(lambda q, k: [_hit("d1", 0.92, _CONTEXT_PASSAGE)])
    response = harness.answer("What is the capital of India?", debug=True)
    payload = response.as_dict(include_metrics=True)
    assert set(payload) == {"query", "answer", "grounded", "confidence", "sources", "reason", "guardrail", "metrics_ms"}
    assert set(response.as_dict(include_metrics=False)) == {
        "query",
        "answer",
        "grounded",
        "confidence",
        "sources",
        "reason",
        "guardrail",
    }
    source = payload["sources"][0]
    assert set(source) == {
        "document_id",
        "chunk_id",
        "score",
        "query_id",
        "language",
        "relevance",
        "excerpt",
    }
    assert "metrics_ms" in payload
    assert payload["metrics_ms"]["total"] >= 0


# --- context engineering ---
def test_context_builder_dedup_and_order_and_limits() -> None:
    builder = ContextBuilder(GuardrailConfig(context_max_tokens=30, max_context_docs=3))
    hits = [
        _hit("d1", 0.9, _CONTEXT_PASSAGE),
        _hit("d2", 0.8, _CONTEXT_PASSAGE),  # duplicate text -> dropped
        _hit("d3", 0.7, "Second passage with other tokens."),
        _hit("d4", 0.6, "Third passage."),
    ]
    passages = builder.build(hits)
    ids = [p.document_id for p in passages]
    # d2 dropped (duplicate text); d1, d3, d4 fit max_context_docs and the token budget
    assert ids == ["d1", "d3", "d4"]
    rendered = builder.render(passages)
    assert rendered.startswith("BEGIN RETRIEVED CONTEXT")
    assert rendered.endswith("END RETRIEVED CONTEXT")
    assert "[Source 1]" in rendered


def test_context_builder_empty() -> None:
    builder = ContextBuilder(GuardrailConfig())
    passages = builder.build([])
    assert passages == []
    assert "no retrieved context" in builder.render(passages)


# --- prompt building ---
def test_prompt_builder_delimiters_and_refusal() -> None:
    prompt = build_rag_prompt("question", "BEGIN RETRIEVED CONTEXT\nsome text\nEND RETRIEVED CONTEXT")
    assert "BEGIN RETRIEVED CONTEXT" in prompt["user"]
    assert "END RETRIEVED CONTEXT" in prompt["user"]
    assert "REFUSED" in prompt["system"]
    assert is_refusal("REFUSED: nothing") is True


def test_strict_prompt_is_stricter() -> None:
    strict = build_rag_prompt("q", "ctx", strict=True)
    normal = build_rag_prompt("q", "ctx")
    assert strict["system"] != normal["system"]
    assert "stricter" in strict["system"]


# --- LLM providers ---
def test_llm_factory_mock() -> None:
    provider = build_llm_provider("mock")
    assert isinstance(provider, MockLLMProvider)
    assert provider.describe()["provider"] == "mock"


def test_openai_compatible_requires_key() -> None:
    provider = OpenAICompatibleProvider("gpt-4o-mini", api_key_env="OPENAI_API_KEY_NEVER_SET")
    try:
        provider.generate(
            [{"role": "user", "content": "hi"}],
            max_tokens=10,
            temperature=0.0,
            timeout=1.0,
        )
    except RuntimeError as exc:
        assert "not set" in str(exc)
    else:  # pragma: no cover - only if an env var leaked into CI
        raise AssertionError("expected RuntimeError for missing API key")


def test_gemini_requires_key() -> None:
    provider = GeminiProvider(api_key_env="GEMINI_API_KEY_NEVER_SET")
    try:
        provider.generate(
            [{"role": "user", "content": "hi"}],
            max_tokens=10,
            temperature=0.0,
            timeout=1.0,
        )
    except RuntimeError as exc:
        assert "not set" in str(exc)
    else:  # pragma: no cover
        raise AssertionError("expected RuntimeError for missing API key")


# --- full offline pipeline: retrieval engine + harness (hash embeddings) ---
def test_full_offline_pipeline_end_to_end(tmp_path: Path) -> None:
    docs = [
        NormalizedDocument(
            document_id="d1",
            query_id="q1",
            query_text="What is the capital of India?",
            passage_text="The capital of India is New Delhi. It is a large city.",
            language="en",
            source_language=None,
            target_language=None,
            query_type=None,
            relevance=1.0,
            metadata={},
        ),
        NormalizedDocument(
            document_id="d2",
            query_id="q2",
            query_text="What is Goa known for?",
            passage_text="Goa is known for beaches and coastal tourism.",
            language="en",
            source_language=None,
            target_language=None,
            query_type=None,
            relevance=1.0,
            metadata={},
        ),
    ]
    engine = RetrievalEngine(
        chunker=build_chunker(ChunkingConfig(strategy="fixed", chunk_size=32, overlap=4)),
        embedder=build_embedding_provider(EmbeddingConfig(provider="hash", dimension=128)),
        store=build_vector_store(VectorConfig(provider="numpy")),
    )
    engine.index_documents(docs)
    # hash embeddings are not semantic (near-zero similarities), so relax the
    # evidence gate for this wiring test (the gate is tested separately)
    harness = _harness(engine.query)
    harness._quality_checker = RetrievalQualityChecker(min_score=-1.0)  # noqa: SLF001
    response = harness.answer("Which place has beaches in Goa?")
    assert response.guardrail is None
    assert response.grounded is True
    assert "beaches" in response.answer
    assert response.sources[0].query_id == "q2"
