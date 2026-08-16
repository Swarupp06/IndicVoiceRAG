"""RAG orchestration harness (Phase 2).

Deterministic pipeline, not an agent:

  request
  -> validate input
  -> safety check
  -> retrieve
  -> retrieval quality check
  -> build context
  -> generate answer (with retries + timeout)
  -> validate output
  -> grounding check (with strict regeneration retry)
  -> structured response

Every stage is timed; timings are returned when `debug=True`.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, Callable

from .config import AppConfig, GuardrailConfig, LLMConfig, RetrievalConfig
from .context import ContextBuilder, ContextPassage
from .grounding import GroundingValidator, build_grounding_validator
from .guardrails import (
    InputValidator,
    RetrievalQualityChecker,
    SafetyChecker,
)
from .llm import LLMProvider, LLMResponse
from .prompts import build_rag_messages, is_refusal, parse_confidence, strip_confidence
from .rag_types import RAGResponse, SourceInfo
from .vector_store import RetrievalHit


FALLBACK_GENERATION_FAILED = (
    "Sorry, I could not generate a response at this time. Please try again."
)


@dataclass(slots=True)
class HarnessComponents:
    retrieve_fn: Callable[[str, int], list[RetrievalHit]]
    llm: LLMProvider
    retrieval_config: RetrievalConfig
    llm_config: LLMConfig
    guardrails: GuardrailConfig


class RAGHarness:
    def __init__(
        self,
        components: HarnessComponents,
        input_validator: InputValidator | None = None,
        safety_checker: SafetyChecker | None = None,
        quality_checker: RetrievalQualityChecker | None = None,
        context_builder: ContextBuilder | None = None,
        grounding_validator: GroundingValidator | None = None,
    ):
        self._components = components
        self._input_validator = input_validator or InputValidator(
            max_query_chars=components.guardrails.max_query_chars
        )
        self._safety_checker = safety_checker or SafetyChecker()
        self._quality_checker = quality_checker or RetrievalQualityChecker(
            min_score=components.guardrails.min_retrieval_score,
            min_hits=components.guardrails.min_hits,
        )
        self._context_builder = context_builder or ContextBuilder(components.guardrails)
        self._grounding_validator = grounding_validator or build_grounding_validator(
            components.guardrails.grounding_threshold
        )

    # -- component accessors --
    @property
    def llm(self) -> LLMProvider:
        return self._components.llm

    @property
    def config(self) -> AppConfig:
        return AppConfig(
            retrieval=self._components.retrieval_config,
            llm=self._components.llm_config,
            guardrails=self._components.guardrails,
        )

    def answer(self, query: str, top_k: int | None = None, debug: bool = False) -> RAGResponse:
        timings: dict[str, float] = {}
        start_total = time.perf_counter()
        step = start_total

        def mark(name: str) -> None:
            nonlocal step
            now = time.perf_counter()
            timings[name] = timings.get(name, 0.0) + (now - step) * 1000.0
            step = now

        top_k = top_k or self._components.retrieval_config.top_k
        guardrails = self._components.guardrails

        def done() -> None:
            timings["total"] = (time.perf_counter() - start_total) * 1000.0

        # 1. validate input
        validation = self._input_validator.validate(query)
        if not validation.valid:
            done()
            return RAGResponse(
                query=query,
                answer="",
                grounded=False,
                confidence=0.0,
                sources=[],
                reason=validation.reason,
                guardrail="invalid_input",
                metrics=timings if debug else None,
            )

        # 2. safety check
        safety = self._safety_checker.check(query)
        if not safety.safe:
            done()
            return RAGResponse(
                query=query,
                answer="",
                grounded=False,
                confidence=0.0,
                sources=[],
                reason=safety.reason,
                guardrail="unsafe_input",
                metrics=timings if debug else None,
            )

        # 3. retrieve
        hits = self._components.retrieve_fn(query, max(1, top_k))
        mark("retrieval")

        # 4. retrieval quality check (off-topic / low relevance / no evidence)
        quality = self._quality_checker.check(hits)
        if not quality.ok:
            done()
            return RAGResponse(
                query=query,
                answer="",
                grounded=False,
                confidence=quality.best_score,
                sources=[],
                reason=quality.reason,
                guardrail=quality.guardrail,
                metrics=timings if debug else None,
            )

        # 5. build context (dedup, ordering, limits, delimited boundaries)
        passages = self._context_builder.build(hits)
        context_text = self._context_builder.render(passages)
        mark("context")

        # 6. generate (with retries + timeout)
        strict = False
        result: LLMResponse | None = None
        last_error: Exception | None = None
        max_retries = max(0, self._components.llm_config.max_retries)
        timeout = self._components.llm_config.timeout_seconds
        for _ in range(max_retries + 1):
            messages = build_rag_messages(query, context_text, strict=strict)
            try:
                result = self._components.llm.generate(
                    messages,
                    max_tokens=self._components.llm_config.max_tokens,
                    temperature=self._components.llm_config.temperature,
                    timeout=timeout,
                )
                break
            except Exception as exc:  # noqa: BLE001 - provider failures handled
                last_error = exc
                result = None
                time.sleep(0.1)
        mark("generation")

        llm_info = self._llm_info(result)
        generated = result.text if result is not None else None

        if generated is None or not generated.strip():
            reason = f"generation failed after {max_retries + 1} attempt(s)"
            if last_error is not None:
                reason += f": {type(last_error).__name__}: {last_error}"
            done()
            return RAGResponse(
                query=query,
                answer=FALLBACK_GENERATION_FAILED,
                grounded=False,
                confidence=0.0,
                sources=[self._source_from_passage(p) for p in passages],
                reason=reason,
                guardrail="generation_failed",
                metrics=timings if debug else None,
                llm=llm_info,
            )

        answer = strip_confidence(generated).strip()
        confidence = parse_confidence(generated)

        # 7. validate output (non-empty)
        if not answer:
            done()
            return RAGResponse(
                query=query,
                answer=FALLBACK_GENERATION_FAILED,
                grounded=False,
                confidence=0.0,
                sources=[self._source_from_passage(p) for p in passages],
                reason="generator returned an empty answer",
                guardrail="generation_failed",
                metrics=timings if debug else None,
                llm=llm_info,
            )

        # 8. grounding check (with strict regeneration retry)
        grounding = self._grounding_validator.validate(answer, context_text, query)
        mark("grounding")

        if not grounding.grounded and guardrails.strict_retry_on_ungrounded:
            for _ in range(max(0, guardrails.max_ungrounded_retries)):
                messages = build_rag_messages(query, context_text, strict=True)
                try:
                    strict_result = self._components.llm.generate(
                        messages,
                        max_tokens=self._components.llm_config.max_tokens,
                        temperature=self._components.llm_config.temperature,
                        timeout=timeout,
                    )
                except Exception:  # noqa: BLE001
                    mark("generation")
                    break
                mark("generation")
                strict_answer = strip_confidence(strict_result.text).strip()
                llm_info = self._llm_info(strict_result)
                if not strict_answer:
                    break
                strict_grounding = self._grounding_validator.validate(
                    strict_answer, context_text, query
                )
                mark("grounding")
                if strict_grounding.grounded or is_refusal(strict_answer):
                    answer = strict_answer
                    grounding = strict_grounding
                    break

        best_score = quality.best_score
        if is_refusal(answer):
            computed = 0.5 * best_score
        elif confidence is not None:
            computed = confidence
        else:
            computed = 0.6 * best_score + 0.4 * grounding.score
        confidence_value = round(max(0.0, min(1.0, computed)), 4)

        timings["total"] = (time.perf_counter() - start_total) * 1000.0

        return RAGResponse(
            query=query,
            answer=answer,
            grounded=grounding.grounded,
            confidence=confidence_value,
            sources=[self._source_from_passage(p) for p in passages],
            reason=grounding.reason,
            guardrail=None if grounding.grounded else "ungrounded",
            metrics=timings if debug else None,
            llm=llm_info,
        )

    def _llm_info(self, result: LLMResponse | None) -> dict[str, Any]:
        if result is None:
            return {"provider": self._components.llm.name, "model": self._components.llm.model_name}
        return {
            "provider": result.provider,
            "model": result.model,
            "latency_ms": round(result.latency_ms, 2),
            "usage": result.usage,
        }

    @staticmethod
    def _source_from_passage(passage: ContextPassage) -> SourceInfo:
        metadata = passage.metadata
        return SourceInfo(
            document_id=passage.document_id,
            chunk_id=passage.chunk_id,
            score=passage.score,
            query_id=str(metadata.get("query_id")) if metadata.get("query_id") is not None else None,
            language=str(metadata.get("language")) if metadata.get("language") else None,
            relevance=float(metadata["relevance"]) if metadata.get("relevance") is not None else None,
            excerpt=passage.text,
        )


