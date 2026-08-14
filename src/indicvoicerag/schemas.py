from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
from typing import Any


def _first_non_empty(source: dict[str, Any], keys: list[str]) -> Any:
    for key in keys:
        value = source.get(key)
        if value is not None and value != "":
            return value
    return None


def _language_code(value: Any) -> str | None:
    """Reduce a language tag like 'hin_Deva' or 'hindi' to a stable code ('hin')."""
    if value is None:
        return None
    text = str(value).strip().lower()
    if not text:
        return None
    code = text.split("_")[0].split("-")[0]
    aliases = {
        "english": "en",
        "eng": "en",
        "en": "en",
        "hindi": "hin",
        "hin": "hin",
        "marathi": "mar",
        "mar": "mar",
        "bengali": "ben",
        "ben": "ben",
        "gujarati": "guj",
        "guj": "guj",
        "tamil": "tam",
        "tam": "tam",
        "telugu": "tel",
        "tel": "tel",
        "kannada": "kan",
        "kan": "kan",
        "malayalam": "mal",
        "mal": "mal",
        "assamese": "asm",
        "asm": "asm",
        "nepali": "nep",
        "nep": "nep",
        "oriya": "ori",
        "odia": "ori",
        "ori": "ori",
        "punjabi": "pan",
        "pan": "pan",
        "sanskrit": "san",
        "san": "san",
        "urdu": "urd",
        "urd": "urd",
    }
    return aliases.get(code, code)


@dataclass(slots=True)
class NormalizedDocument:
    document_id: str
    query_id: str | None
    query_text: str | None
    passage_text: str
    language: str | None
    source_language: str | None
    target_language: str | None
    query_type: str | None
    relevance: float | None
    metadata: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_record(cls, record: dict[str, Any], fallback_language: str | None = None) -> "NormalizedDocument":
        passage_text = _first_non_empty(
            record,
            [
                "passage",
                "passage_text",
                "text",
                "document",
                "doc_text",
                "body",
            ],
        )
        if passage_text is None:
            raise ValueError("Record is missing a passage/text field.")
        passage_text = str(passage_text).strip()
        if not passage_text:
            raise ValueError("Record passage/text is empty.")

        query_id = _first_non_empty(record, ["query_id", "qid", "id_query"])
        query_text = _first_non_empty(record, ["query", "query_text", "question"])
        query_type = _first_non_empty(record, ["query_type", "type"])
        language = _language_code(
            _first_non_empty(record, ["language", "lang", "target_language", "target_lang", "tgt_lang"])
        ) or _language_code(fallback_language)
        source_language = _language_code(
            _first_non_empty(record, ["source_language", "src_lang", "source_lang"])
        )
        target_language = _language_code(
            _first_non_empty(record, ["target_language", "tgt_lang", "target_lang"])
        )
        relevance = _first_non_empty(record, ["relevance", "label", "score", "is_relevant"])

        raw_doc_id = _first_non_empty(record, ["doc_id", "document_id", "passage_id", "id"])
        if raw_doc_id is None:
            stable_seed = f"{query_id}|{passage_text[:80]}"
            raw_doc_id = hashlib.sha1(stable_seed.encode("utf-8")).hexdigest()[:16]

        normalized_known_keys = {
            "query_id",
            "qid",
            "id_query",
            "query",
            "query_text",
            "question",
            "query_type",
            "type",
            "language",
            "lang",
            "source_language",
            "src_lang",
            "source_lang",
            "target_language",
            "tgt_lang",
            "target_lang",
            "relevance",
            "label",
            "score",
            "is_relevant",
            "doc_id",
            "document_id",
            "passage_id",
            "id",
            "passage",
            "passage_text",
            "text",
            "document",
            "doc_text",
            "body",
        }
        metadata = {k: v for k, v in record.items() if k not in normalized_known_keys}

        rel_value: float | None
        if relevance is None:
            rel_value = None
        else:
            try:
                rel_value = float(relevance)
            except (TypeError, ValueError):
                rel_value = None

        return cls(
            document_id=str(raw_doc_id),
            query_id=None if query_id is None else str(query_id),
            query_text=None if query_text is None else str(query_text),
            passage_text=passage_text,
            language=None if language is None else str(language),
            source_language=None if source_language is None else str(source_language),
            target_language=None if target_language is None else str(target_language),
            query_type=None if query_type is None else str(query_type),
            relevance=rel_value,
            metadata=metadata,
        )


@dataclass(slots=True)
class DocumentChunk:
    chunk_id: str
    document_id: str
    text: str
    chunk_index: int
    metadata: dict[str, Any] = field(default_factory=dict)
