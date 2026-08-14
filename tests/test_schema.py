from indicvoicerag.schemas import NormalizedDocument


def test_normalized_document_preserves_known_and_extra_metadata() -> None:
    record = {
        "query_id": "q1",
        "query_text": "capital of goa",
        "passage_text": "Panaji is the capital city of Goa.",
        "language": "en",
        "source_language": "en",
        "target_language": "en",
        "query_type": "factoid",
        "relevance": 1,
        "custom_field": "kept",
    }
    doc = NormalizedDocument.from_record(record)
    assert doc.query_id == "q1"
    assert doc.document_id != ""
    assert doc.passage_text.startswith("Panaji")
    assert doc.relevance == 1.0
    assert doc.metadata["custom_field"] == "kept"


def test_normalized_document_generates_stable_id_without_doc_id() -> None:
    record = {"query_id": "qA", "text": "Test passage."}
    doc_a = NormalizedDocument.from_record(record)
    doc_b = NormalizedDocument.from_record(record)
    assert doc_a.document_id == doc_b.document_id
