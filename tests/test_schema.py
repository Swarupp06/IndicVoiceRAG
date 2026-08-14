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


def test_normalized_document_real_msmarco_xi_keys() -> None:
    record = {
        "query_id": "1102432",
        "query": "कॉर्पोरेशन क्या है?",
        "Eng_Query": "what is a corporation?",
        "query_type": "DESCRIPTION",
        "source_lang": "eng_Latn",
        "target_lang": "hin_Deva",
        "doc_id": "1102432:5",
        "passage_text": "मैकडॉनल्ड कॉर्पोरेशन दुनिया के सबसे पहचानने योग्य निगमों में से एक है।",
        "Eng_passage_text": "McDonald's Corporation is one of the most recognizable corporations.",
        "relevance": 1,
        "is_selected": 1,
        "answer": "एक निगम।",
        "Eng_answer": "A corporation.",
    }
    doc = NormalizedDocument.from_record(record)
    assert doc.document_id == "1102432:5"
    assert doc.query_id == "1102432"
    assert doc.query_text == "कॉर्पोरेशन क्या है?"
    assert doc.language == "hin"
    assert doc.source_language == "en"
    assert doc.target_language == "hin"
    assert doc.query_type == "DESCRIPTION"
    assert doc.relevance == 1.0
    assert doc.metadata["Eng_Query"] == "what is a corporation?"
    assert doc.metadata["Eng_passage_text"] == "McDonald's Corporation is one of the most recognizable corporations."
    assert doc.metadata["answer"] == "एक निगम।"
