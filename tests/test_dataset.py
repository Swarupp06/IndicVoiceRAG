from unittest.mock import Mock, patch
from pathlib import Path

from indicvoicerag.config import DatasetConfig
from indicvoicerag.dataset import DatasetInspector, flatten_msmarco_row


def test_dataset_inspector_reads_hub_file_listing() -> None:
    config = DatasetConfig(repo_id="ai4bharat/MSMARCO-XI", split="train", streaming=True, language="mar")
    inspector = DatasetInspector(config)
    fake_info = Mock()
    fake_info.siblings = [
        Mock(rfilename="train/martrain.parquet"),
        Mock(rfilename="train/hintrain.parquet"),
        Mock(rfilename="README.md"),
    ]
    with patch("indicvoicerag.dataset.get_dataset_config_names", return_value=["default"]), patch(
        "indicvoicerag.dataset.dataset_info",
        return_value=fake_info,
    ):
        info = inspector.inspect_hub_files()
    assert info["repo_id"] == "ai4bharat/MSMARCO-XI"
    assert info["configs"] == ["default"]
    assert info["parquet_file_count"] == 2
    assert info["selected_parquet_files"] == ["train/martrain.parquet"]


def test_dataset_inspector_sample_loading(tmp_path: Path) -> None:
    config = DatasetConfig(
        sample_size=2,
        streaming=True,
        sample_cache_path=str(tmp_path / "none.jsonl"),
    )
    inspector = DatasetInspector(config)
    sample_rows = [
        {"query_id": 1, "passages": {"Translated_passages": ["p1"], "is_selected": [1]}},
        {"query_id": 2, "passages": {"Translated_passages": ["p2"], "is_selected": [1]}},
        {"query_id": 3, "passages": {"Translated_passages": ["p3"], "is_selected": [1]}},
    ]
    with patch.object(inspector, "_read_remote_rows", return_value=sample_rows), patch(
        "indicvoicerag.dataset.DatasetInspector._write_cached_sample"
    ):
        rows = inspector.load_sample()
    assert len(rows) == 2
    assert rows[0]["query_id"] == "1"


def test_dataset_inspector_local_sample_path_loading(tmp_path: Path) -> None:
    sample_file = tmp_path / "sample.jsonl"
    sample_file.write_text('{"id":"1","text":"one"}\n{"id":"2","text":"two"}\n', encoding="utf-8")
    inspector = DatasetInspector(DatasetConfig(local_sample_path=str(sample_file), sample_size=1))
    rows = inspector.load_sample()
    assert len(rows) == 1
    assert rows[0]["id"] == "1"


def test_flatten_msmarco_row_expands_passages_with_relevance() -> None:
    row = {
        "query_id": 1102432,
        "query": "कॉर्पोरेशन क्या है?",
        "Eng_Query": "what is a corporation?",
        "query_type": "DESCRIPTION",
        "source_lang": "eng_Latn",
        "target_lang": "hin_Deva",
        "Answer": "एक निगम।",
        "Eng_Answer": "A corporation.",
        "meta": {"model_name": "ckpt"},
        "passages": {
            "English_passages": ["passage A", "passage B"],
            "Translated_passages": ["अनुच्छेद ए", "अनुच्छेद बी"],
            "is_selected": [0, 1],
        },
    }
    records = flatten_msmarco_row(row)
    assert len(records) == 2
    assert records[0]["relevance"] == 0
    assert records[1]["relevance"] == 1
    assert records[1]["doc_id"] == "1102432:1"
    assert records[1]["query"] == "कॉर्पोरेशन क्या है?"
    assert records[1]["passage_text"] == "अनुच्छेद बी"


def test_flatten_msmarco_row_handles_missing_selected() -> None:
    row = {
        "query_id": 5,
        "query": "q",
        "passages": {"English_passages": ["only english"], "Translated_passages": [], "is_selected": []},
    }
    records = flatten_msmarco_row(row)
    assert len(records) == 1
    assert records[0]["relevance"] == 0
    assert records[0]["passage_text"] == "only english"
