from unittest.mock import Mock, patch
from pathlib import Path

from indicvoicerag.config import DatasetConfig
from indicvoicerag.dataset import DatasetInspector


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


def test_dataset_inspector_sample_loading() -> None:
    config = DatasetConfig(sample_size=2, streaming=True)
    inspector = DatasetInspector(config)
    sample_rows = [{"id": "1", "text": "hello"}, {"id": "2", "text": "world"}, {"id": "3", "text": "extra"}]
    with patch.object(inspector, "_load_streaming_parquet", return_value=sample_rows):
        rows = inspector.load_sample()
    assert len(rows) == 2
    assert rows[0]["id"] == "1"


def test_dataset_inspector_local_sample_path_loading(tmp_path: Path) -> None:
    sample_file = tmp_path / "sample.jsonl"
    sample_file.write_text('{"id":"1","text":"one"}\n{"id":"2","text":"two"}\n', encoding="utf-8")
    inspector = DatasetInspector(DatasetConfig(local_sample_path=str(sample_file), sample_size=1))
    rows = inspector.load_sample()
    assert len(rows) == 1
    assert rows[0]["id"] == "1"
