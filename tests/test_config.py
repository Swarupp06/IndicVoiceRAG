from pathlib import Path

from indicvoicerag.config import load_config


def test_load_config_from_toml(tmp_path: Path) -> None:
    config_path = tmp_path / "config.toml"
    config_path.write_text(
        """
[dataset]
sample_size = 10
language = "hi"

[chunking]
strategy = "semantic"
chunk_size = 42
overlap = 7

[retrieval]
top_k = 3
""".strip(),
        encoding="utf-8",
    )
    cfg = load_config(config_path)
    assert cfg.dataset.sample_size == 10
    assert cfg.dataset.language == "hi"
    assert cfg.chunking.strategy == "semantic"
    assert cfg.chunking.chunk_size == 42
    assert cfg.retrieval.top_k == 3
