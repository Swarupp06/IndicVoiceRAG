from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
import tomllib
from typing import Any


@dataclass(slots=True)
class DatasetConfig:
    repo_id: str = "ai4bharat/MSMARCO-XI"
    split: str = "train"
    config_name: str | None = "default"
    streaming: bool = True
    sample_size: int = 128
    language: str | None = "hin"
    cache_dir: str | None = None
    max_retries: int = 3
    revision: str = "main"
    allow_insecure_ssl: bool = False
    local_sample_path: str | None = None


@dataclass(slots=True)
class ChunkingConfig:
    strategy: str = "fixed"
    chunk_size: int = 120
    overlap: int = 20
    semantic_threshold: float = 0.35
    sentence_separator_regex: str = r"(?<=[.!?])\s+"


@dataclass(slots=True)
class EmbeddingConfig:
    provider: str = "hash"
    model_name: str = "hash-256"
    dimension: int = 256
    normalize: bool = True


@dataclass(slots=True)
class VectorConfig:
    provider: str = "faiss"
    index_path: str = "indexes/dev.index"
    metadata_path: str = "indexes/dev.metadata.jsonl"


@dataclass(slots=True)
class RetrievalConfig:
    top_k: int = 5


@dataclass(slots=True)
class AppConfig:
    dataset: DatasetConfig = field(default_factory=DatasetConfig)
    chunking: ChunkingConfig = field(default_factory=ChunkingConfig)
    embedding: EmbeddingConfig = field(default_factory=EmbeddingConfig)
    vector: VectorConfig = field(default_factory=VectorConfig)
    retrieval: RetrievalConfig = field(default_factory=RetrievalConfig)


def _merge_into_dataclass(instance: Any, payload: dict[str, Any]) -> None:
    for key, value in payload.items():
        if not hasattr(instance, key):
            continue
        current_value = getattr(instance, key)
        if hasattr(current_value, "__dataclass_fields__") and isinstance(value, dict):
            _merge_into_dataclass(current_value, value)
        else:
            setattr(instance, key, value)


def load_config(config_path: str | Path | None = None) -> AppConfig:
    config = AppConfig()
    if config_path is None:
        return config

    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")

    with path.open("rb") as handle:
        payload = tomllib.load(handle)
    _merge_into_dataclass(config, payload)
    return config
