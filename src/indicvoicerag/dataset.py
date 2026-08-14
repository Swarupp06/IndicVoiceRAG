from __future__ import annotations

from dataclasses import asdict
import json
from pathlib import Path
import random
import re
import time
from typing import Any, Iterable

from datasets import IterableDataset, get_dataset_config_names, load_dataset
from huggingface_hub import dataset_info, set_client_factory
import httpx

from .config import DatasetConfig
from .schemas import NormalizedDocument


class DatasetAccessError(RuntimeError):
    """Raised when dataset access fails after retries."""


class DatasetInspector:
    def __init__(self, config: DatasetConfig):
        self.config = config
        if self.config.allow_insecure_ssl:
            set_client_factory(lambda: httpx.Client(verify=False, timeout=60.0))

    def list_configs(self) -> list[str]:
        return get_dataset_config_names(self.config.repo_id)

    def _list_repo_files(self) -> list[str]:
        info = dataset_info(self.config.repo_id, revision=self.config.revision)
        siblings = getattr(info, "siblings", None) or []
        return [s.rfilename for s in siblings if s.rfilename]

    def list_parquet_files(self) -> list[str]:
        return [path for path in self._list_repo_files() if path.endswith(".parquet")]

    def _extract_language_tag(self, parquet_path: str) -> str | None:
        # Examples: train/martrain.parquet, validation/hinval.parquet
        name = parquet_path.split("/")[-1]
        match = re.match(r"^([a-z]{3})", name)
        return match.group(1) if match else None

    def select_parquet_files(self) -> list[str]:
        candidates = [path for path in self.list_parquet_files() if path.startswith(f"{self.config.split}/")]
        if not candidates:
            return []
        if not self.config.language:
            return sorted(candidates)
        lang = self.config.language.lower()
        filtered = [path for path in candidates if self._extract_language_tag(path) == lang]
        return sorted(filtered or candidates)

    def inspect_hub_files(self) -> dict[str, Any]:
        files = self._list_repo_files()
        parquet_files = [path for path in files if path.endswith(".parquet")]
        selected = self.select_parquet_files()
        return {
            "repo_id": self.config.repo_id,
            "revision": self.config.revision,
            "configs": self.list_configs(),
            "split": self.config.split,
            "language_filter": self.config.language,
            "parquet_file_count": len(parquet_files),
            "selected_parquet_files": selected[:20],
            "parquet_examples": parquet_files[:20],
            "total_repo_file_count": len(files),
        }

    def inspect_sample_schema(self, rows: int = 5) -> dict[str, Any]:
        if rows <= 0:
            return {"rows": 0, "columns": []}
        sample = self.load_sample(rows)
        if not sample:
            return {"rows": 0, "columns": []}
        columns = sorted({key for row in sample for key in row.keys()})
        return {"rows": len(sample), "columns": columns}

    def _build_data_files(self) -> dict[str, list[str]]:
        parquet_paths = self.select_parquet_files()
        if not parquet_paths:
            raise DatasetAccessError(
                f"No parquet files found for split={self.config.split} language={self.config.language} in {self.config.repo_id}."
            )
        urls = [f"hf://datasets/{self.config.repo_id}@{self.config.revision}/{path}" for path in parquet_paths]
        return {"train": urls}

    def _load_streaming_parquet(self) -> IterableDataset:
        data_files = self._build_data_files()
        return load_dataset(
            "parquet",
            data_files=data_files,
            split="train",
            streaming=True,
            cache_dir=self.config.cache_dir,
        )

    def _load_local_sample(self) -> list[dict[str, Any]]:
        if not self.config.local_sample_path:
            raise DatasetAccessError("local_sample_path is not configured.")
        path = Path(self.config.local_sample_path)
        if not path.exists():
            raise DatasetAccessError(f"Local sample file not found: {path}")
        rows: list[dict[str, Any]] = []
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                text = line.strip()
                if not text:
                    continue
                rows.append(json.loads(text))
        return rows

    def load_sample(self, size: int | None = None) -> list[dict[str, Any]]:
        target_size = size or self.config.sample_size
        if target_size <= 0:
            raise ValueError("sample size must be positive")

        last_error: Exception | None = None
        if self.config.local_sample_path:
            rows = self._load_local_sample()
            return rows[:target_size]

        for attempt in range(1, self.config.max_retries + 1):
            try:
                dataset = self._load_streaming_parquet()
                sample: list[dict[str, Any]] = []
                for index, record in enumerate(dataset):
                    sample.append(dict(record))
                    if index + 1 >= target_size:
                        break
                if not sample:
                    raise DatasetAccessError("Dataset loaded but returned no sample rows.")
                return sample
            except Exception as exc:  # noqa: BLE001
                last_error = exc
                if attempt >= self.config.max_retries:
                    break
                time.sleep(random.uniform(0.1, 0.4))

        raise DatasetAccessError(
            "Failed to stream MSMARCO-XI sample. "
            f"repo={self.config.repo_id}, split={self.config.split}, language={self.config.language}, retries={self.config.max_retries}. "
            "If this environment has TLS interception, set dataset.allow_insecure_ssl=true or use dataset.local_sample_path for offline smoke tests."
        ) from last_error

    def normalized_sample(self, size: int | None = None) -> list[NormalizedDocument]:
        rows = self.load_sample(size)
        docs: list[NormalizedDocument] = []
        for row in rows:
            try:
                docs.append(NormalizedDocument.from_record(row, fallback_language=self.config.language))
            except ValueError:
                continue
        return docs


def serialize_documents_jsonl(documents: Iterable[NormalizedDocument], path: str | Path) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", encoding="utf-8") as handle:
        for doc in documents:
            handle.write(json.dumps(asdict(doc), ensure_ascii=False) + "\n")
