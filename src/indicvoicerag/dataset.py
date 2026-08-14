from __future__ import annotations

from dataclasses import asdict
import json
from pathlib import Path
import random
import re
import shutil
import time
from typing import Any, Iterable

from datasets import get_dataset_config_names
from huggingface_hub import dataset_info, hf_hub_download, set_client_factory
import httpx
import pyarrow.parquet as pq

from .config import DatasetConfig
from .schemas import NormalizedDocument


class DatasetAccessError(RuntimeError):
    """Raised when dataset access fails after retries."""


def flatten_msmarco_row(row: dict[str, Any]) -> list[dict[str, Any]]:
    """Expand one real MSMARCO-XI record into per-passage flat records.

    Real schema (ai4bharat/MSMARCO-XI, e.g. validation/hinval.parquet):

        source_lang, target_lang, meta (struct), Answer, query_id (int),
        query_type, passages{English_passages[], Translated_passages[],
        is_selected[]}, Eng_Query, Eng_Answer, query

    Each output record carries query id/text, one passage and its relevance
    label so the generic normalizer can turn it into a NormalizedDocument.
    """
    query_id = row.get("query_id")
    query_text = row.get("query")
    eng_query = row.get("Eng_Query")
    query_type = row.get("query_type")
    source_lang = row.get("source_lang")
    target_lang = row.get("target_lang")
    answer = row.get("Answer")
    eng_answer = row.get("Eng_Answer")
    meta = row.get("meta")

    passages = row.get("passages") or {}
    english_passages = passages.get("English_passages") or []
    translated_passages = passages.get("Translated_passages") or []
    is_selected = passages.get("is_selected") or []

    count = max(len(english_passages), len(translated_passages))
    if count == 0:
        return []

    records: list[dict[str, Any]] = []
    for index in range(count):
        translated = translated_passages[index] if index < len(translated_passages) else None
        english = english_passages[index] if index < len(english_passages) else None
        passage_text = translated or english
        if not passage_text:
            continue
        selected = is_selected[index] if index < len(is_selected) else 0
        try:
            selected_int = int(selected)
        except (TypeError, ValueError):
            selected_int = 0
        records.append(
            {
                "query_id": None if query_id is None else str(query_id),
                "query": None if query_text is None else str(query_text),
                "Eng_Query": None if eng_query is None else str(eng_query),
                "query_type": None if query_type is None else str(query_type),
                "source_lang": None if source_lang is None else str(source_lang),
                "target_lang": None if target_lang is None else str(target_lang),
                "doc_id": f"{query_id}:{index}" if query_id is not None else f"passage:{index}",
                "passage_text": str(passage_text).strip(),
                "Eng_passage_text": None if english is None else str(english),
                "relevance": 1 if selected_int > 0 else 0,
                "is_selected": selected_int,
                "answer": None if answer is None else str(answer),
                "Eng_answer": None if eng_answer is None else str(eng_answer),
                "meta": meta,
            }
        )
    return records


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

    # -- remote parquet access (verified: download-first + local pyarrow read) --

    def _resolve_url(self, parquet_path: str) -> str:
        return f"https://huggingface.co/datasets/{self.config.repo_id}/resolve/{self.config.revision}/{parquet_path}"

    def _parquet_cache_dir(self) -> Path:
        return Path(self.config.parquet_cache_dir or "data")

    def _local_parquet_path(self, parquet_path: str) -> Path | None:
        candidate = self._parquet_cache_dir() / Path(parquet_path).name
        return candidate if candidate.exists() else None

    def _ensure_local_parquet(self, parquet_path: str) -> Path:
        """Return a local copy of the real parquet file (download once, then reuse)."""
        existing = self._local_parquet_path(parquet_path)
        if existing is not None:
            return existing

        cache_dir = self._parquet_cache_dir()
        cache_dir.mkdir(parents=True, exist_ok=True)
        target = cache_dir / Path(parquet_path).name
        part = target.with_suffix(target.suffix + ".part")

        downloaded = hf_hub_download(
            repo_id=self.config.repo_id,
            filename=parquet_path,
            repo_type="dataset",
            revision=self.config.revision,
            cache_dir=self.config.cache_dir,
        )
        shutil.copyfile(downloaded, part)
        part.replace(target)
        return target

    def _open_parquet(self, parquet_path: str) -> pq.ParquetFile:
        local_path = self._ensure_local_parquet(parquet_path)
        try:
            return pq.ParquetFile(str(local_path))
        except Exception as exc:
            raise DatasetAccessError(
                f"Failed to open parquet file {local_path}: {type(exc).__name__}: {exc}"
            ) from exc

    def inspect_remote_schema(self, parquet_path: str | None = None) -> dict[str, Any]:
        selected = self.select_parquet_files()
        if not selected:
            raise DatasetAccessError("No parquet files selected.")
        path = parquet_path or selected[0]
        pf = self._open_parquet(path)
        metadata = pf.metadata
        schema = pf.schema_arrow
        local_path = self._local_parquet_path(path)
        return {
            "parquet_file": path,
            "url": self._resolve_url(path),
            "local_path": None if local_path is None else str(local_path),
            "num_row_groups": metadata.num_row_groups,
            "num_rows": metadata.num_rows,
            "columns": [
                {
                    "name": schema.field(i).name,
                    "type": str(schema.field(i).type),
                }
                for i in range(len(schema))
            ],
        }

    def _read_remote_rows(self, size: int) -> list[dict[str, Any]]:
        """Read rows from the selected real parquet file.

        MSMARCO-XI files are single row groups; reading a sample therefore
        requires reading that row group. Row groups are read in order and
        truncated to `size` rows.
        """
        selected = self.select_parquet_files()
        if not selected:
            raise DatasetAccessError(
                f"No parquet files found for split={self.config.split} language={self.config.language} in {self.config.repo_id}."
            )
        pf = self._open_parquet(selected[0])
        rows: list[dict[str, Any]] = []
        num_row_groups = pf.metadata.num_row_groups
        for group_index in range(num_row_groups):
            table = pf.read_row_group(group_index)
            for record in table.to_pylist():
                rows.append(record)
                if len(rows) >= size:
                    return rows
        return rows

    # -- local sample / cache paths --

    def default_sample_cache_path(self) -> Path:
        language = self.config.language or "all"
        return Path("data") / f"msmarco_xi_{language}_{self.config.split}_sample.jsonl"

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

    def _load_cached_sample(self) -> list[dict[str, Any]]:
        cache_path = self.config.sample_cache_path or self.default_sample_cache_path()
        path = Path(cache_path)
        if not path.exists():
            raise DatasetAccessError(f"Real sample cache not found: {path}")
        rows: list[dict[str, Any]] = []
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                text = line.strip()
                if not text:
                    continue
                rows.append(json.loads(text))
        return rows

    def _write_cached_sample(self, rows: list[dict[str, Any]], meta: dict[str, Any]) -> Path:
        cache_path = self.config.sample_cache_path or self.default_sample_cache_path()
        path = Path(cache_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as handle:
            for row in rows:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        meta_path = path.with_suffix(path.suffix + ".meta.json")
        meta["cache_path"] = str(path)
        meta["cached_at"] = time.strftime("%Y-%m-%dT%H:%M:%S%z")
        meta["record_count"] = len(rows)
        with meta_path.open("w", encoding="utf-8") as handle:
            json.dump(meta, handle, ensure_ascii=False, indent=2)
        return path

    def load_sample(self, size: int | None = None) -> list[dict[str, Any]]:
        target_size = size or self.config.sample_size
        if target_size <= 0:
            raise ValueError("sample size must be positive")

        if self.config.local_sample_path:
            return self._load_local_sample()[:target_size]

        # Real-data fast path: reuse a previously fetched real sample.
        cache_path = self.config.sample_cache_path or self.default_sample_cache_path()
        if Path(cache_path).exists():
            return self._load_cached_sample()[:target_size]

        # Live fetch from Hugging Face, then expand + cache the real rows.
        # Each MSMARCO-XI row carries ~10 passages, so read a few raw rows to
        # collect at least `target_size` documents.
        raw_target = max(target_size, min(target_size * 2, 500))
        last_error: Exception | None = None
        for attempt in range(1, self.config.max_retries + 1):
            try:
                raw_rows = self._read_remote_rows(raw_target)
                if not raw_rows:
                    raise DatasetAccessError("Dataset loaded but returned no sample rows.")
                expanded: list[dict[str, Any]] = []
                for row in raw_rows:
                    expanded.extend(flatten_msmarco_row(row))
                if not expanded:
                    raise DatasetAccessError("No passages could be flattened from the sample.")
                selected = self.select_parquet_files()
                meta = {
                    "repo_id": self.config.repo_id,
                    "revision": self.config.revision,
                    "split": self.config.split,
                    "language": self.config.language,
                    "parquet_file": selected[0],
                    "url": self._resolve_url(selected[0]),
                    "source": "huggingface_hub_live_fetch",
                    "raw_rows": len(raw_rows),
                }
                self._write_cached_sample(expanded, meta)
                return expanded[:target_size]
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

    def sample_cache_meta(self) -> dict[str, Any] | None:
        cache_path = self.config.sample_cache_path or self.default_sample_cache_path()
        meta_path = Path(cache_path).with_suffix(Path(cache_path).suffix + ".meta.json")
        if not meta_path.exists():
            return None
        with meta_path.open("r", encoding="utf-8") as handle:
            return json.load(handle)

    def normalized_sample(self, size: int | None = None) -> list[NormalizedDocument]:
        rows = self.load_sample(size)
        docs: list[NormalizedDocument] = []
        for row in rows:
            try:
                docs.append(NormalizedDocument.from_record(row, fallback_language=self.config.language))
            except ValueError:
                continue
        return docs

    def inspect_sample_schema(self, rows: int = 5) -> dict[str, Any]:
        if rows <= 0:
            return {"rows": 0, "columns": []}
        sample = self.load_sample(rows)
        if not sample:
            return {"rows": 0, "columns": []}
        columns = sorted({key for row in sample for key in row.keys()})
        return {"rows": len(sample), "columns": columns}


def serialize_documents_jsonl(documents: Iterable[NormalizedDocument], path: str | Path) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", encoding="utf-8") as handle:
        for doc in documents:
            handle.write(json.dumps(asdict(doc), ensure_ascii=False) + "\n")
