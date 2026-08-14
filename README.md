# IndicVoiceRAG - Phase 1 Foundation

This branch implements the **Phase 1** backend foundation for HH Goa 2026 Task 2:

**Dataset -> Chunking -> Embedding -> Vector Index -> Retrieval**

## What is implemented

- Dataset access layer for `ai4bharat/MSMARCO-XI` with:
  - Hugging Face dataset config inspection
  - Hub parquet file structure inspection
  - split/language parquet selection (`train/<lang>train.parquet`, `validation/<lang>val.parquet`)
  - streaming/partial sample loading (no full dataset materialization)
  - retry-aware loading
- Normalized document schema preserving:
  - query/document identifiers
  - query text and type
  - language/source/target language
  - relevance labels/scores where available
  - additional source metadata
- Configurable chunking framework with 3 strategies:
  - fixed token chunking with overlap
  - sentence-aware chunking
  - semantic chunking using sentence similarity boundaries
- Provider-independent embedding layer:
  - default local `hash` provider (fast, no model download)
  - optional `sentence_transformers` provider
- Vector retrieval layer:
  - vector store interface
  - FAISS-backed store when available
  - automatic NumPy fallback when FAISS is unavailable
  - top-k similarity search with metadata-preserving hits
- Retrieval evaluation utilities:
  - Hit@K / Recall@K
  - MRR primitives
- CLI tools:
  - dataset inspection
  - sample normalization
  - index build
  - retrieval query
  - end-to-end smoke test
- Unit tests covering all required Phase-1 surfaces.

## What is NOT implemented yet

- Voice input integration
- Sarvam/ElevenLabs STT integration
- Answer generation / LLM orchestration
- Guardrails (off-topic / unsafe / hallucination checks)
- Frontend/live demo deployment
- End-to-end latency optimization for sub-200ms target

## Project structure

```text
src/indicvoicerag/
  cli.py            # CLI entrypoint
  config.py         # central TOML config schema + loader
  dataset.py        # dataset inspector/loader/sampler
  schemas.py        # normalized document + chunk schema
  chunking.py       # fixed/sentence/semantic chunkers
  embedding.py      # embedding interface + providers
  vector_store.py   # FAISS/NumPy vector store
  retrieval.py      # retrieval engine
  evaluation.py     # retrieval metrics foundation
tests/
  ...               # unit tests
config.example.toml # sample config
```

## Setup

> Python 3.11.9 expected.

```powershell
# from repository root
.venv\Scripts\python.exe -m pip install -e .[dev]
```

Optional FAISS:

```powershell
.venv\Scripts\python.exe -m pip install -e .[faiss]
```

## Usage

### 1) Inspect dataset structure and schema sample

```powershell
.venv\Scripts\python.exe -m indicvoicerag.cli inspect-dataset --rows 5
```

### 2) Build a small local sample index

```powershell
.venv\Scripts\python.exe -m indicvoicerag.cli build-index --size 64 --save
```

### 3) Run retrieval

```powershell
.venv\Scripts\python.exe -m indicvoicerag.cli retrieve --query "Goa tourism beaches" --size 64 --top-k 5
```

### 4) Run smoke test

```powershell
.venv\Scripts\python.exe -m indicvoicerag.cli smoke-test --size 32 --query "What is Goa known for?"
```

Offline smoke (no live HF access):

```powershell
.venv\Scripts\python.exe -m indicvoicerag.cli --config tests/fixtures/smoke_config.toml smoke-test --query "capital of goa"
```

## Notes on dataset handling

- The code does **not** assume a language-specific config like `"mar"`.
- It inspects available configs from Hugging Face dynamically.
- It inspects repository parquet files via Hub metadata before loading rows.
- It defaults to streaming and samples only small subsets for development.
- If your environment has TLS interception issues, set `dataset.allow_insecure_ssl = true`.
- For offline smoke/debug runs, you can set `dataset.local_sample_path` to a tiny JSONL sample file.