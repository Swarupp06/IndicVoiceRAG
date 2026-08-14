# IndicVoiceRAG - Phase 1 Foundation

Phase 1 backend foundation for HH Goa 2026 Task 2:

**Real MSMARCO-XI -> Normalization -> Chunking -> Real Multilingual Embeddings -> FAISS Vector Index -> Retrieval -> Retrieval Evaluation**

## What is implemented

### Real dataset access (`ai4bharat/MSMARCO-XI`)
- Inspects the hub repository: available configs, parquet file layout, remote schema.
- The dataset exposes only the `default` config; per-language data lives in
  `train/<lang>train.parquet` / `validation/<lang>val.parquet` (14 Indic languages).
- Loads a **small real subset** via download-first access
  (`huggingface_hub.hf_hub_download` + local pyarrow read) - the whole dataset
  (~55 GB) is never materialized.
- Real sample is flattened, normalized and cached to
  `data/msmarco_xi_<lang>_<split>_sample.jsonl` (gitignored) with a provenance
  sidecar recording the source URL, split, language and fetch time.
- Real record schema observed (e.g. `validation/hinval.parquet`, 97,941 rows):
  - `query_id` (int), `query` (translated), `Eng_Query`, `query_type`
  - `passages{English_passages[], Translated_passages[], is_selected[]}` - real relevance labels
  - `Answer` / `Eng_Answer`, `source_lang` / `target_lang`, `meta` (generation params)

### Normalization
- `NormalizedDocument` preserves identifiers, query text/type, source/target
  language, relevance labels and all extra metadata.
- `flatten_msmarco_row()` expands each MSMARCO-XI row into per-passage documents
  carrying the real `is_selected` relevance label.

### Chunking (3 strategies, all tested on real data)
1. Fixed-size token chunks with overlap
2. Sentence-aware chunking
3. Semantic chunking (cosine-similarity sentence boundaries)

### Real multilingual embeddings
- `sentence_transformers` provider (PyTorch) - default
  `intfloat/multilingual-e5-small` (384-dim, e5 `query:`/`passage:` prefixes).
- `fastembed` provider (ONNX runtime) - lightweight CPU alternative.
- `hash` provider retained for offline unit tests only (NOT a semantic embedding).
- Model comparison on the identical real sample (2000 docs / 105 queries,
  Hindi, validation split):

  | Model | Runtime | Dim | Hit@1 | Hit@5 | Hit@10 | MRR | index 2000 docs |
  |---|---|---|---|---|---|---|---|
  | `intfloat/multilingual-e5-small` | PyTorch-CPU | 384 | 0.40 | 0.75 | 0.90 | **0.553** | ~84 s |
  | `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2` | ONNX-CPU | 384 | 0.19 | 0.58 | 0.72 | 0.357 | ~125 s |

  **Selected default: `multilingual-e5-small`** - best retrieval quality on
  Indic data, small (118M params), fast on CPU, 384-dim.

### Vector store
- FAISS (`IndexFlatIP`) with metadata-preserving hits; NumPy fallback.
- Persistence: `save()` writes the FAISS index + metadata JSONL; `load()`
  restores both for retrieval without re-indexing.

### Retrieval & evaluation
- Top-k similarity retrieval with chunk-level hits aggregated to documents.
- Hit@K / Recall@K / MRR evaluated against the dataset's real `is_selected` labels.

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
  cli.py            # CLI entrypoint (inspect/sample/build/retrieve/evaluate/smoke)
  config.py         # central TOML config schema + loader
  dataset.py        # hub inspector + download-first real parquet access + cache
  schemas.py        # normalized document schema + real-record flattening
  chunking.py       # fixed/sentence/semantic chunkers
  embedding.py      # real embedding providers (e5 / fastembed / hash)
  vector_store.py   # FAISS/NumPy store with persistence
  retrieval.py      # retrieval engine
  evaluation.py     # Hit@K / Recall@K / MRR on real relevance labels
tests/
  ...               # offline unit tests
config.example.toml # sample config
```

## Setup

> Python 3.11.9 expected. Real embeddings need torch (CPU) + sentence-transformers;
> FAISS is optional (NumPy fallback).

```powershell
.venv\Scripts\python.exe -m pip install -e .[dev]
.venv\Scripts\python.exe -m pip install torch --index-url https://download.pytorch.org/whl/cpu
.venv\Scripts\python.exe -m pip install sentence-transformers faiss-cpu
# optional lightweight ONNX backend:
.venv\Scripts\python.exe -m pip install fastembed
```

## Usage

### 1) Inspect the real dataset structure and schema

```powershell
.venv\Scripts\python.exe -m indicvoicerag.cli inspect-dataset --rows 5
```

### 2) Print normalized real documents (first run downloads the real sample once)

```powershell
.venv\Scripts\python.exe -m indicvoicerag.cli sample-docs --size 100
```

### 3) Chunk statistics on a real sample (all 3 strategies)

```powershell
.venv\Scripts\python.exe -m indicvoicerag.cli chunk-stats --size 500
```

### 4) Build and persist a real index

```powershell
.venv\Scripts\python.exe -m indicvoicerag.cli build-index --size 1000 --save
```

### 5) Retrieve with a Hindi query

```powershell
.venv\Scripts\python.exe -m indicvoicerag.cli retrieve --query "कॉर्पोरेशन क्या है?" --size 1000 --top-k 5
```

### 6) Real-data retrieval evaluation (Hit@K / Recall@K / MRR)

```powershell
.venv\Scripts\python.exe -m indicvoicerag.cli evaluate-retrieval --size 2000 --top-k 10
```

### 7) Real-data + real-embedding smoke test

```powershell
.venv\Scripts\python.exe -m indicvoicerag.cli smoke-test --query "कॉर्पोरेशन क्या है?" --size 200
```

The smoke-test output flags `real_msmarco_xi_data_used` and
`real_embedding_model_used` along with dataset provenance (source URL, split,
language) and the embedding model description.

Offline smoke (no live HF access, hash embeddings for tests only):

```powershell
.venv\Scripts\python.exe -m indicvoicerag.cli --config tests/fixtures/smoke_config.toml smoke-test --query "capital of goa"
```

## Notes on dataset handling

- The code does **not** assume a language-specific config like `"mar"` - only
  `default` is exposed, so per-language parquet files are selected directly.
- Download-first access downloads only the selected language/split parquet file
  (one-time; `validation/*.parquet` is ~440 MB) into `data/` (gitignored).
  Subsequent runs reuse the flattened sample cache.
- If your environment has TLS interception issues, set
  `dataset.allow_insecure_ssl = true`.
- For fully offline smoke/debug runs, set `dataset.local_sample_path` to a tiny
  JSONL sample file.
