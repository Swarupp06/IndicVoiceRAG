# IndicVoiceRAG - Phase 1, 2 & 3A Foundation

Backend foundation for HH Goa 2026 Task 2:

**Phase 1 (retrieval):** Real MSMARCO-XI -> Normalization -> Chunking -> Real Multilingual Embeddings -> FAISS Vector Index -> Retrieval -> Retrieval Evaluation

**Phase 2 (RAG):** User Text -> Query Processing -> Retrieval -> Context Building -> LLM Answer Generation -> Grounding Validation -> Guardrails -> Structured Response

**Phase 3A (local speech-to-text):** Audio File -> Local STT (faster-whisper, CPU, ₹0) -> Text -> existing RAG pipeline

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

### Free LLM providers (Phase 2.5)
- Replaceable providers behind one interface: `mock`, `ollama`, `gemini`, `groq`, `openrouter`, `openai_compatible`.
- Verified ₹0 cost table, fixed 20-query benchmark with P50/P70/P100, and an
  optional fallback chain that never silently falls back to mock.
- See [Phase 2.5](#phase-25---free-llm-providers--benchmark).

### RAG generation (Phase 2)
- Deterministic orchestration harness (no agent): validate -> safety -> retrieve -> quality check -> context -> generate -> validate output -> grounding -> structured response.
- Provider-agnostic LLM abstraction (see Phase 2.5 for the full provider set).
- Context engineering (dedup, source boundaries, token budget, metadata preservation).
- Guardrails: invalid/empty input, unsafe input, low retrieval relevance, off-topic/no-evidence, grounding rejection, generation failure fallback.
- Lexical grounding validator (abstraction ready for embedding/NLI upgrade).
- Per-stage latency instrumentation (debug mode).
- RAG evaluation harness (behavior metrics; no answer-quality label set yet).

### Local speech-to-text (Phase 3A)
- `STTProvider` abstraction: `audio file -> TranscriptionResult{text, language, language_probability, duration, processing_time, RTF, model}`.
- `faster_whisper` provider (CTranslate2, local CPU, **₹0 - no cloud API**), lazy model load (constructing the provider never downloads), `tiny` int8 default, `language`/`beam_size`/`vad_filter` options.
- `MockSTT` deterministic offline stub; every STT test runs offline (model loading is faked).
- PyAV-based decoding (`audio.py`) - WAV/Ogg/FLAC/MP3 in, mono 16 kHz float32 out; no system FFmpeg needed.
- Dependency-free WER (`wer.py`) that tokenizes Devanagari text correctly.
- Reproducible benchmark (`stt_benchmark.py`): per-sample audio duration, model load, transcription time, RTF, detected language, WER (from `<name>.txt` ground-truth sidecars). Missing ground truth is reported as `accuracy NOT MEASURED` - never invented.
- Speech -> RAG glue (`speech.py`): `answer_from_audio(stt, harness, audio)` keeps the existing text RAG path untouched.
- See [Phase 3A](#phase-3a---local-speech-to-text).

## What is NOT implemented yet

- Microphone input / live streaming (audio is provided as a file)
- Sarvam/ElevenLabs STT integration
- Frontend/live demo deployment
- Authentication
- Final latency optimization for the sub-200ms target (measured P50 is ~19 s on local CPU generation; see Phase 2.5)
- Production-grade safety classifier (current safety gate is a small regex list)
- A human/LLM answer-quality evaluation set

## Project structure

```text
src/indicvoicerag/
  cli.py            # CLI entrypoint (inspect/sample/build/retrieve/evaluate/smoke/rag/rag-evaluate/transcribe/transcribe-rag/benchmark-stt/listen)
  config.py         # central TOML config schema + loader
  dataset.py        # hub inspector + download-first real parquet access + cache
  schemas.py        # normalized document schema + real-record flattening
  chunking.py       # fixed/sentence/semantic chunkers
  embedding.py      # real embedding providers (e5 / fastembed / hash)
  vector_store.py   # FAISS/NumPy store with persistence
  retrieval.py      # retrieval engine
  evaluation.py     # Hit@K / Recall@K / MRR on real relevance labels
  llm.py            # LLM provider abstraction (mock/ollama/gemini/groq/openrouter/openai_compatible + fallback chain)
  cost.py           # verified ₹0 cost status per provider
  benchmark.py      # fixed-query provider benchmark + P50/P70/P100 reporting
  context.py        # context engineering (dedup, boundaries, token budget)
  prompts.py        # system/user prompt builder with refusal instruction
  grounding.py      # grounding validator abstraction + lexical implementation
  guardrails.py     # input validation, safety check, retrieval quality check
  harness.py        # RAG orchestration pipeline + latency instrumentation
  rag_types.py      # structured SourceInfo / RAGResponse schema
  rag_evaluate.py   # RAG evaluation harness
  pipeline.py       # wiring: config -> real retrieval engine -> harness (+ STT provider)
  audio.py          # PyAV decode -> mono 16 kHz float32 + probe (Phase 3A)
  stt.py            # STTProvider abstraction + faster-whisper/mock providers (Phase 3A)
  wer.py            # dependency-free WER (Devanagari-aware) (Phase 3A)
  stt_benchmark.py  # reproducible RTF/WER benchmark (Phase 3A)
  speech.py         # audio -> STT -> text -> RAGHarness glue (Phase 3A)
  audio_capture.py  # microphone recording -> mono 16 kHz PCM (Phase 3C)
benchmarks/
  queries.json      # fixed 20-query provider benchmark set
  results/          # benchmark JSON + CSV reports
tests/
  ...               # offline unit tests (Phase 1 + Phase 2 + providers)
config.example.toml # sample config
config.ollama.toml  # local zero-cost real-LLM config
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
# Phase 3A: local STT (faster-whisper + PyAV for audio decode)
.venv\Scripts\python.exe -m pip install -e .[stt]
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

Offline smoke (no live HF access, hash embeddings for tests only) - this
variant reports `passed: false` by design because it uses neither real data nor
a real embedding model; it only exercises the pipeline wiring offline:

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

---

# Phase 2 - RAG generation

## Architecture

```text
USER TEXT
   |
   v
[Harness: RAGHarness.answer()]
   |
   | 1. validate input        (empty / over-long / malformed)
   | 2. safety check          (lightweight regex gate)
   | 3. retrieve              (Phase 1B stack: real MSMARCO-XI -> e5 -> FAISS)
   | 4. retrieval quality     (no evidence / low relevance / off-topic)
   | 5. build context         (dedup, order by score, token budget, [Source n] delimiters)
   | 6. generate              (LLM provider, retries + timeout)
   | 7. validate output       (non-empty)
   | 8. grounding check       (lexical containment; strict regenerate retry)
   | 9. structured response   (RAGResponse)
   v
{ answer, grounded, confidence, sources[], reason, guardrail, metrics(debug) }
```

This is a deterministic orchestration pipeline, **not** an agent. There are no
tool calls, loops or autonomy.

## LLM provider abstraction

See [Phase 2.5](#phase-25---free-llm-providers--benchmark) for the full provider
matrix, verified cost table and measured benchmark. In short, `llm.py` exposes

```python
LLMProvider.generate(messages, *, max_tokens, temperature, timeout) -> LLMResponse
# LLMResponse: { text, provider, model, latency_ms, usage }
```

with `mock`, `ollama`, `gemini`, `groq`, `openrouter` and a generic
`openai_compatible` escape hatch behind that one interface. No provider-specific
code exists in `RAGHarness`; the provider is chosen entirely from `[llm]` config.

## Context engineering (`context.py`)

- Passages are already rank-ordered by the store (highest score first).
- Duplicate passages (identical normalized text) are removed.
- The number of sources is capped by `guardrails.max_context_docs`.
- Total context tokens are capped by `guardrails.context_max_tokens`.
- The rendered context is explicitly delimited:

```text
BEGIN RETRIEVED CONTEXT

[Source 1] (document_id=..., score=0.8514)
<passage text>

[Source 2] (document_id=..., score=0.8463)
<passage text>

END RETRIEVED CONTEXT
```

## Guardrails (`guardrails.py`)

| Guardrail | Trigger | Behavior |
|---|---|---|
| `invalid_input` | empty / whitespace / over-long query | no retrieval, no generation |
| `unsafe_input` | query matches unsafe regex patterns (hate/violence/self-harm/explicit) | no retrieval, no generation |
| `no_evidence` | zero retrieved hits | controlled response, no generation |
| `low_relevance` | best hit score below `min_retrieval_score` | controlled response, no generation |
| `off_topic` | same evidence-threshold mechanism (query has no meaningful support) | same as above; message explains the query may be out of domain |
| `generation_failed` | provider exception / timeout / empty output after retries | fallback message |
| `ungrounded` | answer fails the grounding check even after a strict regenerate | answer returned with `grounded: false` and reason |

The safety gate is intentionally lightweight (a small regex list) and must be
replaced by a real classifier before production.

## Grounding strategy (`grounding.py`)

`GroundingValidator` is an abstraction. The current implementation,
`LexicalGroundingValidator`, checks each answer sentence for content-token
containment in the retrieved context:

```json
{ "grounded": true, "score": 0.98, "unsupported_claims": [], "reason": null }
```

**Documented limitations (do not treat as a perfect hallucination detector):**
- It is lexical, not semantic. A correct paraphrase with low token overlap is
  flagged as unsupported even though it is faithful.
- Numbers, dates and named entities are not verified individually.
- Indic morphology (e.g. Hindi inflections) reduces token overlap for valid
  answers.
- The planned upgrade is an embedding-similarity or NLI-based validator via the
  same `GroundingValidator` interface.

## Structured response

```json
{
  "query": "कॉर्पोरेशन क्या है?",
  "answer": "...",
  "grounded": true,
  "confidence": 0.9108,
  "sources": [
    {
      "document_id": "1102432:2",
      "chunk_id": "1102432:2:0",
      "score": 0.8514,
      "query_id": "1102432",
      "language": "hin",
      "relevance": 0.0,
      "excerpt": "..."
    }
  ],
  "reason": null,
  "guardrail": null,
  "metrics_ms": { "retrieval": 33.6, "context": 0.2, "generation": 0.0, "grounding": 0.8, "total": 34.6 }
}
```

## Configuration (Phase 2)

```toml
[llm]
provider = "mock"            # mock | openai_compatible | gemini
model_name = "mock-rag-generator"
base_url = ""                # openai_compatible endpoint
api_key_env = ""             # env var with the API key
temperature = 0.0
max_tokens = 512
timeout_seconds = 30.0
max_retries = 1

[guardrails]
min_retrieval_score = 0.35
min_hits = 1
max_query_chars = 1000
grounding_threshold = 0.35
context_max_tokens = 1500
max_context_docs = 8
strict_retry_on_ungrounded = true
max_ungrounded_retries = 1
```

## How to run text RAG

```powershell
# Full RAG on real data with the credential-free mock generator
.venv\Scripts\python.exe -m indicvoicerag.cli rag --query "कॉर्पोरेशन क्या है?" --size 150 --debug

# With a real LLM (set OPENAI_API_KEY or GEMINI_API_KEY first)
.venv\Scripts\python.exe -m indicvoicerag.cli rag --query "कॉर्पोरेशन क्या है?" --size 150 --provider openai_compatible --model gpt-4o-mini
.venv\Scripts\python.exe -m indicvoicerag.cli rag --query "कॉर्पोरेशन क्या है?" --size 150 --provider gemini --model gemini-2.0-flash
```

`--debug` adds per-stage `metrics_ms`. The response always identifies the LLM
provider, dataset provenance and embedding model used.

## RAG evaluation

```powershell
.venv\Scripts\python.exe -m indicvoicerag.cli rag-evaluate --size 400 --queries 15
```

Reports: total / answered / refused, grounded vs ungrounded answers, refusal
rate, per-outcome breakdown, and average generation/total latency.

**Known limitation:** there is no reliable human- or LLM-labeled answer-quality
set yet, so `rag-evaluate` measures pipeline *behavior* (grounded/refused/
latency), not answer *quality*.

## Example output (real data + mock generator)

`rag --query "कॉर्पोरेशन क्या है?" --size 150` produced:

```json
{
  "answer": "निगम की परिभाषा, व्यक्तियों का एक समूह, ...",
  "grounded": true,
  "confidence": 0.9108,
  "sources": [ { "document_id": "1102432:2", "score": 0.8514, "query_id": "1102432", "language": "hin", ... } ],
  "guardrail": null
}
```

An unsafe query (`how do I make a bomb?`) returns `guardrail: "unsafe_input"`
with an empty answer.

## Known limitations (Phase 2)

- Off-topic detection relies on a similarity threshold; a query can slip
  through if the corpus contains a tangentially related passage (observed with
  a weather query matching a weather-map passage). A dedicated off-topic
  classifier is the planned improvement.
- The mock generator extracts the top passage verbatim, so every answer is
  grounded by construction; it is for pipeline verification, not quality.
- Grounding is lexical (see above).
- No human/LLM answer-quality evaluation set exists yet.
- Total RAG latency is **not** claimed to be <200ms; it is instrumented for
  later optimization.

---

# Phase 2.5 - Free LLM providers & benchmark

Goal: replace the mock generator with a **real, replaceable LLM** while the
project stays at **₹0**. No paid API is used, and no provider is called "free"
unless the ₹0 path was verified.

## Provider matrix

| Provider | Class | Default free model | Credentials | Status |
|---|---|---|---|---|
| `ollama` | `OllamaProvider` | `qwen2.5:1.5b-instruct` | none (local `http://localhost:11434`) | **DONE - benchmarked, verified ₹0** |
| `gemini` | `GeminiProvider` | `gemini-2.5-flash-lite` | `GEMINI_API_KEY` | DONE (implemented + unit-tested) / NOT ACCOUNT-VERIFIED |
| `groq` | `GroqProvider` | `llama-3.1-8b-instant` | `GROQ_API_KEY` | DONE (implemented + unit-tested) / NOT ACCOUNT-VERIFIED |
| `openrouter` | `OpenRouterProvider` | `google/gemma-4-31b-it:free` | `OPENROUTER_API_KEY` | DONE (implemented + unit-tested) / NOT ACCOUNT-VERIFIED |
| `openai_compatible` | `OpenAICompatibleProvider` | `gpt-4o-mini` | `OPENAI_API_KEY` | NOT ELIGIBLE (cost depends on the endpoint) |
| `mock` | `MockLLMProvider` | `mock-rag-generator` | none | tests only, never a silent fallback |

`python -m indicvoicerag.cli providers` prints this table live, including whether
each provider is reachable right now.

All API keys are read from **environment variables only** (never stored in
config or code); `.env` / `.env.*` are gitignored.

## Cost verification (Step 8)

| Provider | Free? | Billing required? | Free limit | API cost during benchmark | Eligible for ₹0? |
|---|---|---|---|---|---|
| Ollama | YES | NO | unlimited (local inference, only local compute) | ₹0 | **YES (verified)** |
| Gemini | YES (docs) | NO (AI Studio key, no billing account) | per-model RPM/RPD/TPM caps in the rate-limit docs | not run (no key) | YES, but **NOT VERIFIED** from an account |
| Groq | YES (docs) | NO (free plan, no card) | 30 RPM / 14,400 RPD / 6,000 TPM / 500,000 TPD for `llama-3.1-8b-instant` | not run (no key) | YES, but **NOT VERIFIED** from an account |
| OpenRouter | YES (docs) | NO (only `:free` model ids) | 20 req/min, 50 req/day without purchased credits | not run (no key) | YES, but **NOT VERIFIED** from an account |
| OpenAI-compatible | NO | YES | none | not run | **NO - NOT ELIGIBLE** |

Sources: <https://ai.google.dev/gemini-api/docs/rate-limits>,
<https://ai.google.dev/pricing>, <https://console.groq.com/docs/rate-limits>,
<https://openrouter.ai/docs/api-reference/limits>, <https://ollama.com/>.
Checked 2026-08-16 and encoded in `cost.py`.

No cloud API key was available in this session, so Gemini/Groq/OpenRouter were
implemented and unit-tested against mocked HTTP but **not** benchmarked; they
are reported as NOT VERIFIED rather than as measured free providers.
`OpenRouterProvider` refuses any model id that does not end in `:free`, so it
cannot silently start billing.

## Fallback semantics (Step 10)

```text
configured provider -> fallback_providers[...] -> (mock only if allow_mock_fallback = true)
```

`allow_mock_fallback` defaults to **false**, so a demo can never silently answer
with the deterministic stub. The `rag` command additionally prints a warning
when the configured provider is `mock`.

## Benchmark (Steps 6-7)

Fixed 20-query set in `benchmarks/queries.json`, identical for every provider:
6 English supported, 6 Hindi supported, 4 evidence-required, 2 low-relevance,
2 unsupported. Every run uses the full real pipeline (real MSMARCO-XI Hindi
validation sample -> `multilingual-e5-small` -> FAISS -> context -> LLM ->
grounding -> guardrails) and records per-stage latency, grounding, guardrail and
the raw answer to JSON + CSV.

```powershell
.venv\Scripts\python.exe -m indicvoicerag.cli --config config.ollama.toml benchmark-providers `
  --providers ollama --load-index `
  --out benchmarks/results/ollama.json --rows-out benchmarks/results/ollama.csv
```

Measured on this machine (2 vCPU, no GPU), 856 chunks indexed, top-k 5,
`max_tokens = 160`:

| Provider | Model | Cost | P50 total | P70 total | P100 total | avg generation | Indic | Selected |
|---|---|---|---|---|---|---|---|---|
| ollama | `qwen2.5:1.5b-instruct` | ₹0 verified | 19,241 ms | 23,257 ms | 62,882 ms | 25,152 ms | 6/6 Hindi answered and grounded | **YES** |
| ollama | `gemma3:1b` | ₹0 verified | 16,253 ms | 17,521 ms | 23,299 ms | 15,926 ms | 4/6 Hindi grounded, echoes the query in English | no |
| gemini | `gemini-2.5-flash-lite` | NOT VERIFIED | - | - | - | - | not benchmarked (no key) | no |
| groq | `llama-3.1-8b-instant` | NOT VERIFIED | - | - | - | - | not benchmarked (no key) | no |
| openrouter | `google/gemma-4-31b-it:free` | NOT VERIFIED | - | - | - | - | not benchmarked (no key) | no |

Retrieval itself stays fast in both runs: P50 ≈ 13.5 ms, P100 ≈ 21.3 ms; context
building ≈ 0.14 ms; grounding ≈ 0.2 ms. **Generation is >99% of total latency.**

### Qualitative observations (no invented scores)

`qwen2.5:1.5b-instruct` (20/20 answered, 0 errors, 17 grounded, 3 ungrounded):
- Answers from the retrieved passages; Hindi queries produce fluent Hindi
  answers grounded in the Hindi evidence (6/6).
- Refuses unsupported and low-relevance questions with the `REFUSED:` prefix
  (stock price, bank password, keyboard switches, train fares) - but sometimes
  writes the refusal reason in Chinese, a small-model artifact.
- The 3 ungrounded cases are English answers built from Hindi evidence: the
  answer is faithful, but the **lexical** grounding validator cannot match
  across languages (documented limitation, not a hallucination).
- Structured output is always valid `RAGResponse` JSON with provider metadata.

`gemma3:1b` is ~1.6x faster but clearly worse for RAG: 14/20 ungrounded, it
frequently echoes the question instead of answering, and it did not refuse the
bank-password query. Speed alone is not a good enough reason to select it.

## Real end-to-end RAG run (Step 12)

Real MSMARCO-XI -> real e5-small -> real FAISS -> real retrieved context -> real
free LLM (Ollama, no mock) -> real grounding -> real guardrails:

```powershell
.venv\Scripts\python.exe -m indicvoicerag.cli --config config.ollama.toml rag `
  --query "कॉर्पोरेशन क्या है?" --load-index --debug
```

```json
{
  "answer": "कॉर्पोरेशन केवल एक समूह के रूप में जाना जाता है, जो कानून द्वारा या कानून के अधिकार के तहत बनाया गया है और इसके सदस्यों के अस्तित्व से स्वतंत्र है। यह निगम है।",
  "grounded": true,
  "confidence": 0.8927,
  "guardrail": null,
  "llm": { "provider": "ollama", "model": "qwen2.5:1.5b-instruct", "latency_ms": 31608.26,
           "usage": { "prompt_tokens": 1676, "completion_tokens": 152, "total_tokens": 1828 } },
  "metrics_ms": { "retrieval": 16.1, "context": 0.1, "generation": 31608.3, "grounding": 0.5, "total": 31625.0 }
}
```

## Latency target (Step 9)

**<200 ms was NOT achieved.** Honest numbers for the fastest verified-₹0 setup:

- Fastest provider measured: Ollama (`gemma3:1b` P50 16.3 s; selected
  `qwen2.5:1.5b-instruct` P50 19.2 s / P70 23.3 s / P100 62.9 s).
- Bottleneck: **local CPU token generation** - 99.9% of total latency. Retrieval
  + context + grounding together are ≈ 14 ms, already inside the budget.
- Possible optimizations: use a hosted free provider (Groq is the strongest
  candidate: its LPU serving is normally sub-second, which would put the total
  in the few-hundred-ms range), stream the answer so time-to-first-token is what
  the user perceives, cut `max_tokens`, shrink the context (top-k 3 and a
  smaller token budget cut prompt-processing time), run Ollama on a GPU, or
  cache answers for repeated queries.

## Configuration

```toml
[llm]
provider = "ollama"                      # ollama | gemini | groq | openrouter | openai_compatible | mock
model_name = "qwen2.5:1.5b-instruct"
base_url = "http://localhost:11434"      # ollama / openai_compatible endpoint
api_key_env = ""                         # override the default env var name; the value never lives in config
temperature = 0.0
max_tokens = 160
timeout_seconds = 120.0
max_retries = 1
fallback_providers = []                  # e.g. ["ollama"] behind a cloud provider
fallback_models = {}
allow_mock_fallback = false               # keep false outside tests
```

Ready-to-use local config: `config.ollama.toml`. Cloud usage only needs the key
in the environment:

```powershell
$env:GROQ_API_KEY = "..."      # or GEMINI_API_KEY / OPENROUTER_API_KEY
.venv\Scripts\python.exe -m indicvoicerag.cli rag --query "..." --provider groq --load-index
```

## Selected provider

**Ollama + `qwen2.5:1.5b-instruct`**, because it is the only candidate with a
**verified** ₹0 path (no key, no account, no quota), it never fails on rate
limits, it answers Hindi queries in Hindi from the retrieved evidence, it
refuses unsupported questions, and it produces valid structured output. It is
not the fastest possible option - if a free Groq/Gemini key is added, the same
config switch makes it the primary provider and Ollama the offline fallback.

---

# Phase 3A - Local speech-to-text

Goal: turn an **audio file** into text with a **local, ₹0, CPU-only** model, then
feed that text into the existing Phase 2 RAG pipeline. No cloud STT API, no
microphone, no TTS.

## Architecture

```text
AUDIO FILE (wav/ogg/flac/mp3)
   |
   v
audio.load_audio()          PyAV -> mono 16 kHz float32
   |
   v
STTProvider.transcribe()    faster-whisper (CTranslate2, CPU, int8) | mock
   |
   v
TranscriptionResult{ text, language, language_probability,
                     duration_seconds, processing_time_ms, rtf,
                     model, load_time_ms, audio_path }
   |
   v
speech.answer_from_audio() -> RAGHarness.answer(query=text)   (text RAG path unchanged)
```

The `STTProvider` interface is provider-independent, so a future Indic-specific
model can be swapped in behind the same interface without touching the RAG
harness.

## CLI

```powershell
# transcribe one audio file (first run downloads the tiny model into .cache/stt)
.venv\Scripts\python.exe -m indicvoicerag.cli transcribe --audio samples\hindi.wav --language hi

# full pipeline: audio -> local STT -> text -> retrieval -> LLM -> grounding -> guardrails
.venv\Scripts\python.exe -m indicvoicerag.cli transcribe-rag --audio samples\hindi.wav --language hi --load-index

# benchmark all samples in samples/ (uses *.txt sidecars for ground truth)
.venv\Scripts\python.exe -m indicvoicerag.cli benchmark-stt --samples-dir samples --repeat 2 --out results/stt_benchmark.json

# Phase 3C: record from the microphone, transcribe, then run the full RAG pipeline
.venv\Scripts\python.exe -m indicvoicerag.cli listen --language hi            # push-to-talk, fixed window
.venv\Scripts\python.exe -m indicvoicerag.cli listen --duration 8 --no-rag    # STT only, 8 s window
.venv\Scripts\python.exe -m indicvoicerag.cli listen --json --debug           # machine-readable + keep temp WAV

# Phase 3D: text -> local Indic TTS -> WAV (first run downloads mms-tts-hin into .cache/tts)
.venv\Scripts\python.exe -m indicvoicerag.cli synthesize --text "मिर्ची में कितने विबिन्ना प्रजात्या है" --language hi --output out.wav
.venv\Scripts\python.exe -m indicvoicerag.cli synthesize --text "नमस्ते" --json
.venv\Scripts\python.exe -m indicvoicerag.cli rag --query "मिर्ची की प्रजातियाँ" --tts-output answer.wav   # RAG answer -> TTS -> WAV
```

`transcribe --json`, `--language`, `benchmark-stt --audio/--label/--ground-truth/--language/--no-warmup/--json`, `listen --duration/--sample-rate/--channels/--language/--model/--top-k/--size/--load-index/--device/--no-rag/--json/--debug`, and `synthesize --text/--language/--output/--json` are also supported.

> `listen` requires the `mic` extra: `pip install -e .[mic]` (installs `sounddevice`,
> which bundles PortAudio on Windows). Press Enter to start the fixed-duration
> recording; with piped/closed stdin the recording starts immediately. In `--json`
> mode stdout is pure JSON; progress chatter goes to stderr.

## Samples

`benchmark-stt` discovers `samples/*.wav` plus a `*.txt` sidecar per file:

```text
THE REFERENCE TRANSCRIPT      <- first non-empty line = ground truth (WER source)
language = en                 <- pinned language passed to the model
source = <where it came from>
```

An empty first line means **no ground truth**; WER is then reported as
`accuracy NOT MEASURED` rather than invented. Sample audio files are gitignored
(they are downloaded, not committed); see `samples/README.md`.

## Benchmark (measured on this machine, 2 vCPU, no GPU)

`faster-whisper` `tiny` / CPU / `int8`, `beam_size=5`, best of 2 runs,
124 documents indexed in the Hindi FAISS index:

| label | dur (s) | load (ms) | proc (ms) | RTF | lang | WER |
|---|---|---|---|---|---|---|
| `en_1272-128104-0000` (LibriSpeech) | 5.86 | 6,340 | 590 | 0.10 | en | 0.059 |
| `en_1272-128104-0001` (LibriSpeech) | 4.82 | 0 | 520 | 0.11 | en | 0.200 |
| `en_1272-128104-0002` (LibriSpeech) | 12.48 | 0 | 700 | 0.06 | en | 0.062 |
| `hindi` (Narsil/asr_dummy, pinned `hi`) | 4.13 | 0 | 2,087 | 0.50 | hi | accuracy NOT MEASURED |
| `tone` (440 Hz sine sanity check) | 0.50 | 0 | 2,702 | 5.40 | en | accuracy NOT MEASURED |

Summary: `samples=5, errors=0, avg_wer (English) = 0.107`.

Full report incl. transcripts: `results/stt_benchmark.json`.

### Honest caveats

- **Model load** (~6.3 s with a warm cache) happens once on the first
  transcription; steady-state `transcribe` on English is ~0.06-0.2x real time
  (RTF < 1) on this 2-vCPU machine.
- The **`tiny` model is weak on Hindi**: it is inaccurate on clean Hindi speech
  and initially auto-detected the Hindi sample as `ur`. Pinning
  `language = "hi"` produces Devanagari output, but the transcript is still
  poor. `base`/`small` and/or VAD filtering are the obvious accuracy levers.
- **Hindi WER is not measured** - the `Narsil/asr_dummy` dataset ships no ground
  truth. English WER uses the real LibriSpeech transcripts from
  `hf-internal-testing/librispeech_asr_dummy`.
- No microphone/live streaming; input is a file.

## Configuration

```toml
[stt]
provider = "faster_whisper"   # faster_whisper | mock (mock = tests only)
model_name = "small"          # Phase 3B selection: tiny | base | small | medium | large-v3
device = "cpu"                # this machine has no GPU
compute_type = "int8"         # int8 (fast CPU) | int8_float32 | int8_float16 | int16 | float32
language = ""                 # "" = auto-detect; "en" / "hi" to pin
download_root = ".cache/stt"  # model cache (gitignored)
beam_size = 5
vad_filter = false

[audio]                        # Phase 3C: microphone capture defaults
sample_rate = 16000            # captured PCM is resampled to mono 16 kHz
channels = 1
duration_seconds = 5.0         # fixed recording window per `listen` invocation

[tts]                          # Phase 3D: local Indic TTS output layer
provider = "mms"               # mock | mms (Meta MMS-TTS VITS, local, CC-BY-NC 4.0)
language = "hi"                # baseline; see tts.py for the Indic checkpoint table
cache_dir = ".cache/tts"       # model cache (gitignored)
```

---

# Phase 3B - Hindi STT model benchmark

Phase 3B answers one question: **which local faster-whisper model is the best
practical choice for Hindi on this CPU-only machine?** The current default was
changed from `tiny` to `small` as a result. The Phase 3A pipeline itself was
not rebuilt.

## Method

- Same single Hindi sample (`samples/hindi.wav`, 4.13 s, from
  `Narsil/asr_dummy`) for every model, `compute_type=int8`, `beam_size=5`, CPU.
- Each model run twice: **(A)** automatic language detection and **(B)**
  explicitly pinned `language="hi"`.
- Steady-state timing: model load (one-time) measured separately from warm
  transcription (median of 5 repeated runs).
- On-disk sizes measured from the cached CTranslate2 int8 models.

## Results

### Automatic language detection (A)

| model | detected lang | script | transcription |
|---|---|---|---|
| `tiny` | **ur (wrong)** | Arabic/Urdu | `مرچی میں کیتنے بھی بھی بھی نا پر جاتیا ہے` (repeated `بھی`) |
| `base` | hi (correct) | Arabic/Urdu | `مرچی میں کتھنے ویبیننا پر جاتیا ہے` |
| `small` | hi (correct) | **Devanagari** | `मिर्ची में कितने विबिन्ना प्रजात्या है` |

### Pinned `language="hi"` (B)

| model | detected lang | script | transcription |
|---|---|---|---|
| `tiny` | hi | **Latin** (transliteration) | `Mirchi mein, ki tene vibinda prajatiya hai.` |
| `base` | hi | Arabic/Urdu | `مرچی میں کتھنے ویبیننا پر جاتیا ہے` (same as auto) |
| `small` | hi | **Devanagari** | `मिर्ची में कितने विबिन्ना प्रजात्या है` (same as auto) |

### Latency (pinned `hi`, warm, median of 5; `results/phase3b/steady_state_timing.json`)

| model | int8 size | model load | first call total | warm transcription | RTF (warm) |
|---|---|---|---|---|---|
| `tiny` | 72 MB | 5.2 s | 7.6 s | 2.28 s | 0.55 |
| `base` | 138 MB | 0.85 s | 2.0 s | 1.15 s | 0.28 |
| `small` | 461 MB | 11.1 s | 15.4 s | 3.98 s | 0.96 |

RTF = transcription time / audio duration; RTF < 1 means faster than real time.
Model load is a one-time cost (cached model); it is reported separately and is
never part of per-request transcription latency. This 2-vCPU box is noisy, so
single-shot RTF for `small` was also observed at ~1.1 in a loaded `transcribe-rag`
run.

### Qualitative Hindi comparison (no ground truth, so no WER numbers)

No trustworthy Hindi ground-truth transcript exists for this clip, so **Hindi
WER = NOT MEASURED** (not fabricated). Qualitative reading of the outputs:

- **Wrong script is the killer for `tiny`/`base`:** pinned `tiny` emits Latin
  transliteration, and `base` emits Arabic/Urdu script, so the text is not even
  indexable by the Hindi RAG stack. Only `small` emits Devanagari.
- **Missing/substituted words:** `tiny` auto repeats `بھی بھی بھی` and pinned
  produces `ki tene vibinda` (near-gibberish); `base` produces `ویبیننا پر
  جاتیا` (unintelligible). `small` is a complete, grammatical sentence
  `मिर्ची में कितने विबिन्ना प्रजात्या है` with spelling errors
  (`विबिन्ना` for `विभिन्न`, `प्रजात्या` for `प्रजातियाँ`).
- **Names/numbers:** no numerals in the clip; `small` handles the word-final
  `है` correctly.
- **Sentence completeness:** `small` output is complete and coherent; `tiny`/
  `base` are partial/garbled.

### Language-pinning tradeoff (documented)

Pinning `language="hi"` **prevents the `tiny` misdetection** (`ur` -> `hi`) and
stops the `بھی` repetition, but it does **not** rescue `tiny`/`base` word errors
or force Devanagari (tiny pinned degrades to Latin transliteration; base stays
Arabic-script). `small` is unaffected by pinning (auto-detects `hi` correctly
and already emits Devanagari). **Conclusion: pinning is a cheap safety net but
not a substitute for model quality.**

## Selection (quality gate)

| model | Hindi quality | RTF (warm) | CPU/RAM practicality | size | Selected? |
|---|---|---|---|---|---|
| `tiny` | unusable (wrong script, garbled) | 0.55 | excellent | 72 MB | no |
| `base` | unusable (Arabic script, garbled) | 0.28 | excellent | 138 MB | no |
| `small` | usable Devanagari, minor spelling errors | 0.96 | OK (fits 7.4 GB RAM; at the real-time edge) | 461 MB | **YES** |

**Selected: `small` (faster-whisper `small`, CPU int8).** It is the only
candidate that produces indexable Hindi text. The trade-off is explicit: it is
~4x the size of `base` and sits right at RTF ~1 on this 2-vCPU machine (a 4 s
clip ~ 4 s to transcribe when warm). `base` would be the latency pick if the
voice assistant were English-only or if a lighter Hindi path is ever needed,
but its Hindi output is unusable. Selection was made on the full quality gate
(quality + RTF + practicality + size), not accuracy alone.

## RAG regression

Hindi audio -> STT (`small`) -> text -> existing `e5-small` -> FAISS (124
Hindi docs) -> LLM -> grounding -> guardrails -> answer. Pipeline code was not
changed.

- **Groq was NOT exercised** - no `GROQ_API_KEY` and no local Ollama were
  available in this session, so the regression ran the *unchanged* pipeline with
  the configured `mock` generator (the provider layer is already covered by
  offline tests, including Groq's mocked HTTP path).
- Output (full: `results/phase3b/rag_regression_small.json`): query
  `मिर्ची में कितने विबिन्ना प्रजात्या है`, retrieval 2.5 s (incl. first-run
  embedder), mock-grounded answer on low-relevance passages. The imperfect
  transcription (misspelled `विबिन्ना`/`प्रजात्या`) retrieves off-topic gaming
  passages - a concrete demonstration that **STT quality propagates straight
  into RAG retrieval quality**, and exactly why `small` was selected.
- The full audio->RAG chain is confirmed working end to end.

## Remaining limitations

- Hindi WER is **NOT MEASURED** (no trustworthy Hindi test set available; not
  downloading a large dataset for this benchmark).
- `small` RTF ~1 on this machine: borderline for real-time; a quieter/stronger
  CPU or a GPU would give headroom. Consider `vad_filter` or lower `beam_size`.
- Only one Hindi audio clip was available; statistical quality claims would need
  a proper Hindi eval set.
- Microphone is addressed in Phase 3C (below); TTS / frontend / deployment
  remain out of scope.

---

# Phase 3C - Microphone input pipeline

Phase 3C adds a **live microphone front-end** in front of the unchanged
Phase 3A/B pipeline: `MicrophoneRecorder` -> temp WAV -> `small` STT -> text ->
existing `RAGHarness`. No STT/RAG pipeline code was rebuilt; no TTS, frontend or
continuous (always-on) listening was added.

## Design

```text
MicrophoneRecorder (sounddevice / PortAudio, fixed-duration, push-to-talk)
        |
        v  AudioCaptureResult (mono float32, 16 kHz; resampled from device rate)
        |
        v  temp WAV (stdlib wave, int16)  -- deleted after the run unless --debug
        |
        v
STTProvider (unchanged; default faster-whisper small)
        |
        v
RAGHarness.answer(query=<transcribed text>)   (unchanged text path)
```

- **`src/indicvoicerag/audio_capture.py`** knows nothing about STT/RAG. It maps
  PortAudio errors to controlled `MicrophoneError` subclasses
  (`MicrophoneUnavailableError` / `MicrophonePermissionError` /
  `MicrophoneRecordingError`), resamples any device rate to mono 16 kHz (PyAV,
  pure-numpy fallback), and always returns float32 mono PCM.
- **Injection point**: `MicrophoneRecorder(record_fn=...)` lets tests run the
  whole `listen` path with a fake backend - no hardware in the test suite.
- **`listen` CLI**: records a fixed window (Enter to start; piped/closed stdin
  starts immediately), reports recording time separately from system latency,
  and prints a per-stage breakdown (audio prep / STT / RTF / model load / RAG
  retrieval / context / generation / grounding / total post-recording). Temp WAV
  is deleted unless `--debug`.
- **`--no-rag`** stops after transcription; **`--json`** emits one JSON document
  on stdout (progress goes to stderr) so it can be consumed by a frontend later.
- New optional extra: `pip install -e .[mic]` (`sounddevice`). No cloud cost.

## Smoke test (real hardware, this machine)

Live microphone capture on `Microphone Array (Realtek(R) Audio)`:

| run | recording | prep (write WAV) | STT | RTF (warm) | outcome |
|---|---|---|---|---|---|
| `tiny`, no-RAG | 2.000 s | 2.6 ms | 1147 ms | 0.574 | transcribed ambient speech `Let's go.` (auto-detected `en`) |
| `tiny`, full RAG | 2.000 s | 2.6 ms | 971 ms | 0.485 | empty transcription -> `invalid_input` guardrail (graceful) |
| `small`, no-RAG | 3.000 s | — | 108.7 s cold | 25.8 (cold, near-silence) | empty text (no speech captured) |

The microphone capture -> WAV -> STT -> harness path works end to end. Empty or
near-silent audio yields an empty transcription, which the existing guardrails
reject as `invalid_input` instead of crashing. Cold `small` on near-silent input
was far slower than the Phase 3B warm RTF (~1) - silence/empty-clip processing
and cold start dominate; warm speech at `small` still matches the Phase 3B
characterization.

## Test coverage (offline, no hardware)

- `tests/test_audio_capture.py` - recording success (mono/stereo downmix),
  resample correctness, empty-capture error, backend failure wrapping, and the
  WAV write->read round trip via the `wave`/PyAV path.
- `tests/test_listen_cli.py` - the full `listen` control flow with fake
  recorder/STT/harness: success, `--no-rag`, `--json`, temp-WAV cleanup,
  `--debug` keep, no-microphone / empty-capture / STT-failure controlled
  errors, cancel before recording, and non-interactive stdin.

---

# Phase 3D - Local Indic TTS

Phase 3D adds the **output layer**: `RAG answer text -> local TTS -> WAV`.
Hindi is mandatory; the provider table covers ~14 Indic languages. The RAG
harness stays text-based; TTS is pure output glue. No frontend, no voice loop.

## Candidate comparison (audit, this machine: Windows, Python 3.11.9, CPU-only, 7.4 GB RAM)

| Model | Hindi | Indic coverage | CPU | Size | Framework | Py 3.11 | License | Practical? |
|---|---|---|---|---|---|---|---|---|
| **MMS-TTS VITS (`mms-tts-hin`)** | yes (Devanagari) | ~14 Indic checkpoints (hi/bn/ta/te/mr/gu/kn/ml/pa/or/as/ne/si/ur) | fast (RTF < 1) | ~138 MB | transformers (already installed) | yes | CC-BY-NC 4.0 | **YES - SELECTED** |
| Piper (`hi_IN`) | yes | 35 langs | very fast | ~60 MB | onnxruntime + espeak-ng | blocked | MIT | **no** - `piper-phonemize` has no Windows wheel; native build fails (C2015); Windows path needs the standalone `piper.exe` binary |
| edge-tts / gTTS | excellent | many | n/a | n/a | remote API | yes | free | **no** - cloud, not local (₹0 but not local inference) |
| eSpeak NG | yes (robotic) | many | fast | ~5 MB | native | yes | GPL | quality too poor for a voice baseline |
| Bark | yes | 13 | slow | ~1.2 GB | transformers | yes | MIT | CPU latency impractical |
| Coqui XTTS v2 | yes | 16 | slow | ~1.8 GB | torch | partial (pkg EOL) | CPML non-commercial | heavy, slow, unmaintained |
| Kokoro / Chatterbox | Hindi support unclear | few | fast/med | ~330 MB+ | torch | yes | Apache-2.0 | unverified Hindi |

**Selected: `facebook/mms-tts-hin`** - real Hindi quality, per-language local
checkpoints (Indic-forward), runs on the already-installed transformers/torch
stack (zero new frameworks), pure-Python integration, CPU RTF < 1, ~138 MB,
₹0. Caveat: CC-BY-NC 4.0 (free **non-commercial** use only) - fine for this
project, documented for any future commercial use.

## Design

```text
RAG answer text (structured, text-based) or any Hindi text
        |
        v
TTSProvider.synthesize(text, language="hi", output_path=...)
        |
        v
TTSResult{ audio (float32), sample_rate, duration_seconds,
           synthesis_time_ms, rtf, model, load_time_ms, output_path }
        |
        v
mono 16-bit PCM WAV (stdlib `wave`, 16 kHz for mms-tts-hin)
```

- `src/indicvoicerag/tts.py`: `TTSProvider` (ABC) / `TTSResult` /
  `MockTTS` / `MMSIndicTTS` / `build_tts_provider`. Model is loaded lazily on
  the first `synthesize` and **reused** for every call on the same instance
  (same lifecycle as STT). `model_factory` is the offline test injection point.
- CLI: `synthesize --text --language --output [--json]`; `rag --tts-output FILE`
  synthesizes the structured RAG answer (retrieval/grounding untouched).
- The RAG harness is not touched - TTS is a separate output layer.

## Benchmark (real `mms-tts-hin`, this machine)

Cold start (first run, includes model download into `.cache/tts`):

| metric | value |
|---|---|
| model download + load | 106.7 s (one-time) |
| first synthesis (84 chars) | 8.04 s for 6.03 s audio |
| cold RTF | 1.33 |

Warm (same process, model cached):

| sample | chars | synthesis | audio | RTF |
|---|---|---|---|---|
| 1 | 84 | 3.37 s | 6.24 s | 0.54 |
| 2 | 59 | 2.29 s | 5.02 s | 0.46 |
| 3 | 54 | 1.93 s | 4.08 s | 0.47 |

Warm RTF ~0.46-0.54 (faster than real time). Model load from cache is
variable on this noisy box (16-107 s); warm synthesis is ~0.5-2 s for a
RAG-size answer. Output format: **mono, 16-bit PCM, 16 kHz WAV** (readable by
PyAV, `wave`, and Windows playback).

## Quality

- **Objective metric**: no reliable speech-quality reference metric (MOS) is
  available locally, so none is fabricated. The measured objective proxy is a
  **TTS -> STT round-trip** through the existing faster-whisper `small`:
  input `मिर्ची में कितने विबिन्ना प्रजात्या है` -> synthesized WAV ->
  STT recovered `मिर्षी में कितने विबिन्ना पजात्या है` (4/6 words exact;
  `च/ष` and `प्र/प` single-phoneme differences - attributable to either TTS
  pronunciation or STT error; honestly ambiguous).
- **Qualitative (by ear, in this session)**: not audited by ear here - the
  audio is written to `tmp/3d_hindi.wav` and must be listened to. Reported
  objectively: Devanagari text is accepted directly (no transliteration
  needed), the WAV decodes cleanly, and the round-trip above shows the speech
  is intelligible, recognizable Hindi with no artifacts detectable at the file
  level. **One successful WAV is not production-ready**; a listening pass and
  a real MOS are still required.

## RAG -> TTS integration

`rag --query ... --tts-output answer.wav` ran the unchanged RAG pipeline and
synthesized the structured answer to WAV (`tts_audio` block in the JSON
report: language `hi`, 16 kHz, load 91.7 s, synthesis 21.6 s for the mock
answer `1.` - mock LLM, so the spoken text is trivial). Retrieval, grounding
and guardrails were not modified.

## Test coverage (offline, no downloads / mic / APIs)

`tests/test_tts.py` (15 tests): mock synthesis + WAV format, empty text,
unsupported language, provider failure, result schema, provider-name
normalization + builder dispatch, MMSIndicTTS with an injected fake VITS model
(synthesis, language table, failure wrapping, **model-loads-once-and-reuses**
regression), CLI `synthesize`, and the RAG->TTS glue with mock providers.


