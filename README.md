# IndicVoiceRAG - Phase 1 & 2 Foundation

Backend foundation for HH Goa 2026 Task 2:

**Phase 1 (retrieval):** Real MSMARCO-XI -> Normalization -> Chunking -> Real Multilingual Embeddings -> FAISS Vector Index -> Retrieval -> Retrieval Evaluation

**Phase 2 (RAG):** User Text -> Query Processing -> Retrieval -> Context Building -> LLM Answer Generation -> Grounding Validation -> Guardrails -> Structured Response

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

## What is NOT implemented yet

- Speech-to-text / voice input / microphone
- Sarvam/ElevenLabs STT integration
- Frontend/live demo deployment
- Authentication
- Final latency optimization for the sub-200ms target (measured P50 is ~19 s on local CPU generation; see Phase 2.5)
- Production-grade safety classifier (current safety gate is a small regex list)
- A human/LLM answer-quality evaluation set

## Project structure

```text
src/indicvoicerag/
  cli.py            # CLI entrypoint (inspect/sample/build/retrieve/evaluate/smoke/rag/rag-evaluate)
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
  pipeline.py       # wiring: config -> real retrieval engine -> harness
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
