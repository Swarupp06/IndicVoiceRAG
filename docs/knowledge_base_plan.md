# IndicVoiceRAG — Goa Knowledge Base Plan

## 1. Purpose

IndicVoiceRAG will be developed as a multilingual voice RAG assistant focused initially on Goa.

The assistant will accept spoken questions, convert speech to text, retrieve relevant information from a curated Goa knowledge base, generate a grounded answer, and optionally convert the answer back to speech.

Initial languages:

- Hindi
- English

The system must refuse or clearly state that information is unavailable when the knowledge base does not provide sufficient evidence.

---

## 2. Initial Knowledge Domains

Phase 4 will initially focus on:

### Tourism

- Places to visit
- Beaches
- Heritage attractions
- Activities
- Visitor-oriented information

### Heritage & History

- Historical places
- Cultural heritage
- Important historical events
- Heritage sites and institutions
- Festivals and cultural traditions

Do not expand into unrelated domains during the first dataset build.

---

## 3. Source Policy

The knowledge base should prioritize authoritative and trustworthy sources.

Preferred sources:

1. Official Goa government sources
2. Official Goa tourism sources
3. Official government/institutional heritage sources
4. Other authoritative sources explicitly approved for inclusion

Every document must preserve its provenance.

Do not silently include arbitrary web pages simply because they contain relevant text.

---

## 4. Document Metadata

Every normalized document should preserve metadata including, where available:

- document_id
- title
- source
- source_url
- publisher/organization
- language
- category
- publication date
- access date
- original document type
- content

The metadata must remain available through the retrieval pipeline so that answers can expose their supporting sources.

---

## 5. Language Strategy

The initial system supports:

- Hindi queries
- English queries
- Hindi answers
- English answers

The corpus may contain English and Hindi documents.

Because the retrieval system uses multilingual embeddings, we will evaluate whether a Hindi query can correctly retrieve relevant English source material.

We should not duplicate every source into Hindi unless there is a demonstrated need.

---

## 6. Retrieval Strategy

The existing retrieval architecture will be reused.

Target pipeline:

documents
→ normalization
→ chunking
→ multilingual embeddings
→ FAISS
→ retrieval
→ context construction
→ LLM
→ grounding/guardrails

Do not replace the existing retrieval architecture unless evaluation demonstrates a specific deficiency.

---

## 7. Answerability

A question is considered answerable when the knowledge base contains sufficient evidence to construct a grounded answer.

Examples of answerable questions:

- What are some famous beaches in Goa?
- What is Old Goa known for?
- Which heritage attractions are mentioned in the knowledge base?

Examples of unanswerable questions:

- Information not represented in the knowledge base
- Current facts not covered by the corpus
- Questions unrelated to the selected domains

For unanswerable questions, the system should refuse or state that sufficient information is not available.

The assistant must not hallucinate missing information.

---

## 8. Evaluation Dataset

The first evaluation set should contain approximately:

- 30–50 questions initially

The set should include:

- Hindi questions
- English questions
- Tourism questions
- Heritage/history questions
- Answerable questions
- Unanswerable questions
- Easy questions
- Multi-document questions where appropriate

Each evaluation query should preserve:

- query_id
- question
- language
- category
- answerability
- expected relevant source/document
- notes

The evaluation set will later be expanded to approximately 100–200 queries if useful.

---

## 9. Retrieval Metrics

We will continue using:

- Hit@1
- Hit@3
- Hit@5
- Recall@K
- MRR

We will additionally evaluate:

- grounded-answer rate
- correct-refusal rate
- source attribution correctness

Do not claim improvements without measured evidence.

---

## 10. Voice Evaluation

After the text knowledge base is validated, representative questions will be tested through:

microphone
→ STT
→ retrieval
→ RAG
→ grounding
→ TTS

Voice testing must distinguish between:

- STT errors
- retrieval errors
- generation errors
- grounding/guardrail decisions
- TTS errors

This prevents one subsystem's failure from being incorrectly attributed to another.

---

## 11. Cost Requirement

The project must remain ₹0 for development and evaluation wherever practical.

Preferred components:

- local/open-source models
- existing local embeddings
- FAISS
- existing faster-whisper STT
- existing local MMS TTS
- existing free-tier LLM path where required

No paid service should be introduced without explicit approval.

---

## 12. Scope Boundaries

Phase 4 will NOT initially include:

- frontend development
- web deployment
- continuous conversation
- wake-word detection
- multi-user authentication
- commercial deployment
- broad general-purpose knowledge
- unrelated knowledge domains

These can be considered later.

---

## 13. Phase 4 Success Criteria

Phase 4 will be considered successful when:

1. A curated Goa knowledge corpus exists.
2. Every document has provenance metadata.
3. Documents can be ingested through the existing pipeline.
4. The corpus can be embedded and indexed.
5. A representative evaluation set exists.
6. Retrieval performance is measured.
7. Grounded answer performance is measured.
8. Correct refusal behavior is measured.
9. Hindi and English queries are evaluated.
10. At least one real Hindi voice question can retrieve relevant Goa knowledge and produce a grounded answer.

---

## 14. Implementation Principle

Build incrementally.

Do not collect a massive dataset before testing the pipeline.

Recommended order:

1. Define sources.
2. Collect a small pilot corpus.
3. Normalize documents.
4. Index the pilot corpus.
5. Create evaluation questions.
6. Measure retrieval.
7. Fix data/chunking problems.
8. Expand the corpus.
9. Validate end-to-end voice RAG.

The existing working RAG/STT/TTS architecture should be preserved unless measurements justify changes.