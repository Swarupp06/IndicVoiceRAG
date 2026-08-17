"""Create the evaluation query set for the Goa pilot corpus.

Each record carries query_id, query_text, language, relevance, and
maps to relevant document(s) via document_id.
Multi-document queries use the same query_id across records.
"""
from __future__ import annotations

import json
from pathlib import Path

ACCESS_DATE = "2026-08-17"

# Each entry: (query_id, query_text, language, category, [(doc_id, relevance)])
QUERIES = [
    # ── English queries ──────────────────────────────────────────────
    (
        "q_en_001",
        "What is the capital of Goa?",
        "en",
        "geography",
        [("goa_overview_001", 1.0)],
    ),
    (
        "q_en_002",
        "Which UNESCO World Heritage Site is in Goa?",
        "en",
        "heritage",
        [("goa_unesco_churches_001", 1.0), ("goa_bom_jesus_001", 1.0)],
    ),
    (
        "q_en_003",
        "What are the best beaches to visit in Goa?",
        "en",
        "tourism",
        [("goa_beaches_001", 1.0)],
    ),
    (
        "q_en_004",
        "What is Goan food like?",
        "en",
        "culture",
        [("goa_cuisine_001", 1.0)],
    ),
    (
        "q_en_005",
        "How long did the Portuguese rule Goa?",
        "en",
        "history",
        [("goa_history_001", 1.0)],
    ),
    (
        "q_en_006",
        "What wildlife sanctuaries are in Goa?",
        "en",
        "geography",
        [("goa_wildlife_001", 1.0)],
    ),
    (
        "q_en_007",
        "What is Tiatr theatre in Goa?",
        "en",
        "culture",
        [("goa_theatre_001", 1.0)],
    ),
    (
        "q_en_008",
        "What are the major rivers of Goa?",
        "en",
        "geography",
        [("goa_biodiversity_001", 1.0)],
    ),
    (
        "q_en_009",
        "What is feni and how is it made?",
        "en",
        "culture",
        [("goa_cuisine_001", 1.0)],
    ),
    (
        "q_en_010",
        "What museums can I visit in Goa?",
        "en",
        "tourism",
        [("goa_museums_001", 1.0)],
    ),
    # ── Hindi queries ────────────────────────────────────────────────
    (
        "q_hi_001",
        "गोवा की राजधानी क्या है?",
        "hi",
        "geography",
        [("goa_overview_001", 1.0)],
    ),
    (
        "q_hi_002",
        "गोवा में कौन सी यूनेस्को विश्व धरोहर स्थल है?",
        "hi",
        "heritage",
        [("goa_unesco_churches_001", 1.0), ("goa_bom_jesus_001", 1.0)],
    ),
    (
        "q_hi_003",
        "गोवा की सबसे अच्छी बीच कौन सी हैं?",
        "hi",
        "tourism",
        [("goa_beaches_001", 1.0)],
    ),
    (
        "q_hi_004",
        "गोवाई खाना कैसा होता है?",
        "hi",
        "culture",
        [("goa_cuisine_001", 1.0)],
    ),
    (
        "q_hi_005",
        "पुर्तगालियों ने गोवा पर कितने साल राज किया?",
        "hi",
        "history",
        [("goa_history_001", 1.0)],
    ),
    (
        "q_hi_006",
        "गोवा में कौन से नृत्य और संगीत के पारंपरिक रूप हैं?",
        "hi",
        "culture",
        [("goa_culture_dance_001", 1.0)],
    ),
    (
        "q_hi_007",
        "गोवा में कौन से किले हैं?",
        "hi",
        "heritage",
        [("goa_heritage_sites_001", 1.0)],
    ),
    (
        "q_hi_008",
        "गोवा का मानसून कैसा होता है?",
        "hi",
        "geography",
        [("goa_geography_001", 1.0)],
    ),
]


def main() -> None:
    out_dir = Path(__file__).parent / "normalized"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "goa_pilot_queries.jsonl"

    with out_path.open("w", encoding="utf-8") as f:
        for query_id, query_text, language, category, doc_rels in QUERIES:
            for doc_id, relevance in doc_rels:
                record = {
                    "query_id": query_id,
                    "query_text": query_text,
                    "language": language,
                    "category": category,
                    "document_id": doc_id,
                    "relevance": relevance,
                    "access_date": ACCESS_DATE,
                }
                f.write(json.dumps(record, ensure_ascii=False) + "\n")

    # Count unique queries
    unique_qids = set(q[0] for q in QUERIES)
    print(f"Wrote {len(QUERIES)} query-document pairs ({len(unique_qids)} unique queries) to {out_path}")


if __name__ == "__main__":
    main()
