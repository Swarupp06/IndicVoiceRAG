"""Word error rate (WER) computation, dependency-free.

WER = (substitutions + insertions + deletions) / reference word count,
computed via Levenshtein distance over lowercase word tokens.
"""

from __future__ import annotations

import re

# Devanagari (Hindi) letters plus ASCII letters/digits/apostrophe
_WORD_RE = re.compile(r"[\w\u0900-\u097f']+", re.UNICODE)


def tokenize(text: str) -> list[str]:
    """Lowercase word tokens (handles Latin and Devanagari text)."""
    return [match.group(0) for match in _WORD_RE.finditer((text or "").lower())]


def _edit_distance(reference: list[str], hypothesis: list[str]) -> int:
    prev = list(range(len(hypothesis) + 1))
    for i in range(1, len(reference) + 1):
        current = [i] + [0] * len(hypothesis)
        for j in range(1, len(hypothesis) + 1):
            cost = 0 if reference[i - 1] == hypothesis[j - 1] else 1
            current[j] = min(
                prev[j] + 1,  # deletion
                current[j - 1] + 1,  # insertion
                prev[j - 1] + cost,  # substitution / match
            )
        prev = current
    return prev[len(hypothesis)]


def word_error_rate(reference: str, hypothesis: str) -> float:
    """Word error rate in [0, 1] (0 = perfect, 1 = everything wrong)."""
    ref = tokenize(reference)
    hyp = tokenize(hypothesis)
    if not ref:
        return 0.0 if not hyp else 1.0
    return _edit_distance(ref, hyp) / len(ref)
