"""Phase 3A integration: audio -> STT -> text -> existing RAGHarness.

The text path (`RAGHarness.answer(query=...)`) is untouched; this module only
adds the audio front-end in front of it, so the RAG pipeline stays reusable
with plain text input.
"""

from __future__ import annotations

from typing import Any

from .harness import RAGHarness
from .stt import STTProvider, TranscriptionResult


def transcribe_audio(stt: STTProvider, audio_path: str, *, language: str | None = None) -> TranscriptionResult:
    """Run local speech-to-text on an audio file."""
    return stt.transcribe(audio_path, language=language)


def answer_from_audio(
    stt: STTProvider,
    harness: RAGHarness,
    audio_path: str,
    *,
    language: str | None = None,
    top_k: int | None = None,
    debug: bool = False,
) -> tuple[TranscriptionResult, Any]:
    """audio -> STT -> text -> RAGHarness.answer().

    Returns (transcription, RAGResponse). The query fed to the harness is the
    transcribed text; the existing text-only RAG path is unchanged.
    """
    transcription = transcribe_audio(stt, audio_path, language=language)
    response = harness.answer(query=transcription.text, top_k=top_k, debug=debug)
    return transcription, response
