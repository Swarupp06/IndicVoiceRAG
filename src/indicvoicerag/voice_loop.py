"""Phase 3E: end-to-end voice turn orchestration.

One controlled voice turn:

    user speaks
      -> MicrophoneRecorder
      -> AudioCaptureResult (mono float32 PCM)
      -> STTProvider
      -> TranscriptionResult (text + metadata)
      -> RAGHarness.answer(text)
      -> RAGResponse (answer + grounding + guardrails)
      -> TTSProvider (only on the normal answered path)
      -> TTSResult / WAV output

The RAG core stays text-based.  TTS is pure output glue.
"""

from __future__ import annotations

import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .audio_capture import MicrophoneError, MicrophoneRecorder, write_pcm_wav
from .harness import RAGHarness
from .rag_types import RAGResponse
from .stt import STTError, STTProvider
from .tts import TTSError, TTSProvider


@dataclass(slots=True)
class VoiceTurnResult:
    """Structured outcome of a single voice turn."""

    # --- transcription ---
    transcription: str = ""
    detected_language: str | None = None
    language_probability: float = 0.0

    # --- RAG ---
    rag_response: RAGResponse | None = None

    # --- TTS ---
    tts_audio_path: str | None = None
    tts_duration_seconds: float = 0.0
    tts_sample_rate: int = 0

    # --- latencies (ms) ---
    recording_duration_seconds: float = 0.0
    stt_processing_ms: float = 0.0
    stt_rtf: float = 0.0
    stt_load_ms: float = 0.0
    rag_total_ms: float = 0.0
    rag_retrieval_ms: float = 0.0
    rag_generation_ms: float = 0.0
    rag_grounding_ms: float = 0.0
    tts_synthesis_ms: float = 0.0
    tts_load_ms: float = 0.0
    tts_rtf: float = 0.0
    total_post_recording_ms: float = 0.0

    # --- errors ---
    error: str | None = None
    error_stage: str | None = None  # "stt" | "rag" | "tts" | "mic"

    def as_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "transcription": self.transcription,
            "detected_language": self.detected_language,
            "language_probability": round(self.language_probability, 4),
            "rag": self.rag_response.as_dict(include_metrics=True) if self.rag_response is not None else None,
            "tts": {
                "audio_path": self.tts_audio_path,
                "duration_seconds": round(self.tts_duration_seconds, 3),
                "sample_rate": self.tts_sample_rate,
                "synthesis_ms": round(self.tts_synthesis_ms, 1),
                "load_ms": round(self.tts_load_ms, 1),
                "rtf": round(self.tts_rtf, 4),
            }
            if self.tts_audio_path is not None
            else None,
            "recording_duration_seconds": round(self.recording_duration_seconds, 3),
            "latency": {
                "stt_ms": round(self.stt_processing_ms, 1),
                "stt_rtf": round(self.stt_rtf, 4),
                "rag_total_ms": round(self.rag_total_ms, 1),
                "rag_retrieval_ms": round(self.rag_retrieval_ms, 1),
                "rag_generation_ms": round(self.rag_generation_ms, 1),
                "rag_grounding_ms": round(self.rag_grounding_ms, 1),
                "tts_synthesis_ms": round(self.tts_synthesis_ms, 1),
                "total_post_recording_ms": round(self.total_post_recording_ms, 1),
            },
        }
        if self.error is not None:
            payload["error"] = self.error
            payload["error_stage"] = self.error_stage
        return payload


def run_voice_turn(
    *,
    stt: STTProvider,
    harness: RAGHarness | None,
    tts: TTSProvider | None,
    recorder: MicrophoneRecorder,
    language: str | None = None,
    top_k: int | None = None,
    tts_output: str | None = None,
    debug: bool = False,
) -> VoiceTurnResult:
    """Execute a single voice turn: mic -> STT -> RAG -> TTS.

    Returns a ``VoiceTurnResult`` with every stage's outcome and latency.
    The RAG harness may be ``None`` (STT-only mode).  TTS is only invoked on
    the normal answered path (guardrail refusals produce no audio).
    """
    result = VoiceTurnResult()

    # --- Stage 1: Record ---
    try:
        capture = recorder.record()
    except MicrophoneError as exc:
        result.error = str(exc)
        result.error_stage = "mic"
        return result
    except KeyboardInterrupt:
        result.error = "recording cancelled"
        result.error_stage = "mic"
        return result

    result.recording_duration_seconds = capture.duration_seconds
    if capture.empty:
        result.error = "empty recording (no speech detected)"
        result.error_stage = "mic"
        return result

    # Write temp WAV for STT
    t0 = time.perf_counter()
    fd, tmp_name = tempfile.mkstemp(suffix=".wav", prefix="ivr_voice_")
    import os
    os.close(fd)
    tmp_path = Path(tmp_name)
    write_pcm_wav(tmp_path, capture.samples, capture.sample_rate)
    prep_ms = (time.perf_counter() - t0) * 1000.0

    try:
        # --- Stage 2: STT ---
        t_stt = time.perf_counter()
        try:
            transcription = stt.transcribe(str(tmp_path), language=language)
        except STTError as exc:
            result.error = str(exc)
            result.error_stage = "stt"
            if not debug:
                tmp_path.unlink(missing_ok=True)
            return result

        result.transcription = transcription.text
        result.detected_language = transcription.language
        result.language_probability = transcription.language_probability
        result.stt_processing_ms = transcription.processing_time_ms
        result.stt_rtf = transcription.rtf
        result.stt_load_ms = transcription.load_time_ms or 0.0

        if not transcription.text.strip():
            result.error = "empty transcription (no speech detected)"
            result.error_stage = "stt"
            if not debug:
                tmp_path.unlink(missing_ok=True)
            return result

        # --- Stage 3: RAG ---
        rag_response: RAGResponse | None = None
        if harness is not None:
            t_rag = time.perf_counter()
            try:
                rag_response = harness.answer(query=transcription.text, top_k=top_k, debug=debug)
            except Exception as exc:  # noqa: BLE001
                result.error = f"RAG failed: {type(exc).__name__}: {exc}"
                result.error_stage = "rag"
                if not debug:
                    tmp_path.unlink(missing_ok=True)
                return result
            rag_ms = (time.perf_counter() - t_rag) * 1000.0
            result.rag_response = rag_response
            result.rag_total_ms = rag_ms
            metrics = rag_response.metrics or {}
            result.rag_retrieval_ms = float(metrics.get("retrieval", 0.0))
            result.rag_generation_ms = float(metrics.get("generation", 0.0))
            result.rag_grounding_ms = float(metrics.get("grounding", 0.0))

        # --- Stage 4: TTS (only on normal answered path) ---
        synthesize_text: str | None = None
        if rag_response is not None and rag_response.guardrail is None:
            # Normal RAG path: synthesize the answer
            synthesize_text = rag_response.answer
        elif rag_response is None and tts is not None:
            # No-RAG mode: synthesize the transcription directly
            synthesize_text = transcription.text

        if tts is not None and synthesize_text and synthesize_text.strip():
            out_path = tts_output
            if out_path is None:
                slug = "".join(c for c in synthesize_text.strip().split()[0] if c.isalnum())[:20] if synthesize_text.split() else "speech"
                out_path = f"tmp/{slug}-{language or 'auto'}.wav"
            t_tts = time.perf_counter()
            try:
                tts_result = tts.synthesize(synthesize_text, language=language, output_path=out_path)
            except TTSError as exc:
                result.error = f"TTS failed: {exc}"
                result.error_stage = "tts"
                if not debug:
                    tmp_path.unlink(missing_ok=True)
                return result
            tts_ms = (time.perf_counter() - t_tts) * 1000.0
            result.tts_audio_path = tts_result.output_path
            result.tts_duration_seconds = tts_result.duration_seconds
            result.tts_sample_rate = tts_result.sample_rate
            result.tts_synthesis_ms = tts_ms
            result.tts_load_ms = tts_result.load_time_ms or 0.0
            result.tts_rtf = tts_result.rtf

        # --- Total post-recording latency ---
        total_ms = (time.perf_counter() - t_stt) * 1000.0
        result.total_post_recording_ms = total_ms

        if not debug:
            tmp_path.unlink(missing_ok=True)

    except KeyboardInterrupt:
        if not debug:
            tmp_path.unlink(missing_ok=True)
        result.error = "interrupted"
        result.error_stage = "unknown"
    except Exception as exc:  # noqa: BLE001
        if not debug:
            tmp_path.unlink(missing_ok=True)
        result.error = f"unexpected error: {type(exc).__name__}: {exc}"
        result.error_stage = "unknown"

    return result
