"""Phase 3E offline tests for the voice-turn orchestration.

All tests use mock doubles: no microphone, no model downloads, no API keys.
They verify control flow, error boundaries, latency collection, and the
VoiceTurnResult schema.
"""

from __future__ import annotations

import builtins
import json
from pathlib import Path

import numpy as np
import pytest

from indicvoicerag.audio_capture import MicrophoneRecorder, MicrophoneUnavailableError
from indicvoicerag.cli import run_voice
from indicvoicerag.config import AppConfig
from indicvoicerag.rag_types import RAGResponse, SourceInfo
from indicvoicerag.stt import STTError
from indicvoicerag.tts import TTSError, TTSSynthesisError
from indicvoicerag.voice_loop import VoiceTurnResult, run_voice_turn


# --- helpers ---

def _sine(duration: float = 0.4, rate: int = 16000) -> tuple[np.ndarray, int]:
    t = np.arange(int(duration * rate)) / rate
    return (0.5 * np.sin(2 * np.pi * 440 * t)).astype(np.float32), rate


def _config() -> AppConfig:
    cfg = AppConfig()
    cfg.stt.provider = "mock"
    cfg.tts.provider = "mock"
    cfg.audio.duration_seconds = 0.4
    return cfg


def _recorder() -> MicrophoneRecorder:
    return MicrophoneRecorder(
        sample_rate=16000,
        duration_seconds=0.4,
        record_fn=lambda d, sr, ch, dev: _sine(d, sr or 16000),
    )


def _empty_recorder() -> MicrophoneRecorder:
    return MicrophoneRecorder(
        sample_rate=16000,
        duration_seconds=0.4,
        record_fn=lambda d, sr, ch, dev: (np.array([], dtype=np.float32), sr or 16000),
    )


def _failing_recorder() -> MicrophoneRecorder:
    def _boom(d, sr, ch, dev):
        raise OSError("no microphone")

    return MicrophoneRecorder(
        sample_rate=16000,
        duration_seconds=0.4,
        record_fn=_boom,
    )


class _FakeSTT:
    """Minimal STT double: returns pre-configured transcription."""

    def __init__(self, text: str = "mock transcription", language: str = "hi", error: Exception | None = None):
        self._text = text
        self._language = language
        self._error = error
        self.last_path: str | None = None

    def transcribe(self, audio_path: str, *, language: str | None = None):
        from indicvoicerag.stt import TranscriptionResult
        self.last_path = audio_path
        if self._error is not None:
            raise self._error
        return TranscriptionResult(
            text=self._text,
            language=language or self._language,
            language_probability=0.95,
            duration_seconds=0.4,
            processing_time_ms=120.0,
            rtf=0.3,
            model="mock-stt",
            load_time_ms=0.0,
            audio_path=audio_path,
        )


class _FakeHarness:
    """Minimal harness double: returns pre-configured RAGResponse."""

    def __init__(self, response: RAGResponse | None = None, error: Exception | None = None):
        self._response = response or RAGResponse(
            query="mock",
            answer="उत्तर",
            grounded=True,
            confidence=0.9,
            sources=[SourceInfo(document_id="d1", chunk_id="d1:0", score=0.85, excerpt="पाठ")],
            metrics={"retrieval": 5.0, "context": 0.2, "generation": 3.0, "grounding": 0.4, "total": 8.6},
        )
        self._error = error
        self.last_query: str | None = None

    def answer(self, query: str, top_k: int | None = None, debug: bool = False):
        self.last_query = query
        if self._error is not None:
            raise self._error
        return self._response


class _FakeTTS:
    """Minimal TTS double: returns pre-configured TTSResult."""

    def __init__(self, error: Exception | None = None):
        self._error = error
        self.last_text: str | None = None

    def synthesize(self, text: str, *, language: str | None = None, output_path=None):
        from indicvoicerag.tts import TTSResult, write_tts_wav
        import numpy as np
        self.last_text = text
        if self._error is not None:
            raise self._error
        samples = (0.5 * np.sin(2 * np.pi * 440 * np.arange(16000) / 16000)).astype(np.float32)
        out_path = str(output_path) if output_path else "tmp/mock_answer.wav"
        write_tts_wav(out_path, samples, 16000)
        return TTSResult(
            text=text,
            language=language or "hi",
            sample_rate=16000,
            duration_seconds=1.0,
            synthesis_time_ms=200.0,
            rtf=0.2,
            model="mock-tts-hi",
            provider="mock",
            load_time_ms=0.0,
            output_path=out_path,
            audio=samples,
        )


class _RefusalHarness:
    """Harness that returns a guardrail refusal."""

    def __init__(self):
        self.last_query: str | None = None

    def answer(self, query: str, top_k: int | None = None, debug: bool = False):
        self.last_query = query
        return RAGResponse(
            query=query,
            answer="",
            grounded=False,
            confidence=0.0,
            sources=[],
            guardrail="low_relevance",
            reason="retrieval scores too low",
            metrics={"retrieval": 2.0, "context": 0.1, "generation": 0.0, "grounding": 0.0, "total": 2.1},
        )


# --- run_voice_turn tests ---

def test_voice_turn_success(tmp_path: Path) -> None:
    tts_out = tmp_path / "answer.wav"
    result = run_voice_turn(
        stt=_FakeSTT("नमस्ते भारत"),
        harness=_FakeHarness(),
        tts=_FakeTTS(),
        recorder=_recorder(),
        language="hi",
        tts_output=str(tts_out),
    )
    assert result.transcription == "नमस्ते भारत"
    assert result.detected_language == "hi"
    assert result.rag_response is not None
    assert result.rag_response.answer == "उत्तर"
    assert result.tts_audio_path is not None
    assert Path(result.tts_audio_path).exists()
    assert result.tts_duration_seconds == pytest.approx(1.0, abs=0.01)
    assert result.error is None
    assert result.total_post_recording_ms > 0
    assert result.rag_total_ms > 0


def test_voice_turn_stt_refusal_skips_tts(tmp_path: Path) -> None:
    """Guardrail refusal -> no TTS audio generated."""
    result = run_voice_turn(
        stt=_FakeSTT("some query"),
        harness=_RefusalHarness(),
        tts=_FakeTTS(),
        recorder=_recorder(),
        language="hi",
        tts_output=str(tmp_path / "should_not_exist.wav"),
    )
    assert result.rag_response is not None
    assert result.rag_response.guardrail == "low_relevance"
    assert result.tts_audio_path is None
    assert result.error is None


def test_voice_turn_empty_transcript(tmp_path: Path) -> None:
    result = run_voice_turn(
        stt=_FakeSTT(""),
        harness=_FakeHarness(),
        tts=_FakeTTS(),
        recorder=_recorder(),
        language="hi",
    )
    assert result.error == "empty transcription (no speech detected)"
    assert result.error_stage == "stt"
    assert result.tts_audio_path is None


def test_voice_turn_stt_failure(tmp_path: Path) -> None:
    result = run_voice_turn(
        stt=_FakeSTT(error=STTError("model corrupted")),
        harness=_FakeHarness(),
        tts=_FakeTTS(),
        recorder=_recorder(),
        language="hi",
    )
    assert result.error == "model corrupted"
    assert result.error_stage == "stt"
    assert result.tts_audio_path is None


def test_voice_turn_rag_failure(tmp_path: Path) -> None:
    result = run_voice_turn(
        stt=_FakeSTT("query"),
        harness=_FakeHarness(error=RuntimeError("LLM timeout")),
        tts=_FakeTTS(),
        recorder=_recorder(),
        language="hi",
    )
    assert "LLM timeout" in (result.error or "")
    assert result.error_stage == "rag"
    assert result.tts_audio_path is None


def test_voice_turn_tts_failure(tmp_path: Path) -> None:
    result = run_voice_turn(
        stt=_FakeSTT("query"),
        harness=_FakeHarness(),
        tts=_FakeTTS(error=TTSSynthesisError("model failed")),
        recorder=_recorder(),
        language="hi",
        tts_output=str(tmp_path / "fail.wav"),
    )
    assert "model failed" in (result.error or "")
    assert result.error_stage == "tts"
    assert result.tts_audio_path is None


def test_voice_turn_mic_failure() -> None:
    result = run_voice_turn(
        stt=_FakeSTT(),
        harness=_FakeHarness(),
        tts=_FakeTTS(),
        recorder=_failing_recorder(),
        language="hi",
    )
    assert "no microphone" in (result.error or "")
    assert result.error_stage == "mic"


def test_voice_turn_empty_recording() -> None:
    result = run_voice_turn(
        stt=_FakeSTT(),
        harness=_FakeHarness(),
        tts=_FakeTTS(),
        recorder=_empty_recorder(),
        language="hi",
    )
    assert "empty recording" in (result.error or "")
    assert result.error_stage == "mic"


def test_voice_turn_no_rag(tmp_path: Path) -> None:
    """STT-only mode (no RAG) still runs TTS on transcription."""
    result = run_voice_turn(
        stt=_FakeSTT("just transcription"),
        harness=None,
        tts=_FakeTTS(),
        recorder=_recorder(),
        language="hi",
        tts_output=str(tmp_path / "stt_only.wav"),
    )
    assert result.transcription == "just transcription"
    assert result.rag_response is None
    assert result.tts_audio_path is not None
    assert result.error is None


def test_voice_turn_result_schema() -> None:
    result = VoiceTurnResult(
        transcription="test",
        detected_language="hi",
        language_probability=0.9,
        stt_processing_ms=100.0,
        stt_rtf=0.5,
        rag_total_ms=50.0,
        tts_synthesis_ms=200.0,
        total_post_recording_ms=350.0,
    )
    d = result.as_dict()
    assert d["transcription"] == "test"
    assert d["detected_language"] == "hi"
    assert d["rag"] is None
    assert d["tts"] is None
    assert "error" not in d
    assert d["latency"]["stt_ms"] == 100.0
    assert d["latency"]["total_post_recording_ms"] == 350.0


def test_voice_turn_result_schema_with_error() -> None:
    result = VoiceTurnResult(error="boom", error_stage="stt")
    d = result.as_dict()
    assert d["error"] == "boom"
    assert d["error_stage"] == "stt"
    assert d["rag"] is None
    assert d["tts"] is None


def test_voice_turn_result_schema_with_tts(tmp_path: Path) -> None:
    from indicvoicerag.tts import write_tts_wav
    samples = np.zeros(16000, dtype=np.float32)
    out = tmp_path / "test.wav"
    write_tts_wav(out, samples, 16000)
    result = VoiceTurnResult(
        transcription="test",
        tts_audio_path=str(out),
        tts_duration_seconds=1.0,
        tts_sample_rate=16000,
        tts_synthesis_ms=200.0,
        tts_rtf=0.2,
    )
    d = result.as_dict()
    assert d["tts"]["audio_path"] == str(out)
    assert d["tts"]["duration_seconds"] == 1.0
    assert d["tts"]["rtf"] == 0.2


# --- run_voice (CLI) tests ---

def test_cli_voice_success_mocked(tmp_path: Path, monkeypatch, capsys) -> None:
    monkeypatch.setattr(builtins, "input", lambda *a: "")
    cfg = _config()
    harness = _FakeHarness()
    stt = _FakeSTT("नमस्ते")
    tts = _FakeTTS()

    rc = run_voice(
        cfg,
        language="hi",
        recorder=_recorder(),
        stt=stt,
        harness=harness,
        tts=tts,
        tts_output=str(tmp_path / "out.wav"),
        as_json=True,
    )

    out = json.loads(capsys.readouterr().out)
    assert rc == 0
    assert out["transcription"] == "नमस्ते"
    assert out["rag"]["answer"] == "उत्तर"
    assert out["tts"] is not None
    assert out["tts"]["duration_seconds"] == pytest.approx(1.0, abs=0.01)


def test_cli_voice_success_human_output(monkeypatch, capsys) -> None:
    monkeypatch.setattr(builtins, "input", lambda *a: "")
    cfg = _config()
    harness = _FakeHarness()
    stt = _FakeSTT("query")
    tts = _FakeTTS()

    rc = run_voice(
        cfg,
        language="hi",
        recorder=_recorder(),
        stt=stt,
        harness=harness,
        tts=tts,
    )

    captured = capsys.readouterr().out
    assert rc == 0
    assert "Transcription: query" in captured
    assert "RAG answer:" in captured
    assert "TTS output:" in captured
    assert "total post-recording" in captured


def test_cli_voice_stt_error(monkeypatch, capsys) -> None:
    monkeypatch.setattr(builtins, "input", lambda *a: "")
    cfg = _config()

    rc = run_voice(
        cfg,
        language="hi",
        recorder=_recorder(),
        stt=_FakeSTT(error=STTError("model broken")),
        harness=_FakeHarness(),
        tts=_FakeTTS(),
        as_json=True,
    )

    out = json.loads(capsys.readouterr().out)
    assert rc == 1
    assert out["error_stage"] == "stt"
    assert "model broken" in out["error"]


def test_cli_voice_rag_refusal_no_audio(monkeypatch, capsys) -> None:
    monkeypatch.setattr(builtins, "input", lambda *a: "")
    cfg = _config()

    rc = run_voice(
        cfg,
        language="hi",
        recorder=_recorder(),
        stt=_FakeSTT("query"),
        harness=_RefusalHarness(),
        tts=_FakeTTS(),
        as_json=True,
    )

    out = json.loads(capsys.readouterr().out)
    assert rc == 0
    assert out["rag"]["guardrail"] == "low_relevance"
    assert out["tts"] is None


def test_cli_voice_no_rag_mode(monkeypatch, capsys) -> None:
    monkeypatch.setattr(builtins, "input", lambda *a: "")
    cfg = _config()

    rc = run_voice(
        cfg,
        language="hi",
        recorder=_recorder(),
        stt=_FakeSTT("transcription"),
        harness=None,
        tts=_FakeTTS(),
        no_rag=True,
        as_json=True,
    )

    out = json.loads(capsys.readouterr().out)
    assert rc == 0
    assert out["transcription"] == "transcription"
    assert out["rag"] is None
    assert out["tts"] is not None


def test_cli_voice_cancelled(monkeypatch, capsys) -> None:
    monkeypatch.setattr(builtins, "input", lambda *_: (_ for _ in ()).throw(KeyboardInterrupt()))
    cfg = _config()

    rc = run_voice(cfg, language="hi", recorder=_recorder(), stt=_FakeSTT(), tts=_FakeTTS())
    assert rc == 1


def test_cli_voice_cancelled_before_recording(monkeypatch, capsys) -> None:
    """KeyboardInterrupt at the Enter prompt -> exit code 1."""
    call_count = {"n": 0}

    def _interrupt_once(*a):
        call_count["n"] += 1
        if call_count["n"] == 1:
            raise KeyboardInterrupt()
        return ""

    monkeypatch.setattr(builtins, "input", _interrupt_once)
    cfg = _config()

    rc = run_voice(cfg, language="hi", recorder=_recorder(), stt=_FakeSTT(), tts=_FakeTTS())
    assert rc == 1
