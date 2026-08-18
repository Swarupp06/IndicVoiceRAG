"""Phase 3C CLI integration tests for the `listen` command.

The microphone, STT model and RAG pipeline are all replaced with offline
doubles so these tests run without hardware, downloads or network. They assert
the control flow: success path, latency reporting, error boundaries and the
temporary-WAV cleanup contract.
"""

from __future__ import annotations

import builtins
import json
import tempfile
from pathlib import Path

import numpy as np
import pytest

from indicvoicerag.audio_capture import MicrophoneRecorder, MicrophoneUnavailableError
from indicvoicerag.cli import run_listen
from indicvoicerag.config import AppConfig
from indicvoicerag.rag_types import RAGResponse, SourceInfo
from indicvoicerag.stt import STTError


def _sine(duration: float = 0.4, rate: int = 16000) -> tuple[np.ndarray, int]:
    t = np.arange(int(duration * rate)) / rate
    return (0.5 * np.sin(2 * np.pi * 440 * t)).astype(np.float32), rate


def _config() -> AppConfig:
    cfg = AppConfig()
    cfg.stt.provider = "mock"
    cfg.audio.duration_seconds = 0.4
    return cfg


def _recorder() -> MicrophoneRecorder:
    return MicrophoneRecorder(
        sample_rate=16000,
        duration_seconds=0.4,
        record_fn=lambda d, sr, ch, dev: _sine(d, sr or 16000),
    )


def _response(query: str) -> RAGResponse:
    return RAGResponse(
        query=query,
        answer="उत्तर",
        grounded=True,
        confidence=0.9,
        sources=[SourceInfo(document_id="d1", chunk_id="d1:0", score=0.85, excerpt="पाठ")],
        metrics={"retrieval": 5.0, "context": 0.2, "generation": 3.0, "grounding": 0.4, "total": 8.6},
        llm={"provider": "mock", "model": "mock-rag-generator", "latency_ms": 3.0, "usage": {}},
    )


class _FakeHarness:
    def __init__(self, response: RAGResponse):
        self._response = response
        self.last_query: str | None = None

    def answer(self, query: str, top_k: int | None = None, debug: bool = False) -> RAGResponse:
        self.last_query = query
        return self._response


def test_listen_end_to_end_mocked(monkeypatch, capsys) -> None:
    monkeypatch.setattr(builtins, "input", lambda *a: "")
    harness = _FakeHarness(_response("mock transcription"))

    rc = run_listen(_config(), language="hi", recorder=_recorder(), harness=harness, as_json=True)

    out = json.loads(capsys.readouterr().out)
    assert rc == 0
    assert out["transcription"] == "mock transcription"
    assert out["language"] == "hi"
    assert out["recording_duration_seconds"] == pytest.approx(0.4, abs=0.02)
    assert out["stt"]["processing_ms"] >= 0
    assert out["stt"]["rtf"] >= 0
    assert out["rag"]["answer"] == "उत्तर"
    assert out["rag"]["grounded"] is True
    assert out["rag_total_ms"] == pytest.approx(8.6, abs=0.1)
    assert out["total_post_recording_ms"] > 0
    assert harness.last_query == "mock transcription"


def test_listen_no_rag_skips_harness(monkeypatch, capsys) -> None:
    monkeypatch.setattr(builtins, "input", lambda *a: "")

    rc = run_listen(_config(), language="hi", recorder=_recorder(), no_rag=True, as_json=True)

    out = json.loads(capsys.readouterr().out)
    assert rc == 0
    assert out["transcription"] == "mock transcription"
    assert out["rag"] is None
    assert out["rag_total_ms"] == 0.0


def test_listen_cleans_temp_wav(monkeypatch) -> None:
    monkeypatch.setattr(builtins, "input", lambda *a: "")
    before = set(Path(tempfile.gettempdir()).glob("ivr_listen_*.wav"))

    rc = run_listen(_config(), language="hi", recorder=_recorder(), harness=_FakeHarness(_response("x")))

    after = set(Path(tempfile.gettempdir()).glob("ivr_listen_*.wav"))
    assert rc == 0
    assert after == before


def test_listen_debug_keeps_temp_wav(monkeypatch, capsys) -> None:
    monkeypatch.setattr(builtins, "input", lambda *a: "")

    rc = run_listen(
        _config(), language="hi", recorder=_recorder(), harness=_FakeHarness(_response("x")),
        debug=True, as_json=True,
    )

    out = json.loads(capsys.readouterr().out)
    assert rc == 0
    assert out["temp_wav"]
    kept = Path(out["temp_wav"])
    assert kept.exists()
    kept.unlink(missing_ok=True)


def test_listen_no_microphone_is_controlled_failure(monkeypatch, capsys) -> None:
    monkeypatch.setattr(builtins, "input", lambda *a: "")

    def no_mic(duration: float, sample_rate: int | None, channels: int, device: int | None):
        raise MicrophoneUnavailableError("no input device found")

    rc = run_listen(_config(), recorder=MicrophoneRecorder(record_fn=no_mic), harness=_FakeHarness(_response("x")))

    assert rc == 1
    assert "no input device found" in capsys.readouterr().err


def test_listen_empty_capture_is_controlled_failure(monkeypatch, capsys) -> None:
    monkeypatch.setattr(builtins, "input", lambda *a: "")
    empty = MicrophoneRecorder(record_fn=lambda d, sr, ch, dev: (np.array([], dtype=np.float32), sr or 16000))

    rc = run_listen(_config(), recorder=empty, harness=_FakeHarness(_response("x")))

    assert rc == 1
    assert "captured no audio" in capsys.readouterr().err


def test_listen_stt_failure_is_controlled(monkeypatch, capsys) -> None:
    monkeypatch.setattr(builtins, "input", lambda *a: "")

    class _FailSTT:
        def transcribe(self, audio_path, language=None):
            raise STTError("transcriber exploded")

    monkeypatch.setattr("indicvoicerag.cli._build_stt_for_listen", lambda config, model, provider=None: _FailSTT())

    rc = run_listen(_config(), recorder=_recorder(), harness=_FakeHarness(_response("x")))

    assert rc == 1
    assert "speech-to-text failed" in capsys.readouterr().err


def test_listen_cancelled_before_recording(monkeypatch, capsys) -> None:
    monkeypatch.setattr(builtins, "input", lambda *a: (_ for _ in ()).throw(KeyboardInterrupt()))

    rc = run_listen(_config(), recorder=_recorder(), harness=_FakeHarness(_response("x")))

    assert rc == 1
    assert "cancelled" in capsys.readouterr().err


def test_listen_noninteractive_stdin_still_records(monkeypatch, capsys) -> None:
    monkeypatch.setattr(builtins, "input", lambda *a: (_ for _ in ()).throw(EOFError()))

    rc = run_listen(_config(), language="hi", recorder=_recorder(), harness=_FakeHarness(_response("x")))

    assert rc == 0
    assert "mock transcription" in capsys.readouterr().out
