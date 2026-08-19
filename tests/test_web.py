"""Tests for the minimal web demo layer."""

from __future__ import annotations

import wave
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient


def _write_wav(path: Path, seconds: float = 0.5, rate: int = 16000) -> Path:
    sample_frames = int(seconds * rate)
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(rate)
        handle.writeframes(b"\x00\x00" * sample_frames)
    return path


class TestHealthEndpoint:
    def test_health_returns_ok(self) -> None:
        from indicvoicerag.web import app
        client = TestClient(app)
        resp = client.get("/health")
        assert resp.status_code == 200
        assert resp.json() == {"status": "ok"}


class TestIndexEndpoint:
    def test_index_returns_html(self) -> None:
        from indicvoicerag.web import app
        client = TestClient(app)
        resp = client.get("/")
        assert resp.status_code == 200
        assert "IndicVoiceRAG" in resp.text


class TestAskEndpointValidation:
    @patch("indicvoicerag.web._get_state")
    def test_empty_audio_returns_400(self, mock_state: MagicMock) -> None:
        from indicvoicerag.web import app
        client = TestClient(app)
        resp = client.post(
            "/api/ask",
            files={"audio": ("empty.wav", b"", "audio/wav")},
            data={"stt_provider": "mock", "language": "hi-IN"},
        )
        assert resp.status_code == 400

    @patch("indicvoicerag.web._get_state")
    def test_missing_audio_returns_422(self, mock_state: MagicMock) -> None:
        from indicvoicerag.web import app
        client = TestClient(app)
        resp = client.post("/api/ask", data={"stt_provider": "mock"})
        assert resp.status_code == 422


class TestAskEndpointWithMockPipeline:
    @patch("indicvoicerag.web._get_state")
    def test_valid_audio_with_mock_stt_returns_result(self, mock_state: MagicMock, tmp_path: Path) -> None:
        from indicvoicerag.web import app
        from indicvoicerag.stt import MockSTT
        from indicvoicerag.rag_types import RAGResponse, SourceInfo

        audio = _write_wav(tmp_path / "test.wav", seconds=0.5)

        mock_stt = MockSTT(text="test transcription", language="hi", language_probability=0.95)

        mock_harness = MagicMock()
        mock_harness.answer.return_value = RAGResponse(
            query="test transcription",
            answer="test answer from RAG",
            grounded=True,
            confidence=0.85,
            sources=[SourceInfo(document_id="doc1", chunk_id="c1", score=0.9, excerpt="test excerpt")],
        )

        mock_tts = MagicMock()
        tts_result = MagicMock()
        tts_result.as_dict.return_value = {"text": "test answer", "provider": "mock"}
        tts_result.output_path = str(tmp_path / "tts_out.wav")
        tts_result.synthesis_time_ms = 100.0
        mock_tts.synthesize.return_value = tts_result

        mock_state.return_value = {
            "config": MagicMock(),
            "harness": mock_harness,
            "stt": mock_stt,
            "tts": mock_tts,
        }

        # Write a dummy TTS output file
        _write_wav(tmp_path / "tts_out.wav", seconds=0.5)

        client = TestClient(app)
        resp = client.post(
            "/api/ask",
            files={"audio": ("test.wav", audio.read_bytes(), "audio/wav")},
            data={"stt_provider": "mock", "language": "hi-IN"},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["transcription"] == "test transcription"
        assert data["language"] is not None
        assert data["rag"]["answer"] == "test answer from RAG"
        assert data["rag"]["grounded"] is True
        assert data["rag"]["confidence"] == 0.85
        assert len(data["rag"]["sources"]) == 1
        assert "stt_ms" in data
        assert "rag_total_ms" in data
        assert "total_ms" in data

    @patch("indicvoicerag.web._get_state")
    def test_stt_failure_returns_422(self, mock_state: MagicMock, tmp_path: Path) -> None:
        from indicvoicerag.web import app
        from indicvoicerag.stt import STTError

        audio = _write_wav(tmp_path / "fail.wav", seconds=0.5)

        mock_stt = MagicMock()
        mock_stt.transcribe.side_effect = STTError("STT exploded")

        mock_state.return_value = {
            "config": MagicMock(),
            "harness": MagicMock(),
            "stt": mock_stt,
            "tts": MagicMock(),
        }

        client = TestClient(app)
        resp = client.post(
            "/api/ask",
            files={"audio": ("fail.wav", audio.read_bytes(), "audio/wav")},
            data={"stt_provider": "mock", "language": "hi-IN"},
        )
        assert resp.status_code == 422
        data = resp.json()
        assert "STT failed" in data["error"]
        assert data["error_stage"] == "stt"

    @patch("indicvoicerag.web._get_state")
    def test_no_secrets_in_response(self, mock_state: MagicMock, tmp_path: Path) -> None:
        from indicvoicerag.web import app
        from indicvoicerag.stt import MockSTT
        from indicvoicerag.rag_types import RAGResponse

        audio = _write_wav(tmp_path / "sec.wav", seconds=0.5)
        mock_stt = MockSTT(text="query", language="hi")
        mock_harness = MagicMock()
        mock_harness.answer.return_value = RAGResponse(
            query="query", answer="safe answer", grounded=True, confidence=0.9,
        )
        mock_tts = MagicMock()
        tts_result = MagicMock()
        tts_result.as_dict.return_value = {"text": "safe answer", "provider": "mock"}
        tts_result.output_path = str(tmp_path / "tts_out.wav")
        tts_result.synthesis_time_ms = 100.0
        mock_tts.synthesize.return_value = tts_result
        _write_wav(tmp_path / "tts_out.wav", seconds=0.5)

        mock_state.return_value = {
            "config": MagicMock(),
            "harness": mock_harness,
            "stt": mock_stt,
            "tts": mock_tts,
        }

        client = TestClient(app)
        resp = client.post(
            "/api/ask",
            files={"audio": ("sec.wav", audio.read_bytes(), "audio/wav")},
            data={"stt_provider": "mock", "language": "hi-IN"},
        )
        assert resp.status_code == 200
        body = resp.text
        assert "sk_" not in body
        assert "gsk_" not in body
        assert "SARVAM_API_KEY" not in body
        assert "GROQ_API_KEY" not in body


class TestTTSServing:
    def test_missing_tts_file_returns_404(self) -> None:
        from indicvoicerag.web import app
        client = TestClient(app)
        resp = client.get("/api/tts/nonexistent.wav")
        assert resp.status_code == 404


class TestConfigSelection:
    def test_load_config_with_defaults_when_env_unset(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("INDICVOICERAG_CONFIG", raising=False)
        from indicvoicerag import web

        with patch("indicvoicerag.web.load_config") as mock_load:
            mock_load.return_value = MagicMock()
            web._load_app_config()

        mock_load.assert_called_once_with()

    def test_load_config_uses_env_selected_path(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("INDICVOICERAG_CONFIG", "config.codespaces.toml")
        from indicvoicerag import web

        with patch("indicvoicerag.web.load_config") as mock_load:
            mock_load.return_value = MagicMock()
            web._load_app_config()

        mock_load.assert_called_once_with("config.codespaces.toml")

    def test_config_selection_does_not_expose_secrets(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Config selection forwards only the config path; API key values must
        never be read, stored, or serialized by the selection logic."""
        from dataclasses import asdict
        import json

        config_path = Path(__file__).resolve().parents[1] / "config.codespaces.toml"
        monkeypatch.setenv("INDICVOICERAG_CONFIG", str(config_path))
        monkeypatch.setenv("GROQ_API_KEY", "gsk_groq_secret_sentinel")
        monkeypatch.setenv("SARVAM_API_KEY", "sarvam_secret_sentinel")
        from indicvoicerag import web

        cfg = web._load_app_config()

        assert cfg.llm.api_key_env == "GROQ_API_KEY"
        assert cfg.stt.api_key_env == "SARVAM_API_KEY"
        serialized = json.dumps(asdict(cfg), ensure_ascii=False)
        assert "gsk_groq_secret_sentinel" not in serialized
        assert "sarvam_secret_sentinel" not in serialized
