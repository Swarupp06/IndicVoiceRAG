"""Offline unit tests for the Phase 3A STT abstraction.

No test downloads a model: faster-whisper behavior is faked with an injected
model factory, and MockSTT covers the offline paths.
"""

from __future__ import annotations

import wave
from pathlib import Path

import pytest

from indicvoicerag.audio import AudioError, probe_audio
from indicvoicerag.config import STTConfig
from indicvoicerag.stt import (
    FasterWhisperSTT,
    MockSTT,
    STTAudioError,
    STTConfigurationError,
    STTError,
    STTTranscriptionError,
    TranscriptionResult,
    build_stt_provider,
    normalize_stt_provider_name,
)


def _write_wav(path: Path, seconds: float = 0.5, frames: int | None = None, rate: int = 16000) -> Path:
    sample_frames = frames if frames is not None else int(seconds * rate)
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(rate)
        handle.writeframes(b"\x00\x00" * sample_frames)
    return path


class _FakeInfo:
    language = "hi"
    language_probability = 0.91
    duration = 1.25


class _FakeSegment:
    def __init__(self, text: str, start: float = 0.0, end: float = 1.0):
        self.text = text
        self.start = start
        self.end = end


class _FakeWhisperModel:
    """Stands in for faster_whisper.WhisperModel; records the request."""

    def __init__(self, segments=None, info=None, error: Exception | None = None):
        self._segments = segments
        self._info = info
        self._error = error
        self.calls: list[tuple[object, dict]] = []

    def transcribe(self, audio, **kwargs):
        self.calls.append((audio, kwargs))
        if self._error is not None:
            raise self._error
        return iter(self._segments or []), self._info


# --- structured result validation ---
def test_transcription_result_as_dict_schema() -> None:
    result = TranscriptionResult(
        text="नमस्ते",
        language="hi",
        language_probability=0.91,
        duration_seconds=1.5,
        processing_time_ms=250.0,
        model="tiny",
        load_time_ms=100.0,
        audio_path="samples/hi.wav",
    )
    payload = result.as_dict()
    assert set(payload) == {
        "text",
        "language",
        "language_probability",
        "duration_seconds",
        "processing_time_ms",
        "rtf",
        "model",
        "load_time_ms",
        "audio_path",
    }
    assert payload["text"] == "नमस्ते"
    assert payload["language"] == "hi"
    assert payload["rtf"] > 0


def test_rtf_is_real_time_factor() -> None:
    result = TranscriptionResult(
        text="x", duration_seconds=2.0, processing_time_ms=500.0
    )
    assert result.rtf == pytest.approx(0.25)


# --- valid transcription ---
def test_mock_stt_valid_transcription(tmp_path: Path) -> None:
    audio = _write_wav(tmp_path / "valid.wav")
    stt = MockSTT(text="What is a corporation?", language_probability=0.98)
    result = stt.transcribe(str(audio))
    assert result.text == "What is a corporation?"
    assert result.language == "en"
    assert result.language_probability == pytest.approx(0.98)
    assert result.duration_seconds == pytest.approx(0.5)
    assert result.model == "mock-stt"
    assert result.processing_time_ms >= 0
    assert result.rtf >= 0


def test_mock_stt_empty_behavior_returns_empty_text(tmp_path: Path) -> None:
    audio = _write_wav(tmp_path / "quiet.wav")
    stt = MockSTT(behavior="empty")
    result = stt.transcribe(str(audio))
    assert result.text == ""
    assert result.duration_seconds == pytest.approx(0.5)


# --- empty audio ---
def test_empty_audio_raises(tmp_path: Path) -> None:
    audio = _write_wav(tmp_path / "empty.wav", frames=0)
    stt = MockSTT()
    with pytest.raises(STTAudioError):
        stt.transcribe(str(audio))


# --- invalid audio ---
def test_invalid_audio_raises(tmp_path: Path) -> None:
    audio = tmp_path / "garbage.wav"
    audio.write_bytes(b"this is not an audio file at all\x00\x01\x02")
    stt = MockSTT()
    with pytest.raises(STTAudioError):
        stt.transcribe(str(audio))


def test_missing_audio_raises(tmp_path: Path) -> None:
    stt = MockSTT()
    with pytest.raises(STTAudioError):
        stt.transcribe(str(tmp_path / "nope.wav"))


# --- provider failure ---
def test_provider_failure_raises(tmp_path: Path) -> None:
    audio = _write_wav(tmp_path / "ok.wav")
    stt = MockSTT(behavior="raise")
    with pytest.raises(STTTranscriptionError):
        stt.transcribe(str(audio))


# --- language handling ---
def test_language_override(tmp_path: Path) -> None:
    audio = _write_wav(tmp_path / "lang.wav")
    stt = MockSTT(text="बैंकॉक में कल का मौसम", language="hi")
    result = stt.transcribe(str(audio), language="hi")
    assert result.language == "hi"

    # explicit hint wins over the provider default
    stt_default_en = MockSTT(text="hello", language="en")
    hinted = stt_default_en.transcribe(str(audio), language="hi")
    assert hinted.language == "hi"


# --- faster-whisper wiring with an injected fake model ---
def test_faster_whisper_maps_segments_to_result(tmp_path: Path) -> None:
    audio = _write_wav(tmp_path / "fw.wav")
    fake = _FakeWhisperModel(
        segments=[_FakeSegment("Hello world", 0.0, 0.9)],
        info=_FakeInfo(),
    )
    stt = FasterWhisperSTT(model_name="tiny", compute_type="int8", model_factory=lambda: fake)
    result = stt.transcribe(str(audio))
    assert result.text == "Hello world"
    assert result.language == "hi"
    assert result.language_probability == pytest.approx(0.91)
    assert result.duration_seconds == pytest.approx(1.25)
    assert result.model == "tiny"
    assert result.load_time_ms is not None
    assert fake.calls[0][1]["language"] is None


def test_faster_whisper_passes_language_hint(tmp_path: Path) -> None:
    audio = _write_wav(tmp_path / "fw_hint.wav")
    fake = _FakeWhisperModel(segments=[], info=_FakeInfo())
    stt = FasterWhisperSTT(model_name="tiny", model_factory=lambda: fake)
    stt.transcribe(str(audio), language="en")
    assert fake.calls[0][1]["language"] == "en"


def test_faster_whisper_empty_language_config_means_auto_detect(tmp_path: Path) -> None:
    """language='' from a TOML config must behave like None (auto-detect)."""
    audio = _write_wav(tmp_path / "fw_empty.wav")
    fake = _FakeWhisperModel(segments=[], info=_FakeInfo())
    stt = FasterWhisperSTT(model_name="tiny", language="", model_factory=lambda: fake)
    stt.transcribe(str(audio))
    assert fake.calls[0][1]["language"] is None


def test_faster_whisper_provider_failure(tmp_path: Path) -> None:
    audio = _write_wav(tmp_path / "fw_err.wav")
    fake = _FakeWhisperModel(error=RuntimeError("model exploded"))
    stt = FasterWhisperSTT(model_name="tiny", model_factory=lambda: fake)
    with pytest.raises(STTTranscriptionError):
        stt.transcribe(str(audio))


def test_faster_whisper_rejects_invalid_compute_type() -> None:
    with pytest.raises(STTConfigurationError):
        FasterWhisperSTT(compute_type="half8")


def test_faster_whisper_audio_error_before_model_load(tmp_path: Path) -> None:
    # the audio file is validated before the model is ever loaded
    stt = FasterWhisperSTT(model_name="tiny")
    with pytest.raises(STTAudioError):
        stt.transcribe(str(tmp_path / "nope.wav"))


def test_faster_whisper_model_loaded_once_and_reused_across_turns(tmp_path: Path) -> None:
    """Phase 3C.1 regression: one provider instance must build the model exactly
    once and reuse it for every transcribe call (no per-request reload)."""
    audio = _write_wav(tmp_path / "fw_reuse.wav")
    fake = _FakeWhisperModel(segments=[_FakeSegment("नमस्ते", 0.0, 0.5)], info=_FakeInfo())
    calls = {"builds": 0, "instance": None}

    def factory():
        calls["builds"] += 1
        calls["instance"] = fake
        return fake

    stt = FasterWhisperSTT(model_name="tiny", model_factory=factory)

    first = stt.transcribe(str(audio))
    second = stt.transcribe(str(audio))
    third = stt.transcribe(str(audio))

    assert calls["builds"] == 1
    assert len(fake.calls) == 3
    assert stt.model_load_seconds is not None
    assert first.load_time_ms == pytest.approx(stt.model_load_seconds * 1000.0, abs=1.0)
    assert second.load_time_ms == first.load_time_ms  # load is not repeated
    assert third.load_time_ms == first.load_time_ms
    assert stt._model is fake  # noqa: SLF001 - the cached instance is the factory's model


# --- builder ---
def test_build_stt_provider_mock() -> None:
    provider = build_stt_provider(provider="mock")
    assert isinstance(provider, MockSTT)


def test_build_stt_provider_faster_whisper() -> None:
    provider = build_stt_provider(provider="faster_whisper", model_name="tiny")
    assert isinstance(provider, FasterWhisperSTT)
    assert provider.model_name == "tiny"
    assert provider.device == "cpu"
    assert provider.compute_type == "int8"


def test_build_stt_provider_unknown() -> None:
    with pytest.raises(ValueError):
        build_stt_provider(provider="skynet")


def test_normalize_stt_provider_name() -> None:
    assert normalize_stt_provider_name("faster-whisper") == "faster_whisper"
    assert normalize_stt_provider_name("Mock") == "mock"


def test_stt_config_defaults_to_real_local_provider() -> None:
    cfg = STTConfig()
    assert cfg.provider == "faster_whisper"
    assert cfg.device == "cpu"


# --- audio probing (low level) ---
def test_probe_audio_metadata(tmp_path: Path) -> None:
    audio = _write_wav(tmp_path / "meta.wav", seconds=1.0, rate=8000)
    info = probe_audio(audio)
    assert info["duration_seconds"] == pytest.approx(1.0)
    assert info["sample_rate"] == 8000
    assert info["channels"] == 1


def test_probe_audio_rejects_garbage(tmp_path: Path) -> None:
    audio = tmp_path / "bad.wav"
    audio.write_bytes(b"garbage")
    with pytest.raises(AudioError):
        probe_audio(audio)


def test_all_stt_errors_share_base() -> None:
    assert issubclass(STTAudioError, STTError)
    assert issubclass(STTTranscriptionError, STTError)
    assert issubclass(STTConfigurationError, STTError)


# --- Sarvam STT provider tests (mocked, no real API calls) ---
from unittest.mock import MagicMock, patch

from indicvoicerag.stt import SarvamSTT


class TestSarvamSTT:
    """Mocked unit tests for the Sarvam STT provider."""

    def test_sarvam_stt_transcribe_hindi(self, tmp_path: Path) -> None:
        """Test Hindi transcription with mocked Sarvam API."""
        audio = _write_wav(tmp_path / "hindi.wav", seconds=1.0)
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "request_id": "test-123",
            "transcript": "नमस्ते, आप कैसे हैं?",
            "language_code": "hi-IN",
            "language_probability": 0.95,
        }

        provider = SarvamSTT(api_key_env="SARVAM_API_KEY_TEST", model="saaras:v3", language_code="hi-IN")
        with patch("indicvoicerag.stt.os.environ", {"SARVAM_API_KEY_TEST": "test_key_123"}):
            with patch("indicvoicerag.stt.httpx.Client") as mock_client_cls:
                mock_client = MagicMock()
                mock_client_cls.return_value.__enter__.return_value = mock_client
                mock_client.post.return_value = mock_response
                result = provider.transcribe(audio)

        assert result.text == "नमस्ते, आप कैसे हैं?"
        assert result.language == "hi-IN"
        assert result.language_probability == 0.95
        assert result.model == "saaras:v3"
        assert result.processing_time_ms > 0

    def test_sarvam_stt_transcribe_english(self, tmp_path: Path) -> None:
        """Test English transcription with mocked Sarvam API."""
        audio = _write_wav(tmp_path / "english.wav", seconds=1.0)
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "request_id": "test-456",
            "transcript": "Hello, how are you?",
            "language_code": "en-IN",
            "language_probability": 0.92,
        }

        provider = SarvamSTT(api_key_env="SARVAM_API_KEY_TEST", model="saaras:v3", language_code="en-IN")
        with patch("indicvoicerag.stt.os.environ", {"SARVAM_API_KEY_TEST": "test_key_123"}):
            with patch("indicvoicerag.stt.httpx.Client") as mock_client_cls:
                mock_client = MagicMock()
                mock_client_cls.return_value.__enter__.return_value = mock_client
                mock_client.post.return_value = mock_response
                result = provider.transcribe(audio)

        assert result.text == "Hello, how are you?"
        assert result.language == "en-IN"

    def test_sarvam_stt_transcribe_with_metadata(self, tmp_path: Path) -> None:
        """Test that transcribe returns TranscriptionResult with all metadata."""
        audio = _write_wav(tmp_path / "meta.wav", seconds=2.0)
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "request_id": "test-789",
            "transcript": "यह एक परीक्षण है",
            "language_code": "hi-IN",
            "language_probability": 0.88,
        }

        provider = SarvamSTT(api_key_env="SARVAM_API_KEY_TEST")
        with patch("indicvoicerag.stt.os.environ", {"SARVAM_API_KEY_TEST": "test_key_123"}):
            with patch("indicvoicerag.stt.httpx.Client") as mock_client_cls:
                mock_client = MagicMock()
                mock_client_cls.return_value.__enter__.return_value = mock_client
                mock_client.post.return_value = mock_response
                result = provider.transcribe(audio)

        assert isinstance(result, TranscriptionResult)
        assert result.text == "यह एक परीक्षण है"
        assert result.duration_seconds == pytest.approx(2.0)
        assert result.processing_time_ms > 0
        assert result.audio_path is not None

    def test_sarvam_stt_available_with_key(self) -> None:
        """Test available() returns True when API key is set."""
        with patch("indicvoicerag.stt.os.environ", {"SARVAM_API_KEY_TEST": "test_key_123"}):
            provider = SarvamSTT(api_key_env="SARVAM_API_KEY_TEST")
            assert provider.available() is True

    def test_sarvam_stt_not_available_without_key(self) -> None:
        """Test available() returns False when API key is missing."""
        with patch("indicvoicerag.stt.os.environ", {}):
            provider = SarvamSTT(api_key_env="SARVAM_API_KEY_TEST")
            assert provider.available() is False

    def test_sarvam_stt_missing_key_raises_config_error(self, tmp_path: Path) -> None:
        """Test that missing API key raises STTConfigurationError."""
        audio = _write_wav(tmp_path / "nokey.wav", seconds=1.0)
        provider = SarvamSTT(api_key_env="SARVAM_API_KEY_TEST")
        with patch("indicvoicerag.stt.os.environ", {}):
            with pytest.raises(STTConfigurationError, match="Sarvam API key not found"):
                provider.transcribe(audio)

    def test_sarvam_stt_http_401_raises_auth_error(self, tmp_path: Path) -> None:
        """Test that HTTP 401 raises STTTranscriptionError."""
        audio = _write_wav(tmp_path / "auth.wav", seconds=1.0)
        mock_response = MagicMock()
        mock_response.status_code = 401
        mock_response.text = "Unauthorized"

        provider = SarvamSTT(api_key_env="SARVAM_API_KEY_TEST")
        with patch("indicvoicerag.stt.os.environ", {"SARVAM_API_KEY_TEST": "bad_key"}):
            with patch("indicvoicerag.stt.httpx.Client") as mock_client_cls:
                mock_client = MagicMock()
                mock_client_cls.return_value.__enter__.return_value = mock_client
                mock_client.post.return_value = mock_response
                with pytest.raises(STTTranscriptionError, match="authentication failed"):
                    provider.transcribe(audio)

    def test_sarvam_stt_http_500_raises_server_error(self, tmp_path: Path) -> None:
        """Test that HTTP 500 raises STTTranscriptionError."""
        audio = _write_wav(tmp_path / "server.wav", seconds=1.0)
        mock_response = MagicMock()
        mock_response.status_code = 500
        mock_response.text = "Internal Server Error"

        provider = SarvamSTT(api_key_env="SARVAM_API_KEY_TEST")
        with patch("indicvoicerag.stt.os.environ", {"SARVAM_API_KEY_TEST": "test_key"}):
            with patch("indicvoicerag.stt.httpx.Client") as mock_client_cls:
                mock_client = MagicMock()
                mock_client_cls.return_value.__enter__.return_value = mock_client
                mock_client.post.return_value = mock_response
                with pytest.raises(STTTranscriptionError, match="API error 500"):
                    provider.transcribe(audio)

    def test_sarvam_stt_timeout_raises_timeout_error(self, tmp_path: Path) -> None:
        """Test that timeout raises STTTranscriptionError."""
        import httpx as httpx_module
        audio = _write_wav(tmp_path / "timeout.wav", seconds=1.0)

        provider = SarvamSTT(api_key_env="SARVAM_API_KEY_TEST")
        with patch("indicvoicerag.stt.os.environ", {"SARVAM_API_KEY_TEST": "test_key"}):
            with patch("indicvoicerag.stt.httpx.Client") as mock_client_cls:
                mock_client = MagicMock()
                mock_client_cls.return_value.__enter__.return_value = mock_client
                mock_client.post.side_effect = httpx_module.TimeoutException("Request timed out")
                with pytest.raises(STTTranscriptionError, match="timed out"):
                    provider.transcribe(audio)

    def test_sarvam_stt_malformed_json_raises_parse_error(self, tmp_path: Path) -> None:
        """Test that malformed JSON response raises STTTranscriptionError."""
        audio = _write_wav(tmp_path / "malformed.wav", seconds=1.0)
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.side_effect = ValueError("Invalid JSON")

        provider = SarvamSTT(api_key_env="SARVAM_API_KEY_TEST")
        with patch("indicvoicerag.stt.os.environ", {"SARVAM_API_KEY_TEST": "test_key"}):
            with patch("indicvoicerag.stt.httpx.Client") as mock_client_cls:
                mock_client = MagicMock()
                mock_client_cls.return_value.__enter__.return_value = mock_client
                mock_client.post.return_value = mock_response
                with pytest.raises(STTTranscriptionError, match="invalid JSON"):
                    provider.transcribe(audio)

    def test_sarvam_stt_transcription_result_schema_compat(self, tmp_path: Path) -> None:
        """Test that SarvamSTT returns TranscriptionResult compatible with as_dict()."""
        audio = _write_wav(tmp_path / "schema.wav", seconds=1.5)
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "request_id": "test-schema",
            "transcript": "परीक्षण",
            "language_code": "hi-IN",
            "language_probability": 0.90,
        }

        provider = SarvamSTT(api_key_env="SARVAM_API_KEY_TEST")
        with patch("indicvoicerag.stt.os.environ", {"SARVAM_API_KEY_TEST": "test_key"}):
            with patch("indicvoicerag.stt.httpx.Client") as mock_client_cls:
                mock_client = MagicMock()
                mock_client_cls.return_value.__enter__.return_value = mock_client
                mock_client.post.return_value = mock_response
                result = provider.transcribe(audio)

        payload = result.as_dict()
        assert "text" in payload
        assert "language" in payload
        assert "language_probability" in payload
        assert "duration_seconds" in payload
        assert "processing_time_ms" in payload
        assert "rtf" in payload
        assert "model" in payload
        assert "audio_path" in payload
        assert payload["text"] == "परीक्षण"

    def test_build_stt_provider_sarvam(self) -> None:
        """Test that build_stt_provider creates SarvamSTT correctly."""
        provider = build_stt_provider(
            provider="sarvam",
            model_name="saaras:v3",
            api_key_env="SARVAM_API_KEY_TEST",
            language_code="hi-IN",
        )
        assert isinstance(provider, SarvamSTT)
        assert provider.model_name == "saaras:v3"

    def test_build_stt_provider_sarvam_ignores_whisper_model(self) -> None:
        """Regression: Sarvam must use saaras:v3 even if model_name is a faster-whisper name."""
        provider = build_stt_provider(
            provider="sarvam",
            model_name="small",
            api_key_env="SARVAM_API_KEY_TEST",
        )
        assert isinstance(provider, SarvamSTT)
        assert provider.model_name == "saaras:v3"

    def test_normalize_stt_provider_name_sarvam(self) -> None:
        """Test that Sarvam aliases are normalized correctly."""
        assert normalize_stt_provider_name("sarvam") == "sarvam"
        assert normalize_stt_provider_name("saaras") == "sarvam"
        assert normalize_stt_provider_name("SARVAM") == "sarvam"
