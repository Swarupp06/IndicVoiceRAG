"""Phase 3D offline unit tests for the TTS abstraction.

No test downloads a model, touches the microphone, or calls any external API:
MockTTS covers the offline paths and MMSIndicTTS is exercised with an injected
fake VITS model.
"""

from __future__ import annotations

import wave
from pathlib import Path

import numpy as np
import pytest

from indicvoicerag.audio import load_audio
from indicvoicerag.config import AppConfig
from indicvoicerag.stt import MockSTT
from indicvoicerag.tts import (
    MMSIndicTTS,
    MMS_LANGUAGE_MODELS,
    MockTTS,
    TTSConfigurationError,
    TTSError,
    TTSSynthesisError,
    TTSTextError,
    TTSUnsupportedLanguageError,
    build_tts_provider,
    normalize_tts_provider_name,
)
from indicvoicerag.cli import run_synthesize, synthesize_answer_audio


# --- mock provider ---
def test_mock_tts_valid_synthesis(tmp_path: Path) -> None:
    tts = MockTTS()
    result = tts.synthesize("नमस्ते भारत", language="hi", output_path=tmp_path / "out.wav")
    assert result.language == "hi"
    assert result.model == "mock-tts-hi"
    assert result.provider == "mock"
    assert result.sample_rate == 22050
    assert result.duration_seconds == pytest.approx(1.0, abs=0.01)
    assert result.synthesis_time_ms >= 0
    assert result.rtf >= 0
    assert result.output_path == str(tmp_path / "out.wav")
    assert Path(result.output_path).exists()
    assert result.audio is not None and result.audio.size > 0


def test_mock_tts_wav_is_playable_pcm(tmp_path: Path) -> None:
    tts = MockTTS()
    result = tts.synthesize("नमस्ते", output_path=tmp_path / "out.wav")
    with wave.open(str(result.output_path), "rb") as handle:
        assert handle.getnchannels() == 1
        assert handle.getsampwidth() == 2
        assert handle.getframerate() == 22050
        assert handle.getnframes() == int(1.0 * 22050)


def test_mock_tts_empty_text_raises() -> None:
    with pytest.raises(TTSTextError):
        MockTTS().synthesize("   ", language="hi")


def test_mock_tts_unsupported_language_raises() -> None:
    with pytest.raises(TTSUnsupportedLanguageError):
        MockTTS().synthesize("नमस्ते", language="xx")


def test_mock_tts_provider_failure_raises() -> None:
    with pytest.raises(TTSSynthesisError):
        MockTTS(behavior="raise").synthesize("नमस्ते", language="hi")


def test_tts_result_schema() -> None:
    from indicvoicerag.tts import TTSResult

    result = TTSResult(
        text="नमस्ते",
        language="hi",
        sample_rate=22050,
        duration_seconds=1.0,
        synthesis_time_ms=120.0,
        rtf=0.12,
        model="mms-tts-hin",
        provider="mms",
        load_time_ms=3000.0,
        output_path="tmp/x.wav",
    )
    payload = result.as_dict()
    assert set(payload) == {
        "text",
        "language",
        "sample_rate",
        "duration_seconds",
        "synthesis_time_ms",
        "rtf",
        "model",
        "provider",
        "load_time_ms",
        "output_path",
    }


# --- provider name normalization / builder ---
def test_normalize_tts_provider_name_aliases() -> None:
    assert normalize_tts_provider_name("mms") == "mms"
    assert normalize_tts_provider_name("mms-tts") == "mms"
    assert normalize_tts_provider_name("VITS") == "mms"
    assert normalize_tts_provider_name("mock") == "mock"


def test_build_tts_provider_mock_and_mms() -> None:
    assert isinstance(build_tts_provider("mock"), MockTTS)
    provider = build_tts_provider("mms", language="hi")
    assert isinstance(provider, MMSIndicTTS)
    assert provider._language == "hi"  # noqa: SLF001 - inspect defaults
    with pytest.raises(TTSConfigurationError):
        build_tts_provider("bogus")


# --- MMSIndicTTS with an injected fake VITS model ---
class _FakeWaveform:
    def __init__(self, samples: np.ndarray):
        self._samples = samples

    def detach(self):
        return self

    def numpy(self) -> np.ndarray:
        return self._samples


class _BatchWaveform:
    """Mimics the [batch, samples] waveform tensor returned by VitsModel."""

    def __init__(self, samples: np.ndarray):
        self._samples = samples

    def __getitem__(self, index: int):
        return _FakeWaveform(self._samples[index])


class _FakeOutput:
    def __init__(self, samples: np.ndarray):
        self.waveform = _BatchWaveform(samples)

    def __getitem__(self, index: int):
        return self.waveform[index]


class _FakeVitsModel:
    def __init__(self, samples: np.ndarray, error: Exception | None = None):
        self._samples = samples
        self._error = error
        self.config = type("Config", (), {"sampling_rate": 22050})()
        self.eval_called = False

    def eval(self):
        self.eval_called = True
        return self

    def __call__(self, **inputs):
        if self._error is not None:
            raise self._error
        return _FakeOutput(self._samples)


class _FakeTokenizer:
    def __call__(self, text: str, return_tensors: str = "pt"):
        assert return_tensors == "pt"
        return {"input_ids": [1, 2, 3]}


def _sine_audio(duration: float = 1.0, rate: int = 22050) -> np.ndarray:
    t = np.arange(int(duration * rate)) / rate
    return (0.5 * np.sin(2 * np.pi * 440 * t)).astype(np.float32)[None, :]  # [batch, samples]


def test_mms_tts_synthesis_with_fake_model(tmp_path: Path) -> None:
    built = {}

    def factory(model_id: str, language: str, cache_dir) -> tuple[_FakeVitsModel, _FakeTokenizer]:
        built["model_id"] = model_id
        built["language"] = language
        return _FakeVitsModel(_sine_audio(1.0)), _FakeTokenizer()

    tts = MMSIndicTTS(language="hi", model_factory=factory)
    result = tts.synthesize("मिर्ची में कितने विभिन्न प्रजातियाँ हैं", output_path=tmp_path / "mms.wav")

    assert built["model_id"] == "facebook/mms-tts-hin"
    assert built["language"] == "hi"
    assert result.provider == "mms"
    assert result.model == "mms-tts-hin"
    assert result.sample_rate == 22050
    assert result.duration_seconds == pytest.approx(1.0, abs=0.01)
    assert result.load_time_ms is not None and result.load_time_ms > 0
    assert Path(result.output_path).exists()
    info = load_audio(result.output_path)
    assert not info.empty
    assert info.duration_seconds == pytest.approx(1.0, abs=0.02)


def test_mms_tts_unsupported_language_raises() -> None:
    tts = MMSIndicTTS(language="hi", model_factory=lambda m, l, c: (_FakeVitsModel(_sine_audio()), _FakeTokenizer()))
    with pytest.raises(TTSUnsupportedLanguageError):
        tts.synthesize("नमस्ते", language="xx")


def test_mms_tts_synthesis_failure_wrapped(tmp_path: Path) -> None:
    tts = MMSIndicTTS(
        language="hi",
        model_factory=lambda m, l, c: (_FakeVitsModel(_sine_audio(), error=RuntimeError("boom")), _FakeTokenizer()),
    )
    with pytest.raises(TTSSynthesisError):
        tts.synthesize("नमस्ते", output_path=tmp_path / "f.wav")


def test_mms_tts_model_loaded_once_and_reused(tmp_path: Path) -> None:
    """Phase 3D regression: one provider instance builds the VITS model exactly
    once and reuses it for every synthesize call (no per-call reload)."""
    builds = {"count": 0}

    def factory(model_id: str, language: str, cache_dir) -> tuple[_FakeVitsModel, _FakeTokenizer]:
        builds["count"] += 1
        return _FakeVitsModel(_sine_audio(1.0)), _FakeTokenizer()

    tts = MMSIndicTTS(language="hi", model_factory=factory)
    first = tts.synthesize("पहला", output_path=tmp_path / "a.wav")
    second = tts.synthesize("दूसरा", output_path=tmp_path / "b.wav")
    third = tts.synthesize("तीसरा", output_path=tmp_path / "c.wav")

    assert builds["count"] == 1
    assert first.load_time_ms == second.load_time_ms == third.load_time_ms
    assert tts.model_load_seconds is not None


def test_mms_language_table_has_hindi_and_indic_entries() -> None:
    assert "hi" in MMS_LANGUAGE_MODELS
    assert MMS_LANGUAGE_MODELS["hi"] == "facebook/mms-tts-hin"
    for code in ("bn", "ta", "te", "mr", "gu", "kn", "ml", "pa", "or", "as", "ne", "si", "ur"):
        assert code in MMS_LANGUAGE_MODELS


# --- CLI + RAG integration glue (mock providers, offline) ---
def test_cli_run_synthesize_mock(tmp_path: Path, capsys) -> None:
    cfg = AppConfig()
    cfg.tts.provider = "mock"
    out = tmp_path / "cli.wav"
    run_synthesize(cfg, text="नमस्ते भारत", language="hi", output=str(out), as_json=False)
    captured = capsys.readouterr().out
    assert "language         : hi" in captured
    assert "model            : mock-tts-hi" in captured
    assert "RTF" in captured
    assert str(out) in captured
    assert out.exists()


def test_rag_to_tts_glue_mock(tmp_path: Path) -> None:
    """Hindi RAG answer -> TTS -> WAV output layer (harness stays text-based)."""
    cfg = AppConfig()
    cfg.tts.provider = "mock"
    cfg.tts.language = "hi"
    answer = "मिर्ची में कितने विभिन्न प्रजातियाँ हैं"
    out = tmp_path / "answer.wav"
    result = synthesize_answer_audio(cfg, answer, str(out))
    assert result.output_path == str(out)
    assert Path(result.output_path).exists()
    assert result.language == "hi"
    info = load_audio(out)
    assert not info.empty
