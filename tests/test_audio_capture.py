"""Phase 3C unit tests: microphone capture abstraction (no real hardware).

Every test injects a fake ``record_fn`` so nothing touches sounddevice or an
actual microphone. We still exercise the real resample / WAV-write / WAV-read
path offline through PyAV and the ``wave`` module.
"""

from __future__ import annotations

import wave

import numpy as np
import pytest

from indicvoicerag.audio import load_audio
from indicvoicerag.audio_capture import (
    AudioCaptureResult,
    MicrophoneRecorder,
    MicrophoneRecordingError,
    MicrophoneUnavailableError,
    resample_pcm,
    write_pcm_wav,
)


def _sine_pcm(duration: float = 0.5, rate: int = 44100, channels: int = 1) -> tuple[np.ndarray, int]:
    t = np.arange(int(duration * rate)) / rate
    signal = (0.5 * np.sin(2 * np.pi * 440 * t)).astype(np.float32)
    if channels > 1:
        return np.repeat(signal[:, None], channels, axis=1), rate
    return signal, rate


def test_record_success_mono_16k() -> None:
    rec = MicrophoneRecorder(sample_rate=16000, record_fn=lambda d, sr, ch, dev: _sine_pcm(d, 44100, 1))
    result = rec.record(0.5)
    assert isinstance(result, AudioCaptureResult)
    assert result.sample_rate == 16000
    assert not result.empty
    assert result.duration_seconds == pytest.approx(0.5, abs=0.02)


def test_record_stereo_is_downmixed_to_mono() -> None:
    rec = MicrophoneRecorder(sample_rate=16000, record_fn=lambda d, sr, ch, dev: _sine_pcm(d, 48000, 2))
    result = rec.record(0.5)
    assert result.samples.ndim == 1
    assert result.sample_rate == 16000


def test_record_no_microphone_propagates_controlled_error() -> None:
    def no_mic(duration: float, sample_rate: int | None, channels: int, device: int | None) -> tuple[np.ndarray, int]:
        raise MicrophoneUnavailableError("no input device found")

    rec = MicrophoneRecorder(record_fn=no_mic)
    with pytest.raises(MicrophoneUnavailableError):
        rec.record()


def test_record_empty_capture_raises() -> None:
    rec = MicrophoneRecorder(
        record_fn=lambda d, sr, ch, dev: (np.array([], dtype=np.float32), sr or 16000)
    )
    with pytest.raises(MicrophoneRecordingError):
        rec.record()


def test_record_backend_exception_wrapped() -> None:
    def boom(duration: float, sample_rate: int | None, channels: int, device: int | None) -> tuple[np.ndarray, int]:
        raise RuntimeError("backend blew up")

    rec = MicrophoneRecorder(record_fn=boom)
    with pytest.raises(MicrophoneRecordingError):
        rec.record()


def test_resample_same_rate_returns_mono_unchanged() -> None:
    mono, _ = _sine_pcm(0.5, 16000, 1)
    out = resample_pcm(mono, 16000, 16000, 1)
    assert out.ndim == 1
    assert np.allclose(out, mono, atol=1e-6)


def test_resample_rate_conversion_lengthens_signal() -> None:
    mono, _ = _sine_pcm(0.5, 16000, 1)
    up = resample_pcm(mono, 16000, 44100, 1)
    assert len(up) > len(mono)
    down = resample_pcm(up, 44100, 16000, 1)
    assert len(down) == pytest.approx(len(mono), abs=2)


def test_write_pcm_wav_round_trips_through_wave_module(tmp_path) -> None:
    mono, _ = _sine_pcm(0.5, 16000, 1)
    path = write_pcm_wav(tmp_path / "capture.wav", mono, 16000)
    assert path.exists()
    with wave.open(str(path), "rb") as handle:
        assert handle.getnchannels() == 1
        assert handle.getsampwidth() == 2
        assert handle.getframerate() == 16000
        assert handle.getnframes() == len(mono)


def test_write_pcm_wav_loads_back_through_av(tmp_path) -> None:
    mono, _ = _sine_pcm(0.5, 16000, 1)
    path = write_pcm_wav(tmp_path / "capture.wav", mono, 16000)
    info = load_audio(path)

    assert not info.empty
    assert info.duration_seconds == pytest.approx(0.5, abs=0.02)
