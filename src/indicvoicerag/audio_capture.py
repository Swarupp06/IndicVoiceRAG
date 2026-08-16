"""Microphone capture (Phase 3C).

Isolates microphone hardware from speech recognition: this module knows nothing
about STT or RAG. It records a fixed duration of PCM audio from a microphone
and hands a mono 16 kHz float32 buffer up to the caller, which is then written
to a temporary WAV and given to ``STTProvider``.

Flow::

    MicrophoneRecorder
         |
         v
    AudioCaptureResult   (mono float32 PCM + sample rate + duration)
         |
         v
    STTProvider
         |
         v
    TranscriptionResult
         |
         v
    RAGHarness

Recording uses ``sounddevice`` (PortAudio, bundled in the wheel on Windows -
no system install). The capture backend is injectable so the unit tests run
with a fake backend and never touch real hardware.
"""

from __future__ import annotations

import wave
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np

# default capture parameters (overridable via [audio] config / CLI)
DEFAULT_SAMPLE_RATE = 16000
DEFAULT_CHANNELS = 1
DEFAULT_DURATION_SECONDS = 5.0


class MicrophoneError(RuntimeError):
    """Base class for microphone capture failures."""


class MicrophoneUnavailableError(MicrophoneError):
    """No audio capture backend, no input device, or no permission to open it."""


class MicrophonePermissionError(MicrophoneError):
    """The OS denied access to the microphone (privacy settings)."""


class MicrophoneRecordingError(MicrophoneError):
    """The capture backend could not record (device busy / returned no audio)."""


@dataclass(slots=True)
class AudioCaptureResult:
    """Captured PCM audio (mono float32) plus its metadata."""

    samples: np.ndarray
    sample_rate: int
    duration_seconds: float

    @property
    def empty(self) -> bool:
        return self.samples.size == 0


def _av_resample_mono(samples: np.ndarray, source_rate: int, target_rate: int) -> np.ndarray:
    """Resample mono PCM with PyAV (already a dependency via faster-whisper)."""
    import av

    resampler = av.AudioResampler(format="flt", layout="mono", rate=target_rate)
    frame = av.AudioFrame.from_ndarray(samples.reshape(1, -1), format="flt", layout="mono")
    frame.sample_rate = source_rate
    chunks: list[np.ndarray] = []
    for out_frame in resampler.resample(frame):
        chunks.append(out_frame.to_ndarray())
    for out_frame in resampler.resample(None):
        chunks.append(out_frame.to_ndarray())
    return np.concatenate(chunks, axis=1).reshape(-1).astype(np.float32)


def _linear_resample_mono(samples: np.ndarray, source_rate: int, target_rate: int) -> np.ndarray:
    """Dependency-free linear-interpolation fallback (nearest-neighbour blend)."""
    n_out = max(1, int(round(len(samples) * target_rate / source_rate)))
    x = np.linspace(0, len(samples) - 1, num=n_out)
    x0 = np.floor(x).astype(np.int64)
    x1 = np.minimum(x0 + 1, len(samples) - 1)
    frac = (x - x0).astype(np.float32)
    return (samples[x0] * (1.0 - frac) + samples[x1] * frac).astype(np.float32)


def resample_pcm(samples: np.ndarray, source_rate: int, target_rate: int, channels: int = 1) -> np.ndarray:
    """Convert captured PCM to mono float32 at ``target_rate``."""
    mono = samples[:, 0] if samples.ndim > 1 and samples.shape[1] > 1 else samples.ravel()
    mono = np.asarray(mono, dtype=np.float32)
    if mono.size == 0:
        return mono
    if source_rate == target_rate:
        return mono
    try:
        return _av_resample_mono(mono, int(source_rate), int(target_rate))
    except Exception:  # noqa: BLE001 - fall back to pure-numpy interpolation
        return _linear_resample_mono(mono, int(source_rate), int(target_rate))


def write_pcm_wav(path: str | Path, samples: np.ndarray, sample_rate: int) -> Path:
    """Write mono float32 PCM as a 16-bit PCM WAV file (stdlib ``wave``)."""
    out = Path(path)
    pcm16 = (np.clip(np.asarray(samples, dtype=np.float32), -1.0, 1.0) * 32767.0).astype(np.int16)
    with wave.open(str(out), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(int(sample_rate))
        handle.writeframes(pcm16.tobytes())
    return out


def sounddevice_record(
    duration: float, sample_rate: int | None, channels: int, device: int | None = None
) -> tuple[np.ndarray, int]:
    """Fixed-duration recording through sounddevice (PortAudio).

    Returns ``(samples float32, actual_sample_rate)``. Raises the mapped
    ``MicrophoneError`` subclasses; never leaks a raw traceback to callers.
    """
    try:
        import sounddevice as sd
    except ImportError as exc:  # pragma: no cover - install-time guard
        raise MicrophoneUnavailableError(
            "no audio capture backend installed; run `pip install -e .[mic]` "
            "(sounddevice) to enable microphone input"
        ) from exc

    def _map_error(exc: Exception, action: str) -> MicrophoneError:
        message = str(exc).lower()
        if "permission" in message or "access denied" in message:
            return MicrophonePermissionError(f"microphone permission denied: {exc}")
        if "invalid device" in message or "device unavailable" in message:
            return MicrophoneUnavailableError(f"microphone unavailable: {exc}")
        return MicrophoneRecordingError(f"microphone {action} failed: {exc}")

    try:
        if device is None:
            device = sd.default.device[0]
        rate = sample_rate
        try:
            sd.check_input_settings(device=device, samplerate=rate, channels=channels)
        except (sd.PortAudioError, ValueError):
            info = sd.query_devices(device, "input")
            rate = int(info["default_samplerate"])
        frames = max(1, int(duration * rate))
        data = sd.rec(frames, samplerate=rate, channels=channels, dtype="float32", device=device, blocking=True)
        return np.asarray(data, dtype=np.float32), rate
    except MicrophoneError:
        raise
    except sd.PortAudioError as exc:
        raise _map_error(exc, "recording") from exc
    except Exception as exc:  # noqa: BLE001 - wrap anything else as a recording failure
        raise MicrophoneRecordingError(f"microphone recording failed: {type(exc).__name__}: {exc}") from exc


class MicrophoneRecorder:
    """Fixed-duration microphone recorder (push-to-talk building block).

    ``record_fn`` is an injection point for tests: it receives
    ``(duration, sample_rate, channels, device)`` and returns
    ``(samples, actual_rate)``.
    """

    def __init__(
        self,
        sample_rate: int = DEFAULT_SAMPLE_RATE,
        channels: int = DEFAULT_CHANNELS,
        duration_seconds: float = DEFAULT_DURATION_SECONDS,
        device: int | None = None,
        record_fn: Callable[[float, int | None, int, int | None], tuple[np.ndarray, int]] | None = None,
    ):
        self.sample_rate = sample_rate
        self.channels = channels
        self.duration_seconds = duration_seconds
        self.device = device
        self._record_fn = record_fn

    def available(self) -> bool:
        """True when a capture backend and at least one input device exist."""
        try:
            import sounddevice as sd

            devices = sd.query_devices()
        except Exception:  # noqa: BLE001 - treat any backend failure as unavailable
            return False
        return any(d["max_input_channels"] > 0 for d in devices)

    def record(self, duration_seconds: float | None = None) -> AudioCaptureResult:
        """Record a fixed duration and return mono ``target_rate`` PCM."""
        duration = duration_seconds or self.duration_seconds
        record_fn = self._record_fn or sounddevice_record
        try:
            data, rate = record_fn(duration, self.sample_rate, self.channels, self.device)
        except MicrophoneError:
            raise
        except Exception as exc:  # noqa: BLE001 - backend contract violation
            raise MicrophoneRecordingError(
                f"microphone recording failed: {type(exc).__name__}: {exc}"
            ) from exc

        samples = np.asarray(data, dtype=np.float32)
        if samples.size == 0:
            raise MicrophoneRecordingError("microphone captured no audio (empty recording)")
        mono = resample_pcm(samples, int(rate), self.sample_rate, self.channels)
        if mono.size == 0:
            raise MicrophoneRecordingError("microphone captured no audio after resampling")
        return AudioCaptureResult(
            samples=mono,
            sample_rate=self.sample_rate,
            duration_seconds=mono.size / self.sample_rate,
        )
