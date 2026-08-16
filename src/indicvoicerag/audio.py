"""Provider-independent audio loading (PyAV-based).

Decodes an audio file into mono 16 kHz float32 samples and reports its
duration. PyAV bundles its own FFmpeg libraries, so no system FFmpeg
executable is required for WAV (or Ogg/FLAC/MP3) input.

This module has no STT-specific logic: the STT providers only need the
decoded samples (faster-whisper accepts raw float32 PCM) and the duration
(for latency / RTF reporting).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Union

import numpy as np

AudioPath = Union[str, Path]

TARGET_SAMPLE_RATE = 16000


class AudioError(RuntimeError):
    """Raised when an audio file cannot be located or decoded."""


@dataclass(slots=True)
class AudioInfo:
    """Decoded audio: mono float32 samples at ``TARGET_SAMPLE_RATE``."""

    samples: np.ndarray
    sample_rate: int
    duration_seconds: float

    @property
    def empty(self) -> bool:
        return self.samples.size == 0


def load_audio(path: AudioPath) -> AudioInfo:
    """Decode an audio file into mono 16 kHz float32 samples.

    Raises :class:`AudioError` for missing/empty/invalid files or files
    without a decodable audio stream.
    """
    audio_path = Path(path)
    if not audio_path.exists():
        raise AudioError(f"audio file not found: {audio_path}")
    if audio_path.stat().st_size == 0:
        raise AudioError(f"audio file is empty: {audio_path}")

    try:
        import av
    except ImportError as exc:  # pragma: no cover - depends on install
        raise AudioError(
            "PyAV (av) is required to decode audio; install faster-whisper "
            "or the 'stt' project extra."
        ) from exc

    try:
        container = av.open(str(audio_path))
    except Exception as exc:  # noqa: BLE001 - av raises many exception types
        raise AudioError(
            f"could not open audio file {audio_path}: {type(exc).__name__}: {exc}"
        ) from exc

    try:
        stream = next(s for s in container.streams if s.type == "audio")
    except StopIteration:
        container.close()
        raise AudioError(f"no audio stream found in {audio_path}")

    resampler = av.AudioResampler(format="flt", layout="mono", rate=TARGET_SAMPLE_RATE)
    chunks: list[np.ndarray] = []
    try:
        for frame in container.decode(stream):
            for resampled in resampler.resample(frame):
                chunks.append(resampled.to_ndarray())
        for resampled in resampler.resample(None):
            chunks.append(resampled.to_ndarray())
    except Exception as exc:  # noqa: BLE001 - decode failures surface as AudioError
        raise AudioError(
            f"failed to decode audio {audio_path}: {type(exc).__name__}: {exc}"
        ) from exc
    finally:
        container.close()

    if not chunks:
        raise AudioError(f"no audio samples could be decoded from {audio_path}")
    samples = np.concatenate(chunks, axis=1).reshape(-1).astype(np.float32)
    if samples.size == 0:
        raise AudioError(f"decoded audio is empty: {audio_path}")

    return AudioInfo(
        samples=samples,
        sample_rate=TARGET_SAMPLE_RATE,
        duration_seconds=samples.size / TARGET_SAMPLE_RATE,
    )


def probe_audio(path: AudioPath) -> dict[str, Any]:
    """Read container metadata without decoding: duration / rate / channels.

    Raises :class:`AudioError` for missing, empty or undecodable files. When
    the container does not expose a duration, falls back to a full decode.
    """
    audio_path = Path(path)
    if not audio_path.exists():
        raise AudioError(f"audio file not found: {audio_path}")
    if audio_path.stat().st_size == 0:
        raise AudioError(f"audio file is empty: {audio_path}")

    try:
        import av
    except ImportError as exc:  # pragma: no cover - depends on install
        raise AudioError(
            "PyAV (av) is required to decode audio; install faster-whisper "
            "or the 'stt' project extra."
        ) from exc

    try:
        container = av.open(str(audio_path))
        stream = next(s for s in container.streams if s.type == "audio")
        duration = stream.duration * stream.time_base if stream.duration else None
        sample_rate = stream.rate
        channels = stream.layout.nb_channels if stream.layout else None
        container.close()
    except StopIteration:
        raise AudioError(f"no audio stream found in {audio_path}")
    except Exception as exc:  # noqa: BLE001 - av raises many exception types
        raise AudioError(
            f"could not open audio file {audio_path}: {type(exc).__name__}: {exc}"
        ) from exc

    if duration is None:
        duration = load_audio(audio_path).duration_seconds
    return {
        "duration_seconds": float(duration),
        "sample_rate": int(sample_rate) if sample_rate else None,
        "channels": channels,
    }
