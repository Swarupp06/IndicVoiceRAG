"""Local Speech-to-Text abstraction (Phase 3A).

    audio file
        |
        v
    STTProvider.transcribe(audio_path)
        |
        v
    TranscriptionResult{ text, language, ... }
        |
        v
    text  ->  existing RAGHarness

Providers:
- ``mock``           deterministic offline stub (tests / pipeline wiring only)
- ``faster_whisper`` faster-whisper (CTranslate2) local models, CPU by default

The result type is provider-independent, so a future Indic-specific model can
be swapped in behind the same interface without touching the RAG harness.
"""

from __future__ import annotations

import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Callable

from .audio import AudioPath, AudioError, load_audio

# compute types supported by faster-whisper on CPU backends (CTranslate2)
_CPU_COMPUTE_TYPES = ("int8", "int8_float32", "int8_float16", "int16", "float16", "float32")


class STTError(RuntimeError):
    """Base class for speech-to-text failures."""


class STTConfigurationError(STTError):
    """Provider cannot run (e.g. invalid compute type or missing package)."""


class STTAudioError(STTError):
    """The input audio file cannot be loaded or decoded."""


class STTTranscriptionError(STTError):
    """The provider failed while transcribing valid audio."""


@dataclass(slots=True)
class TranscriptionResult:
    """Provider-independent transcription outcome.

    ``rtf`` (real-time factor) = processing_time / audio_duration, where
    ``processing_time_ms`` is the model transcription time (model loading is
    reported separately via ``load_time_ms``). RTF < 1 means faster than real
    time.
    """

    text: str
    language: str | None = None
    language_probability: float = 0.0
    duration_seconds: float = 0.0
    processing_time_ms: float = 0.0
    rtf: float = 0.0
    model: str | None = None
    load_time_ms: float | None = None
    audio_path: str | None = None

    def __post_init__(self) -> None:
        if self.duration_seconds and self.duration_seconds > 0:
            self.rtf = (self.processing_time_ms / 1000.0) / self.duration_seconds
        else:
            self.rtf = 0.0

    def as_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "text": self.text,
            "language": self.language,
            "language_probability": round(self.language_probability, 4),
            "duration_seconds": round(self.duration_seconds, 3),
            "processing_time_ms": round(self.processing_time_ms, 2),
            "rtf": round(self.rtf, 4),
        }
        if self.model is not None:
            payload["model"] = self.model
        if self.load_time_ms is not None:
            payload["load_time_ms"] = round(self.load_time_ms, 2)
        if self.audio_path is not None:
            payload["audio_path"] = self.audio_path
        return payload


class STTProvider(ABC):
    """Speech-to-text abstraction: audio file path -> TranscriptionResult."""

    name: str = "abstract"
    _model_name: str = "unknown"

    @abstractmethod
    def transcribe(self, audio_path: AudioPath, *, language: str | None = None) -> TranscriptionResult:
        """Transcribe an audio file into structured text."""
        raise NotImplementedError

    @property
    def model_name(self) -> str:
        return self._model_name

    def available(self) -> bool:
        return True

    def describe(self) -> dict[str, Any]:
        return {"provider": self.name, "model": self.model_name}

    def warmup(self, timeout: float = 60.0) -> None:
        """Warm model/caches so the first timed call is fast (no-op default)."""
        return


class MockSTT(STTProvider):
    """Deterministic offline stub.

    Still validates the audio file through ``load_audio`` so the audio
    pipeline is exercised, but it never downloads or runs a model.
    """

    name = "mock"

    def __init__(
        self,
        text: str = "mock transcription",
        language: str | None = "en",
        language_probability: float = 0.95,
        behavior: str = "ok",
        model_name: str = "mock-stt",
    ):
        if behavior not in ("ok", "empty", "raise"):
            raise ValueError(f"unknown mock behavior: {behavior!r}")
        self._text = text
        self._language = language
        self._language_probability = language_probability
        self._behavior = behavior
        self._model_name = model_name

    def transcribe(self, audio_path: AudioPath, *, language: str | None = None) -> TranscriptionResult:
        try:
            info = load_audio(audio_path)
        except AudioError as exc:
            raise STTAudioError(str(exc)) from exc
        if self._behavior == "raise":
            raise STTTranscriptionError("mock provider failure")
        if self._behavior == "empty":
            return TranscriptionResult(
                text="",
                language=language or self._language,
                duration_seconds=info.duration_seconds,
                processing_time_ms=0.0,
                model=self._model_name,
            )
        started = time.perf_counter()
        detected = language or self._language
        return TranscriptionResult(
            text=self._text,
            language=detected,
            language_probability=self._language_probability
            if detected == self._language
            else 0.9,
            duration_seconds=info.duration_seconds,
            processing_time_ms=(time.perf_counter() - started) * 1000.0,
            model=self._model_name,
            audio_path=str(audio_path),
        )

    def describe(self) -> dict[str, Any]:
        return {"provider": self.name, "model": self._model_name, "behavior": self._behavior}


class FasterWhisperSTT(STTProvider):
    """Local faster-whisper provider (CTranslate2), CPU by default.

    The model is loaded lazily on the first ``transcribe`` call, so
    constructing the provider never downloads anything. ``model_factory`` is an
    injection point for offline unit tests (no model download).
    """

    name = "faster_whisper"

    def __init__(
        self,
        model_name: str = "tiny",
        device: str = "cpu",
        compute_type: str = "int8",
        language: str | None = None,
        beam_size: int = 5,
        vad_filter: bool = False,
        download_root: str | None = None,
        model_factory: Callable[[], Any] | None = None,
    ):
        if compute_type not in _CPU_COMPUTE_TYPES:
            raise STTConfigurationError(
                f"unsupported compute_type {compute_type!r}; "
                f"choose from {sorted(_CPU_COMPUTE_TYPES)}"
            )
        self._model_name = model_name
        self._device = device
        self._compute_type = compute_type
        self._language = language
        self._beam_size = beam_size
        self._vad_filter = vad_filter
        self._download_root = download_root
        self._model_factory = model_factory
        self._model: Any | None = None
        self._model_load_seconds: float | None = None

    @property
    def device(self) -> str:
        return self._device

    @property
    def compute_type(self) -> str:
        return self._compute_type

    @property
    def model_load_seconds(self) -> float | None:
        """Seconds spent constructing the model (includes first-run download)."""
        return self._model_load_seconds

    def _ensure_model(self) -> Any:
        if self._model is None:
            started = time.perf_counter()
            if self._model_factory is not None:
                self._model = self._model_factory()
            else:
                try:
                    from faster_whisper import WhisperModel
                except ImportError as exc:
                    raise STTConfigurationError(
                        "faster-whisper is not installed; install the 'stt' "
                        "extra (`pip install faster-whisper`) or switch "
                        "stt.provider to 'mock'."
                    ) from exc
                try:
                    self._model = WhisperModel(
                        self._model_name,
                        device=self._device,
                        compute_type=self._compute_type,
                        download_root=self._download_root,
                    )
                except Exception as exc:  # noqa: BLE001 - model load failures
                    raise STTConfigurationError(
                        f"could not load faster-whisper model {self._model_name!r} "
                        f"on {self._device}/{self._compute_type}: "
                        f"{type(exc).__name__}: {exc}"
                    ) from exc
            self._model_load_seconds = time.perf_counter() - started
        return self._model

    def available(self) -> bool:
        try:
            import faster_whisper  # noqa: F401
        except ImportError:
            return False
        return True

    def transcribe(self, audio_path: AudioPath, *, language: str | None = None) -> TranscriptionResult:
        try:
            info = load_audio(audio_path)
        except AudioError as exc:
            raise STTAudioError(str(exc)) from exc
        model = self._ensure_model()
        requested_language = language or self._language
        started = time.perf_counter()
        try:
            segments, segment_info = model.transcribe(
                info.samples,
                language=requested_language,
                beam_size=self._beam_size,
                vad_filter=self._vad_filter,
            )
            parts = [segment.text.strip() for segment in segments if getattr(segment, "text", None)]
            text = " ".join(parts).strip()
        except Exception as exc:  # noqa: BLE001 - provider failures surface as STTTranscriptionError
            raise STTTranscriptionError(
                f"faster-whisper transcription failed: {type(exc).__name__}: {exc}"
            ) from exc
        processing_ms = (time.perf_counter() - started) * 1000.0
        detected_language = getattr(segment_info, "language", None) or requested_language or "unknown"
        language_probability = float(getattr(segment_info, "language_probability", 0.0) or 0.0)
        duration = getattr(segment_info, "duration", None)
        duration = float(duration) if duration else info.duration_seconds
        return TranscriptionResult(
            text=text,
            language=detected_language,
            language_probability=language_probability,
            duration_seconds=duration,
            processing_time_ms=processing_ms,
            model=self._model_name,
            load_time_ms=(self._model_load_seconds or 0.0) * 1000.0,
            audio_path=str(audio_path),
        )

    def describe(self) -> dict[str, Any]:
        return {
            "provider": self.name,
            "model": self._model_name,
            "device": self._device,
            "compute_type": self._compute_type,
            "language": self._language,
        }


_STT_ALIASES = {
    "faster-whisper": "faster_whisper",
    "fasterwhisper": "faster_whisper",
    "whisper": "faster_whisper",
    "local": "faster_whisper",
}


def normalize_stt_provider_name(provider: str) -> str:
    normalized = provider.strip().lower().replace("-", "_")
    return _STT_ALIASES.get(normalized, normalized)


def build_stt_provider(
    provider: str,
    model_name: str = "tiny",
    device: str = "cpu",
    compute_type: str = "int8",
    language: str | None = None,
    beam_size: int = 5,
    vad_filter: bool = False,
    download_root: str | None = None,
) -> STTProvider:
    """Build a single STT provider by name."""
    normalized = normalize_stt_provider_name(provider)
    if normalized == "mock":
        return MockSTT()
    if normalized == "faster_whisper":
        return FasterWhisperSTT(
            model_name=model_name,
            device=device,
            compute_type=compute_type,
            language=language,
            beam_size=beam_size,
            vad_filter=vad_filter,
            download_root=download_root,
        )
    raise ValueError(f"Unsupported STT provider: {provider}")
