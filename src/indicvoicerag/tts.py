"""Local Indic Text-to-Speech abstraction (Phase 3D).

    text
      |
      v
    TTSProvider.synthesize(text)
      |
      v
    TTSResult{ audio, duration, synthesis_time, rtf, ... }
      |
      v
    WAV output file (mono 16-bit PCM)

Providers:
- ``mock``   deterministic offline stub (tests / pipeline wiring only)
- ``mms``    Meta MMS-TTS VITS checkpoints per language (``facebook/mms-tts-*``)
             via the already-installed transformers/torch stack; Hindi is the
             baseline, and the per-language checkpoints cover ~14 Indic
             languages. CC-BY-NC 4.0 (free for non-commercial use).

The result type is provider-independent, so a future Indic-specific model can be
swapped in behind the same interface without touching the RAG harness or the
audio-output layer. The RAG harness stays text-based; TTS is a pure output
layer.
"""

from __future__ import annotations

import time
import wave
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import numpy as np

# language -> facebook/mms-tts-* checkpoint (ISO 639-3 repo suffix)
MMS_LANGUAGE_MODELS: dict[str, str] = {
    "hi": "facebook/mms-tts-hin",
    "hin": "facebook/mms-tts-hin",
    "bn": "facebook/mms-tts-ben",
    "ben": "facebook/mms-tts-ben",
    "ta": "facebook/mms-tts-tam",
    "tam": "facebook/mms-tts-tam",
    "te": "facebook/mms-tts-tel",
    "tel": "facebook/mms-tts-tel",
    "mr": "facebook/mms-tts-mar",
    "mar": "facebook/mms-tts-mar",
    "gu": "facebook/mms-tts-guj",
    "guj": "facebook/mms-tts-guj",
    "kn": "facebook/mms-tts-kan",
    "kan": "facebook/mms-tts-kan",
    "ml": "facebook/mms-tts-mal",
    "mal": "facebook/mms-tts-mal",
    "pa": "facebook/mms-tts-pan",
    "pan": "facebook/mms-tts-pan",
    "or": "facebook/mms-tts-ori",
    "ori": "facebook/mms-tts-ori",
    "as": "facebook/mms-tts-asm",
    "asm": "facebook/mms-tts-asm",
    "ne": "facebook/mms-tts-npi",
    "npi": "facebook/mms-tts-npi",
    "si": "facebook/mms-tts-sin",
    "sin": "facebook/mms-tts-sin",
    "ur": "facebook/mms-tts-urd",
    "urd": "facebook/mms-tts-urd",
}

DEFAULT_SAMPLE_RATE = 22050  # VITS sampling_rate; MockTTS mimics the same rate
_MAX_TEXT_CHARS = 400


class TTSError(RuntimeError):
    """Base class for text-to-speech failures."""


class TTSConfigurationError(TTSError):
    """Provider cannot run (e.g. missing package or unknown provider)."""


class TTSTextError(TTSError):
    """The input text is empty or outside the supported length."""


class TTSUnsupportedLanguageError(TTSError):
    """The provider has no model for the requested language."""


class TTSSynthesisError(TTSError):
    """The provider failed while synthesizing valid text."""


@dataclass(slots=True)
class TTSResult:
    """Provider-independent synthesis outcome.

    ``rtf`` (real-time factor) = synthesis_time / generated_audio_duration,
    where ``synthesis_time_ms`` is model inference only (model loading is
    reported separately via ``load_time_ms``). RTF < 1 means faster than real
    time.
    """

    text: str
    language: str | None = None
    sample_rate: int = DEFAULT_SAMPLE_RATE
    duration_seconds: float = 0.0
    synthesis_time_ms: float = 0.0
    rtf: float = 0.0
    model: str | None = None
    provider: str | None = None
    load_time_ms: float | None = None
    output_path: str | None = None
    audio: np.ndarray | None = None

    def as_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "text": self.text,
            "language": self.language,
            "sample_rate": self.sample_rate,
            "duration_seconds": round(self.duration_seconds, 3),
            "synthesis_time_ms": round(self.synthesis_time_ms, 2),
            "rtf": round(self.rtf, 4),
        }
        if self.model is not None:
            payload["model"] = self.model
        if self.provider is not None:
            payload["provider"] = self.provider
        if self.load_time_ms is not None:
            payload["load_time_ms"] = round(self.load_time_ms, 2)
        if self.output_path is not None:
            payload["output_path"] = self.output_path
        return payload


class TTSProvider(ABC):
    """Text-to-speech abstraction: text -> WAV audio (bytes / path / samples)."""

    name: str = "abstract"

    @abstractmethod
    def synthesize(
        self,
        text: str,
        *,
        language: str | None = None,
        output_path: str | Path | None = None,
    ) -> TTSResult:
        """Synthesize speech from text into mono PCM WAV audio."""
        raise NotImplementedError

    def available(self) -> bool:
        return True

    def describe(self) -> dict[str, Any]:
        return {"provider": self.name}

    def warmup(self, timeout: float = 120.0) -> None:
        """Load the model so the first timed synthesis is fast (no-op default)."""
        return


def _validate_text(text: str) -> str:
    if text is None or not str(text).strip():
        raise TTSTextError("cannot synthesize empty text")
    text = str(text).strip()
    if len(text) > _MAX_TEXT_CHARS:
        raise TTSTextError(
            f"text too long ({len(text)} chars; max {_MAX_TEXT_CHARS}); "
            "the RAG answers are short by design"
        )
    return text


def _normalize_language(language: str | None) -> str | None:
    if language is None:
        return None
    return language.strip().lower()


def write_tts_wav(path: str | Path, samples: np.ndarray, sample_rate: int) -> Path:
    """Write mono float32 PCM as 16-bit PCM WAV (stdlib ``wave``)."""
    out = Path(path)
    pcm16 = (np.clip(np.asarray(samples, dtype=np.float32), -1.0, 1.0) * 32767.0).astype(np.int16)
    with wave.open(str(out), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(int(sample_rate))
        handle.writeframes(pcm16.tobytes())
    return out


class MockTTS(TTSProvider):
    """Deterministic offline stub.

    Emits a short 440 Hz sine tone (so the WAV is audible/verifiable) without
    any model. Supports any language in ``MMS_LANGUAGE_MODELS`` plus a default
    ``hi``; pass ``language`` to pick a supported one.
    """

    name = "mock"

    def __init__(
        self,
        sample_rate: int = DEFAULT_SAMPLE_RATE,
        tone_hz: float = 440.0,
        duration_seconds: float = 1.0,
        behavior: str = "ok",
    ):
        if behavior not in ("ok", "raise"):
            raise ValueError(f"unknown mock behavior: {behavior!r}")
        self._sample_rate = sample_rate
        self._tone_hz = tone_hz
        self._duration = duration_seconds
        self._behavior = behavior

    def synthesize(
        self,
        text: str,
        *,
        language: str | None = None,
        output_path: str | Path | None = None,
    ) -> TTSResult:
        text = _validate_text(text)
        requested = _normalize_language(language) or "hi"
        if requested not in MMS_LANGUAGE_MODELS:
            raise TTSUnsupportedLanguageError(
                f"mock provider has no model for language {language!r}"
            )
        if self._behavior == "raise":
            raise TTSSynthesisError("mock provider failure")
        started = time.perf_counter()
        t = np.arange(int(self._duration * self._sample_rate)) / self._sample_rate
        samples = (0.5 * np.sin(2 * np.pi * self._tone_hz * t)).astype(np.float32)
        synthesis_ms = (time.perf_counter() - started) * 1000.0
        result = TTSResult(
            text=text,
            language=requested,
            sample_rate=self._sample_rate,
            duration_seconds=self._duration,
            synthesis_time_ms=synthesis_ms,
            rtf=(synthesis_ms / 1000.0) / self._duration,
            model=f"mock-tts-{requested}",
            provider=self.name,
            output_path=None,
            audio=samples,
        )
        if output_path is not None:
            result.output_path = str(write_tts_wav(output_path, samples, self._sample_rate))
        return result

    def describe(self) -> dict[str, Any]:
        return {"provider": self.name, "model": "mock", "tone_hz": self._tone_hz}


class MMSIndicTTS(TTSProvider):
    """Meta MMS-TTS VITS per-language checkpoints on CPU (local, ₹0).

    The model is loaded lazily on the first ``synthesize`` call, so constructing
    the provider never downloads anything. ``model_factory`` is an injection
    point for offline unit tests (no model download). The same provider instance
    caches the model and reuses it for every subsequent call.
    """

    name = "mms"

    def __init__(
        self,
        language: str = "hi",
        cache_dir: str | Path | None = None,
        model_factory: Callable[[str, str, Any], tuple[Any, Any]] | None = None,
    ):
        self._language = _normalize_language(language) or "hi"
        self._cache_dir = Path(cache_dir) if cache_dir else None
        self._model_factory = model_factory
        self._model: Any | None = None
        self._tokenizer: Any | None = None
        self._model_id: str | None = None
        self._model_load_seconds: float | None = None

    @property
    def model_load_seconds(self) -> float | None:
        """Seconds spent constructing the model (includes first-run download)."""
        return self._model_load_seconds

    def _resolve_model_id(self, language: str) -> str:
        normalized = _normalize_language(language)
        if normalized not in MMS_LANGUAGE_MODELS:
            raise TTSUnsupportedLanguageError(
                f"no MMS-TTS checkpoint for language {language!r}; "
                f"supported: {sorted(set(MMS_LANGUAGE_MODELS))}"
            )
        return MMS_LANGUAGE_MODELS[normalized]

    def _ensure_model(self, language: str) -> tuple[Any, Any, str]:
        model_id = self._resolve_model_id(language)
        if self._model is not None and self._model_id == model_id:
            return self._model, self._tokenizer, model_id
        started = time.perf_counter()
        if self._model_factory is not None:
            self._model, self._tokenizer = self._model_factory(model_id, language, self._cache_dir)
        else:
            try:
                from transformers import AutoTokenizer, VitsModel
            except ImportError as exc:
                raise TTSConfigurationError(
                    "transformers is not installed; install the project extras "
                    "(`pip install -e .[tts]`) to enable local TTS."
                ) from exc
            try:
                self._tokenizer = AutoTokenizer.from_pretrained(model_id, cache_dir=self._cache_dir)
                self._model = VitsModel.from_pretrained(model_id, cache_dir=self._cache_dir)
                self._model.eval()
            except Exception as exc:  # noqa: BLE001 - model load failures
                raise TTSConfigurationError(
                    f"could not load MMS-TTS model {model_id!r}: {type(exc).__name__}: {exc}"
                ) from exc
        self._model_id = model_id
        self._model_load_seconds = time.perf_counter() - started
        return self._model, self._tokenizer, model_id

    def synthesize(
        self,
        text: str,
        *,
        language: str | None = None,
        output_path: str | Path | None = None,
    ) -> TTSResult:
        text = _validate_text(text)
        requested = _normalize_language(language) or self._language
        model, tokenizer, model_id = self._ensure_model(requested)
        started = time.perf_counter()
        try:
            inputs = tokenizer(text, return_tensors="pt")
            output = model(**inputs).waveform
            samples = np.asarray(output[0].detach().numpy(), dtype=np.float32)
        except TTSError:
            raise
        except Exception as exc:  # noqa: BLE001 - provider failures surface as TTSSynthesisError
            raise TTSSynthesisError(
                f"MMS-TTS synthesis failed: {type(exc).__name__}: {exc}"
            ) from exc
        synthesis_ms = (time.perf_counter() - started) * 1000.0
        sample_rate = int(getattr(model.config, "sampling_rate", DEFAULT_SAMPLE_RATE))
        duration = samples.size / sample_rate
        result = TTSResult(
            text=text,
            language=requested,
            sample_rate=sample_rate,
            duration_seconds=duration,
            synthesis_time_ms=synthesis_ms,
            rtf=(synthesis_ms / 1000.0) / duration if duration > 0 else 0.0,
            model=model_id.split("/")[-1],
            provider=self.name,
            load_time_ms=(self._model_load_seconds or 0.0) * 1000.0,
            output_path=None,
            audio=samples,
        )
        if output_path is not None:
            result.output_path = str(write_tts_wav(output_path, samples, sample_rate))
        return result

    def describe(self) -> dict[str, Any]:
        return {
            "provider": self.name,
            "language": self._language,
            "model": self._model_id or MMS_LANGUAGE_MODELS.get(self._language),
        }


_TTS_ALIASES = {
    "mms": "mms",
    "mms_tts": "mms",
    "vits": "mms",
    "facebook": "mms",
    "local": "mms",
}


def normalize_tts_provider_name(provider: str) -> str:
    normalized = provider.strip().lower().replace("-", "_")
    return _TTS_ALIASES.get(normalized, normalized)


def build_tts_provider(
    provider: str,
    language: str = "hi",
    cache_dir: str | Path | None = None,
) -> TTSProvider:
    """Build a single TTS provider by name."""
    normalized = normalize_tts_provider_name(provider)
    if normalized == "mock":
        return MockTTS()
    if normalized == "mms":
        return MMSIndicTTS(language=language, cache_dir=cache_dir)
    raise TTSConfigurationError(f"Unsupported TTS provider: {provider}")
