"""Reproducible CPU STT benchmark (Phase 3A).

Measures per sample: audio duration, model load time (first sample only, may
include a first-run model download), transcription time, total processing
time, real-time factor (RTF), detected language and transcript. When a ground
-truth reference exists (``samples/<name>.txt`` sidecar) the WER is computed.

Sidecar format::

    THE REFERENCE TRANSCRIPT TEXT
    language = en
    source = ...            # optional metadata lines (key = value)

An empty reference (or a missing sidecar) means "no ground truth"; WER is then
reported as not measured rather than invented.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .stt import STTProvider
from .wer import word_error_rate


@dataclass(slots=True)
class STTSample:
    path: str
    label: str
    language_expected: str | None = None
    reference: str | None = None
    source: str | None = None


def load_samples(samples_dir: str | Path = "samples") -> list[STTSample]:
    """Discover ``*.wav`` files plus their ``*.txt`` ground-truth sidecars."""
    base = Path(samples_dir)
    if not base.is_dir():
        raise FileNotFoundError(f"samples directory not found: {base}")
    samples: list[STTSample] = []
    for wav in sorted(base.glob("*.wav")):
        label = wav.stem
        reference: str | None = None
        language_expected: str | None = None
        source: str | None = None
        sidecar = wav.with_suffix(".txt")
        if sidecar.exists():
            lines = sidecar.read_text(encoding="utf-8").splitlines()
            if lines and lines[0].strip():
                reference = lines[0].strip()
            for line in lines[1:]:
                if "=" in line:
                    key, _, value = line.partition("=")
                    if key.strip() == "language":
                        language_expected = value.strip() or None
                    elif key.strip() == "source":
                        source = value.strip() or None
        samples.append(
            STTSample(
                path=str(wav),
                label=label,
                language_expected=language_expected,
                reference=reference,
                source=source,
            )
        )
    return samples


def _best_run(provider: STTProvider, sample: STTSample, repeat: int) -> dict[str, Any] | None:
    """Run the provider and return the fastest valid run (or an error row)."""
    best: dict[str, Any] | None = None
    for _ in range(max(1, repeat)):
        started = time.perf_counter()
        try:
            result = provider.transcribe(sample.path, language=sample.language_expected)
        except Exception as exc:  # noqa: BLE001 - report, do not abort the benchmark
            return {"error": f"{type(exc).__name__}: {exc}"}
        total_ms = (time.perf_counter() - started) * 1000.0
        record = {"result": result, "total_ms": total_ms}
        if best is None or total_ms < best["total_ms"]:
            best = record
    return best


def run_stt_benchmark(
    provider: STTProvider,
    samples: list[STTSample],
    *,
    repeat: int = 1,
    warmup: bool = True,
) -> dict[str, Any]:
    """Run the provider over all samples and return a structured report.

    The first sample also pays the lazy model load; when the provider exposes
    ``model_load_seconds`` it is reported separately so transcription time and
    RTF measure inference only (per-sample total includes the load).
    """
    if warmup:
        provider.warmup()

    rows: list[dict[str, Any]] = []
    for index, sample in enumerate(samples):
        row: dict[str, Any] = {
            "label": sample.label,
            "path": sample.path,
            "language_expected": sample.language_expected,
            "has_reference": bool(sample.reference),
        }
        best = _best_run(provider, sample, repeat)
        if best is None or "error" in best:
            if best is not None:
                row.update(best)
            else:
                row["error"] = "no successful run"
            rows.append(row)
            continue

        result = best["result"]
        # the model loads lazily on the first successful transcription only
        model_load_seconds = getattr(provider, "model_load_seconds", None)
        if index > 0:
            model_load_seconds = None
        row["model_load_ms"] = round(model_load_seconds * 1000.0, 1) if model_load_seconds else 0.0
        row["detected_language"] = result.language
        row["language_probability"] = round(result.language_probability, 4)
        row["audio_duration_seconds"] = round(result.duration_seconds, 3)
        row["transcription_time_ms"] = round(result.processing_time_ms, 2)
        row["total_processing_ms"] = round(best["total_ms"], 2)
        row["rtf"] = round(result.rtf, 4)
        row["rtf_including_load"] = round(
            (best["total_ms"] / 1000.0) / result.duration_seconds if result.duration_seconds else 0.0,
            4,
        )
        row["text"] = result.text
        if sample.reference:
            row["wer"] = round(word_error_rate(sample.reference, result.text), 4)
        else:
            row["wer"] = None
            row["wer_note"] = "accuracy NOT MEASURED: no ground-truth transcript"
        rows.append(row)

    measured = [r for r in rows if "wer" in r and r.get("wer") is not None]
    with_reference = [r for r in rows if r.get("has_reference")]
    summary = {
        "samples": len(rows),
        "with_reference": len(with_reference),
        "wer_measured": len(measured),
        "errors": len([r for r in rows if "error" in r]),
        "avg_wer": round(sum(r["wer"] for r in measured) / len(measured), 4) if measured else None,
    }
    return {
        "provider": provider.describe(),
        "repeat": repeat,
        "summary": summary,
        "samples": rows,
    }
