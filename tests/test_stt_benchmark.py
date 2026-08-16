"""Offline unit tests for the Phase 3A STT benchmark (measurements + WER)."""

from __future__ import annotations

import wave
from pathlib import Path

import pytest

from indicvoicerag.stt import MockSTT, STTProvider
from indicvoicerag.stt_benchmark import STTSample, load_samples, run_stt_benchmark
from indicvoicerag.wer import word_error_rate


class _RecordingProvider(STTProvider):
    name = "recording"

    def __init__(self) -> None:
        self.calls: list[str | None] = []

    def transcribe(self, audio_path, *, language=None):
        self.calls.append(language)
        return MockSTT(text="x").transcribe(audio_path, language=language)


def _write_wav(path: Path, seconds: float = 0.5, rate: int = 16000) -> Path:
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(rate)
        handle.writeframes(b"\x00\x00" * int(seconds * rate))
    return path


# --- WER ---
def test_wer_identical_text() -> None:
    assert word_error_rate("the cat sat on the mat", "the cat sat on the mat") == 0.0


def test_wer_fully_different() -> None:
    assert word_error_rate("one two three", "alpha beta gamma") == 1.0


def test_wer_single_substitution() -> None:
    assert word_error_rate("the cat sat", "the dog sat") == pytest.approx(1 / 3)


def test_wer_case_and_punctuation_insensitive() -> None:
    assert (
        word_error_rate("The cat, sat on the mat!", "the cat sat on the mat") == 0.0
    )


def test_wer_empty_reference() -> None:
    assert word_error_rate("", "anything at all") == 1.0
    assert word_error_rate("", "") == 0.0


def test_wer_hindi_words() -> None:
    ref = "भारत की राजधानी नई दिल्ली है"
    hyp = "भारत की राजधानी दिल्ली है"
    wer = word_error_rate(ref, hyp)
    assert 0.0 < wer < 1.0


# --- sample discovery ---
def test_load_samples_reads_sidecar_ground_truth(tmp_path: Path) -> None:
    _write_wav(tmp_path / "en.wav")
    (tmp_path / "en.txt").write_text(
        "THE REFERENCE TRANSCRIPT\nlanguage = en\nsource = test\n", encoding="utf-8"
    )
    samples = load_samples(tmp_path)
    assert len(samples) == 1
    assert samples[0].label == "en"
    assert samples[0].reference == "THE REFERENCE TRANSCRIPT"
    assert samples[0].language_expected == "en"
    assert samples[0].source == "test"


def test_load_samples_missing_dir_raises(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        load_samples(tmp_path / "nope")


def test_load_samples_empty_first_line_means_no_reference(tmp_path: Path) -> None:
    _write_wav(tmp_path / "hi.wav")
    (tmp_path / "hi.txt").write_text(
        "\nlanguage = hi\nsource = dummy\n", encoding="utf-8"
    )
    sample = load_samples(tmp_path)[0]
    assert sample.reference is None
    assert sample.language_expected == "hi"


# --- benchmark measurements ---
def test_benchmark_measures_all_fields(tmp_path: Path) -> None:
    audio = _write_wav(tmp_path / "a.wav", seconds=1.0)
    stt = MockSTT(text="hello world")
    report = run_stt_benchmark(
        stt,
        [STTSample(path=str(audio), label="case-a")],
        warmup=True,
    )
    assert report["provider"]["provider"] == "mock"
    assert len(report["samples"]) == 1
    case = report["samples"][0]
    assert case["label"] == "case-a"
    assert case["audio_duration_seconds"] == pytest.approx(1.0)
    assert case["transcription_time_ms"] >= 0
    assert case["rtf"] >= 0
    assert case["text"] == "hello world"
    assert case["detected_language"] == "en"
    assert case["wer"] is None
    assert "accuracy NOT MEASURED" in case["wer_note"]


def test_benchmark_computes_wer_with_ground_truth(tmp_path: Path) -> None:
    audio = _write_wav(tmp_path / "b.wav")
    stt = MockSTT(text="the cat sat")
    report = run_stt_benchmark(
        stt,
        [STTSample(path=str(audio), label="b", reference="the dog sat")],
        warmup=False,
    )
    assert report["samples"][0]["wer"] == pytest.approx(1 / 3, abs=0.001)


def test_benchmark_reports_case_error_without_aborting(tmp_path: Path) -> None:
    good = _write_wav(tmp_path / "good.wav", seconds=0.5)
    bad = tmp_path / "bad.wav"
    bad.write_bytes(b"not audio")
    stt = MockSTT(text="The capital of India is New Delhi.")
    report = run_stt_benchmark(
        stt,
        [STTSample(path=str(good), label="good"), STTSample(path=str(bad), label="bad")],
        warmup=True,
    )
    assert len(report["samples"]) == 2
    assert report["samples"][0]["text"] == "The capital of India is New Delhi."
    assert "error" not in report["samples"][0]
    assert "error" in report["samples"][1]
    assert report["summary"]["errors"] == 1


def test_benchmark_forwards_pinned_language(tmp_path: Path) -> None:
    audio = _write_wav(tmp_path / "pin.wav")
    provider = _RecordingProvider()
    report = run_stt_benchmark(
        provider,
        [STTSample(path=str(audio), label="pin", language_expected="hi")],
        warmup=False,
    )
    assert provider.calls == ["hi"]
    assert report["samples"][0]["detected_language"] == "hi"


def test_benchmark_summary_counts_references(tmp_path: Path) -> None:
    a = _write_wav(tmp_path / "a.wav")
    b = _write_wav(tmp_path / "b.wav")
    stt = MockSTT(text="same")
    report = run_stt_benchmark(
        stt,
        [
            STTSample(path=str(a), label="a", reference="same"),
            STTSample(path=str(b), label="b"),
        ],
        warmup=False,
    )
    assert report["summary"]["samples"] == 2
    assert report["summary"]["with_reference"] == 1
    assert report["summary"]["wer_measured"] == 1
    assert report["summary"]["avg_wer"] == 0.0
