"""Minimal FastAPI web server for the IndicVoiceRAG live demo.

Usage:
    python -m indicvoicerag.web
    # or
    uvicorn indicvoicerag.web:app --host 0.0.0.0 --port 8000
"""

from __future__ import annotations

import os
import tempfile
import time
from pathlib import Path
from typing import Any

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from .config import load_config
from .pipeline import (
    build_harness,
    build_retrieval_engine,
    build_stt_from_config,
    build_tts_from_config,
)
from .dataset import DatasetInspector

app = FastAPI(title="IndicVoiceRAG", version="0.1.0")

_STATIC_DIR = Path(__file__).parent / "static"
_TTS_DIR = Path(tempfile.gettempdir()) / "indicvoicerag_tts"
_TTS_DIR.mkdir(exist_ok=True)

# Lazy-loaded singletons (initialized on first request)
_state: dict[str, Any] = {}


def _get_state() -> dict[str, Any]:
    if not _state:
        config = load_config()
        inspector = DatasetInspector(config.dataset)
        engine = build_retrieval_engine(config)

        # Try to load persisted index; build fresh if not available
        try:
            engine.store.load()
        except Exception:
            docs = inspector.normalized_sample(config.dataset.sample_size)
            engine.index_documents(docs)

        harness = build_harness(config, engine.query)
        stt = build_stt_from_config(config)
        tts = build_tts_from_config(config)

        _state["config"] = config
        _state["harness"] = harness
        _state["stt"] = stt
        _state["tts"] = tts
    return _state


@app.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/", response_class=HTMLResponse)
async def index() -> HTMLResponse:
    html_path = _STATIC_DIR / "index.html"
    if html_path.exists():
        return HTMLResponse(content=html_path.read_text(encoding="utf-8"))
    return HTMLResponse(content="<h1>IndicVoiceRAG</h1><p>Frontend not found.</p>")


@app.post("/api/ask")
async def ask(
    audio: UploadFile = File(...),
    stt_provider: str = Form("sarvam"),
    language: str = Form("hi-IN"),
) -> JSONResponse:
    """Accept an audio file, run STT -> RAG -> TTS, return JSON."""
    state = _get_state()
    stt = state["stt"]
    harness = state["harness"]
    tts = state["tts"]

    # Save uploaded audio to temp file
    suffix = Path(audio.filename or "upload.wav").suffix or ".wav"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix, dir=str(_TTS_DIR)) as tmp:
        content = await audio.read()
        if not content:
            raise HTTPException(status_code=400, detail="Empty audio file")
        tmp.write(content)
        tmp_path = Path(tmp.name)

    try:
        result: dict[str, Any] = {}

        # Stage 1: STT
        t0 = time.perf_counter()
        try:
            transcription = stt.transcribe(str(tmp_path), language=language if language != "auto" else None)
            result["transcription"] = transcription.text
            result["language"] = transcription.language
            result["stt_ms"] = round(transcription.processing_time_ms, 1)
            result["stt_rtf"] = round(transcription.rtf, 4)
            result["stt_model"] = transcription.model
        except Exception as exc:
            result["error"] = f"STT failed: {type(exc).__name__}"
            result["error_stage"] = "stt"
            return JSONResponse(content=result, status_code=422)

        # Stage 2: RAG
        t1 = time.perf_counter()
        try:
            response = harness.answer(query=transcription.text, top_k=5, debug=True)
            result["rag"] = response.as_dict(include_metrics=True)
        except Exception as exc:
            result["error"] = f"RAG failed: {type(exc).__name__}"
            result["error_stage"] = "rag"
            return JSONResponse(content=result, status_code=422)

        rag_ms = (time.perf_counter() - t1) * 1000.0
        result["rag_total_ms"] = round(rag_ms, 1)

        # Stage 3: TTS (only if RAG answered normally)
        t2 = time.perf_counter()
        answer_text = response.answer
        if answer_text and not response.guardrail:
            try:
                tts_result = tts.synthesize(
                    answer_text,
                    language=language.split("-")[0] if language else "hi",
                    output_path=str(_TTS_DIR / f"answer_{int(time.time()*1000)}.wav"),
                )
                result["tts"] = tts_result.as_dict()
                result["tts_audio_url"] = f"/api/tts/{Path(tts_result.output_path).name}"
                result["tts_ms"] = round(tts_result.synthesis_time_ms, 1)
            except Exception as exc:
                result["tts_error"] = f"TTS failed: {type(exc).__name__}"

        total_ms = (time.perf_counter() - t0) * 1000.0
        result["total_ms"] = round(total_ms, 1)

        return JSONResponse(content=result)

    finally:
        tmp_path.unlink(missing_ok=True)


@app.get("/api/tts/{filename}")
async def serve_tts(filename: str) -> FileResponse:
    """Serve a TTS audio file."""
    file_path = _TTS_DIR / filename
    if not file_path.exists():
        raise HTTPException(status_code=404, detail="Audio file not found")
    return FileResponse(path=str(file_path), media_type="audio/wav", filename=filename)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("indicvoicerag.web:app", host="0.0.0.0", port=8000, log_level="info")
