"""FastAPI service for nvidia/nemotron-3.5-asr-streaming-0.6b.

Endpoints
---------
GET  /                 Browser demo: live mic streaming + file upload.
GET  /health           Liveness/readiness + device info.
GET  /languages        Supported language-ID prompt keys.
POST /transcribe       One-shot transcription of an uploaded audio file.
WS   /ws/transcribe    Cache-aware streaming ASR over a WebSocket.
GET  /docs             Swagger UI (auto-generated for the REST endpoints).
"""
from __future__ import annotations

import asyncio
import json
import logging
import os
from contextlib import asynccontextmanager
from pathlib import Path

# Prevent parallel compiler workers from exhausting Docker Desktop host memory.
os.environ.setdefault("TORCHINDUCTOR_COMPILE_THREADS", "1")

import numpy as np
import torch
import anyio
from fastapi import (
    FastAPI, File, Form, HTTPException, Query, UploadFile, WebSocket,
    WebSocketDisconnect,
)
from fastapi.responses import FileResponse, JSONResponse

from .asr import ASREngine
from .audio import StreamingResampler, decode_to_mono16k_async, pcm16_to_float32

DEFAULT_MODEL_PATH = os.path.join(
    os.path.dirname(__file__), "..", "model_cache",
    "nemotron-3.5-asr-streaming-0.6b", "nemotron-3.5-asr-streaming-0.6b.nemo",
)
MODEL_PATH = os.environ.get("MODEL_PATH", DEFAULT_MODEL_PATH)
DEVICE = os.environ.get("ASR_DEVICE", "cuda" if torch.cuda.is_available() else "cpu")
DIAR_MODEL_PATH = os.environ.get(
    "DIAR_MODEL_PATH", str(Path(__file__).resolve().parent.parent /
                           "model_cache/Nemotron-3-Diarization/Nemotron-3-Diarization.nemo"),
)

engine: ASREngine | None = None
diarization = None
diarization_error = "Diarization is disabled."


@asynccontextmanager
async def lifespan(app: FastAPI):
    global engine, diarization, diarization_error
    diarization = None
    diarization_error = "Diarization is disabled."
    engine = ASREngine(MODEL_PATH, device=DEVICE)
    if os.environ.get("DIAR_ENABLED", "1").lower() not in ("0", "false", "no"):
        try:
            from .diarization import DiarizationEngine

            diarization = DiarizationEngine(engine, DIAR_MODEL_PATH)
            await asyncio.to_thread(diarization.warmup)
            diarization_error = None
        except Exception as exc:
            diarization = None
            logging.getLogger(__name__).exception("Diarization unavailable; ordinary ASR remains enabled")
            diarization_error = str(exc)
    yield
    diarization = None
    engine = None


app = FastAPI(
    title="Nemotron 3.5 ASR Streaming",
    description=(
        "Cache-aware FastConformer-RNNT multilingual streaming ASR "
        "(nvidia/nemotron-3.5-asr-streaming-0.6b) served on GPU."
    ),
    version="1.0.0",
    lifespan=lifespan,
)


def _engine() -> ASREngine:
    if engine is None:
        raise HTTPException(status_code=503, detail="Model still loading")
    return engine


@app.get("/health", tags=["meta"])
async def health():
    if engine is None:
        return JSONResponse({"status": "loading"}, status_code=503)
    return {
        "status": "ok",
        "model": os.path.basename(MODEL_PATH),
        "device": engine.device,
        "cuda": torch.cuda.is_available(),
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "sample_rate": engine.sample_rate,
        "num_languages": len(engine.languages),
        "diarization": {
            "available": diarization is not None,
            "detail": diarization_error,
            "max_speakers": 8,
            "buffer_seconds": diarization.buffer_seconds if diarization else None,
        },
    }


@app.get("/languages", tags=["meta"])
async def languages():
    e = _engine()
    return {"count": len(e.languages), "languages": e.languages,
            "note": "Use 'auto' for automatic language detection."}


@app.post("/transcribe", tags=["asr"])
async def transcribe(
    file: UploadFile = File(..., description="Audio file (wav/flac/ogg/mp3/webm/...)."),
    language: str = Form("auto", description="Language key or 'auto'."),
    strip_lang_tags: bool = Form(True, description="Remove emitted <lang> tags."),
):
    """Transcribe a complete audio file. Decoded to 16kHz mono via ffmpeg, then
    run through the cache-aware streaming model in a single pass."""
    e = _engine()
    if not e.is_supported_language(language):
        raise HTTPException(400, f"Unsupported language '{language}'. See /languages.")
    data = await file.read()
    if not data:
        raise HTTPException(400, "Empty upload.")
    try:
        samples = await decode_to_mono16k_async(data, e.sample_rate)
    except ValueError as ex:
        raise HTTPException(400, str(ex))
    if samples.size == 0:
        raise HTTPException(400, "No decodable audio.")
    text = await asyncio.to_thread(
        e.transcribe_array, samples, language, strip_lang_tags
    )
    return {
        "text": text,
        "language": language,
        "duration_sec": round(samples.size / e.sample_rate, 3),
        "filename": file.filename,
    }


@app.websocket("/ws/transcribe")
async def ws_transcribe(
    ws: WebSocket,
    language: str = Query("auto"),
    sample_rate: int = Query(16000, ge=8000, le=192000, description="Sample rate of the PCM you send."),
    strip_lang_tags: bool = Query(True),
    diarize: bool = Query(False),
):
    """Streaming ASR.

    Send **binary** frames of little-endian int16 PCM (mono) at ``sample_rate``.
    Send the **text** message ``{"type":"end"}`` (or just close) to flush.
    The server replies with JSON: ``{"type":"partial"|"final","text":...}``.
    """
    await ws.accept()
    e = _engine()
    if not e.is_supported_language(language):
        await ws.send_json({"type": "error", "detail": f"Unsupported language '{language}'"})
        await ws.close()
        return
    if diarize and diarization is None:
        await ws.send_json({"type": "error", "detail": diarization_error or "Diarization unavailable"})
        await ws.close(code=1013)
        return
    session = None
    disconnected = False
    try:
        factory = diarization.new_session if diarize else e.new_session
        session = await asyncio.to_thread(factory, lang=language, strip_tags=strip_lang_tags)
        resampler = StreamingResampler(sample_rate, e.sample_rate)
        await ws.send_json({"type": "ready", "language": language, "diarize": diarize,
                            "model_sample_rate": e.sample_rate, "input_sample_rate": sample_rate})

        def result(kind):
            message = {"type": kind, "text": session.text}
            if diarize:
                message["segments"] = session.segments
            return message

        while True:
            msg = await ws.receive()
            if msg.get("type") == "websocket.disconnect":
                disconnected = True
                break
            if (data := msg.get("bytes")) is not None:
                samples = pcm16_to_float32(data)
                samples = await asyncio.to_thread(resampler.process, samples)
                outs = await asyncio.to_thread(session.add_audio, samples) if samples.size else []
                if outs:
                    await ws.send_json(result("partial"))
            elif (text := msg.get("text")) is not None:
                try:
                    ctrl = json.loads(text)
                except json.JSONDecodeError:
                    ctrl = {"type": text.strip()}
                if isinstance(ctrl, dict) and ctrl.get("type") in ("end", "finalize", "stop"):
                    tail = await asyncio.to_thread(
                        resampler.process, np.zeros(0, dtype=np.float32), final=True,
                    )
                    if tail.size:
                        await asyncio.to_thread(session.add_audio, tail)
                    await asyncio.to_thread(session.finalize)
                    await ws.send_json(result("final"))
                    break
    except WebSocketDisconnect:
        disconnected = True
    except Exception:
        logging.getLogger(__name__).exception("Streaming transcription failed")
        if not disconnected:
            try:
                await ws.send_json({"type": "error", "detail": "Transcription failed. Check server logs; your displayed text is preserved."})
            except RuntimeError:
                pass
    finally:
        if session is not None and hasattr(session, "close"):
            # ASGI servers cancel disconnected handlers. Shield cleanup so the
            # queued GPU operation finishes before its session caches are freed.
            with anyio.CancelScope(shield=True):
                await anyio.to_thread.run_sync(session.close)
        try:
            await ws.close()
        except RuntimeError:
            pass


@app.get("/", response_class=FileResponse, tags=["meta"])
async def index():
    # Both ports serve the same UI, including file upload and live diarization.
    return FileResponse(Path(__file__).resolve().parent.parent / "web/index.html")
