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
import os
from contextlib import asynccontextmanager

import torch
from fastapi import (
    FastAPI, File, Form, HTTPException, Query, UploadFile, WebSocket,
    WebSocketDisconnect,
)
from fastapi.responses import HTMLResponse, JSONResponse

from .asr import ASREngine
from .audio import decode_to_mono16k_async, pcm16_to_float32, resample

DEFAULT_MODEL_PATH = os.path.join(
    os.path.dirname(__file__), "..", "model_cache",
    "nemotron-3.5-asr-streaming-0.6b", "nemotron-3.5-asr-streaming-0.6b.nemo",
)
MODEL_PATH = os.environ.get("MODEL_PATH", DEFAULT_MODEL_PATH)
DEVICE = os.environ.get("ASR_DEVICE", "cuda" if torch.cuda.is_available() else "cpu")

engine: ASREngine | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global engine
    engine = ASREngine(MODEL_PATH, device=DEVICE)
    yield
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
    sample_rate: int = Query(16000, description="Sample rate of the PCM you send."),
    strip_lang_tags: bool = Query(True),
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
    session = e.new_session(lang=language, strip_tags=strip_lang_tags)
    await ws.send_json({"type": "ready", "language": language,
                        "model_sample_rate": e.sample_rate, "input_sample_rate": sample_rate})
    try:
        while True:
            msg = await ws.receive()
            if msg.get("type") == "websocket.disconnect":
                break
            if (data := msg.get("bytes")) is not None:
                samples = pcm16_to_float32(data)
                if sample_rate != e.sample_rate:
                    samples = await asyncio.to_thread(resample, samples, sample_rate, e.sample_rate)
                outs = await asyncio.to_thread(session.add_audio, samples)
                if outs:
                    await ws.send_json({"type": "partial", "text": outs[-1]})
            elif (text := msg.get("text")) is not None:
                try:
                    ctrl = json.loads(text)
                except json.JSONDecodeError:
                    ctrl = {"type": text.strip()}
                if ctrl.get("type") in ("end", "finalize", "stop"):
                    final = await asyncio.to_thread(session.finalize)
                    await ws.send_json({"type": "final", "text": final})
                    break
    except WebSocketDisconnect:
        pass
    finally:
        if not session.finished:
            await asyncio.to_thread(session.finalize)
        try:
            await ws.close()
        except RuntimeError:
            pass


@app.get("/", response_class=HTMLResponse, tags=["meta"])
async def index():
    return HTMLResponse(INDEX_HTML)


INDEX_HTML = """<!doctype html>
<html><head><meta charset="utf-8"><title>Nemotron 3.5 ASR Streaming</title>
<style>
 body{font-family:system-ui,sans-serif;max-width:820px;margin:2rem auto;padding:0 1rem;color:#111}
 h1{font-size:1.4rem} button{font-size:1rem;padding:.5rem 1rem;margin-right:.5rem;cursor:pointer}
 #out{white-space:pre-wrap;border:1px solid #ccc;border-radius:8px;padding:1rem;min-height:5rem;margin-top:1rem}
 .row{margin:.8rem 0} select,input{font-size:1rem;padding:.3rem}
 a{color:#06c} .muted{color:#666;font-size:.9rem}
</style></head><body>
<h1>Nemotron 3.5 ASR — streaming demo</h1>
<p class="muted">Cache-aware FastConformer-RNNT on GPU. See the
 <a href="/docs">API docs</a>. WebSocket endpoint: <code>/ws/transcribe</code>.</p>
<div class="row">
  Language: <select id="lang"><option value="auto">auto</option></select>
  <label><input type="checkbox" id="strip" checked> strip lang tags</label>
</div>
<div class="row">
  <button id="start">🎤 Start mic streaming</button>
  <button id="stop" disabled>⏹ Stop</button>
</div>
<div class="row">
  <input type="file" id="fileInput" accept="audio/*">
  <button id="upload">Transcribe file</button>
</div>
<div id="out"></div>
<script>
const out=document.getElementById('out'), langSel=document.getElementById('lang');
fetch('/languages').then(r=>r.json()).then(d=>{
  for(const l of d.languages){ if(l==='auto')continue;
    const o=document.createElement('option');o.value=o.textContent=l;langSel.appendChild(o);} });
let ws,ctx,proc,src,stream;
function pcm16(f32){const b=new Int16Array(f32.length);
  for(let i=0;i<f32.length;i++){let s=Math.max(-1,Math.min(1,f32[i]));b[i]=s<0?s*0x8000:s*0x7fff;}return b;}
document.getElementById('start').onclick=async()=>{
  out.textContent=''; stream=await navigator.mediaDevices.getUserMedia({audio:true});
  ctx=new (window.AudioContext||window.webkitAudioContext)();
  const sr=ctx.sampleRate, strip=document.getElementById('strip').checked;
  const proto=location.protocol==='https:'?'wss':'ws';
  ws=new WebSocket(`${proto}://${location.host}/ws/transcribe?language=${langSel.value}&sample_rate=${sr}&strip_lang_tags=${strip}`);
  ws.binaryType='arraybuffer';
  ws.onmessage=e=>{const m=JSON.parse(e.data);
    if(m.type==='partial'||m.type==='final')out.textContent=m.text;
    if(m.type==='error')out.textContent='ERROR: '+m.detail;};
  ws.onopen=()=>{src=ctx.createMediaStreamSource(stream);
    proc=ctx.createScriptProcessor(4096,1,1);
    proc.onaudioprocess=ev=>{if(ws.readyState===1)
      ws.send(pcm16(ev.inputBuffer.getChannelData(0)).buffer);};
    src.connect(proc);proc.connect(ctx.destination);};
  document.getElementById('start').disabled=true;document.getElementById('stop').disabled=false;
};
document.getElementById('stop').onclick=()=>{
  if(proc){proc.disconnect();src.disconnect();}
  if(stream)stream.getTracks().forEach(t=>t.stop());
  if(ctx)ctx.close();
  if(ws&&ws.readyState===1)ws.send(JSON.stringify({type:'end'}));
  document.getElementById('start').disabled=false;document.getElementById('stop').disabled=true;
};
document.getElementById('upload').onclick=async()=>{
  const f=document.getElementById('fileInput').files[0]; if(!f){out.textContent='Pick a file first.';return;}
  out.textContent='Transcribing '+f.name+' …';
  const fd=new FormData();fd.append('file',f);fd.append('language',langSel.value);
  fd.append('strip_lang_tags',document.getElementById('strip').checked);
  const r=await fetch('/transcribe',{method:'POST',body:fd});
  const d=await r.json(); out.textContent=r.ok?d.text:('ERROR: '+(d.detail||JSON.stringify(d)));
};
</script></body></html>"""
