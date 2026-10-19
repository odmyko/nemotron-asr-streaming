# Nemotron 3.5 ASR — real-time streaming speech-to-text

Self-hosted, GPU-accelerated **streaming** speech recognition built on
[`nvidia/nemotron-3.5-asr-streaming-0.6b`](https://huggingface.co/nvidia/nemotron-3.5-asr-streaming-0.6b)
— a multilingual (40 language-locales) cache-aware **FastConformer-RNNT** model.

It ships as two containers: a FastAPI inference server on your GPU and a small
web UI, so you can open a page on your phone or laptop and watch words appear
**as you speak**. Clone it and run one script.

```bash
git clone git@github.com:odmyko/nemotron-asr-streaming.git
cd nemotron-asr-streaming
./run.sh
# → http://localhost:8080   (mic in the browser, live transcription)
# → http://localhost:8000/docs   (REST API)
```

---

## Contents

- [What you get](#what-you-get)
- [Architecture](#architecture)
- [Requirements](#requirements)
- [Quick start](#quick-start)
- [Microphone & HTTPS (read this for phones)](#microphone--https-read-this-for-phones)
- [Using the web UI](#using-the-web-ui)
- [API reference](#api-reference)
- [Local development (no Docker)](#local-development-no-docker)
- [How it works](#how-it-works)
- [Configuration](#configuration)
- [Dependency versions & reproducibility](#dependency-versions--reproducibility)
- [Testing & validation](#testing--validation)
- [Troubleshooting](#troubleshooting)
- [Managing the service](#managing-the-service)
- [License](#license)

## What you get

- **Real-time streaming ASR** over WebSockets — low-latency partial hypotheses
  that refine as audio arrives (cache-aware, non-overlapping chunks).
- **Browser UI** with **Start / Stop / Clear** and a live transcript.
- **One-shot file transcription** REST endpoint (any format via ffmpeg).
- **Multilingual**: 40 language-locales from one model, or `auto` language
  detection.
- **Runs fully offline** after the initial model download.
- **One-command bootstrap** (`./run.sh`) and a Docker-free dev path
  (`scripts/dev.sh`).

## Architecture

```
                                  ┌──────────────────────────────────────────┐
 Browser (phone / laptop)         │             Docker host (GPU)            │
 ┌───────────────────────┐        │                                          │
 │  http://host:8080      │        │   ┌────────────────────────────────┐    │
 │  • mic → PCM16 frames  │  ws    │   │ web (nginx)  :8080             │    │
 │  • Start/Stop/Clear    │ ─────► │   │  • serves the UI               │    │
 │  • live transcript     │  http  │   │  • reverse-proxies /ws,/transcribe, │
 └───────────────────────┘        │   │    /languages,/health,/docs ──┐│    │
                                   │   └───────────────────────────────┼┘    │
                                   │                                   ▼      │
                                   │   ┌────────────────────────────────┐    │
                                   │   │ asr (FastAPI + NeMo)  :8000    │    │
                                   │   │  • Nemotron 3.5 ASR on CUDA    │    │
                                   │   │  • cache-aware streaming engine│    │
                                   │   └────────────────────────────────┘    │
                                   │        ▲ mounts ./model_cache (ro)       │
                                   └────────┼─────────────────────────────────┘
                                            │
                                    nvidia/nemotron-3.5-asr-streaming-0.6b.nemo
```

| Service | Container                | Port | Role                                            |
|---------|--------------------------|------|-------------------------------------------------|
| `asr`   | `nemotron-asr-streaming` | 8000 | FastAPI model server on the GPU                 |
| `web`   | `nemotron-asr-web`       | 8080 | nginx: static UI + reverse proxy to the API     |

Only the API talks to the GPU. The web container is a thin nginx front so the
page is single-origin (no CORS, no mixed-content) and you only expose one port.

## Requirements

**Hardware**
- NVIDIA GPU with ≥ 6 GB VRAM (developed/tested on an RTX 3090, 24 GB; the model
  uses ~3–5 GB).
- ~15 GB free disk (model ≈ 2.4 GB, API image ≈ 10 GB).

**Software**
- Linux with a recent NVIDIA driver (CUDA 12.4+ capable; tested on driver 580 / CUDA 13).
- **Docker** 20.10+ with the **NVIDIA Container Toolkit** (for `--gpus`).
- For the no-Docker dev path: [`uv`](https://docs.astral.sh/uv/) (it provisions Python 3.12 for you).

<details>
<summary>Install Docker + NVIDIA Container Toolkit (Ubuntu)</summary>

```bash
# Docker
curl -fsSL https://get.docker.com | sudo sh
sudo usermod -aG docker "$USER"   # then log out/in

# NVIDIA Container Toolkit
curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey \
  | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list \
  | sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' \
  | sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list
sudo apt-get update && sudo apt-get install -y nvidia-container-toolkit
sudo nvidia-ctk runtime configure --runtime=docker
sudo systemctl restart docker

# verify
docker run --rm --gpus all nvidia/cuda:12.4.1-base-ubuntu22.04 nvidia-smi
```
</details>

## Quick start

```bash
git clone git@github.com:odmyko/nemotron-asr-streaming.git
cd nemotron-asr-streaming
./run.sh
```

`run.sh` will:
1. check Docker / Compose / GPU runtime,
2. download the model into `model_cache/` (idempotent — skipped if present),
3. `docker compose up -d --build`,
4. wait for the model to load, then print the URLs.

First run takes a while (CUDA base image + torch + NeMo build, plus the model
download). Subsequent starts are ~30–60 s (just model load).

Prefer to do it by hand?

```bash
bash scripts/download_model.sh     # weights -> ./model_cache
docker compose up -d --build
curl http://localhost:8000/health
```

## Microphone & HTTPS (read this for phones)

Browsers only allow microphone capture in a **secure context**. That means:

- ✅ `http://localhost:8080` on the **host machine** works (localhost is trusted).
- ❌ `http://<lan-ip>:8080` from a **phone or another laptop** will **not** get
  mic access over plain HTTP — you need HTTPS.

The web container serves **HTTP only**. To use the mic from other devices, put
TLS in front. Easiest options:

- **Tailscale Serve / Funnel** or **Cloudflare Tunnel** — gives you an HTTPS URL
  that proxies to `http://localhost:8080`, no certs to manage.
- **A reverse proxy with a cert** (Caddy is one line):
  ```
  asr.example.com {
      reverse_proxy localhost:8080
  }
  ```
- **Self-signed cert on nginx** — add a `443`/`8443` TLS `server` block to
  `web/default.conf.template` and a cert; the UI auto-uses `wss://` when loaded
  over HTTPS.

File-upload transcription in the UI works over plain HTTP from any device — only
live mic capture needs the secure context.

## Using the web UI

Open the UI, pick a **language** (or leave `auto-detect`), then:

- **▶ Start** — grant mic access; partial text streams in and refines live.
- **⏹ Stop** — flushes the tail and shows the final transcript.
- **🗑 Clear** — wipes the transcript.

There's also a file picker to transcribe an existing audio file.

## API reference

Interactive docs (REST only): **http://localhost:8000/docs** ·
ReDoc: **/redoc**.

| Method | Path             | Purpose                                          |
|--------|------------------|--------------------------------------------------|
| GET    | `/`              | Browser demo (mic streaming + file upload)       |
| GET    | `/health`        | Status, device, GPU name                         |
| GET    | `/languages`     | Supported language keys (`auto` = detect)        |
| POST   | `/transcribe`    | One-shot transcription of an uploaded file       |
| WS     | `/ws/transcribe` | Cache-aware streaming ASR                        |

### `POST /transcribe`

```bash
curl -X POST http://localhost:8000/transcribe \
  -F "file=@test_data/sample_en.wav" \
  -F "language=auto" \
  -F "strip_lang_tags=true"
```

```json
{
  "text": "Well, I don't wish to see it any more, observed Phoebe, ...",
  "language": "auto",
  "duration_sec": 7.435,
  "filename": "sample_en.wav"
}
```

Any ffmpeg-decodable format is accepted (wav/flac/ogg/mp3/m4a/webm/opus…); audio
is converted to 16 kHz mono internally.

### `WS /ws/transcribe`

Query params: `language` (key or `auto`), `sample_rate` (Hz of the PCM you
send — resampled to 16 kHz server-side), `strip_lang_tags` (default `true`).

Protocol:
- Send **binary** frames of little-endian **int16 PCM, mono** at `sample_rate`.
- Send the text message `{"type":"end"}` (or just close) to flush the tail.
- Receive JSON: `{"type": "ready"|"partial"|"final"|"error", "text": ...}`.

Minimal client: see [`scripts/ws_client.py`](scripts/ws_client.py)

```bash
.venv/bin/python scripts/ws_client.py test_data/sample_en.wav en-US
```

> WebSocket endpoints don't appear in the OpenAPI/Swagger page (a FastAPI
> limitation), so `/ws/transcribe` is documented here.

## Local development (no Docker)

Runs the API directly in a venv with autoreload — handy for hacking on
`app/`. Needs `uv` and a GPU.

```bash
scripts/dev.sh        # creates .venv, installs deps, downloads model, runs uvicorn --reload
```

Then in another shell:

```bash
.venv/bin/python scripts/validate_engine.py     # streaming output == NeMo reference
.venv/bin/python scripts/ws_client.py test_data/sample_en.wav en-US
```

## How it works

- **Cache-aware streaming.** Unlike "buffered" streaming that re-processes
  overlapping windows, this model keeps encoder caches and processes each ~320 ms
  chunk exactly once. `app/asr.py` maintains per-session cache state and feeds
  chunks through NeMo's `conformer_stream_step`. The implementation is validated
  to reproduce NeMo's reference `CacheAwareStreamingAudioBuffer` output
  bit-for-bit (`scripts/validate_engine.py`).
- **Prompt-conditioned languages.** The model is conditioned on a language-ID
  prompt. `auto` lets it detect and emit a `<lang>` tag after terminal
  punctuation (stripped from output by default).
- **Single-GPU serialization.** The language prompt is global model state, so
  all GPU work is serialized behind one lock and the prompt is set per chunk from
  each session's language — correct for concurrent multi-language streams on one
  GPU.
- **Why NeMo from git.** The prompt-conditioned streaming class
  (`EncDecRNNTBPEModelWithPrompt`) only exists on NeMo `main`; the pinned commit
  is in `requirements.txt`.

## Configuration

Environment variables (set in `docker-compose.yml` or your shell):

| Variable       | Default                                   | Used by | Meaning                          |
|----------------|-------------------------------------------|---------|----------------------------------|
| `MODEL_PATH`   | `/models/.../*.nemo` (container)          | `asr`   | Path to the `.nemo` weights      |
| `ASR_DEVICE`   | `cuda` (falls back to `cpu` if no GPU)    | `asr`   | Torch device                     |
| `API_UPSTREAM` | `asr:8000`                                | `web`   | Where nginx proxies the API      |
| `MODEL_REPO_ID`| `nvidia/nemotron-3.5-asr-streaming-0.6b`  | scripts | HF repo for the downloader       |

## Dependency versions & reproducibility

`requirements.txt` declares direct dependencies and pins NeMo Speech to commit
`a3afbd901807bfa5b46ae386573af31dced77834`. `requirements.lock.txt` is the full
Linux/Python 3.12 snapshot used by Docker (together with the direct manifest)
and local development. CUDA wheels are resolved from the cu128 index.

The GPU stack is PyTorch/torchaudio 2.7.1+cu128, Triton 3.3.1, and CUDA 12.8.
Application packages are isolated in `/opt/venv`; the image runs `pip check`
and verifies compiled SM120 support. GCC is required by Triton's runtime helper.

## Testing & validation

| Script                          | What it checks                                             |
|---------------------------------|-----------------------------------------------------------|
| `scripts/load_test.py`          | Model restores onto the GPU; prints streaming config       |
| `scripts/streaming_probe.py`    | Reference cache-aware streaming over a file                |
| `scripts/validate_engine.py`    | `ASREngine` streaming == reference output (asserts match)  |
| `scripts/ws_client.py`          | End-to-end WebSocket streaming client                      |

A 7 s English sample is included at `test_data/sample_en.wav`.

## Troubleshooting

**`could not select device driver "nvidia"` / no GPU in container**
NVIDIA Container Toolkit isn't installed/configured. See the install box above,
then `sudo nvidia-ctk runtime configure --runtime=docker && sudo systemctl restart docker`.

**Mic button says "needs a secure context"**
Expected over plain HTTP from a non-localhost device — see
[Microphone & HTTPS](#microphone--https-read-this-for-phones).

**`/health` slow to come up on first start**
The model loads on startup (~30–60 s) and the first build compiles NeMo. Watch
`docker compose logs -f asr`.

**`Unknown prompt key` / offline `transcribe()` errors**
Use the provided endpoints — everything runs through the streaming path on
purpose (the one-shot Lhotse path needs per-utterance language metadata).

**Out of memory**
Close other GPU apps; check `nvidia-smi`. The model needs ~3–5 GB.

## Managing the service

```bash
docker compose ps                 # status
docker compose logs -f            # tail logs (both services)
docker compose logs -f asr        # API only
docker compose restart            # restart
docker compose up -d --build      # rebuild after code changes
docker compose down               # stop & remove
```

## License

Wrapper code in this repo: **MIT** (see [LICENSE](LICENSE)).

Model weights `nvidia/nemotron-3.5-asr-streaming-0.6b` are distributed by NVIDIA
under **OpenMDW-1.1** — review the terms on the
[model card](https://huggingface.co/nvidia/nemotron-3.5-asr-streaming-0.6b).

## Acknowledgements

- NVIDIA NeMo & the Nemotron Speech team for the model and streaming toolkit.
- Structure inspired by
  [vibevoice-asr-api](https://github.com/tehtommeh/vibevoice-asr-api).

## Live speaker separation

The **Separate speakers** checkbox is available on both `http://localhost:8080`
and the API demo at `http://localhost:8000`. Enable it before Start. The page
shows numbered speakers with consistent colors and updates their transcript
while recording. Stop enters a finishing state until the server flushes the
last audio. Clear clears displayed text without resetting speaker identities.
When opened during model loading, the page waits and retries readiness checks automatically.

Both models run inside the ASR container, using shared GPU weights and separate
session caches. The implementation uses NVIDIA NeMo Speech's `SpeakerTaggedASR`
with Nemotron 3.5 ASR and `nvidia/Nemotron-3-Diarization` (up to eight speakers).
Ordinary transcription retains its original streaming context when the checkbox
is off. Files are transcribed without diarization in this release.

### Install or update

`./run.sh` downloads both checkpoints before building. To update manually:

```bash
bash scripts/download_diarization.sh
docker compose up -d --build
```

For Windows/PowerShell without host Python dependencies:

```powershell
docker run --rm --entrypoint sh -v "${PWD}:/work:ro" -v "${PWD}/model_cache:/work/model_cache" -w /work python:3.12-slim -c "pip install huggingface_hub==1.18.0 && python download_diarization.py"
docker compose up -d --build
```

The diarization checkpoint adds about 199 MB on disk. Both models load from the
read-only `/models` mount; runtime remains offline. Missing/broken diarization
weights leave ordinary ASR available and disable the checkbox with an explanation.

| Variable | Default | Purpose |
|---|---|---|
| `DIAR_ENABLED` | `1` | Load the optional diarization model; set `0` to omit it and skip its bootstrap download. |
| `DIAR_MODEL_PATH` | `/models/Nemotron-3-Diarization/Nemotron-3-Diarization.nemo` | Local checkpoint inside the container. |
| `DIAR_MODEL_DIR` | `model_cache/Nemotron-3-Diarization` | Host download destination; update the volume/path if overridden. |

Setting `DIAR_ENABLED=0` avoids loading the second model and saves GPU memory.
The checkbox is still off by default when the model is available.

### WebSocket extension

Connect to `/ws/transcribe?language=auto&sample_rate=48000&diarize=true`.
Audio remains mono little-endian PCM16. `ready`, `partial`, `final`, and `error`
retain their existing meanings; `partial` and `final` include a complete snapshot:

```json
{"type":"partial","text":"Hello there","segments":[{"text":"Hello there","start":0.32,"end":1.12,"speaker_id":"speaker_0","final":false}]}
```

Replace the previous segments with each new snapshot; do not append snapshots.
Times are seconds from the start of this connection. Text and timing can change
until `final`. Speaker identities survive pauses but reset on a new connection.
No names, persistent voice profiles, or audio recordings are stored. Completed
text transcripts are backed up as described below. `/health` reports
`diarization.available`, `detail`, `max_speakers`, and `buffer_seconds`.
If a requested diarizer is unavailable, the server returns an explicit error.
If inference fails, the UI preserves text already received.

### Latency and limitations

The coupled pipeline derives chunk geometry from ASR; `/health` reports the
actual input buffer duration. This excludes compute, network, and initial model
warmup/JIT compilation. Startup warms both ASR paths and speaker batch sizes before accepting connections;
initial startup can take longer while Triton compiles GPU kernels. Compilation
uses one worker (`TORCHINDUCTOR_COMPILE_THREADS=1`) to limit host memory usage.
Diarization uses BF16 on supported CUDA devices to keep attention
kernels within the RTX 5090 shared-memory limit.

Speaker numbers are anonymous and local to a recording. Similar voices, noise,
short interjections, and overlapping speech can cause incorrect attribution.
The multilingual ASR model is preserved; changing languages does not deliberately
create a new speaker. Accurate simultaneous transcription of overlapping voices
is not guaranteed. Many active speakers increase GPU cost and total latency.

### Verification

```bash
python -m unittest discover -s tests -v
node tests/ui.cjs
python scripts/validate_diarization.py
python scripts/validate_diarization.py --audio dialog.wav --min-speakers 2
python scripts/validate_diarization.py --audio dialog.wav --minutes 10 --realtime
```

Run Python checks in the built ASR environment. The GPU validator tests partials
before Stop, final timestamps, repeated finalization, fresh-session isolation,
and interleaving ordinary ASR with speaker-tagged ASR. It reports first partial,
step duration, input-buffer size, and peak allocated GPU memory. Use mono 16 kHz
audio and a real multi-speaker fixture to assess speaker accuracy.
The Node check exercises UI lifecycle, revised snapshots, Clear, errors, safe text
rendering and restart using simulated microphone/transport adapters.

A paced 600-second replay of NVIDIA's eight-voice demo on RTX 5090 completed in
600.13 seconds of wall time, using 62.38 seconds of processing time (compute RTF
0.104). It produced eight speaker channels; the 95th-percentile processing step
was 93 ms and peak allocated GPU memory was approximately 4.88 GiB. The input
buffer was 1.12 seconds. Startup compilation is handled by the warmup above.
Save a full transcript/metrics artifact with `--report model_cache/validation/report.json`.

References: [NVIDIA ASR integration guide](https://huggingface.co/nvidia/Nemotron-3-Diarization/blob/main/ASR_INTEGRATION_GUIDE.md),
[diarization model](https://huggingface.co/nvidia/Nemotron-3-Diarization).

## Transcript exports and automatic backups

Use **Export TXT**, **Export MD**, or **Export JSON** below the transcript to
download the currently visible text. Exports work for microphone recordings and
uploaded files, with or without speaker separation. They also work during a
recording or after an error; unfinished segments remain marked `final: false`
in JSON. TXT and Markdown include speaker labels and available timestamps.
JSON preserves segment timestamps, speaker IDs, language, source, and recording
IDs. Separate recording IDs distinguish speakers whose numbering restarts in
each recording. Clear removes text from browser exports.

After **Stop**, the server flushes the remaining audio, finalizes recognition,
and automatically writes a complete UTF-8 TXT transcript. Speaker turns include
readable labels and timestamps; ordinary transcripts contain plain text. Successful file
uploads are saved too. Each result has its own unique filename. The UI confirms
the save or warns if the disk write failed, while keeping the transcript
available for export. The final WebSocket response and upload response include
`transcript_id` and `backup` (`status`: `saved`, `disabled`, or `error`;
`filename` is present when saved).

Docker Compose mounts **`./transcripts` on the host** at `/transcripts` in the
ASR container, so backups survive container recreation. Rebuild/recreate the
services after updating: `docker compose up -d --build`. For local development,
the default directory is `transcripts` relative to the server working directory.

| Variable | Default | Purpose |
|---|---|---|
| `TRANSCRIPT_SAVE_ENABLED` | `1` | Set `0` to disable automatic transcript backups. |
| `TRANSCRIPT_DIR` | `transcripts` (`/transcripts` in Compose) | Server directory for TXT backups; must be writable. |

Backups contain the complete transcript for one recording, including text
hidden in the browser by Clear. They never store audio. Use browser JSON export
when structured segment metadata is needed. Existing JSON backups are left untouched.
They are excluded from Git and Docker build context. Files remain until you
delete them; there is no automatic retention policy or web download endpoint.
An interrupted connection or server crash before finalization does not create
a completed backup; export any partial text still displayed in the browser.
