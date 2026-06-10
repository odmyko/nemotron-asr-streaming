# Nemotron 3.5 ASR — real-time streaming speech-to-text

Self-hosted, GPU-accelerated **streaming** speech recognition built on
[`nvidia/nemotron-3.5-asr-streaming-0.6b`](https://huggingface.co/nvidia/nemotron-3.5-asr-streaming-0.6b)
— a multilingual (40 language-locales) cache-aware **FastConformer-RNNT** model.

It ships as two containers: a FastAPI inference server on your GPU and a small
web UI, so you can open a page on your phone or laptop and watch words appear
**as you speak**. Clone it and run one script.

```bash
git clone git@github.com:tehtommeh/nemotron-asr-streaming.git
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
git clone git@github.com:tehtommeh/nemotron-asr-streaming.git
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

- [`requirements.txt`](requirements.txt) — the install manifest used by the
  Docker image and `scripts/dev.sh`. NeMo is pinned to an exact git commit;
  torch is installed separately from the cu124 wheel index (see the Dockerfile /
  `dev.sh`).
- [`requirements.lock.txt`](requirements.lock.txt) — a full `uv pip freeze` of a
  known-good environment (exact versions of every transitive dependency) for
  reference / debugging.

Key versions: `torch==2.6.0+cu124`, `nemo-toolkit @ git+…@73cfbcf`,
`fastapi==0.136.3`, `uvicorn==0.49.0`, Python 3.12, CUDA 12.4.

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
