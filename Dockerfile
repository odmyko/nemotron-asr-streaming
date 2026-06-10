# Nemotron 3.5 ASR streaming — GPU inference server.
# Mirrors the locally-validated Python 3.12 + torch cu124 + NeMo (git) stack.
FROM nvidia/cuda:12.4.1-cudnn-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_HUB_OFFLINE=1 \
    TRANSFORMERS_OFFLINE=1

# System deps: ffmpeg (audio decoding), libsndfile1 (soundfile), Python 3.12.
RUN apt-get update && apt-get install -y --no-install-recommends \
        software-properties-common ca-certificates curl git ffmpeg libsndfile1 && \
    add-apt-repository -y ppa:deadsnakes/ppa && apt-get update && \
    apt-get install -y --no-install-recommends \
        python3.12 python3.12-dev python3.12-venv && \
    curl -sS https://bootstrap.pypa.io/get-pip.py | python3.12 && \
    ln -sf /usr/bin/python3.12 /usr/local/bin/python && \
    ln -sf /usr/bin/python3.12 /usr/local/bin/python3 && \
    rm -rf /var/lib/apt/lists/*

# Build deps then CUDA torch (cu124, matching the host driver / 3090).
RUN python -m pip install --upgrade pip Cython packaging setuptools wheel && \
    python -m pip install torch==2.6.0 torchaudio==2.6.0 \
        --index-url https://download.pytorch.org/whl/cu124

# NeMo (from git) + serving deps. This layer is the slow one; keep it cached.
COPY requirements.txt /tmp/requirements.txt
RUN python -m pip install -r /tmp/requirements.txt

WORKDIR /app
COPY app/ /app/app/

# Model is mounted at runtime (see docker-compose.yml / README) to keep the
# image small and reuse the on-disk download cache.
ENV MODEL_PATH=/models/nemotron-3.5-asr-streaming-0.6b/nemotron-3.5-asr-streaming-0.6b.nemo \
    ASR_DEVICE=cuda

EXPOSE 8000
CMD ["python", "-m", "uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
