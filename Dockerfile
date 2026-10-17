
# Nemotron 3.5 Streaming ASR
# RTX 5090 / Blackwell SM120

FROM nvidia/cuda:12.8.1-cudnn-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_HUB_OFFLINE=1 \
    TRANSFORMERS_OFFLINE=1

# System dependencies + Python 3.12
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    software-properties-common \
    ca-certificates curl git ffmpeg libsndfile1 && \
    add-apt-repository -y ppa:deadsnakes/ppa && \
    apt-get update && \
    apt-get install -y --no-install-recommends \
    python3.12 python3.12-dev python3.12-venv && \
    python3.12 -m venv /opt/venv && \
    rm -rf /var/lib/apt/lists/*

# Isolate application packages from Ubuntu's system Python packages (e.g. PyGObject).
ENV VIRTUAL_ENV=/opt/venv \
    PATH="/opt/venv/bin:${PATH}"

# Build tools
RUN python -m pip install --upgrade \
    pip Cython packaging setuptools wheel

# PyTorch CUDA 12.8
RUN python -m pip install \
    torch==2.7.1+cu128 \
    torchaudio==2.7.1+cu128 \
    --index-url https://download.pytorch.org/whl/cu128

# Install NeMo and its dependencies.
# Extra PyTorch index allows resolver to locate CUDA wheels.
COPY requirements.txt /tmp/requirements.txt

RUN python -m pip install \
    -r /tmp/requirements.txt \
    --extra-index-url https://download.pytorch.org/whl/cu128

# Validate dependency compatibility
RUN python -m pip check

# Inspect compiled CUDA architectures without requiring GPU access during the build.
RUN python -c "import torch; \
    arch_flags = torch._C._cuda_getArchFlags().split(); \
    print('PyTorch:', torch.__version__); \
    print('CUDA:', torch.version.cuda); \
    print('Architectures:', arch_flags); \
    assert torch.__version__.startswith('2.7.1+cu128'); \
    assert 'sm_120' in arch_flags"

# Application
WORKDIR /app
COPY app/ /app/app/

ENV MODEL_PATH=/models/nemotron-3.5-asr-streaming-0.6b/nemotron-3.5-asr-streaming-0.6b.nemo \
    ASR_DEVICE=cuda

EXPOSE 8000

CMD ["python", "-m", "uvicorn", "app.main:app", \
    "--host", "0.0.0.0", "--port", "8000"]
