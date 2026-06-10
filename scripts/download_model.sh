#!/usr/bin/env bash
# Fetch the model weights into ./model_cache. Idempotent.
# Uses `uv` if available (fast, ephemeral env), else a python3 venv,
# else falls back to a throwaway Docker container — so it works even with
# no Python on the host.
set -euo pipefail

cd "$(dirname "$0")/.."

MODEL_NAME="nemotron-3.5-asr-streaming-0.6b"
NEMO_FILE="model_cache/${MODEL_NAME}/${MODEL_NAME}.nemo"

if [[ -f "$NEMO_FILE" ]]; then
  echo "✓ Model already downloaded: $NEMO_FILE"
  exit 0
fi

mkdir -p model_cache
echo "Downloading model weights (~2.4 GB)…"

if command -v uv >/dev/null 2>&1; then
  uv run --no-project --with "huggingface_hub==0.36.2" --with hf_transfer \
    python download_model.py
elif command -v python3 >/dev/null 2>&1; then
  python3 -m venv .venv-tools
  .venv-tools/bin/pip install -q --upgrade pip "huggingface_hub==0.36.2" hf_transfer
  .venv-tools/bin/python download_model.py
else
  echo "No uv/python3 found — downloading via Docker…"
  docker run --rm -e HF_HUB_ENABLE_HF_TRANSFER=1 \
    -v "$PWD/model_cache:/work/model_cache" -w /work python:3.12-slim \
    bash -c "pip install -q 'huggingface_hub==0.36.2' hf_transfer && python - <<'PY'
from huggingface_hub import snapshot_download
snapshot_download('nvidia/${MODEL_NAME}', local_dir='model_cache/${MODEL_NAME}')
PY"
fi

echo "✓ Model ready: $NEMO_FILE"
