#!/usr/bin/env bash
# Local development without Docker: create a Python 3.12 venv, install the exact
# dependency stack, download the model, and run the API with autoreload.
# Requires `uv` (https://docs.astral.sh/uv/) and an NVIDIA GPU + driver.
set -euo pipefail
cd "$(dirname "$0")/.."

command -v uv >/dev/null 2>&1 || { echo "Install uv first: https://docs.astral.sh/uv/"; exit 1; }

if [[ ! -d .venv ]]; then
  echo "==> Creating Python 3.12 venv"
  uv venv --python 3.12 .venv
fi

echo "==> Installing build deps"
uv pip install --python .venv/bin/python Cython packaging setuptools wheel

echo "==> Installing CUDA torch (cu128)"
uv pip install --python .venv/bin/python torch==2.7.1+cu128 torchaudio==2.7.1+cu128 \
  --index-url https://download.pytorch.org/whl/cu128

echo "==> Installing NeMo (git) + serving deps"
uv pip install --python .venv/bin/python -r requirements.lock.txt \
  --extra-index-url https://download.pytorch.org/whl/cu128

echo "==> Fetching model weights"
bash scripts/download_model.sh
if [[ "${DIAR_ENABLED:-1}" != "0" ]]; then
  bash scripts/download_diarization.sh
fi

echo "==> Starting API on http://0.0.0.0:8000  (Ctrl-C to stop)"
exec .venv/bin/python -m uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
