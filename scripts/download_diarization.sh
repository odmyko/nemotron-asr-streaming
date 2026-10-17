#!/usr/bin/env bash
# Optional local checkpoint; runtime never downloads weights.
set -euo pipefail
cd "$(dirname "$0")/.."
if [[ -f model_cache/Nemotron-3-Diarization/Nemotron-3-Diarization.nemo ]]; then
  echo "Diarization model already downloaded."
elif command -v uv >/dev/null 2>&1; then
  uv run --no-project --with 'huggingface_hub==1.18.0' python download_diarization.py
elif command -v python3 >/dev/null 2>&1; then
  python3 -m venv .venv-tools
  .venv-tools/bin/python -m pip install -q 'huggingface_hub==1.18.0'
  .venv-tools/bin/python download_diarization.py
else
  mkdir -p model_cache
  docker run --rm -v "$PWD:/work:ro" -v "$PWD/model_cache:/work/model_cache" \
    -w /work python:3.12-slim sh -c \
    "pip install -q 'huggingface_hub==1.18.0' && python download_diarization.py"
fi
