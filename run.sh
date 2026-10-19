#!/usr/bin/env bash
# One-command bootstrap: download the model, build the containers, and start the
# GPU ASR service + web UI. Re-runnable; safe to run again to apply changes.
#
#   git clone git@github.com:tehtommeh/nemotron-asr-streaming.git
#   cd nemotron-asr-streaming
#   ./run.sh
set -euo pipefail
cd "$(dirname "$0")"

bold() { printf "\033[1m%s\033[0m\n" "$*"; }
warn() { printf "\033[33m%s\033[0m\n" "$*"; }
die()  { printf "\033[31mERROR: %s\033[0m\n" "$*" >&2; exit 1; }

bold "==> Preflight checks"
command -v docker >/dev/null 2>&1 || die "docker not found. Install Docker: https://docs.docker.com/get-docker/"
docker compose version >/dev/null 2>&1 || die "docker compose v2 not found. Install the Compose plugin."
if command -v nvidia-smi >/dev/null 2>&1; then
  nvidia-smi --query-gpu=name,memory.total --format=csv,noheader | sed 's/^/    GPU: /'
else
  warn "    nvidia-smi not found — the service needs an NVIDIA GPU to run."
fi
if ! docker info 2>/dev/null | grep -qi 'Runtimes:.*nvidia'; then
  warn "    NVIDIA Docker runtime not detected. Install the NVIDIA Container Toolkit:"
  warn "    https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html"
fi

bold "==> Fetching model weights (idempotent)"
bash scripts/download_model.sh
if [[ "${DIAR_ENABLED:-1}" != "0" ]]; then
  bash scripts/download_diarization.sh
fi

bold "==> Building & starting containers"
docker compose up -d --build

bold "==> Waiting for the model to load on the GPU…"
deadline=$(( $(date +%s) + 600 ))
until curl -fsS http://127.0.0.1:8000/health >/dev/null 2>&1; do
  status=$(docker inspect -f '{{.State.Status}}' nemotron-asr-streaming 2>/dev/null || echo missing)
  [[ "$status" == "running" ]] || die "API container is '$status'. Check: docker compose logs asr"
  [[ $(date +%s) -lt $deadline ]] || die "Timed out waiting for /health. Check: docker compose logs -f asr"
  sleep 3
done

LAN_IP=$(hostname -I 2>/dev/null | tr ' ' '\n' | grep -E '^(192|10|172)\.' | head -1 || true)
bold "==> Ready ✅"
echo
echo "  Web UI (this machine):  http://localhost:8080"
[[ -n "${LAN_IP:-}" ]] && echo "  Web UI (LAN devices):   http://${LAN_IP}:8080   (mic needs a secure context — see README)"
echo "  API docs (Swagger):     http://localhost:8000/docs"
echo "  Health:                 http://localhost:8000/health"
echo
echo "  Logs:  docker compose logs -f"
echo "  Stop:  docker compose down"
