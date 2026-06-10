#!/usr/bin/env python3
"""Download the Nemotron 3.5 ASR streaming weights into ./model_cache.

Idempotent: if the .nemo file is already present it does nothing. Only needs
`huggingface_hub` installed (the helper scripts run it in a throwaway env).
"""
import os
import sys

REPO_ID = os.environ.get("MODEL_REPO_ID", "nvidia/nemotron-3.5-asr-streaming-0.6b")
OUT_DIR = os.environ.get("MODEL_DIR", os.path.join("model_cache", REPO_ID.split("/")[-1]))
NEMO_FILE = os.path.join(OUT_DIR, f"{REPO_ID.split('/')[-1]}.nemo")


def main() -> int:
    if os.path.isfile(NEMO_FILE):
        print(f"Model already present: {NEMO_FILE}")
        return 0
    os.environ.setdefault("HF_HUB_ENABLE_HF_TRANSFER", "1")
    from huggingface_hub import snapshot_download

    print(f"Downloading {REPO_ID} -> {OUT_DIR} (~2.4 GB) …")
    path = snapshot_download(repo_id=REPO_ID, local_dir=OUT_DIR)
    print("DOWNLOADED_TO:", path)
    if not os.path.isfile(NEMO_FILE):
        print(f"ERROR: expected {NEMO_FILE} after download", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
