"""Fetch only the optional diarization checkpoint, before offline runtime."""
import os
from pathlib import Path


def main():
    from huggingface_hub import hf_hub_download

    repo = "nvidia/Nemotron-3-Diarization"
    directory = Path(os.environ.get("DIAR_MODEL_DIR", "model_cache/Nemotron-3-Diarization"))
    filename = "Nemotron-3-Diarization.nemo"
    if (directory / filename).is_file():
        print(f"Diarization model already present: {directory / filename}")
        return
    path = hf_hub_download(repo, filename, revision="f667ed7", local_dir=directory)
    print(f"Diarization model ready: {path}")


if __name__ == "__main__":
    main()
