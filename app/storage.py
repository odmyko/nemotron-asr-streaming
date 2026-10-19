"""Atomic server-side backups of completed transcripts (audio is never saved)."""
from __future__ import annotations

import os
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4


def transcript_record(*, text, language, diarize=False, segments=None,
                      duration_sec=None, source="microphone", filename=None):
    return {
        "schema_version": 1,
        "id": uuid4().hex,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "source": source,
        "filename": filename,
        "language": language,
        "diarize": diarize,
        "duration_sec": duration_sec,
        "text": text,
        "segments": segments or [],
    }


def transcript_text(record):
    """Readable speaker turns, or the full text for ordinary transcription."""
    lines = []
    for segment in record["segments"]:
        speaker = segment.get("speaker_id")
        label = ""
        if speaker:
            number = str(speaker).removeprefix("speaker_")
            label = f"Speaker {int(number) + 1}" if number.isdecimal() else str(speaker)
        start, end = segment.get("start"), segment.get("end")
        if start is not None and end is not None:
            label += f" [{start:.2f}–{end:.2f}s]"
        lines.append((label.strip() + ": " if label else "") + segment["text"])
    return ("\n\n".join(lines) if lines else record["text"]) + "\n"


def save_transcript(record):
    """Publish a complete UTF-8 TXT file; readers never see a partial write."""
    if os.environ.get("TRANSCRIPT_SAVE_ENABLED", "1").lower() in ("0", "false", "no"):
        return {"status": "disabled"}
    directory = Path(os.environ.get("TRANSCRIPT_DIR", "transcripts"))
    directory.mkdir(parents=True, exist_ok=True)
    name = f"transcript-{record['id']}.txt"
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=directory,
                                         prefix=".transcript-", suffix=".tmp", delete=False) as stream:
            temporary = Path(stream.name)
            stream.write(transcript_text(record))
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, directory / name)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
    return {"status": "saved", "filename": name}
