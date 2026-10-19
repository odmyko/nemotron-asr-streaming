import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from app.storage import save_transcript, transcript_record


class TranscriptStorageTests(unittest.TestCase):
    def test_unicode_unique_recordings_and_atomic_failure(self):
        with tempfile.TemporaryDirectory() as directory:
            with patch.dict(os.environ, {"TRANSCRIPT_DIR": directory, "TRANSCRIPT_SAVE_ENABLED": "1"}):
                first = transcript_record(text="Привіт 🌍", language="auto")
                second = transcript_record(text="Another recording", language="en-US")
                saved = save_transcript(first)
                save_transcript(second)
                self.assertNotEqual(first["id"], second["id"])
                self.assertTrue(saved["filename"].endswith(".txt"))
                self.assertEqual((Path(directory) / saved["filename"]).read_text(encoding="utf-8"), first["text"] + "\n")
                with patch("app.storage.os.replace", side_effect=OSError("Disk full")):
                    with self.assertRaises(OSError):
                        save_transcript(transcript_record(text="incomplete", language="auto"))
                self.assertEqual(len(list(Path(directory).iterdir())), 2)

    def test_speaker_turns_are_readable(self):
        record = transcript_record(text="Привіт! Hello!", language="auto", diarize=True, segments=[
            {"text": "Привіт!", "speaker_id": "speaker_0", "start": 0, "end": 1.25},
            {"text": "Hello!", "speaker_id": "speaker_1", "start": 2, "end": 3.5},
        ])
        with tempfile.TemporaryDirectory() as directory:
            with patch.dict(os.environ, {"TRANSCRIPT_DIR": directory, "TRANSCRIPT_SAVE_ENABLED": "1"}):
                saved = save_transcript(record)
            self.assertEqual((Path(directory) / saved["filename"]).read_text(encoding="utf-8"),
                             "Speaker 1 [0.00–1.25s]: Привіт!\n\nSpeaker 2 [2.00–3.50s]: Hello!\n")

    def test_disabled_autosave_does_not_create_directory(self):
        with tempfile.TemporaryDirectory() as directory:
            target = Path(directory) / "disabled"
            with patch.dict(os.environ, {"TRANSCRIPT_DIR": str(target), "TRANSCRIPT_SAVE_ENABLED": "0"}):
                result = save_transcript(transcript_record(text="Hello", language="auto"))
            self.assertEqual(result, {"status": "disabled"})
            self.assertFalse(target.exists())
