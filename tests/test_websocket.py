"""Transport contract tests with a deterministic recognizer, no model downloads."""
import importlib
import sys
import threading
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
from fastapi.testclient import TestClient


class FakeSession:
    def __init__(self, **kwargs):
        self.text = ""
        self.segments = []
        self.finished = False
        self.closed = False

    def add_audio(self, samples):
        self.text = "live text"
        self.segments = [{"text": self.text, "speaker_id": "speaker_0", "start": 0, "end": .1, "final": False}]
        return [self.text]

    def finalize(self):
        self.finished = True
        for segment in self.segments:
            segment["final"] = True
        return self.text

    def close(self):
        self.closed = True


class WebsocketTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        with patch.dict(sys.modules, {"app.asr": SimpleNamespace(ASREngine=object)}):
            cls.api = importlib.import_module("app.main")

    def setUp(self):
        self.sessions = []

        def factory(**kwargs):
            session = FakeSession(**kwargs)
            self.sessions.append(session)
            return session

        self.api.engine = SimpleNamespace(sample_rate=16000, is_supported_language=lambda lang: lang == "auto",
                                          new_session=factory)
        self.api.diarization = SimpleNamespace(new_session=factory)
        self.client = TestClient(self.api.app)

    def test_plain_and_diarized_partial_before_stop_and_final(self):
        for diarize in (False, True):
            with self.client.websocket_connect(f"/ws/transcribe?diarize={str(diarize).lower()}") as ws:
                self.assertEqual(ws.receive_json()["type"], "ready")
                ws.send_bytes(np.zeros(5120, dtype="<i2").tobytes())
                partial = ws.receive_json()
                self.assertEqual(partial["type"], "partial")
                self.assertEqual("segments" in partial, diarize)
                ws.send_json({"type": "end"})
                final = ws.receive_json()
                self.assertEqual(final["type"], "final")
                if diarize:
                    self.assertTrue(final["segments"][0]["final"])
            self.assertTrue(self.sessions[-1].finished)
            self.assertTrue(self.sessions[-1].closed)

    def test_missing_diarizer_is_explicit_and_plain_still_works(self):
        self.api.diarization = None
        self.api.diarization_error = "Weights missing"
        with self.client.websocket_connect("/ws/transcribe?diarize=true") as ws:
            self.assertEqual(ws.receive_json(), {"type": "error", "detail": "Weights missing"})
        with self.client.websocket_connect("/ws/transcribe") as ws:
            self.assertEqual(ws.receive_json()["type"], "ready")
            ws.send_json({"type": "end"})
            self.assertEqual(ws.receive_json()["type"], "final")

    def test_disconnect_does_not_finalize_absent_clients_audio(self):
        with self.client.websocket_connect("/ws/transcribe?diarize=true") as ws:
            ws.receive_json()
        self.assertTrue(self.sessions[-1].closed)
        self.assertFalse(self.sessions[-1].finished)

    def test_cancelled_handler_waits_for_inflight_operation_before_cleanup(self):
        started, release = threading.Event(), threading.Event()
        gpu_lock = threading.Lock()

        class BusySession(FakeSession):
            def add_audio(self, samples):
                with gpu_lock:
                    started.set()
                    if not release.wait(2):
                        raise TimeoutError("Test operation did not finish")
                    return super().add_audio(samples)

            def close(self):
                with gpu_lock:
                    super().close()

        session = BusySession()
        self.api.diarization.new_session = lambda **kwargs: session
        timer = None
        try:
            with self.client.websocket_connect("/ws/transcribe?diarize=true") as ws:
                ws.receive_json()
                ws.send_bytes(np.zeros(5120, dtype="<i2").tobytes())
                self.assertTrue(started.wait(2))
                timer = threading.Timer(.05, release.set)
                timer.start()
            self.assertTrue(session.closed)
            self.assertFalse(session.finished)
        finally:
            release.set()
            if timer is not None:
                timer.join()

    def test_malformed_pcm_reports_error(self):
        with self.client.websocket_connect("/ws/transcribe") as ws:
            ws.receive_json()
            ws.send_bytes(b"x")
            self.assertEqual(ws.receive_json()["type"], "error")

    def test_both_ui_features_are_served_by_api(self):
        page = self.client.get("/")
        self.assertEqual(page.status_code, 200)
        self.assertIn('id="diarize"', page.text)
        self.assertIn('id="fileInput"', page.text)


if __name__ == "__main__":
    unittest.main()
