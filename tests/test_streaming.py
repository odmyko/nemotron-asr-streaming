import threading
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import soxr
import torch

from app.audio import StreamingResampler
from app.diarization import DiarizedSession, segment_snapshot


class ResamplingTests(unittest.TestCase):
    def test_packet_boundaries_do_not_change_waveform(self):
        for rate in (44100, 48000):
            wave = np.sin(np.arange(rate * 2) * (2 * np.pi * 440 / rate)).astype("float32")
            resampler = StreamingResampler(rate)
            pieces = [resampler.process(wave[i:i + 4096]) for i in range(0, wave.size, 4096)]
            pieces.append(resampler.process(np.zeros(0, dtype="float32"), final=True))
            actual = np.concatenate(pieces)
            expected = soxr.resample(wave, rate, 16000)
            self.assertEqual(actual.size, 32000)
            np.testing.assert_allclose(actual, expected, atol=1e-5)


class FakeStreamer:
    def __init__(self, *args):
        self.calls = []
        self.instance_manager = SimpleNamespace(batch_asr_states=[])

    def perform_parallel_streaming_stt_spk(self, **kwargs):
        self.calls.append(kwargs)
        self.instance_manager.batch_asr_states = [SimpleNamespace(seglsts=[{
            "words": "hello <en-US>", "speaker": "speaker_0", "start_time": 0, "end_time": 1,
        }])]


def make_session():
    engine = SimpleNamespace(
        _lock=threading.Lock(), sample_rate=16000, compute_dtype=torch.float32, device="cpu",
        activate_context=lambda _: None,
        model=SimpleNamespace(set_inference_prompt=lambda _: None,
                              decoding=SimpleNamespace(set_strip_lang_tags=lambda _: None)),
        _features=lambda frame: torch.zeros(1, 8, frame.size // 160),
    )
    backend = SimpleNamespace(
        engine=engine, model=object(), use_amp=False, context=[56, 13], cfg={}, cache_frames=9, stride=160,
        chunk_frames=112, frame_samples=121*160, hop_samples=112*160,
        scfg=SimpleNamespace(drop_extra_pre_encoded=2),
    )
    module = SimpleNamespace(SpeakerTaggedASR=FakeStreamer)
    with patch.dict("sys.modules", {"nemo.collections.asr.parts.utils.multispk_transcribe_utils": module}):
        return DiarizedSession(backend, "auto", True)


class SessionTests(unittest.TestCase):
    def test_live_partials_and_short_final_tail(self):
        session = make_session()
        self.assertEqual(session.add_audio(np.ones(5000, dtype="float32")), [])
        self.assertEqual(session.add_audio(np.ones(20000, dtype="float32")), ["hello"])
        self.assertFalse(session.streamer.calls[-1]["is_buffer_empty"])
        self.assertEqual(session.finalize(), "hello")
        self.assertTrue(session.streamer.calls[-1]["is_buffer_empty"])
        count = len(session.streamer.calls)
        self.assertEqual(session.finalize(), "hello")
        self.assertEqual(len(session.streamer.calls), count)
        self.assertTrue(all(s["final"] for s in session.segments))

    def test_tiny_recording_and_empty_recording(self):
        session = make_session()
        session.add_audio(np.ones(400, dtype="float32"))
        self.assertEqual(session.finalize(), "hello")
        self.assertEqual(session.segments[0]["end"], 0.025)
        self.assertEqual(make_session().finalize(), "")

    def test_sessions_and_disconnect_are_independent(self):
        first, second = make_session(), make_session()
        first.add_audio(np.ones(1000, dtype="float32"))
        first.close()
        self.assertEqual(first.pending.size, 0)
        self.assertIsNone(first.streamer)
        self.assertEqual(second.step_num, 0)
        self.assertFalse(second.finished)

    def test_snapshots_replace_revised_text_and_clamp_padding(self):
        initial = [{"words": "hel", "speaker": "speaker_0", "start_time": -1, "end_time": 2}]
        revised = [{**initial[0], "words": "hello <en-US>"}]
        self.assertEqual(segment_snapshot(initial, 1, True, False)[0]["text"], "hel")
        result = segment_snapshot(revised, 1, True, True)
        self.assertEqual(result, [{"text": "hello", "start": 0, "end": 1,
                                   "speaker_id": "speaker_0", "final": True}])


if __name__ == "__main__":
    unittest.main()
