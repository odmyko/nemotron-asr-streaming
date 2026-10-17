"""GPU integration/performance check, using local weights and audio.

python scripts/validate_diarization.py [--audio dialog.wav] [--minutes 10] [--realtime]
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
os.environ.setdefault("TORCHINDUCTOR_COMPILE_THREADS", "1")

import numpy as np
import soundfile as sf
import torch
from nemo.utils import logging

logging.setLevel(logging.ERROR)
from app.asr import ASREngine
from app.diarization import DiarizationEngine


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--audio", default="test_data/sample_en.wav")
    parser.add_argument("--minutes", type=float, default=0)
    parser.add_argument("--realtime", action="store_true")
    parser.add_argument("--min-speakers", type=int, default=1)
    parser.add_argument("--report", help="Write full transcript and metrics to a JSON artifact")
    args = parser.parse_args()
    engine = ASREngine("model_cache/nemotron-3.5-asr-streaming-0.6b/nemotron-3.5-asr-streaming-0.6b.nemo")
    audio, sr = sf.read(args.audio, dtype="float32")
    assert sr == 16000 and audio.ndim == 1, "Use mono 16 kHz audio"
    plain = engine.transcribe_array(audio, "en-US")
    assert plain.strip(), "Ordinary ASR returned no text"
    backend = DiarizationEngine(engine, "model_cache/Nemotron-3-Diarization/Nemotron-3-Diarization.nemo")
    warmup_started = time.perf_counter()
    backend.warmup()
    warmup_seconds = time.perf_counter() - warmup_started
    print("WARMUP_SECONDS", warmup_seconds, flush=True)
    session = backend.new_session()
    other = engine.new_session("en-US")
    if args.minutes:
        audio = np.resize(audio, int(sr * args.minutes * 60))
    step = 5120
    started = time.perf_counter()
    first_partial = None
    max_step = 0
    compute_seconds = 0
    max_lag = 0
    step_times = []
    for offset in range(0, audio.size, step):
        audio_time = min(offset + step, audio.size) / sr
        if args.realtime:
            time.sleep(max(0, audio_time - (time.perf_counter()-started)))
        tick = time.perf_counter()
        updates = session.add_audio(audio[offset:offset + step])
        # Interleave an ordinary session to exercise global prompt/context isolation.
        if offset < min(audio.size, 16000 * 7):
            other.add_audio(audio[offset:offset + step])
        elapsed = time.perf_counter() - tick
        step_times.append(elapsed)
        max_step = max(max_step, elapsed)
        compute_seconds += elapsed
        if args.realtime:
            max_lag = max(max_lag, time.perf_counter()-started-audio_time)
        if updates and session.text and first_partial is None:
            first_partial = audio_time
            print("FIRST_PARTIAL", json.dumps(session.segments, ensure_ascii=False), flush=True)
        if offset % (step * 100) == 0:
            print("PROGRESS", round(offset/sr, 2), flush=True)
    text = session.finalize()
    assert text and session.finalize() == text
    assert first_partial is not None, "No nonempty partial before Stop"
    assert all(s["final"] and 0 <= s["start"] <= s["end"] <= audio.size/sr for s in session.segments)
    assert len({s["speaker_id"] for s in session.segments}) >= args.min_speakers
    assert other.finalize(), "Interleaved ordinary ASR failed"
    fresh = backend.new_session()
    assert fresh.segments == [] and fresh.step_num == 0
    fresh.close()
    report = {
        "plain_text": plain, "text": text, "segments": session.segments,
        "duration_sec": audio.size/sr, "elapsed_sec": time.perf_counter()-started,
        "first_partial_audio_sec": first_partial, "max_step_sec": max_step,
        "buffer_sec": backend.buffer_seconds,
        "compute_sec": compute_seconds, "max_lag_sec": max_lag,
        "warmup_sec": warmup_seconds,
        "peak_gpu_bytes": torch.cuda.max_memory_allocated(),
        "speakers": sorted({s["speaker_id"] for s in session.segments}),
        "num_segments": len(session.segments),
        "p95_step_sec": float(np.percentile(step_times, 95)),
        "compute_real_time_factor": compute_seconds / (audio.size/sr),
    }
    if args.report:
        destination = Path(args.report)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    summary = {k: v for k, v in report.items() if k not in ("segments", "text", "plain_text")}
    summary["text_preview"] = text[:300]
    print("RESULT", json.dumps(summary, ensure_ascii=False), flush=True)
    session.close()


if __name__ == "__main__":
    main()
