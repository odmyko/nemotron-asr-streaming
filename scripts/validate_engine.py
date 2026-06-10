"""Feed the sample wav through ASREngine in 320ms chunks; compare to expected."""
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import numpy as np
import soundfile as sf
from app.asr import ASREngine

MODEL = os.path.join("model_cache", "nemotron-3.5-asr-streaming-0.6b",
                     "nemotron-3.5-asr-streaming-0.6b.nemo")
WAV = os.path.join("test_data", "sample_en.wav")
EXPECTED = ("Well, I don't wish to see it any more, observed Phoebe, turning "
            "away her eyes. It is certainly very like the old portrait.")

eng = ASREngine(MODEL)
audio, sr = sf.read(WAV, dtype="float32")
assert sr == eng.sample_rate, (sr, eng.sample_rate)

s = eng.new_session(lang="en-US", strip_tags=True)
chunk = int(0.32 * sr)  # 320ms blocks, like a live mic
for i in range(0, len(audio), chunk):
    outs = s.add_audio(audio[i:i + chunk])
    if outs:
        print(f"[{i/sr:5.2f}s] {outs[-1]!r}")
final = s.finalize()
print("\nFINAL :", repr(final))
print("EXPECT:", repr(EXPECTED))
print("\nMATCH:", final == EXPECTED)
