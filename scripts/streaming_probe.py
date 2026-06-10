"""Validate offline + cache-aware streaming transcription and dump streaming cfg."""
import os, time
os.environ.setdefault("HF_HUB_OFFLINE", "1")
import torch
import nemo.collections.asr as nemo_asr
from nemo.collections.asr.parts.utils.streaming_utils import CacheAwareStreamingAudioBuffer

HERE = os.path.dirname(__file__)
MODEL_PATH = os.path.join(HERE, "..", "model_cache",
                          "nemotron-3.5-asr-streaming-0.6b",
                          "nemotron-3.5-asr-streaming-0.6b.nemo")
WAV = os.path.join(HERE, "..", "test_data", "sample_en.wav")
LANG = "en-US"

m = nemo_asr.models.ASRModel.restore_from(MODEL_PATH, map_location="cuda")
m.eval()

# ---- offline transcribe() is skipped: the prompt dataset needs per-cut
# language metadata; we drive everything through cache-aware streaming. ----

# ---- set up streaming ----
if hasattr(m, "set_inference_prompt"):
    m.set_inference_prompt(LANG)
m.encoder.setup_streaming_params()
scfg = m.encoder.streaming_cfg
print("\n=== STREAMING CFG ===")
for k in ["chunk_size", "shift_size", "pre_encode_cache_size", "drop_extra_pre_encoded",
          "last_channel_cache_size", "valid_out_len"]:
    print(f"  {k}: {getattr(scfg, k, None)}")
print("  att_context_size(default):", m.encoder.att_context_size)
print("  sample_rate:", m.cfg.sample_rate)
print("  window_stride:", m.cfg.preprocessor.window_stride)

# ---- cache-aware streaming over the file ----
buf = CacheAwareStreamingAudioBuffer(model=m, online_normalization=False)
buf.append_audio_file(WAV, stream_id=-1)
cache_lc, cache_lt, cache_len = m.encoder.get_initial_cache_state(batch_size=1)
prev_hyp = None
pred_out = None
texts = None
def drop(step):
    return 0 if step == 0 else m.encoder.streaming_cfg.drop_extra_pre_encoded
for i, (chunk, clen) in enumerate(buf):
    with torch.inference_mode():
        (pred_out, texts, cache_lc, cache_lt, cache_len, prev_hyp) = m.conformer_stream_step(
            processed_signal=chunk, processed_signal_length=clen,
            cache_last_channel=cache_lc, cache_last_time=cache_lt, cache_last_channel_len=cache_len,
            keep_all_outputs=buf.is_buffer_empty(), previous_hypotheses=prev_hyp,
            previous_pred_out=pred_out, drop_extra_pre_encoded=drop(i), return_transcription=True,
        )
    txt = texts[0].text if hasattr(texts[0], "text") else texts[0]
    print(f"  step {i:2d}: {txt!r}")
print("\n=== STREAMING FINAL ===")
print(texts[0].text if hasattr(texts[0], "text") else texts[0])
