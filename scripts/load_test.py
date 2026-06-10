"""Quick smoke test: load the .nemo model offline onto GPU and inspect it."""
import os
os.environ.setdefault("HF_HUB_OFFLINE", "1")
import torch
import nemo.collections.asr as nemo_asr

MODEL_PATH = os.path.join(
    os.path.dirname(__file__), "..",
    "model_cache", "nemotron-3.5-asr-streaming-0.6b",
    "nemotron-3.5-asr-streaming-0.6b.nemo",
)

print("Restoring model from", MODEL_PATH)
model = nemo_asr.models.ASRModel.restore_from(restore_path=MODEL_PATH, map_location="cuda")
model.eval()
print("Loaded:", type(model).__name__)
print("Class MRO:", [c.__name__ for c in type(model).__mro__][:6])
cfg = model.cfg
print("sample_rate:", cfg.get("sample_rate"))
# encoder streaming config
enc = cfg.get("encoder", {})
print("encoder._target_:", enc.get("_target_"))
print("att_context_size:", enc.get("att_context_size"))
print("att_context_style:", enc.get("att_context_style"))
# check streaming support
print("has setup_streaming_params:", hasattr(model.encoder, "setup_streaming_params"))
print("has conformer_stream_step:", hasattr(model, "conformer_stream_step"))
print("decoding type:", type(model.decoding).__name__ if hasattr(model, "decoding") else None)
print("GPU mem allocated MB:", round(torch.cuda.memory_allocated()/1e6, 1))
