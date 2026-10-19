"""Inference engine for nvidia/nemotron-3.5-asr-streaming-0.6b.

Wraps the NeMo cache-aware FastConformer-RNNT model with:
  * a one-shot file transcription path, and
  * an incremental, per-session streaming path used by the WebSocket endpoint.

All GPU work is serialized behind a single lock. The model exposes the
language prompt as *global* state (``set_inference_prompt`` mutates the shared
model), so we set it immediately before every encoder step from the session's
own language. On a single GPU the inference is serialized regardless, so this
is both correct and cheap.
"""
from __future__ import annotations

import os
import threading
from copy import deepcopy
from dataclasses import dataclass, field
from typing import List, Optional

os.environ.setdefault("TORCHINDUCTOR_COMPILE_THREADS", "1")

import numpy as np
import torch

from .transcript import strip_lang_tags

# Keep HF fully offline at runtime; everything we need is in the local cache.
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

import nemo.collections.asr as nemo_asr  # noqa: E402


@dataclass
class StreamingSession:
    """Holds the cache-aware streaming state for a single audio stream."""

    engine: "ASREngine"
    lang: str = "auto"
    strip_tags: bool = True

    cache_last_channel: Optional[torch.Tensor] = None
    cache_last_time: Optional[torch.Tensor] = None
    cache_last_channel_len: Optional[torch.Tensor] = None
    previous_hypotheses: object = None
    pred_out_stream: object = None
    step_num: int = 0
    buffer_idx: int = 0  # feature frames already consumed
    _raw: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=np.float32))
    last_text: str = ""
    finished: bool = False

    def __post_init__(self):
        m = self.engine.model
        (
            self.cache_last_channel,
            self.cache_last_time,
            self.cache_last_channel_len,
        ) = m.encoder.get_initial_cache_state(batch_size=1)

    # -- public API -----------------------------------------------------
    def add_audio(self, samples: np.ndarray) -> List[str]:
        """Feed mono float32 [-1, 1] samples @16kHz. Returns text after each new chunk."""
        if samples.size:
            self._raw = np.concatenate([self._raw, samples.astype(np.float32, copy=False)])
        return self.engine._run_session_steps(self, flush=False)

    def finalize(self) -> str:
        """Flush any trailing audio as a final (partial) chunk and return final text."""
        if not self.finished:
            self.engine._run_session_steps(self, flush=True)
            self.finished = True
        return self.text

    @property
    def text(self) -> str:
        return strip_lang_tags(self.last_text) if self.strip_tags else self.last_text


class ASREngine:
    def __init__(self, model_path: str, device: str = "cuda"):
        self.device = device
        self.model = nemo_asr.models.ASRModel.restore_from(
            restore_path=model_path, map_location=device
        )
        self.model.eval()
        self.model.encoder.setup_streaming_params()
        self.scfg = deepcopy(self.model.encoder.streaming_cfg)
        self.context = list(self.model.encoder.att_context_size)
        self.compute_dtype = next(self.model.parameters()).dtype
        self.sample_rate = int(self.model.cfg.sample_rate)
        self.prompt_dictionary = dict(
            self.model.cfg.model_defaults.get("prompt_dictionary", {})
        )
        self.languages = sorted(self.prompt_dictionary.keys())
        # A dedicated preprocessor with dither/pad disabled and no normalization,
        # matching NeMo's CacheAwareStreamingAudioBuffer.extract_preprocessor().
        self._preprocessor = self._build_streaming_preprocessor()
        self._lock = threading.Lock()

    # -- setup ----------------------------------------------------------
    def _build_streaming_preprocessor(self):
        from omegaconf import OmegaConf, open_dict

        cfg = OmegaConf.to_container(self.model.cfg.preprocessor, resolve=True)
        cfg = OmegaConf.create(cfg)
        with open_dict(cfg):
            cfg.dither = 0.0
            cfg.pad_to = 0
            if cfg.get("normalize", "NA") not in ("per_feature", "all_features"):
                cfg.normalize = "None"
        pre = self.model.from_config_dict(cfg)
        return pre.to(self.device)

    def is_supported_language(self, lang: str) -> bool:
        return lang in self.prompt_dictionary

    # -- feature extraction --------------------------------------------
    def activate_context(self, context):
        """Called under _lock: session caches require their original geometry."""
        if list(self.model.encoder.att_context_size) != list(context):
            self.model.encoder.set_default_att_context_size(context)
            self.model.encoder.setup_streaming_params()

    def _features(self, raw: np.ndarray) -> torch.Tensor:
        sig = torch.from_numpy(raw).unsqueeze(0).to(self.device)
        length = torch.tensor([raw.shape[0]], device=self.device)
        with torch.inference_mode():
            feats, _ = self._preprocessor(input_signal=sig, length=length)
        return feats  # [1, n_feat, T]

    # -- streaming core -------------------------------------------------
    def _run_session_steps(self, s: StreamingSession, flush: bool) -> List[str]:
        if s._raw.size == 0:
            return []
        scfg = self.scfg
        texts: List[str] = []
        with self._lock, torch.inference_mode():
            self.activate_context(self.context)
            self.model.decoding.set_strip_lang_tags(False)
            feats = self._features(s._raw)
            total = feats.size(-1)
            while True:
                first = s.step_num == 0
                chunk_size = scfg.chunk_size[0] if first else scfg.chunk_size[1]
                shift_size = scfg.shift_size[0] if first else scfg.shift_size[1]
                pre = scfg.pre_encode_cache_size[0] if first else scfg.pre_encode_cache_size[1]

                end = s.buffer_idx + chunk_size
                # Mid-stream: only consume when a full chunk is buffered.
                # On flush: consume whatever frames remain as the last chunk.
                if end > total:
                    if not flush or s.buffer_idx >= total:
                        break
                    end = total

                chunk = feats[:, :, s.buffer_idx:end]
                if pre > 0:
                    lo = max(0, s.buffer_idx - pre)
                    chunk = torch.cat([feats[:, :, lo:s.buffer_idx], chunk], dim=-1)
                chunk = chunk.to(self.compute_dtype)
                clen = torch.tensor([chunk.size(-1)], device=self.device)
                drop = 0 if first else scfg.drop_extra_pre_encoded
                keep_all = flush  # keep tail outputs only on the final flush

                self.model.set_inference_prompt(s.lang)
                (
                    s.pred_out_stream,
                    transcribed,
                    s.cache_last_channel,
                    s.cache_last_time,
                    s.cache_last_channel_len,
                    s.previous_hypotheses,
                ) = self.model.conformer_stream_step(
                    processed_signal=chunk,
                    processed_signal_length=clen,
                    cache_last_channel=s.cache_last_channel,
                    cache_last_time=s.cache_last_time,
                    cache_last_channel_len=s.cache_last_channel_len,
                    keep_all_outputs=keep_all,
                    previous_hypotheses=s.previous_hypotheses,
                    previous_pred_out=s.pred_out_stream,
                    drop_extra_pre_encoded=drop,
                    return_transcription=True,
                )
                hyp = transcribed[0]
                s.last_text = hyp.text if hasattr(hyp, "text") else hyp
                s.buffer_idx += shift_size
                s.step_num += 1
                texts.append(s.text)
                if flush and s.buffer_idx >= total:
                    break
        return texts

    def new_session(self, lang: str = "auto", strip_tags: bool = True) -> StreamingSession:
        if not self.is_supported_language(lang):
            raise ValueError(f"Unsupported language '{lang}'")
        with self._lock:
            self.activate_context(self.context)
            return StreamingSession(engine=self, lang=lang, strip_tags=strip_tags)

    # -- one-shot file transcription -----------------------------------
    def transcribe_array(self, samples: np.ndarray, lang: str = "auto",
                         strip_tags: bool = True) -> str:
        """Transcribe a complete utterance by streaming it through in one pass."""
        s = self.new_session(lang=lang, strip_tags=strip_tags)
        s.add_audio(samples)
        return s.finalize()

    def warmup(self):
        """Initialize the native FP32 streaming path before reporting readiness."""
        session = self.new_session()
        silence = np.zeros(round(self.sample_rate * .32), dtype=np.float32)
        for _ in range(4):
            session.add_audio(silence)
        session.finalize()
