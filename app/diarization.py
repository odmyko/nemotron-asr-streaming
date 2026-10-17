"""Live adapter for the pinned NeMo Speech speaker-tagged ASR pipeline.

Weights are shared, but each connection owns its SpeakerTaggedASR instance.
All calls touching the shared models run under ASREngine's GPU lock.
"""
from __future__ import annotations

import os
from copy import deepcopy

import numpy as np
import torch
from omegaconf import OmegaConf

from .transcript import strip_lang_tags

def segment_snapshot(seglsts, duration: float, strip_tags: bool, final: bool):
    """Convert NeMo SegLST to a replaceable, JSON-safe transcript snapshot."""
    segments = []
    for segment in seglsts:
        text = segment["words"].strip()
        if strip_tags:
            text = strip_lang_tags(text)
        if not text:
            continue
        start = min(duration, max(0.0, float(segment["start_time"])))
        end = min(duration, max(start, float(segment["end_time"])))
        segments.append({
            "text": text, "start": round(start, 3), "end": round(end, 3),
            "speaker_id": segment["speaker"], "final": final,
        })
    return sorted(segments, key=lambda s: (s["start"], s["end"], s["speaker_id"]))


class DiarizationEngine:
    def __init__(self, engine, model_path: str):
        from nemo.collections.asr.models import SortformerEncLabelModel
        from nemo.collections.asr.parts.utils.multispk_transcribe_utils import (
            configure_diar_streaming, validate_feature_frame_strides,
        )

        if not os.path.isfile(model_path):
            raise FileNotFoundError("Diarization weights missing. Run scripts/download_diarization.sh first.")
        self.engine = engine
        self.model = SortformerEncLabelModel.restore_from(model_path, map_location=engine.device)
        self.model.eval()
        self.use_amp = str(engine.device).startswith("cuda") and torch.cuda.is_bf16_supported()
        if self.use_amp:
            self.model.to(dtype=torch.bfloat16)
        self.model.rttms_mask_mats = None
        self.cfg = OmegaConf.create({
            "deploy_mode": True, "streaming_mode": True, "max_num_of_spks": 8,
            "batch_size": 1, "parallel_speaker_strategy": True, "masked_asr": True,
            "mask_preencode": False, "single_speaker_mode": False,
            "cache_gating": True, "cache_gating_buffer_size": 2,
            "binary_diar_preds": True, "spkcache_len": None,
            "spkcache_update_period": 222, "fifo_len": 264, "diar_right_context": 0,
            "att_context_size": [56, 13], "sent_break_sec": 1.0,
            "fix_prev_words_count": 5, "update_prev_words_sentence": 5,
            "ignored_initial_frame_steps": 5, "generate_realtime_scripts": False,
            "log": False,
        })
        self.context = [56, 13]
        try:
            engine.activate_context(self.context)
            self.scfg = deepcopy(engine.model.encoder.streaming_cfg)
            validate_feature_frame_strides(engine.model, self.model)
            scfg = self.scfg
            configure_diar_streaming(
                self.model, self.cfg, engine.model.encoder.subsampling_factor,
                diar_chunk_len=scfg.valid_out_len + scfg.cache_drop_size,
            )
        finally:
            engine.activate_context(engine.context)
        # Match the upstream FeatureStacking path: pad the first pre-encode cache
        # and drop it even on step zero, so diarization never consumes cache frames.
        self.stride = round(float(engine.model.cfg.preprocessor.window_stride) * engine.sample_rate)
        self.chunk_frames = int(scfg.chunk_size[-1])
        self.shift_frames = int(scfg.shift_size[-1])
        self.cache_frames = int(scfg.pre_encode_cache_size[-1])
        self.frame_samples = (self.chunk_frames + self.cache_frames) * self.stride
        self.hop_samples = self.shift_frames * self.stride
        self.buffer_seconds = self.chunk_frames * self.stride / engine.sample_rate

    def new_session(self, lang="auto", strip_tags=True):
        return DiarizedSession(self, lang, strip_tags)

    def warmup(self):
        """Compile attention shapes before accepting microphone connections."""
        session = self.new_session()
        try:
            silence = np.zeros(self.hop_samples, dtype=np.float32)
            # Fill both FIFO and speaker-cache geometries, including cache rollover.
            for _ in range(48):
                session.add_audio(silence)
            session.finalize()
        finally:
            session.close()


class DiarizedSession:
    def __init__(self, backend, lang, strip_tags):
        from nemo.collections.asr.parts.utils.multispk_transcribe_utils import SpeakerTaggedASR

        self.backend = backend
        self.engine = backend.engine
        self.lang = lang
        self.strip_tags = strip_tags
        self.finished = False
        self.step_num = 0
        self.total_samples = 0
        self.processed_samples = 0
        self.pending = np.zeros(backend.cache_frames * backend.stride, dtype=np.float32)
        self.segments = []
        with self.engine._lock, torch.inference_mode(), torch.autocast(
            device_type="cuda", dtype=torch.bfloat16, enabled=backend.use_amp,
        ):
            self.engine.activate_context(backend.context)
            self.streamer = SpeakerTaggedASR(backend.cfg, self.engine.model, backend.model)

    @property
    def text(self):
        return " ".join(s["text"] for s in self.segments)

    def add_audio(self, samples):
        if self.finished:
            raise ValueError("Session is already finished")
        self.total_samples += samples.size
        self.pending = np.concatenate((self.pending, samples.astype(np.float32, copy=False)))
        return self._run(flush=False)

    def _run(self, flush):
        changed = False
        backend = self.backend
        with self.engine._lock, torch.inference_mode(), torch.autocast(
            device_type="cuda", dtype=torch.bfloat16, enabled=backend.use_amp,
        ):
            self.engine.activate_context(backend.context)
            while self.processed_samples < self.total_samples:
                if not flush and self.pending.size < backend.frame_samples:
                    break
                frame = self.pending[:backend.frame_samples]
                if frame.size < backend.frame_samples:
                    frame = np.pad(frame, (0, backend.frame_samples - frame.size))
                features = self.engine._features(frame)
                expected = backend.cache_frames + backend.chunk_frames
                features = features[:, :, :expected].to(self.engine.compute_dtype)
                final_step = flush and self.processed_samples + backend.hop_samples >= self.total_samples
                self.engine.model.set_inference_prompt(self.lang)
                self.engine.model.decoding.set_strip_lang_tags(False)
                self.streamer.perform_parallel_streaming_stt_spk(
                    step_num=self.step_num, chunk_audio=features,
                    chunk_lengths=torch.tensor([features.shape[-1]], device=self.engine.device),
                    is_buffer_empty=final_step,
                    drop_extra_pre_encoded=backend.scfg.drop_extra_pre_encoded,
                )
                self.step_num += 1
                self.processed_samples += backend.hop_samples
                self.pending = self.pending[backend.hop_samples:]
                changed = True
            states = self.streamer.instance_manager.batch_asr_states
            if states:
                self.segments = segment_snapshot(
                    states[0].seglsts, self.total_samples / self.engine.sample_rate,
                    self.strip_tags, flush,
                )
        return [self.text] if changed else []

    def finalize(self):
        if not self.finished:
            self._run(flush=True)
            self.finished = True
        return self.text

    def close(self):
        # Disconnect must not launch expensive inference for an absent client.
        with self.engine._lock:
            self.finished = True
            self.pending = np.zeros(0, dtype=np.float32)
            self.streamer = None
