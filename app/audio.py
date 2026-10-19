"""Audio decoding / resampling helpers."""
from __future__ import annotations

import asyncio
import subprocess
from typing import Optional

import numpy as np

TARGET_SR = 16000


class StreamingResampler:
    """One resampler per connection; retain filter history and flush its tail."""

    def __init__(self, orig_sr: int, target_sr: int = TARGET_SR):
        self._stream = None
        if orig_sr != target_sr:
            import soxr

            self._stream = soxr.ResampleStream(orig_sr, target_sr, 1, dtype="float32")

    def process(self, samples: np.ndarray, *, final: bool = False) -> np.ndarray:
        if self._stream is None:
            return samples
        return self._stream.resample_chunk(samples, last=final)


def decode_to_mono16k(data: bytes, target_sr: int = TARGET_SR) -> np.ndarray:
    """Decode arbitrary audio bytes (wav/flac/ogg/mp3/webm/opus/...) to mono
    float32 [-1, 1] at ``target_sr`` using ffmpeg. Returns a 1-D numpy array."""
    proc = subprocess.run(
        [
            "ffmpeg", "-nostdin", "-hide_banner", "-loglevel", "error",
            "-i", "pipe:0",
            "-f", "f32le", "-acodec", "pcm_f32le",
            "-ac", "1", "-ar", str(target_sr),
            "pipe:1",
        ],
        input=data,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    if proc.returncode != 0:
        raise ValueError(
            f"ffmpeg failed to decode audio: {proc.stderr.decode(errors='ignore')[:500]}"
        )
    return np.frombuffer(proc.stdout, dtype=np.float32).copy()


def pcm16_to_float32(data: bytes) -> np.ndarray:
    """Interpret raw little-endian int16 PCM bytes as float32 [-1, 1]."""
    ints = np.frombuffer(data, dtype="<i2")
    return (ints.astype(np.float32)) / 32768.0


def resample(samples: np.ndarray, orig_sr: int, target_sr: int = TARGET_SR) -> np.ndarray:
    if orig_sr == target_sr or samples.size == 0:
        return samples
    import soxr

    return soxr.resample(samples, orig_sr, target_sr).astype(np.float32)


async def decode_to_mono16k_async(data: bytes, target_sr: int = TARGET_SR) -> np.ndarray:
    return await asyncio.to_thread(decode_to_mono16k, data, target_sr)
