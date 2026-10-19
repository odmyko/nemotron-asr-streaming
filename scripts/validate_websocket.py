"""Check the running API with real PCM and require text before sending Stop."""
import argparse
import asyncio
import json
import time

import numpy as np
import soundfile as sf
import soxr
import websockets


async def run(base_url, audio, rate, diarize):
    query = f"language=en-US&sample_rate={rate}&diarize={str(diarize).lower()}"
    pcm = (np.clip(audio, -1, 1) * 32767).astype("<i2").tobytes()
    partial_received = asyncio.Event()
    started = time.perf_counter()
    first_partial_seconds = None
    async with websockets.connect(f"{base_url}/ws/transcribe?{query}", max_size=None) as ws:
        ready = json.loads(await ws.recv())
        assert ready["type"] == "ready", ready

        async def sender():
            step = int(rate * .32) * 2
            for offset in range(0, len(pcm), step):
                await ws.send(pcm[offset:offset + step])
                await asyncio.sleep(.03)
            await asyncio.wait_for(partial_received.wait(), 90)
            await ws.send(json.dumps({"type": "end"}))

        task = asyncio.create_task(sender())
        try:
            while True:
                result = json.loads(await asyncio.wait_for(ws.recv(), 120))
                assert result["type"] != "error", result
                if result["type"] == "partial" and result["text"]:
                    if first_partial_seconds is None:
                        first_partial_seconds = time.perf_counter() - started
                    partial_received.set()
                if result["type"] == "final":
                    assert result["text"]
                    if diarize:
                        assert result["segments"] and all(s["final"] for s in result["segments"])
                    print(json.dumps({"diarize": diarize, "sample_rate": rate,
                                      "first_partial_wall_sec": first_partial_seconds,
                                      "text": result["text"][:300],
                                      "speakers": sorted({s["speaker_id"] for s in result.get("segments", [])})}), flush=True)
                    break
            await task
            return result["text"]
        finally:
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)


async def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default="ws://127.0.0.1:8000")
    parser.add_argument("--audio", default="test_data/sample_en.wav")
    args = parser.parse_args()
    audio, rate = sf.read(args.audio, dtype="float32")
    plain_text = None
    for diarize, target_rate in ((False, 16000), (True, 48000), (True, 44100), (False, 16000)):
        samples = soxr.resample(audio, rate, target_rate) if rate != target_rate else audio
        text = await run(args.url, samples, target_rate, diarize)
        if not diarize:
            if plain_text is not None:
                assert text == plain_text, "Ordinary ASR changed after a diarized session"
            plain_text = text
    # A new connection must not inherit transcript/speaker caches from prior speech.
    async with websockets.connect(f"{args.url}/ws/transcribe?diarize=true", max_size=None) as ws:
        assert json.loads(await ws.recv())["type"] == "ready"
        await ws.send(np.zeros(24000, dtype="<i2").tobytes())
        await ws.send(json.dumps({"type": "end"}))
        while True:
            result = json.loads(await asyncio.wait_for(ws.recv(), 120))
            assert result["type"] != "error", result
            if result["type"] == "final":
                assert not result["text"].strip() and not result["segments"], result
                print("Fresh silent session: no transcript leakage", flush=True)
                break


if __name__ == "__main__":
    asyncio.run(main())
