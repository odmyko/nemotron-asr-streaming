"""Stream a wav to the /ws/transcribe endpoint as 320ms PCM16 chunks."""
import asyncio, json, sys
import numpy as np, soundfile as sf, websockets

WAV = sys.argv[1] if len(sys.argv) > 1 else "test_data/sample_en.wav"
LANG = sys.argv[2] if len(sys.argv) > 2 else "en-US"
URL = f"ws://127.0.0.1:8000/ws/transcribe?language={LANG}&sample_rate=16000"


async def main():
    audio, sr = sf.read(WAV, dtype="float32")
    pcm = (np.clip(audio, -1, 1) * 32767).astype("<i2").tobytes()
    step = int(0.32 * sr) * 2  # bytes per 320ms
    async with websockets.connect(URL, max_size=None) as ws:
        print("server:", await ws.recv())
        async def sender():
            for i in range(0, len(pcm), step):
                await ws.send(pcm[i:i + step])
                await asyncio.sleep(0.05)  # pace it like real time-ish
            await ws.send(json.dumps({"type": "end"}))
        send_task = asyncio.create_task(sender())
        final = None
        while True:
            m = json.loads(await ws.recv())
            if m["type"] == "partial":
                print(f"  partial: {m['text']!r}")
            elif m["type"] == "final":
                final = m["text"]; print(f"\nFINAL: {final!r}"); break
            elif m["type"] == "error":
                print("ERROR:", m); break
        await send_task
    return final

asyncio.run(main())
