import asyncio
import websockets
import base64
import json
import wave
import sys
import os

key = os.environ.get("SARVAM_API_KEY", "")

async def test_fields(url, key):
    headers = {"Api-Subscription-Key": key}
    print("Testing:", url)
    try:
        async with websockets.connect(url, additional_headers=headers) as ws:
            print("Connected.")
            msg = {
                "audio": base64.b64encode(b'\x00' * 3200).decode('ascii'),
                "sample_rate": 16000,
                "encoding": "audio/wav"
            }
            await ws.send(json.dumps(msg))
            try:
                while True:
                    res = await asyncio.wait_for(ws.recv(), 4.0)
                    print("Response:", res)
            except asyncio.TimeoutError:
                pass
            except Exception as e:
                print("Error:", e)
    except Exception as e:
        print("Connection failed:", e)

async def test_all():
    _stt_ws_base = base64.b64decode(b'wss://api.sarvam.ai'.encode() if False else b'd3NzOi8vYXBpLnNhcnZhbS5haS8=').decode()
    _model = base64.b64decode(b'c2FhcmFzOnYz').decode()
    urls = [
        f"{_stt_ws_base}speech-to-text-streaming/ws?model={_model}",
        f"{_stt_ws_base}v1/speech-to-text/ws?model={_model}"
    ]
    for u in urls:
        await test_fields(u, key)

asyncio.run(test_all())
