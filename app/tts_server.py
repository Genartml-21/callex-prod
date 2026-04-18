import io
import time
import torch
import numpy as np
from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse
import uvicorn
import logging
import torchaudio
import torchaudio.transforms as T

# Mute heavy FastAPI system logs to keep terminal clean
logging.getLogger("uvicorn.access").setLevel(logging.WARNING)

app = FastAPI()

print("[Microservice TTS] Booting up state-of-the-art XTTS_v2 Engine...")
device = "cuda" if torch.cuda.is_available() else "cpu"

try:
    from TTS.api import TTS
    global_tts = TTS("tts_models/multilingual/multi-dataset/xtts_v2").to(device)
    print(f"[Microservice TTS] ✅ XTTS_v2 initialized successfully on {device}.")
except Exception as e:
    print(f"[Microservice TTS] ⚠️ Could not load XTTS_v2. Ensure 'TTS' pip library is installed: {e}")
    global_tts = None


@app.post("/stream_tts")
async def generate_tts(request: Request):
    """
    Receives JSON payload containing 'text' and streams back raw 16000Hz PCM 16-bit bytes.
    Matches the exact audio specification the old ElevenLabs API provided.
    """
    payload = await request.json()
    text = payload.get("text", "")
    language = payload.get("language", "hi")
    
    # We require a sample voice to clone instantly. (You can provide any 5 second .wav file!)
    reference_voice_path = payload.get("reference_voice", "data/callex_reference.wav")

    def pcm_generator():
        if not global_tts:
            yield b""
            return
            
        try:
            print(f"[Microservice TTS] Generating audio natively for: '{text[:50]}...'")
            start = time.time()
            
            # Generate highly expressive XTTS_v2 Audio Tensor Sequence in memory
            wav_list = global_tts.tts(text=text, speaker_wav=reference_voice_path, language=language)
            
            # Coqui usually generates at 24000Hz. PBX expects 16000Hz explicitly.
            audio_tensor = torch.tensor(wav_list).unsqueeze(0)
            resampler = T.Resample(orig_freq=24000, new_freq=16000)
            audio_16k = resampler(audio_tensor)
            
            # Convert float32 array down to PCM Int16 strict byte stream
            audio_int16 = (audio_16k * 32767.0).to(torch.int16).squeeze(0).numpy().tobytes()
            
            print(f"[Microservice TTS] -> Generated in {time.time() - start:.2f}s")
            
            # Yield in smaller chunks to emulate streaming latency (e.g. 0.1s slices)
            chunk_size = 3200 
            for i in range(0, len(audio_int16), chunk_size):
                yield audio_int16[i:i+chunk_size]
                
        except Exception as e:
            print(f"[Microservice TTS] Error generating PCM: {e}. (Ensure {reference_voice_path} exists!)")
            yield b""

    # Use standard streaming response format natively hooked to async loop
    return StreamingResponse(pcm_generator(), media_type="application/octet-stream")

if __name__ == "__main__":
    uvicorn.run("tts_server:app", host="127.0.0.1", port=8124, log_level="info")
