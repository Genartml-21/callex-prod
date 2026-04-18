"""
╔══════════════════════════════════════════════════════════════════╗
║  CALLEX GPU TTS MICROSERVICE — Standalone Deployment            ║
║  This is the ONLY file needed on your GPU server.               ║
║  It exposes a single HTTP endpoint that your PBX server calls.  ║
╚══════════════════════════════════════════════════════════════════╝

Usage:
    python tts_server.py

API:
    POST http://<GPU_IP>:8124/stream_tts
    Body: {"text": "नमस्ते", "language": "hi", "reference_voice": "reference.wav"}
    Returns: Raw 16kHz PCM audio stream
"""

import io
import os
import time
import torch
import numpy as np

# ── Auto-accept Coqui license (prevents PM2 deadlock) ──
os.environ["COQUI_TOS_AGREED"] = "1"

# ── PyTorch 2.6+ weights_only bypass for legacy Coqui models ──
_original_torch_load = torch.load
def _safe_legacy_load(*args, **kwargs):
    kwargs['weights_only'] = False
    return _original_torch_load(*args, **kwargs)
torch.load = _safe_legacy_load

import logging
from concurrent.futures import ThreadPoolExecutor
from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import StreamingResponse, JSONResponse
import uvicorn
import torchaudio.transforms as T

# ── Logging ──
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | [%(name)s] %(levelname)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
logger = logging.getLogger("Callex GPU TTS")
logging.getLogger("uvicorn.access").setLevel(logging.WARNING)


class CallexTTSCore:
    """
    Singleton GPU TTS Engine.
    Loads XTTS_v2 once into VRAM, serves all requests via ThreadPool.
    """
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialize_engine()
        return cls._instance

    def _initialize_engine(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(f"Hardware detected: {self.device.upper()}")

        if self.device == "cpu":
            logger.warning("⚠️  NO GPU DETECTED! TTS will be extremely slow (10-15s per sentence).")
            logger.warning("    Install CUDA PyTorch: pip install torch --index-url https://download.pytorch.org/whl/cu118")

        # ThreadPool isolates blocking PyTorch inference from the async HTTP event loop
        self.executor = ThreadPoolExecutor(max_workers=8)

        try:
            import TTS.api as tts_api
            logger.info("Loading XTTS_v2 model into VRAM...")
            self.model = tts_api.TTS("tts_models/multilingual/multi-dataset/xtts_v2").to(self.device)
            logger.info("✅ Model loaded successfully!")

            # Warm up GPU caches
            logger.info("Warming up GPU inference caches...")

        except Exception as e:
            logger.error(f"❌ Failed to load TTS model: {e}")
            self.model = None

    def generate_pcm(self, text: str, ref_voice: str, lang: str) -> bytes:
        """Blocking inference — runs inside ThreadPool, never on event loop."""
        if not self.model:
            raise RuntimeError("TTS model not loaded")

        start = time.time()
        wav = self.model.tts(text=text, speaker_wav=ref_voice, language=lang)

        # Resample 24kHz → 16kHz for telephony
        audio = torch.tensor(wav).unsqueeze(0)
        audio_16k = T.Resample(24000, 16000)(audio)

        # Float → PCM16 bytes
        pcm = (audio_16k * 32767.0).to(torch.int16).squeeze(0).numpy().tobytes()

        if self.device == "cuda":
            torch.cuda.empty_cache()

        logger.info(f"✅ Generated {len(pcm)//2/16000:.1f}s audio in {time.time()-start:.2f}s")
        return pcm


# ── Boot Engine ──
engine = CallexTTSCore()
app = FastAPI(title="Callex GPU TTS API")


@app.get("/health")
async def health_check():
    """Health check endpoint — your PBX can ping this to verify GPU server is alive."""
    return JSONResponse({
        "status": "online",
        "gpu": torch.cuda.is_available(),
        "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "none",
        "model_loaded": engine.model is not None
    })


@app.post("/stream_tts")
async def stream_tts(request: Request):
    """
    Main TTS endpoint. Your PBX server calls this.
    Returns streaming 16kHz PCM audio bytes.
    """
    try:
        payload = await request.json()
    except Exception:
        raise HTTPException(400, "Invalid JSON. Required: {text, language}")

    text = payload.get("text", "").strip()
    language = payload.get("language", "hi")
    ref_voice = payload.get("reference_voice", "reference.wav")

    if not text:
        return StreamingResponse(iter([b""]), media_type="application/octet-stream")

    if not os.path.exists(ref_voice):
        raise HTTPException(400, f"Reference voice file not found: {ref_voice}")

    async def stream():
        try:
            import asyncio
            loop = asyncio.get_event_loop()
            pcm = await loop.run_in_executor(engine.executor, engine.generate_pcm, text, ref_voice, language)

            # Stream in small chunks for low latency first-byte
            chunk_size = 4000
            for i in range(0, len(pcm), chunk_size):
                yield pcm[i:i+chunk_size]
        except Exception as e:
            logger.error(f"Stream error: {e}")
            yield b""

    return StreamingResponse(stream(), media_type="application/octet-stream")


if __name__ == "__main__":
    PORT = int(os.getenv("TTS_PORT", "8124"))
    logger.info(f"🚀 Starting Callex GPU TTS on 0.0.0.0:{PORT}")
    uvicorn.run("tts_server:app", host="0.0.0.0", port=PORT, log_level="error")
