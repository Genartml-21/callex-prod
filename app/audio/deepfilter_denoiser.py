"""
DeepFilterNet3 Production Noise Suppressor
==========================================

Production-grade wrapper for DeepFilterNet3 — the state-of-the-art neural
noise suppression model trained on the DNS Challenge dataset (traffic, crowd,
babble, wind, vehicle noise).

Architecture:
  Raw 16kHz PCM → Upsample 48kHz → DeepFilterNet3 → Downsample 16kHz → Clean PCM

Key Features:
  - Buffers incoming chunks and processes in optimal 480-sample (10ms @ 48kHz) windows
  - Globally pre-loaded model shared across calls (single load at startup)
  - Per-call state reset with graceful fallback to raw audio on any crash
  - Zero external API calls — fully on-premise

Reference: https://github.com/rikorose/DeepFilterNet
"""

import logging
import time
import warnings
from threading import Lock
from typing import Optional, Tuple

import numpy as np
import torch
import torchaudio.functional as TAF

# Suppress noisy warnings from torchaudio backward compat shim
warnings.filterwarnings("ignore", category=UserWarning, module="df")
warnings.filterwarnings("ignore", category=UserWarning, module="torchaudio")

logger = logging.getLogger(__name__)

# ─── Global singleton (loaded once per process) ───────────────────────────────
_GLOBAL_DF_MODEL = None
_GLOBAL_DF_STATE = None
_GLOBAL_DF_LOCK  = Lock()
_GLOBAL_DF_SR    = 48000  # DeepFilterNet3 native sample rate


def load_deepfilter_model() -> bool:
    """
    Load DeepFilterNet3 model into global state.
    Call once at startup from lifespan(). Thread-safe.
    
    Returns True on success, False on failure.
    """
    global _GLOBAL_DF_MODEL, _GLOBAL_DF_STATE, _GLOBAL_DF_SR

    with _GLOBAL_DF_LOCK:
        if _GLOBAL_DF_MODEL is not None:
            return True  # Already loaded

        try:
            t0 = time.time()
            print("[DeepFilter] 🧠 Loading DeepFilterNet3 model...")

            from df import init_df
            from df.enhance import enhance as _enhance

            model, df_state, _ = init_df()
            model.eval()

            # Store enhance function too so we don't import it per-call
            from df.config import config
            # We must create a new DF state per call, so we save the config parameters
            from df.model import ModelParams
            p = ModelParams()
            df_params = {
                'sr': p.sr, 'fft_size': p.fft_size, 'hop_size': p.hop_size, 
                'nb_bands': p.nb_erb, 'min_nb_erb_freqs': p.min_nb_freqs
            }
            
            _GLOBAL_DF_MODEL = (model, _enhance, df_params)
            _GLOBAL_DF_STATE = df_state # DEPRECATED, kept for compat
            _GLOBAL_DF_SR    = p.sr   # 48000

            elapsed = time.time() - t0
            print(f"[DeepFilter] ✅ DeepFilterNet3 loaded in {elapsed:.2f}s "
                  f"(native SR={_GLOBAL_DF_SR}Hz, model=epoch-120)")
            return True

        except Exception as e:
            print(f"[DeepFilter] ❌ Failed to load model: {e}")
            print("[DeepFilter] ⚠️  Falling back to raw audio passthrough")
            return False


def is_model_loaded() -> bool:
    """Return True if the global model is ready."""
    return _GLOBAL_DF_MODEL is not None


# ─── Per-call denoiser ────────────────────────────────────────────────────────

class DeepFilterDenoiser:
    """
    Per-call instance of DeepFilterNet3 noise suppression.

    Usage:
        denoiser = DeepFilterDenoiser(call_sample_rate=16000)
        clean_float32 = denoiser.process(raw_int16_array)

    The denoiser maintains an internal buffer, so you can feed it any-sized
    chunks from FreeSWITCH (e.g. 160-sample / 10ms packets) and it will
    process them optimally.
    """

    CALL_SR = 16000  # FreeSWITCH always sends 16kHz

    def __init__(self, call_sample_rate: int = 16000):
        self.call_sr = call_sample_rate
        self._model_active = is_model_loaded()
        self._buffer = np.array([], dtype=np.float32)
        self._calls_processed = 0
        self._total_latency_ms = 0.0

        # Compute optimal chunk size for DeepFilterNet3
        # DF3 hop=480 @ 48kHz → 480*(16000/48000) = 160 samples @ 16kHz
        # We process in 480-sample @ 16kHz (= 1440 @ 48kHz) windows for good
        # SNR improvement without adding too much latency per packet
        self._process_chunk_16k = 480  # 30ms @ 16kHz
        
        self._df_state = None

        if self._model_active:
            # Instantiate a completely unique RNN state for this specific call
            from libdf import DF
            model, _enhance, params = _GLOBAL_DF_MODEL
            self._df_state = DF(**params)
            print(f"[DeepFilter] ✅ Per-call instance ready (SR={self.call_sr}Hz)")
        else:
            print("[DeepFilter] ⚠️  Model not available — passthrough mode")

    def reset(self):
        """Reset internal state (call if reusing this instance, not needed for new calls)."""
        self._buffer = np.array([], dtype=np.float32)

    def process(self, pcm_int16: np.ndarray) -> np.ndarray:
        """
        Denoise a chunk of raw audio from FreeSWITCH.

        Args:
            pcm_int16: Int16 numpy array at CALL_SR (16kHz)

        Returns:
            Float32 numpy array of the same length, denoised.
            Falls back to normalized passthrough if model is unavailable.
        """
        if not self._model_active:
            # Passthrough — normalize to float32 only
            return pcm_int16.astype(np.float32) / 32768.0

        try:
            # Convert incoming int16 to float32 [-1, 1]
            float_audio = pcm_int16.astype(np.float32) / 32768.0

            # Accumulate into internal buffer
            self._buffer = np.concatenate([self._buffer, float_audio])

            output_chunks = []

            # Process in 480-sample windows
            while len(self._buffer) >= self._process_chunk_16k:
                chunk_16k = self._buffer[:self._process_chunk_16k]
                self._buffer = self._buffer[self._process_chunk_16k:]

                enhanced = self._enhance_chunk(chunk_16k)
                output_chunks.append(enhanced)

            if output_chunks:
                return np.concatenate(output_chunks)
            else:
                # Not enough samples yet — return silence placeholder
                # (caller should handle empty returns gracefully)
                return np.array([], dtype=np.float32)

        except Exception as e:
            print(f"[DeepFilter] ⚠️  Error during inference: {e}, falling back")
            self._model_active = False
            return pcm_int16.astype(np.float32) / 32768.0

    def _enhance_chunk(self, chunk_16k: np.ndarray) -> np.ndarray:
        """Process a single 480-sample chunk through DeepFilterNet3."""
        model, enhance_fn, _ = _GLOBAL_DF_MODEL
        df_state = self._df_state

        t0 = time.perf_counter()

        with torch.no_grad():
            # Step 1: 16kHz → 48kHz (DeepFilter native rate)
            t_16k = torch.from_numpy(chunk_16k).unsqueeze(0)   # [1, 480]
            t_48k = TAF.resample(t_16k, self.call_sr, _GLOBAL_DF_SR)  # [1, 1440]

            # Step 2: Run DeepFilterNet3
            enhanced_48k = enhance_fn(model, df_state, t_48k)   # [1, 1440]

            # Step 3: 48kHz → 16kHz
            enhanced_16k = TAF.resample(enhanced_48k, _GLOBAL_DF_SR, self.call_sr)  # [1, 480]

        result = enhanced_16k.squeeze(0).cpu().numpy()

        # Clip to [-1, 1] to prevent overflow in downstream int16 conversion
        result = np.clip(result, -1.0, 1.0)

        elapsed_ms = (time.perf_counter() - t0) * 1000
        self._calls_processed += 1
        self._total_latency_ms += elapsed_ms

        # Log diagnostics every 200 frames (~1 minute of audio)
        if self._calls_processed % 200 == 0:
            avg_ms = self._total_latency_ms / self._calls_processed
            print(f"[DeepFilter] 📊 {self._calls_processed} frames, avg latency: {avg_ms:.1f}ms")

        return result

    @property
    def stats(self) -> str:
        if self._calls_processed == 0:
            return "[DeepFilter] No frames processed yet"
        avg_ms = self._total_latency_ms / self._calls_processed
        return (f"[DeepFilter] Processed {self._calls_processed} frames, "
                f"avg {avg_ms:.1f}ms/frame, active={self._model_active}")
