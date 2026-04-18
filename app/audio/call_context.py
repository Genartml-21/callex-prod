"""
CallAudioContext — Production-Grade Per-Call Audio Isolation
=============================================================

Ensures that every concurrent phone call gets its own completely
isolated audio processing pipeline. This prevents:
  - VAD hidden-state crosstalk (barge-in failures)
  - DeepFilter RNN buffer leakage (garbled audio)
  - Speaker verification embedding confusion
  - Noise filter buffer corruption

Architecture:
  Global (shared, read-only):
    - PyTorch model weights (Silero VAD, DeepFilterNet3, Resemblyzer)
    - YAMNet TF model (stateless inference)

  Per-Call (isolated, mutable):
    - Silero VAD model clone (deep-copied RNN state)
    - DeepFilterNet3 DF state (unique FFT/ISTFT buffers)
    - SpeakerVerifier enrollment + embedding state
    - NoiseFilter PCM buffer + filter state
    - SemanticFilter instance

Usage:
    ctx = CallAudioContext(sample_rate=16000)
    # Use ctx.silero_vad, ctx.deepfilter, ctx.speaker_verifier, etc.
    # When call ends:
    ctx.cleanup()
"""

import copy
import time
import numpy as np
from typing import Optional

from app.audio import vad_silero as _vad_module
from app.audio.vad_silero import SileroVADFilter
from app.audio.deepfilter_denoiser import DeepFilterDenoiser
from app.audio.speaker_verifier import SpeakerVerifier
from app.audio.semantic import SemanticFilter


class CallAudioContext:
    """
    Bundles all per-call audio processing state into one object.
    
    Every WebSocket connection (= phone call) MUST create its own
    CallAudioContext. This guarantees zero state leakage between
    concurrent calls.
    """

    def __init__(
        self,
        call_uuid: str,
        sample_rate: int = 16000,
        use_silero: bool = True,
        silero_threshold: float = 0.50,
        speaker_enrollment_seconds: float = 3.0,
        speaker_similarity_threshold: float = 0.65,
        semantic_min_length: int = 1,
        yamnet_classifier=None,
    ):
        self.call_uuid = call_uuid
        self.sample_rate = sample_rate
        self._creation_time = time.time()

        # ── 1. Silero VAD (deep-copied model for isolated RNN state) ──
        # Access the global model through the MODULE (not a captured import value)
        # so we always see the latest value after startup has loaded it.
        global_silero = _vad_module._GLOBAL_SILERO_MODEL
        self.silero_vad: Optional[SileroVADFilter] = None
        if use_silero and global_silero is not None:
            t0 = time.time()
            self.silero_vad = SileroVADFilter(
                sample_rate=sample_rate,
                threshold=silero_threshold,
            )
            # CRITICAL: Deep-copy the shared PyTorch model to get isolated
            # hidden state (h, c tensors). Without this, concurrent calls
            # corrupt each other's RNN state → barge-in breaks completely.
            self.silero_vad.model = copy.deepcopy(global_silero)
            self.silero_vad.model.eval()
            self.silero_vad.reset_noise_profile()
            elapsed_ms = (time.time() - t0) * 1000
            print(f"[CallContext:{call_uuid[:8]}] ✅ Silero VAD cloned ({elapsed_ms:.1f}ms)")
        else:
            print(f"[CallContext:{call_uuid[:8]}] ⚠️ Silero VAD not available (use_silero={use_silero}, model_loaded={global_silero is not None})")

        # ── 2. DeepFilterNet3 (per-call DF state for isolated FFT buffers) ──
        self.deepfilter = DeepFilterDenoiser(call_sample_rate=sample_rate)

        # ── 3. Speaker Verifier (per-call enrollment + embeddings) ──
        self.speaker_verifier = SpeakerVerifier(
            sample_rate=sample_rate,
            enrollment_seconds=speaker_enrollment_seconds,
            similarity_threshold=speaker_similarity_threshold,
        )

        # ── 4. Semantic Filter (per-call instance) ──
        self.semantic_filter = SemanticFilter(
            language='hi',
            min_length=semantic_min_length,
        )

        # ── 5. YAMNet classifier (stateless — safe to share globally) ──
        self.classifier = yamnet_classifier

        print(f"[CallContext:{call_uuid[:8]}] ✅ All audio pipelines isolated for this call")

    @property
    def use_silero(self) -> bool:
        return self.silero_vad is not None

    def cleanup(self):
        """Release per-call resources. Call when the WebSocket disconnects."""
        call_id = self.call_uuid[:8]
        duration = time.time() - self._creation_time

        # Release the deep-copied model to free GPU/CPU memory
        if self.silero_vad is not None:
            del self.silero_vad.model
            self.silero_vad = None

        # Release DeepFilter per-call state
        if self.deepfilter is not None:
            if hasattr(self.deepfilter, '_df_state') and self.deepfilter._df_state is not None:
                del self.deepfilter._df_state
            self.deepfilter = None

        # Release speaker verifier buffers
        if self.speaker_verifier is not None:
            self.speaker_verifier.reset()
            self.speaker_verifier = None

        self.semantic_filter = None
        self.classifier = None

        print(f"[CallContext:{call_id}] 🧹 Cleaned up after {duration:.0f}s call")
