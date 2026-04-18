"""
Speaker Verifier — Production-Grade Caller Voice Identification
=================================================================

Uses Resemblyzer to generate speaker embeddings and verify that
incoming speech belongs to the actual caller (not background speakers).

Production Pipeline:
  1. Enrollment: First ~3 seconds of high-quality speech → reference embedding
  2. Rolling Update: Reference embedding adapts (80% old + 20% new) on confirmed speech
  3. Verification Buffer: Accumulates 1.5s+ of speech before comparing
  4. Energy Gate: Only enroll from loud, clear speech (not bot echo/noise)

Reference: https://github.com/resemble-ai/Resemblyzer
"""

import numpy as np
from typing import Tuple, Optional, List
import time


_GLOBAL_VOICE_ENCODER = None

class SpeakerVerifier:
    """
    Production-grade caller voice verification with rolling embeddings.
    
    Features:
    - Energy-gated enrollment (rejects quiet/noisy audio)
    - Rolling reference embedding (improves over time)
    - Verification buffer (accumulates speech for reliable comparison)
    - Multi-embedding enrollment (uses multiple samples for robust voice print)
    """

    def __init__(
        self,
        sample_rate: int = 16000,
        enrollment_seconds: float = 3.0,
        similarity_threshold: float = 0.72,
    ):
        self.sample_rate = sample_rate
        self.enrollment_seconds = enrollment_seconds
        self.similarity_threshold = similarity_threshold

        # Lazy-load the encoder to avoid slowing down imports
        self._encoder = None
        self._model_loaded = False

        # Enrollment state
        self.reference_embedding: Optional[np.ndarray] = None
        self.enrollment_buffer: List[float] = []
        self.enrollment_duration: float = 0.0
        self.is_enrolled: bool = False
        self._enrollment_energy_sum: float = 0.0
        self._enrollment_chunk_count: int = 0

        # Rolling verification buffer (accumulate speech for reliable comparison)
        self._verify_buffer: List[float] = []
        self._verify_buffer_duration: float = 0.0
        self.MIN_VERIFY_DURATION: float = 0.6  # minimum seconds of speech needed for reliable embedding

        # Rolling embedding update
        self._verified_utterance_count: int = 0
        self._EMBEDDING_UPDATE_WEIGHT: float = 0.20  # 20% new, 80% old
        self._MAX_UPDATES: int = 10  # Stop updating after 10 verified utterances

        # Enrollment quality gate
        self._MIN_ENROLLMENT_ENERGY_DB: float = -35.0  # Minimum avg energy for enrollment

        print(f"[Speaker Verifier] Initialized (enrollment={enrollment_seconds}s, threshold={similarity_threshold})")

    def _load_model(self):
        """Lazy-load Resemblyzer encoder on first use, caching it globally for concurrent calls."""
        if self._model_loaded:
            return

        global _GLOBAL_VOICE_ENCODER
        try:
            if _GLOBAL_VOICE_ENCODER is None:
                from resemblyzer import VoiceEncoder
                print("[Speaker Verifier] Loading Resemblyzer voice encoder globally...")
                start = time.time()
                _GLOBAL_VOICE_ENCODER = VoiceEncoder(device="cpu")
                print(f"[Speaker Verifier] ✅ Shared Encoder loaded ({time.time()-start:.1f}s)")

            self._encoder = _GLOBAL_VOICE_ENCODER
            self._model_loaded = True
        except ImportError:
            print("[Speaker Verifier] ❌ resemblyzer not installed. Run: pip install resemblyzer")
            self._encoder = None
            self._model_loaded = True  # Don't retry
        except Exception as e:
            print(f"[Speaker Verifier] ❌ Failed to load encoder: {e}")
            self._encoder = None
            self._model_loaded = True

    def reset(self):
        """Reset enrollment (call at start of each new call)."""
        self.reference_embedding = None
        self.enrollment_buffer = []
        self.enrollment_duration = 0.0
        self.is_enrolled = False
        self._enrollment_energy_sum = 0.0
        self._enrollment_chunk_count = 0
        self._verify_buffer = []
        self._verify_buffer_duration = 0.0
        self._verified_utterance_count = 0
        print("[Speaker Verifier] Reset for new call")

    def _get_embedding(self, audio: np.ndarray) -> Optional[np.ndarray]:
        """Generate a speaker embedding from audio."""
        if self._encoder is None:
            return None

        try:
            # Resemblyzer expects float32 audio normalized to [-1, 1]
            if audio.dtype != np.float32:
                audio = audio.astype(np.float32)

            max_val = np.max(np.abs(audio))
            if max_val > 1.0:
                audio = audio / max_val

            # Minimum length check (Resemblyzer needs at least ~0.5s)
            min_samples = int(self.sample_rate * 0.5)
            if len(audio) < min_samples:
                return None

            embedding = self._encoder.embed_utterance(audio)
            return embedding

        except Exception as e:
            print(f"[Speaker Verifier] Embedding error: {e}")
            return None

    def _cosine_similarity(self, a: np.ndarray, b: np.ndarray) -> float:
        """Compute cosine similarity between two embeddings."""
        dot = np.dot(a, b)
        norm_a = np.linalg.norm(a)
        norm_b = np.linalg.norm(b)
        if norm_a == 0 or norm_b == 0:
            return 0.0
        return float(dot / (norm_a * norm_b))

    def _audio_energy_db(self, audio: np.ndarray) -> float:
        """Compute energy of audio in dB."""
        energy = np.sqrt(np.mean(audio * audio))
        return float(20 * np.log10(energy + 1e-9))

    def enroll(self, audio_chunk: np.ndarray) -> bool:
        """
        Feed audio during enrollment phase. Returns True when enrollment is complete.
        
        Quality gate: Only accepts chunks with sufficient energy (rejects bot echo/noise).
        """
        if self.is_enrolled:
            return True

        self._load_model()
        if self._encoder is None:
            # No model available — auto-pass everything
            self.is_enrolled = True
            return True

        # Energy gate: reject quiet audio (likely bot echo or background noise)
        chunk_db = self._audio_energy_db(audio_chunk)
        if chunk_db < self._MIN_ENROLLMENT_ENERGY_DB:
            return False

        # Track average energy for quality check
        self._enrollment_energy_sum += chunk_db
        self._enrollment_chunk_count += 1

        self.enrollment_buffer.extend(audio_chunk.tolist())
        self.enrollment_duration += len(audio_chunk) / self.sample_rate

        if self.enrollment_duration >= self.enrollment_seconds:
            # Check average energy quality
            avg_energy = self._enrollment_energy_sum / max(1, self._enrollment_chunk_count)
            if avg_energy < self._MIN_ENROLLMENT_ENERGY_DB:
                print(f"[Speaker Verifier] ⚠️ Enrollment audio too quiet ({avg_energy:.1f}dB), collecting more...")
                self.enrollment_seconds += 1.0
                return False

            # Generate multi-segment reference embedding for robustness
            enrollment_audio = np.array(self.enrollment_buffer, dtype=np.float32)
            
            # Split enrollment audio into overlapping segments and average embeddings
            segment_len = int(self.sample_rate * 1.6)  # 1.6s segments
            hop = segment_len // 2  # 50% overlap
            embeddings = []
            
            for i in range(0, len(enrollment_audio) - segment_len + 1, hop):
                segment = enrollment_audio[i:i + segment_len]
                emb = self._get_embedding(segment)
                if emb is not None:
                    embeddings.append(emb)
            
            # Also generate full-audio embedding
            full_emb = self._get_embedding(enrollment_audio)
            if full_emb is not None:
                embeddings.append(full_emb)

            if len(embeddings) >= 2:
                # Average all embeddings for a robust reference
                self.reference_embedding = np.mean(embeddings, axis=0)
                # Normalize
                norm = np.linalg.norm(self.reference_embedding)
                if norm > 0:
                    self.reference_embedding = self.reference_embedding / norm
                
                self.is_enrolled = True
                self.enrollment_buffer = []  # Free memory
                print(f"[Speaker Verifier] ✅ Caller enrolled ({self.enrollment_duration:.1f}s, "
                      f"{len(embeddings)} segments, avg energy: {avg_energy:.1f}dB)")
                return True
            elif len(embeddings) == 1:
                self.reference_embedding = embeddings[0]
                self.is_enrolled = True
                self.enrollment_buffer = []
                print(f"[Speaker Verifier] ✅ Caller enrolled ({self.enrollment_duration:.1f}s, "
                      f"single segment, avg energy: {avg_energy:.1f}dB)")
                return True
            else:
                # Failed to generate any embedding, try with more audio
                print("[Speaker Verifier] ⚠️ Enrollment embedding failed, collecting more audio...")
                self.enrollment_seconds += 1.0
                return False

        return False

    def feed_verify_buffer(self, audio_chunk: np.ndarray):
        """
        Feed speech into the rolling verification buffer.
        Call this on every valid speech chunk (even before barge-in check).
        The buffer accumulates audio for more reliable embedding comparison.
        """
        self._verify_buffer.extend(audio_chunk.tolist())
        self._verify_buffer_duration += len(audio_chunk) / self.sample_rate
        
        # Keep buffer from growing too large (max 5 seconds)
        max_samples = int(self.sample_rate * 5.0)
        if len(self._verify_buffer) > max_samples:
            excess = len(self._verify_buffer) - max_samples
            self._verify_buffer = self._verify_buffer[excess:]
            self._verify_buffer_duration = len(self._verify_buffer) / self.sample_rate

    def clear_verify_buffer(self):
        """Clear the verification buffer (call after speech ends)."""
        self._verify_buffer = []
        self._verify_buffer_duration = 0.0

    def verify(self, audio_chunk: np.ndarray) -> Tuple[bool, float]:
        """
        Verify if the audio belongs to the enrolled caller.
        
        Uses the accumulated verification buffer for more reliable comparison.
        Falls back to individual chunk if buffer is too short.
        """
        # If model failed to load, pass everything through (fail-open)
        if self._encoder is None:
            return True, 1.0

        # If not enrolled yet, try to enroll with this chunk
        if not self.is_enrolled:
            self.enroll(audio_chunk)
            # During enrollment, assume it's the caller (they're the first speaker)
            return True, 1.0

        # Use verification buffer if it has enough audio, otherwise use chunk directly
        if self._verify_buffer_duration >= self.MIN_VERIFY_DURATION:
            verify_audio = np.array(self._verify_buffer, dtype=np.float32)
        else:
            # Combine buffer + current chunk
            combined = list(self._verify_buffer) + audio_chunk.tolist()
            combined_duration = len(combined) / self.sample_rate
            if combined_duration >= self.MIN_VERIFY_DURATION:
                verify_audio = np.array(combined, dtype=np.float32)
            else:
                # Not enough audio yet — soft pass but with 0.0 confidence 
                # so that barge_in_confirm waits for a real voice match
                return True, 0.0

        # Generate embedding for incoming speech
        chunk_embedding = self._get_embedding(verify_audio)
        if chunk_embedding is None:
            return True, 0.0

        # Compare with reference
        similarity = self._cosine_similarity(self.reference_embedding, chunk_embedding)
        is_caller = similarity >= self.similarity_threshold

        if is_caller:
            # Update rolling reference embedding (improves over time)
            self._update_reference(chunk_embedding)
        else:
            print(f"[Speaker Verifier] 🛡️ Background speaker rejected (sim={similarity:.2f}, "
                  f"threshold={self.similarity_threshold})")

        return is_caller, similarity

    def _update_reference(self, new_embedding: np.ndarray):
        """
        Rolling update of reference embedding with verified speech.
        Weighted average: 80% old reference + 20% new verified embedding.
        """
        if self._verified_utterance_count >= self._MAX_UPDATES:
            return  # Voice print is stable enough
        
        if self.reference_embedding is None:
            return

        weight = self._EMBEDDING_UPDATE_WEIGHT
        updated = (1.0 - weight) * self.reference_embedding + weight * new_embedding
        
        # Re-normalize
        norm = np.linalg.norm(updated)
        if norm > 0:
            self.reference_embedding = updated / norm
        
        self._verified_utterance_count += 1


# Test
if __name__ == "__main__":
    print("Testing Speaker Verifier...")

    sv = SpeakerVerifier(sample_rate=16000, enrollment_seconds=2.0, similarity_threshold=0.72)

    # Simulate enrollment with sine wave "voice"
    t = np.linspace(0, 2, 32000)
    fake_caller = (np.sin(2 * np.pi * 300 * t) * 0.5).astype(np.float32)

    # Enroll
    enrolled = sv.enroll(fake_caller)
    print(f"Enrolled: {enrolled}")

    # Verify same voice
    is_caller, sim = sv.verify(fake_caller[:8000])
    print(f"Same voice: is_caller={is_caller}, similarity={sim:.3f}")

    # Verify different voice
    different = (np.sin(2 * np.pi * 800 * t) * 0.5).astype(np.float32)
    is_caller, sim = sv.verify(different[:8000])
    print(f"Different voice: is_caller={is_caller}, similarity={sim:.3f}")

    print("\n✅ Speaker Verifier tests complete!")
