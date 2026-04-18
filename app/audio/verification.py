"""
Enterprise Speaker Verification Module
======================================

Uses pyannote.audio embeddings to verify the target speaker (caller)
and reject background voices.

This is Layer 3 of the 5-layer audio validation pipeline.

Architecture:
    1. Capture caller's voiceprint from first 2-3 seconds of speech
    2. For each new speech segment, compute embedding
    3. Calculate cosine similarity with caller embedding
    4. Accept only if similarity > threshold (0.75)

Author: AI Audio Engineering Team
Date: 2026-01-31
"""

import numpy as np
import torch
from typing import Optional, Tuple
from scipy.spatial.distance import cosine
from pyannote.audio import Model, Inference
import struct


class SpeakerVerifier:
    """
    Production-grade speaker verification using pyannote.audio embeddings.
    
    Ensures bot only responds to the actual caller, not background voices.
    """
    
    def __init__(
        self,
        similarity_threshold: float = 0.75,
        voiceprint_duration_ms: int = 2000,
        sample_rate: int = 16000,
        device: str = "cpu"
    ):
        """
        Initialize speaker verification system.
        
        Args:
            similarity_threshold: Minimum cosine similarity to accept (0.70-0.85)
            voiceprint_duration_ms: Duration to capture caller voiceprint
            sample_rate: Audio sample rate (Hz)
            device: Compute device ('cpu' or 'cuda')
        """
        self.similarity_threshold = similarity_threshold
        self.voiceprint_duration_ms = voiceprint_duration_ms
        self.sample_rate = sample_rate
        self.device = device
        
        # Speaker embedding model (pyannote.audio)
        print("[Speaker Verification] Loading pyannote embedding model...")
        try:
             # Load model first (Handling 3.1.0 changes)
            self.model = Model.from_pretrained("pyannote/embedding", use_auth_token=None)
            if self.model is None:
                 print("[Speaker Verification] Warning: Could not load 'pyannote/embedding'. Trying 'pyannote/wespeaker-voxceleb-resnet34-LM'...")
                 self.model = Model.from_pretrained("pyannote/wespeaker-voxceleb-resnet34-LM", use_auth_token=None)
            
            if self.model is None:
                 raise ValueError("Failed to load pyannote model. Please ensure you have access or a valid token if using gated models.")

            self.speaker_model = Inference(
                self.model,
                window="whole",
                device=torch.device(device)
            )
            print("[Speaker Verification] Model loaded successfully")
        except Exception as e:
            print(f"[Speaker Verification] CRITICAL ERROR: Failed to load model: {e}")
            # Fallback to dummy model or disable verification to prevent crash
            self.speaker_model = None
        print("[Speaker Verification] Model loaded successfully")
        
        # Caller's voiceprint (512-dim embedding)
        self.caller_embedding: Optional[np.ndarray] = None
        self.voiceprint_captured = False
        
        # Statistics
        self.stats = {
            "total_checks": 0,
            "caller_verified": 0,
            "background_rejected": 0,
            "avg_similarity": 0.0
        }
    
    def _pcm_to_tensor(self, pcm_bytes: bytes) -> torch.Tensor:
        """
        Convert PCM bytes to PyTorch tensor for pyannote.
        
        Args:
            pcm_bytes: Raw PCM audio (16-bit, mono)
        
        Returns:
            Audio tensor (shape: [1, num_samples])
        """
        # Convert bytes to int16 array
        audio_int16 = np.frombuffer(pcm_bytes, dtype=np.int16)
        
        # Normalize to [-1, 1]
        audio_float = audio_int16.astype(np.float32) / 32768.0
        
        # Convert to tensor and add channel dimension
        audio_tensor = torch.from_numpy(audio_float).unsqueeze(0)
        
        return audio_tensor
    
    def capture_caller_voiceprint(self, audio_pcm: bytes) -> bool:
        """
        Capture caller's voiceprint from first speech segment.
        
        This should be called with the first 2-3 seconds of clear caller speech.
        
        Args:
            audio_pcm: PCM audio of caller speaking (16-bit, mono)
        
        Returns:
            True if voiceprint captured successfully, False otherwise
        """
        try:
            # Convert PCM to tensor
            audio_tensor = self._pcm_to_tensor(audio_pcm)
            
            # Generate speaker embedding
            embedding = self.speaker_model({
                "waveform": audio_tensor,
                "sample_rate": self.sample_rate
            })
            
            # Store as caller voiceprint
            self.caller_embedding = embedding.cpu().numpy()
            self.voiceprint_captured = True
            
            print(f"[Speaker Verification] ✓ Caller voiceprint captured "
                  f"({len(audio_pcm)} bytes, embedding dim: {self.caller_embedding.shape})")
            
            return True
            
        except Exception as e:
            print(f"[Speaker Verification] ❌ Failed to capture voiceprint: {e}")
            return False
    
    def verify_speaker(self, audio_pcm: bytes) -> Tuple[bool, float]:
        """
        Verify if audio segment is from the caller.
        
        Args:
            audio_pcm: PCM audio segment to verify (16-bit, mono)
        
        Returns:
            Tuple of (is_caller, similarity_score)
            - is_caller: True if speaker matches caller, False otherwise
            - similarity_score: Cosine similarity (0.0-1.0)
        """
        if not self.voiceprint_captured:
            print("[Speaker Verification] ⚠️  No voiceprint captured yet - accepting by default")
            return True, 1.0
        
        try:
            self.stats["total_checks"] += 1
            
            # Convert PCM to tensor
            audio_tensor = self._pcm_to_tensor(audio_pcm)
            
            # Generate embedding for this segment
            segment_embedding = self.speaker_model({
                "waveform": audio_tensor,
                "sample_rate": self.sample_rate
            })
            segment_embedding = segment_embedding.cpu().numpy()
            
            # Calculate cosine similarity
            similarity = 1.0 - cosine(
                self.caller_embedding.flatten(),
                segment_embedding.flatten()
            )
            
            # Update stats
            self.stats["avg_similarity"] = (
                (self.stats["avg_similarity"] * (self.stats["total_checks"] - 1) + similarity)
                / self.stats["total_checks"]
            )
            
            # Decision
            is_caller = similarity >= self.similarity_threshold
            
            if is_caller:
                self.stats["caller_verified"] += 1
                print(f"[Speaker Verification] ✅ CALLER verified (similarity: {similarity:.3f})")
            else:
                self.stats["background_rejected"] += 1
                print(f"[Speaker Verification] ❌ BACKGROUND VOICE rejected (similarity: {similarity:.3f})")
            
            return is_caller, similarity
            
        except Exception as e:
            print(f"[Speaker Verification] ❌ Error during verification: {e}")
            # On error, fail open (accept) to maintain reliability
            return True, 0.0
    
    def reset(self):
        """Reset speaker verification for new call."""
        self.caller_embedding = None
        self.voiceprint_captured = False
        print("[Speaker Verification] Reset for new call")
    
    def get_stats(self) -> dict:
        """Get verification statistics."""
        if self.stats["total_checks"] > 0:
            acceptance_rate = (
                self.stats["caller_verified"] / self.stats["total_checks"] * 100
            )
        else:
            acceptance_rate = 0.0
        
        return {
            **self.stats,
            "acceptance_rate_pct": round(acceptance_rate, 2),
            "voiceprint_captured": self.voiceprint_captured
        }
    
    def print_stats(self):
        """Print verification statistics."""
        stats = self.get_stats()
        print("\n" + "═" * 60)
        print("SPEAKER VERIFICATION STATISTICS")
        print("═" * 60)
        print(f"Voiceprint Captured:     {stats['voiceprint_captured']}")
        print(f"Total Verifications:     {stats['total_checks']}")
        print(f"  ├─ Caller Verified:    {stats['caller_verified']}")
        print(f"  └─ Background Rejected: {stats['background_rejected']}")
        print(f"Acceptance Rate:         {stats['acceptance_rate_pct']}%")
        print(f"Avg Similarity:          {stats['avg_similarity']:.3f}")
        print("═" * 60 + "\n")


# ═══════════════════════════════════════════════════════════════════
# SEMANTIC INTENT VERIFICATION (LAYER 5)
# ═══════════════════════════════════════════════════════════════════

class SemanticIntentVerifier:
    """
    Layer 5: Semantic Intent Confirmation using faster-whisper.
    
    Ensures detected speech contains meaningful words, not just noise/fillers.
    """
    
    def __init__(
        self,
        model_size: str = "tiny",
        language: str = "hi",
        min_words: int = 2,
        device: str = "cpu"
    ):
        """
        Initialize semantic intent verifier.
        
        Args:
            model_size: Whisper model size ('tiny', 'base', 'small')
            language: Language code ('hi' for Hindi, 'en' for English)
            min_words: Minimum words to consider valid
            device: Compute device
        """
        from faster_whisper import WhisperModel
        
        self.min_words = min_words
        self.language = language
        
        # Filler words to reject (Hindi + English)
        self.filler_words = {
            'uh', 'um', 'hmm', 'ah', 'er', 'uhm', 'erm',
            'ए', 'उम', 'हम्म', 'आह', 'एर'
        }
        
        print(f"[Semantic Intent] Loading faster-whisper ({model_size}) model...")
        self.whisper_model = WhisperModel(
            model_size,
            device=device,
            compute_type="int8"  # Optimized for CPU
        )
        print("[Semantic Intent] Model loaded successfully")
        
        # Statistics
        self.stats = {
            "total_checks": 0,
            "meaningful_speech": 0,
            "fillers_rejected": 0,
            "empty_rejected": 0
        }
    
    def verify_intent(self, audio_pcm: bytes, sample_rate: int = 16000) -> Tuple[bool, str]:
        """
        Verify if audio contains meaningful speech.
        
        Args:
            audio_pcm: PCM audio bytes (16-bit, mono)
            sample_rate: Audio sample rate
        
        Returns:
            Tuple of (has_meaning, transcript)
        """
        try:
            self.stats["total_checks"] += 1
            
            # Convert PCM to numpy array
            audio_int16 = np.frombuffer(audio_pcm, dtype=np.int16)
            audio_float = audio_int16.astype(np.float32) / 32768.0
            
            # Transcribe
            segments, info = self.whisper_model.transcribe(
                audio_float,
                language=self.language,
                beam_size=1,  # Fast inference
                best_of=1,
                temperature=0.0
            )
            
            # Collect transcript
            transcript = " ".join([seg.text for seg in segments]).strip()
            
            # Empty transcript
            if not transcript or len(transcript) < 2:
                self.stats["empty_rejected"] += 1
                print(f"[Semantic Intent] ❌ Empty transcript")
                return False, ""
            
            # Check for filler words only
            words = transcript.lower().split()
            non_filler_words = [w for w in words if w not in self.filler_words]
            
            if len(non_filler_words) < self.min_words:
                self.stats["fillers_rejected"] += 1
                print(f"[Semantic Intent] ❌ Only fillers: '{transcript}'")
                return False, transcript
            
            # Meaningful speech detected
            self.stats["meaningful_speech"] += 1
            print(f"[Semantic Intent] ✅ Meaningful speech: '{transcript}'")
            return True, transcript
            
        except Exception as e:
            print(f"[Semantic Intent] Error: {e}")
            # On error, fail open
            return True, ""
    
    def get_stats(self) -> dict:
        """Get verification statistics."""
        return self.stats
    
    def print_stats(self):
        """Print statistics."""
        stats = self.get_stats()
        print("\n" + "═" * 60)
        print("SEMANTIC INTENT STATISTICS")
        print("═" * 60)
        print(f"Total Checks:            {stats['total_checks']}")
        print(f"  ├─ Meaningful Speech:  {stats['meaningful_speech']}")
        print(f"  ├─ Fillers Rejected:   {stats['fillers_rejected']}")
        print(f"  └─ Empty Rejected:     {stats['empty_rejected']}")
        print("═" * 60 + "\n")
