"""
Silero VAD Filter - Production-Level Voice Activity Detection
==============================================================

Uses pre-trained Silero VAD model for accurate speech detection.
- 99%+ accuracy across diverse noise conditions
- Trained on 6000+ languages
- Lightweight: ~2MB model, <1ms inference per chunk
- Adaptive noise profiling

Reference: https://github.com/snakers4/silero-vad
"""

import torch
import numpy as np
from typing import Tuple, Optional
import warnings

# Suppress torch warnings for cleaner logs
warnings.filterwarnings('ignore', category=UserWarning, module='torch')


_GLOBAL_SILERO_MODEL = None
_GLOBAL_SILERO_UTILS = None

class SileroVADFilter:
    """
    Production-grade Voice Activity Detection using Silero VAD.
    
    Key features:
    - ML-based speech detection (not just energy thresholds)
    - Adaptive noise floor learning
    - Confidence scoring (0.0 = noise, 1.0 = speech)
    - Hysteresis to prevent rapid on/off switching
    """
    
    def __init__(self, sample_rate: int = 16000, threshold: float = 0.5, device: str = "cpu"):
        """
        Initialize Silero VAD filter.
        
        Args:
            sample_rate: Audio sample rate (must be 8000 or 16000)
            threshold: Speech probability threshold (0.0-1.0)
            device: 'cpu' or 'cuda'
        """
        global _GLOBAL_SILERO_MODEL, _GLOBAL_SILERO_UTILS
        self.sample_rate = sample_rate
        self.threshold = threshold
        self.device = torch.device(device)
        
        # Load Silero VAD model globally so we don't redownload/reload it on each call
        if _GLOBAL_SILERO_MODEL is None:
            print("[Silero VAD] Loading model to global cache...")
            try:
                model, utils = torch.hub.load(
                    repo_or_dir='snakers4/silero-vad',
                    model='silero_vad',
                    force_reload=False,
                    onnx=False,  # Use PyTorch version for better performance
                    trust_repo=True
                )
                model.to(self.device)
                model.eval()
                _GLOBAL_SILERO_MODEL = model
                _GLOBAL_SILERO_UTILS = utils
                print(f"[Silero VAD] ✅ Model cached on {device}")
            except Exception as e:
                print(f"[Silero VAD] ❌ Failed to load model: {e}")
                raise
                
        self.model = _GLOBAL_SILERO_MODEL
        utils = _GLOBAL_SILERO_UTILS
        
        # Extract utility functions
        (self.get_speech_timestamps,
         self.save_audio,
         self.read_audio,
         self.VADIterator,
         self.collect_chunks) = utils
        
        # Internal state
        self.noise_floor_samples = []
        self.noise_floor_learned = False
        self.noise_floor_avg = 0.0
        self.hysteresis_state = False  # Prevents rapid switching
        self.hysteresis_counter = 0
        
        # Configuration
        self.noise_learning_frames = 10  # Learn from first 10 frames (~0.5s)
        self.hysteresis_on_threshold = 3   # Need 3 consecutive "speech" frames
        self.hysteresis_off_threshold = 5  # Need 5 consecutive "noise" frames
        
    def reset_noise_profile(self):
        """Reset noise profiling (call at start of each call)"""
        self.noise_floor_samples = []
        self.noise_floor_learned = False
        self.noise_floor_avg = 0.0
        self.hysteresis_state = False
        self.hysteresis_counter = 0
        print("[Silero VAD] Noise profile reset")
    
    def finalize_noise_profile(self):
        """Finalize noise learning (call after opener completes)"""
        if len(self.noise_floor_samples) > 0:
            self.noise_floor_avg = np.mean(self.noise_floor_samples)
            self.noise_floor_learned = True
            print(f"[Silero VAD] Noise floor learned: {self.noise_floor_avg:.3f}")
        else:
            print("[Silero VAD] Warning: No noise samples collected")
    
    def is_speech(self, audio_chunk: np.ndarray) -> Tuple[bool, float]:
        """
        Detect if audio chunk contains speech.
        
        Args:
            audio_chunk: Audio data as float32 numpy array (-1.0 to 1.0)
            
        Returns:
            (is_speech, confidence)
            - is_speech: True if speech detected
            - confidence: Speech probability (0.0-1.0)
        """
        try:
            # Ensure correct format
            if audio_chunk.dtype != np.float32:
                audio_chunk = audio_chunk.astype(np.float32)
            
            # Normalize if needed
            max_val = np.max(np.abs(audio_chunk))
            if max_val > 1.0:
                audio_chunk = audio_chunk / max_val
            
            # CRITICAL: Silero VAD requires exactly 512 samples for 16kHz (or 256 for 8kHz)
            required_samples = 512 if self.sample_rate == 16000 else 256
            
            # If chunk is too small, pad with zeros
            if len(audio_chunk) < required_samples:
                padding = np.zeros(required_samples - len(audio_chunk), dtype=np.float32)
                audio_chunk = np.concatenate([audio_chunk, padding])
            
            # If chunk is too large, process in windows and average
            if len(audio_chunk) > required_samples:
                # Process in non-overlapping windows
                num_windows = len(audio_chunk) // required_samples
                probabilities = []
                
                for i in range(num_windows):
                    start = i * required_samples
                    end = start + required_samples
                    window = audio_chunk[start:end]
                    
                    # Convert to torch tensor
                    audio_tensor = torch.from_numpy(window).to(self.device)
                    
                    # Run inference
                    with torch.no_grad():
                        speech_prob = self.model(audio_tensor, self.sample_rate).item()
                    
                    probabilities.append(speech_prob)
                
                # Average the probabilities
                speech_prob = np.mean(probabilities)
            else:
                # Exactly the right size
                audio_tensor = torch.from_numpy(audio_chunk).to(self.device)
                
                # Run inference
                with torch.no_grad():
                    speech_prob = self.model(audio_tensor, self.sample_rate).item()
            
            # Learn noise floor during initial frames
            if not self.noise_floor_learned and len(self.noise_floor_samples) < self.noise_learning_frames:
                if speech_prob < 0.3:  # Only learn from low-confidence frames
                    self.noise_floor_samples.append(speech_prob)
            
            # Adjust confidence based on learned noise floor
            adjusted_confidence = speech_prob
            if self.noise_floor_learned:
                # Subtract noise floor and normalize
                adjusted_confidence = max(0.0, speech_prob - self.noise_floor_avg)
                adjusted_confidence = min(1.0, adjusted_confidence / (1.0 - self.noise_floor_avg + 1e-6))
            
            # Apply hysteresis to prevent rapid switching
            if adjusted_confidence >= self.threshold:
                if not self.hysteresis_state:
                    # Currently OFF, need consecutive frames to turn ON
                    self.hysteresis_counter += 1
                    if self.hysteresis_counter >= self.hysteresis_on_threshold:
                        self.hysteresis_state = True
                        self.hysteresis_counter = 0
                        is_speech = True
                    else:
                        is_speech = False
                else:
                    # Already ON, stay ON
                    self.hysteresis_counter = 0
                    is_speech = True
            else:
                if self.hysteresis_state:
                    # Currently ON, need consecutive frames to turn OFF
                    self.hysteresis_counter += 1
                    if self.hysteresis_counter >= self.hysteresis_off_threshold:
                        self.hysteresis_state = False
                        self.hysteresis_counter = 0
                        is_speech = False
                    else:
                        is_speech = True  # Stay ON during hysteresis
                else:
                    # Already OFF, stay OFF
                    self.hysteresis_counter = 0
                    is_speech = False
            
            return is_speech, adjusted_confidence
            
        except Exception as e:
            print(f"[Silero VAD] Error during inference: {e}")
            # Fail-safe: assume speech if error
            return True, 0.5
    
    def get_speech_segments(self, audio: np.ndarray, return_seconds: bool = True):
        """
        Get all speech segments from longer audio.
        
        Args:
            audio: Full audio array
            return_seconds: Return timestamps in seconds (vs samples)
            
        Returns:
            List of (start, end) tuples
        """
        try:
            audio_tensor = torch.from_numpy(audio.astype(np.float32)).to(self.device)
            
            speech_timestamps = self.get_speech_timestamps(
                audio_tensor,
                self.model,
                sampling_rate=self.sample_rate,
                threshold=self.threshold,
                return_seconds=return_seconds
            )
            
            return [(ts['start'], ts['end']) for ts in speech_timestamps]
            
        except Exception as e:
            print(f"[Silero VAD] Error in get_speech_segments: {e}")
            return []


# Test function
if __name__ == "__main__":
    print("Testing Silero VAD Filter...")
    
    vad = SileroVADFilter(sample_rate=16000, threshold=0.5)
    
    # Test 1: Silence (zeros)
    silence = np.zeros(16000, dtype=np.float32)
    is_speech, conf = vad.is_speech(silence)
    print(f"Silence test: is_speech={is_speech}, confidence={conf:.3f}")
    assert not is_speech, "Silence should not be detected as speech"
    
    # Test 2: White noise
    noise = np.random.randn(16000).astype(np.float32) * 0.1
    is_speech, conf = vad.is_speech(noise)
    print(f"Noise test: is_speech={is_speech}, confidence={conf:.3f}")
    
    # Test 3: Simulated speech (sine wave with modulation)
    t = np.linspace(0, 1, 16000)
    speech = (np.sin(2 * np.pi * 300 * t) * np.sin(2 * np.pi * 5 * t)).astype(np.float32)
    is_speech, conf = vad.is_speech(speech)
    print(f"Speech test: is_speech={is_speech}, confidence={conf:.3f}")
    
    print("\n✅ Silero VAD Filter tests complete!")
