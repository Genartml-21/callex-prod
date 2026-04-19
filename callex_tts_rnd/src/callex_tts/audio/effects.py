"""
╔══════════════════════════════════════════════════════════════════════╗
║  CALLEX TTS — Professional Audio Effects Chain                      ║
║                                                                      ║
║  Production-grade audio post-processing effects for telephony TTS:   ║
║    • Dynamic Range Compressor (prevents clipping, evens volume)      ║
║    • Peak Limiter (hard ceiling at -1 dBFS for broadcast safety)     ║
║    • Noise Gate (cuts background hiss below threshold)               ║
║    • De-Esser (reduces sibilance on /s/ and /ʃ/ sounds)             ║
║    • Warmth Filter (low-shelf boost for natural fullness)            ║
║    • EBU R128 Loudness Normalization                                 ║
║    • High-Pass Filter (removes sub-bass rumble)                      ║
║                                                                      ║
║  All effects operate on raw PCM waveforms and are individually       ║
║  configurable via YAML.                                              ║
╚══════════════════════════════════════════════════════════════════════╝
"""

from __future__ import annotations

import math
import logging
from dataclasses import dataclass
from typing import Optional

import numpy as np

logger = logging.getLogger("callex.tts.audio.effects")


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  Individual Effect Configurations
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

@dataclass
class CompressorConfig:
    enabled: bool = True
    threshold_db: float = -20.0
    ratio: float = 4.0
    attack_ms: float = 5.0
    release_ms: float = 50.0
    makeup_gain_db: float = 4.0
    knee_db: float = 6.0          # Soft knee width


@dataclass
class LimiterConfig:
    enabled: bool = True
    ceiling_db: float = -1.0
    release_ms: float = 10.0
    lookahead_ms: float = 5.0


@dataclass
class NoiseGateConfig:
    enabled: bool = True
    threshold_db: float = -45.0
    attack_ms: float = 1.0
    release_ms: float = 20.0
    hold_ms: float = 50.0        # Hold gate open after signal drops below threshold


@dataclass
class DeEsserConfig:
    enabled: bool = True
    frequency_hz: float = 6000.0
    threshold_db: float = -15.0
    ratio: float = 3.0
    bandwidth_hz: float = 4000.0  # Width of sibilant band


@dataclass
class WarmthConfig:
    enabled: bool = True
    frequency_hz: float = 200.0
    gain_db: float = 2.0
    q: float = 0.7               # Filter Q factor


@dataclass
class LoudnessConfig:
    enabled: bool = True
    target_lufs: float = -16.0


@dataclass
class HighPassConfig:
    enabled: bool = True
    cutoff_hz: float = 80.0
    order: int = 2


@dataclass
class EffectsChainConfig:
    """Master configuration for the entire effects chain."""
    sample_rate: int = 24000
    high_pass: HighPassConfig = None
    compressor: CompressorConfig = None
    deesser: DeEsserConfig = None
    warmth: WarmthConfig = None
    limiter: LimiterConfig = None
    noise_gate: NoiseGateConfig = None
    loudness: LoudnessConfig = None

    def __post_init__(self):
        self.high_pass = self.high_pass or HighPassConfig()
        self.compressor = self.compressor or CompressorConfig()
        self.deesser = self.deesser or DeEsserConfig()
        self.warmth = self.warmth or WarmthConfig()
        self.limiter = self.limiter or LimiterConfig()
        self.noise_gate = self.noise_gate or NoiseGateConfig()
        self.loudness = self.loudness or LoudnessConfig()


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  DSP Utility Functions
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def db_to_linear(db: float) -> float:
    """Convert decibels to linear amplitude."""
    return 10.0 ** (db / 20.0)


def linear_to_db(linear: float) -> float:
    """Convert linear amplitude to decibels."""
    return 20.0 * math.log10(max(linear, 1e-10))


def rms_db(audio: np.ndarray) -> float:
    """Compute RMS level in dB."""
    rms = np.sqrt(np.mean(audio ** 2) + 1e-10)
    return linear_to_db(rms)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  Individual Effects
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class DynamicRangeCompressor:
    """
    Feed-forward dynamic range compressor with soft knee.
    
    Reduces the dynamic range of audio by attenuating signals
    above the threshold. Uses envelope following for smooth
    gain reduction.
    """

    def __init__(self, config: CompressorConfig, sample_rate: int = 24000):
        self.config = config
        self.sr = sample_rate
        self.threshold = db_to_linear(config.threshold_db)
        self.makeup_gain = db_to_linear(config.makeup_gain_db)
        
        # Time constants → coefficient conversion
        self.attack_coeff = 1.0 - math.exp(-1.0 / (config.attack_ms * sample_rate / 1000.0))
        self.release_coeff = 1.0 - math.exp(-1.0 / (config.release_ms * sample_rate / 1000.0))

    def process(self, audio: np.ndarray) -> np.ndarray:
        """Apply compression to audio signal."""
        if not self.config.enabled:
            return audio

        output = np.copy(audio)
        envelope = 0.0
        knee_half = self.config.knee_db / 2.0

        for i in range(len(output)):
            level = abs(output[i])
            level_db = linear_to_db(level) if level > 1e-10 else -96.0

            # Compute gain reduction with soft knee
            if level_db < (self.config.threshold_db - knee_half):
                # Below knee — no compression
                gain_db = 0.0
            elif level_db > (self.config.threshold_db + knee_half):
                # Above knee — full compression
                gain_db = (self.config.threshold_db + (level_db - self.config.threshold_db) / self.config.ratio) - level_db
            else:
                # In knee region — smooth transition
                x = level_db - self.config.threshold_db + knee_half
                gain_db = ((1.0 / self.config.ratio - 1.0) * x * x) / (2.0 * self.config.knee_db)

            gain = db_to_linear(gain_db)

            # Envelope follower (attack/release smoothing)
            if gain < envelope:
                envelope = self.attack_coeff * gain + (1.0 - self.attack_coeff) * envelope
            else:
                envelope = self.release_coeff * gain + (1.0 - self.release_coeff) * envelope

            output[i] *= envelope

        # Apply makeup gain
        output *= self.makeup_gain

        return output


class PeakLimiter:
    """
    True-peak limiter with lookahead.
    
    Hard-limits peaks to prevent clipping. Essential for broadcast
    and telephony compliance.
    """

    def __init__(self, config: LimiterConfig, sample_rate: int = 24000):
        self.config = config
        self.ceiling = db_to_linear(config.ceiling_db)
        self.release_coeff = 1.0 - math.exp(-1.0 / (config.release_ms * sample_rate / 1000.0))
        self.lookahead = int(config.lookahead_ms * sample_rate / 1000.0)

    def process(self, audio: np.ndarray) -> np.ndarray:
        """Apply peak limiting."""
        if not self.config.enabled:
            return audio

        output = np.copy(audio)

        # Find peaks that exceed ceiling
        peak_mask = np.abs(output) > self.ceiling
        
        if not np.any(peak_mask):
            return output

        # Simple lookahead limiter
        gain = np.ones(len(output))
        envelope = 1.0

        for i in range(len(output)):
            # Lookahead: check upcoming samples
            lookahead_end = min(i + self.lookahead, len(output))
            peak = np.max(np.abs(output[i:lookahead_end])) if i < len(output) else abs(output[i])

            if peak > self.ceiling:
                target_gain = self.ceiling / peak
            else:
                target_gain = 1.0

            # Smooth gain changes
            if target_gain < envelope:
                envelope = target_gain  # Instant attack
            else:
                envelope = self.release_coeff * target_gain + (1.0 - self.release_coeff) * envelope

            gain[i] = envelope

        output *= gain
        return output


class NoiseGate:
    """
    Noise gate with hold time.
    
    Mutes audio below the threshold to remove background noise
    in pauses. Hold time prevents chattering on transients.
    """

    def __init__(self, config: NoiseGateConfig, sample_rate: int = 24000):
        self.config = config
        self.threshold = db_to_linear(config.threshold_db)
        self.attack_coeff = 1.0 - math.exp(-1.0 / (config.attack_ms * sample_rate / 1000.0))
        self.release_coeff = 1.0 - math.exp(-1.0 / (config.release_ms * sample_rate / 1000.0))
        self.hold_samples = int(config.hold_ms * sample_rate / 1000.0)

    def process(self, audio: np.ndarray) -> np.ndarray:
        """Apply noise gating."""
        if not self.config.enabled:
            return audio

        output = np.copy(audio)
        gate_open = False
        hold_counter = 0
        envelope = 0.0

        # RMS-based envelope with block processing
        block_size = 256
        for i in range(0, len(output), block_size):
            block = output[i:i + block_size]
            block_rms = np.sqrt(np.mean(block ** 2) + 1e-10)

            if block_rms > self.threshold:
                gate_open = True
                hold_counter = self.hold_samples
            else:
                if hold_counter > 0:
                    hold_counter -= block_size
                else:
                    gate_open = False

            # Smooth gain transition
            target = 1.0 if gate_open else 0.0
            if target > envelope:
                coeff = self.attack_coeff
            else:
                coeff = self.release_coeff

            for j in range(len(block)):
                envelope = coeff * target + (1.0 - coeff) * envelope
                output[i + j] *= envelope

        return output


class DeEsser:
    """
    Frequency-selective de-esser.
    
    Detects sibilant energy in the 4-8 kHz range and applies
    targeted gain reduction to reduce harshness.
    """

    def __init__(self, config: DeEsserConfig, sample_rate: int = 24000):
        self.config = config
        self.sr = sample_rate
        self.threshold = db_to_linear(config.threshold_db)

    def process(self, audio: np.ndarray) -> np.ndarray:
        """Apply de-essing."""
        if not self.config.enabled:
            return audio

        # Work in frequency domain with overlapping blocks
        block_size = 2048
        hop = block_size // 4
        output = np.copy(audio)
        window = np.hanning(block_size)

        # Compute frequency bin range for sibilant band
        freq_resolution = self.sr / block_size
        low_bin = int(self.config.frequency_hz / freq_resolution)
        high_bin = int((self.config.frequency_hz + self.config.bandwidth_hz) / freq_resolution)
        high_bin = min(high_bin, block_size // 2)

        for i in range(0, len(audio) - block_size, hop):
            block = audio[i:i + block_size] * window
            spectrum = np.fft.rfft(block)
            magnitudes = np.abs(spectrum)

            # Measure sibilant energy
            sibilant_energy = np.mean(magnitudes[low_bin:high_bin])

            if sibilant_energy > self.threshold:
                # Apply gain reduction to sibilant band
                reduction = (self.threshold / sibilant_energy) ** (1.0 - 1.0 / self.config.ratio)
                spectrum[low_bin:high_bin] *= reduction

                # Reconstruct
                processed = np.fft.irfft(spectrum) * window
                output[i:i + block_size] += (processed - block) * 0.5

        return output


class WarmthFilter:
    """
    Low-shelf boost filter for adding warmth to voice audio.
    
    Implements a second-order IIR low-shelf filter using the
    bilinear transform.
    """

    def __init__(self, config: WarmthConfig, sample_rate: int = 24000):
        self.config = config
        self.sr = sample_rate
        self._compute_coefficients()

    def _compute_coefficients(self):
        """Compute biquad filter coefficients for low-shelf."""
        A = db_to_linear(self.config.gain_db / 2.0)  # sqrt of gain
        w0 = 2.0 * math.pi * self.config.frequency_hz / self.sr
        alpha = math.sin(w0) / (2.0 * self.config.q)

        cos_w0 = math.cos(w0)
        sqrt_A = math.sqrt(A)
        two_sqrt_A_alpha = 2.0 * sqrt_A * alpha

        # Low-shelf coefficients (Audio EQ Cookbook)
        self.b0 = A * ((A + 1) - (A - 1) * cos_w0 + two_sqrt_A_alpha)
        self.b1 = 2 * A * ((A - 1) - (A + 1) * cos_w0)
        self.b2 = A * ((A + 1) - (A - 1) * cos_w0 - two_sqrt_A_alpha)
        self.a0 = (A + 1) + (A - 1) * cos_w0 + two_sqrt_A_alpha
        self.a1 = -2 * ((A - 1) + (A + 1) * cos_w0)
        self.a2 = (A + 1) + (A - 1) * cos_w0 - two_sqrt_A_alpha

        # Normalize
        self.b0 /= self.a0
        self.b1 /= self.a0
        self.b2 /= self.a0
        self.a1 /= self.a0
        self.a2 /= self.a0

    def process(self, audio: np.ndarray) -> np.ndarray:
        """Apply low-shelf warmth filter using Direct Form II."""
        if not self.config.enabled:
            return audio

        output = np.copy(audio)
        # Direct Form II Transposed
        z1 = 0.0
        z2 = 0.0

        for i in range(len(output)):
            x = output[i]
            y = self.b0 * x + z1
            z1 = self.b1 * x - self.a1 * y + z2
            z2 = self.b2 * x - self.a2 * y
            output[i] = y

        return output


class HighPassFilter:
    """
    Butterworth high-pass filter to remove sub-bass rumble.
    """

    def __init__(self, config: HighPassConfig, sample_rate: int = 24000):
        self.config = config
        self.sr = sample_rate
        self._compute_coefficients()

    def _compute_coefficients(self):
        """Compute 2nd-order Butterworth HPF coefficients."""
        w0 = 2.0 * math.pi * self.config.cutoff_hz / self.sr
        cos_w0 = math.cos(w0)
        alpha = math.sin(w0) / (2.0 * (1.0 / math.sqrt(2.0)))  # Q = 1/sqrt(2) for Butterworth

        self.b0 = (1 + cos_w0) / 2
        self.b1 = -(1 + cos_w0)
        self.b2 = (1 + cos_w0) / 2
        a0 = 1 + alpha
        self.a1 = -2 * cos_w0 / a0
        self.a2 = (1 - alpha) / a0
        self.b0 /= a0
        self.b1 /= a0
        self.b2 /= a0

    def process(self, audio: np.ndarray) -> np.ndarray:
        """Apply high-pass filter."""
        if not self.config.enabled:
            return audio

        output = np.copy(audio)
        z1 = 0.0
        z2 = 0.0

        for i in range(len(output)):
            x = output[i]
            y = self.b0 * x + z1
            z1 = self.b1 * x - self.a1 * y + z2
            z2 = self.b2 * x - self.a2 * y
            output[i] = y

        return output


class LoudnessNormalizer:
    """
    EBU R128 loudness normalization.
    
    Normalizes integrated loudness to a target LUFS value.
    Uses a simplified LUFS measurement (ITU-R BS.1770-4).
    """

    def __init__(self, config: LoudnessConfig, sample_rate: int = 24000):
        self.config = config
        self.sr = sample_rate

    def process(self, audio: np.ndarray) -> np.ndarray:
        """Normalize loudness to target LUFS."""
        if not self.config.enabled:
            return audio

        current_lufs = self._measure_lufs(audio)
        
        if current_lufs < -70.0:
            # Signal is too quiet / silence — skip
            return audio

        gain_db = self.config.target_lufs - current_lufs
        gain = db_to_linear(gain_db)

        return audio * gain

    def _measure_lufs(self, audio: np.ndarray) -> float:
        """
        Simplified LUFS measurement.
        
        Full ITU-R BS.1770-4 includes K-weighting and gated measurement.
        This simplified version uses K-weighted RMS as an approximation
        suitable for single-channel speech.
        """
        # K-weighting stage 1: high-shelf boost at ~1500 Hz
        # (Simplified — a full implementation would use the exact filter)
        # For speech, RMS-based measurement is a reasonable approximation
        
        rms = np.sqrt(np.mean(audio ** 2) + 1e-10)
        lufs = 20.0 * math.log10(rms) - 0.691  # Approximate LUFS offset

        return lufs


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  Master Effects Chain
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class AudioEffectsChain:
    """
    Production audio effects chain.
    
    Processes audio through a configurable chain of effects in the
    correct signal-flow order:
    
        Input → HPF → Noise Gate → Compressor → De-Esser → 
        Warmth → Loudness Normalization → Limiter → Output
    
    Usage:
        config = EffectsChainConfig(sample_rate=24000)
        chain = AudioEffectsChain(config)
        processed = chain.process(raw_audio_numpy)
    """

    def __init__(self, config: Optional[EffectsChainConfig] = None):
        self.config = config or EffectsChainConfig()
        sr = self.config.sample_rate

        # Initialize effects in signal-flow order
        self.high_pass = HighPassFilter(self.config.high_pass, sr)
        self.noise_gate = NoiseGate(self.config.noise_gate, sr)
        self.compressor = DynamicRangeCompressor(self.config.compressor, sr)
        self.deesser = DeEsser(self.config.deesser, sr)
        self.warmth = WarmthFilter(self.config.warmth, sr)
        self.loudness = LoudnessNormalizer(self.config.loudness, sr)
        self.limiter = PeakLimiter(self.config.limiter, sr)

        logger.info(
            "AudioEffectsChain initialized: HPF=%s, Gate=%s, Comp=%s, DeEss=%s, "
            "Warmth=%s, Loudness=%s, Limiter=%s",
            config.high_pass.enabled, config.noise_gate.enabled,
            config.compressor.enabled, config.deesser.enabled,
            config.warmth.enabled, config.loudness.enabled,
            config.limiter.enabled,
        )

    def process(self, audio: np.ndarray) -> np.ndarray:
        """
        Process audio through the entire effects chain.
        
        Args:
            audio: Raw float32 audio samples, range [-1, 1]
            
        Returns:
            Processed audio (same shape)
        """
        if len(audio) == 0:
            return audio

        # Ensure float64 for processing precision
        audio = audio.astype(np.float64)

        # Signal chain (order matters!)
        audio = self.high_pass.process(audio)     # Remove sub-bass rumble
        audio = self.noise_gate.process(audio)    # Gate out background noise
        audio = self.compressor.process(audio)    # Even out dynamics
        audio = self.deesser.process(audio)       # Tame sibilance
        audio = self.warmth.process(audio)        # Add low-end body
        audio = self.loudness.process(audio)      # Normalize loudness
        audio = self.limiter.process(audio)       # Prevent clipping

        return audio.astype(np.float32)

    def process_torch(self, audio_tensor) -> "torch.Tensor":
        """
        Process a PyTorch tensor through the effects chain.
        Converts to numpy, processes, and converts back.
        """
        import torch
        device = audio_tensor.device
        dtype = audio_tensor.dtype
        
        audio_np = audio_tensor.cpu().numpy().astype(np.float32)
        
        # Process each batch element
        if audio_np.ndim == 1:
            processed = self.process(audio_np)
        else:
            processed = np.stack([self.process(a) for a in audio_np])
        
        return torch.from_numpy(processed).to(device=device, dtype=dtype)
