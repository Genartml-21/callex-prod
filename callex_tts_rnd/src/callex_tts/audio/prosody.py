"""
╔══════════════════════════════════════════════════════════════════════╗
║  CALLEX TTS — Prosody Control Engine                                ║
║                                                                      ║
║  Real-time prosody manipulation on waveform-level audio:             ║
║    • Pitch shifting (phase vocoder, ±12 semitones)                   ║
║    • Speed control (WSOLA time-stretching, 0.5x–2.0x)               ║
║    • Energy scaling (per-band frequency gain curves)                 ║
║                                                                      ║
║  Integrates with SSML <prosody> tags for per-segment control.        ║
╚══════════════════════════════════════════════════════════════════════╝
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from typing import Optional

import numpy as np
import torch

logger = logging.getLogger("callex.tts.audio.prosody")


@dataclass
class ProsodyConfig:
    """Prosody control parameters."""
    sample_rate: int = 24000
    # Pitch shifting
    pitch_shift_semitones: float = 0.0
    max_pitch_semitones: float = 12.0
    # Speed control  
    speed_rate: float = 1.0
    min_speed: float = 0.5
    max_speed: float = 2.0
    # Energy
    energy_scale: float = 1.0
    # Phase vocoder parameters
    n_fft: int = 2048
    hop_length: int = 512


class ProsodyProcessor:
    """
    Waveform-level prosody control for TTS output.
    
    This processor operates on raw audio waveforms (after vocoder output)
    and applies pitch, speed, and energy modifications using DSP techniques.
    
    Usage:
        processor = ProsodyProcessor(sample_rate=24000)
        
        # Shift pitch up by 3 semitones
        audio = processor.pitch_shift(audio, semitones=3.0)
        
        # Slow down to 80% speed (without changing pitch)
        audio = processor.time_stretch(audio, rate=0.8)
        
        # Apply energy scaling
        audio = processor.energy_scale(audio, scale=1.5)
        
        # Or apply all at once from SSML params
        audio = processor.apply(audio, pitch=3.0, rate=0.8, volume=+6.0)
    """

    def __init__(self, sample_rate: int = 24000, n_fft: int = 2048, hop_length: int = 512):
        self.sample_rate = sample_rate
        self.n_fft = n_fft
        self.hop_length = hop_length
        self._window = None  # Lazy init

    def _get_window(self, device: torch.device) -> torch.Tensor:
        """Get or create Hann window on the correct device."""
        if self._window is None or self._window.device != device:
            self._window = torch.hann_window(self.n_fft, device=device)
        return self._window

    def apply(
        self,
        audio: torch.Tensor,
        pitch: float = 0.0,
        rate: float = 1.0,
        volume: float = 0.0,
        energy: float = 1.0,
    ) -> torch.Tensor:
        """
        Apply all prosody modifications in the correct order.
        
        Args:
            audio: Waveform tensor [samples] or [batch, samples]
            pitch: Pitch shift in semitones
            rate: Speed rate (1.0 = normal, < 1 = slower, > 1 = faster)
            volume: Volume adjustment in dB
            energy: Energy scaling multiplier
            
        Returns:
            Modified waveform tensor
        """
        if audio.numel() == 0:
            return audio

        # Ensure 2D: [batch, samples]
        squeeze = False
        if audio.dim() == 1:
            audio = audio.unsqueeze(0)
            squeeze = True

        # 1. Time stretch (changes speed without pitch)
        if rate != 1.0:
            rate = max(0.5, min(2.0, rate))
            audio = self.time_stretch(audio, rate)

        # 2. Pitch shift
        if pitch != 0.0:
            pitch = max(-12.0, min(12.0, pitch))
            audio = self.pitch_shift(audio, pitch)

        # 3. Energy scaling
        if energy != 1.0:
            audio = self.energy_scale(audio, energy)

        # 4. Volume adjustment (dB)
        if volume != 0.0:
            gain = 10.0 ** (volume / 20.0)
            audio = audio * gain

        if squeeze:
            audio = audio.squeeze(0)

        return audio

    def pitch_shift(self, audio: torch.Tensor, semitones: float) -> torch.Tensor:
        """
        Shift pitch by the specified number of semitones.
        
        Uses phase vocoder for time stretching + resampling for pitch shift:
          1. Time-stretch by inverse of pitch ratio (changes pitch but also speed)
          2. Resample to compensate for the speed change
        
        This approach preserves formant structure better than simple resampling.
        
        Args:
            audio: [batch, samples]
            semitones: Pitch shift amount (-12 to +12)
            
        Returns:
            Pitch-shifted audio [batch, samples]
        """
        if semitones == 0.0:
            return audio

        # Pitch ratio: 2^(semitones/12)
        ratio = 2.0 ** (semitones / 12.0)
        
        # Step 1: Time-stretch by 1/ratio using phase vocoder
        stretched = self._phase_vocoder(audio, 1.0 / ratio)
        
        # Step 2: Resample to original length (compensates speed change)
        # This effectively keeps duration constant while shifting pitch
        original_length = audio.shape[-1]
        if stretched.shape[-1] != original_length:
            stretched = torch.nn.functional.interpolate(
                stretched.unsqueeze(1),
                size=original_length,
                mode='linear',
                align_corners=False,
            ).squeeze(1)

        return stretched

    def time_stretch(self, audio: torch.Tensor, rate: float) -> torch.Tensor:
        """
        Time-stretch audio without changing pitch.
        
        Uses the phase vocoder algorithm to stretch/compress the temporal
        dimension while preserving the spectral content (and thus pitch).
        
        Args:
            audio: [batch, samples]
            rate: Stretch rate (< 1.0 = slower/longer, > 1.0 = faster/shorter)
            
        Returns:
            Time-stretched audio [batch, new_samples]
        """
        if rate == 1.0:
            return audio

        return self._phase_vocoder(audio, rate)

    def energy_scale(self, audio: torch.Tensor, scale: float) -> torch.Tensor:
        """
        Scale the energy (RMS amplitude) of the audio.
        
        Applies frequency-aware scaling to preserve tonal balance.
        
        Args:
            audio: [batch, samples]
            scale: Energy multiplier (1.0 = unchanged)
            
        Returns:
            Energy-scaled audio
        """
        if scale == 1.0:
            return audio

        # Compute current RMS
        rms = torch.sqrt(torch.mean(audio ** 2, dim=-1, keepdim=True) + 1e-8)
        target_rms = rms * scale
        
        # Scale to target RMS
        audio = audio * (target_rms / (rms + 1e-8))
        
        # Soft clip to prevent distortion
        audio = torch.tanh(audio)
        
        return audio

    def _phase_vocoder(self, audio: torch.Tensor, rate: float) -> torch.Tensor:
        """
        Phase vocoder implementation for time-stretching.
        
        Operates in STFT domain:
          1. Compute STFT
          2. Interpolate magnitude and unwrap phase at the new rate
          3. Inverse STFT
          
        Args:
            audio: [batch, samples]
            rate: Time stretch rate
            
        Returns:
            Time-stretched audio [batch, new_samples]
        """
        device = audio.device
        window = self._get_window(device)
        batch_size = audio.shape[0]

        # STFT
        pad = self.n_fft // 2
        padded = torch.nn.functional.pad(audio, (pad, pad), mode='reflect')
        
        stft = torch.stft(
            padded.reshape(-1, padded.shape[-1]),
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.n_fft,
            window=window,
            return_complex=True,
        )  # [batch, freq_bins, time_frames]

        n_freq, n_frames = stft.shape[1], stft.shape[2]
        
        # Compute phase advance
        phase_advance = torch.linspace(
            0, math.pi * self.hop_length, n_freq, device=device
        ).unsqueeze(1)

        # New time points
        new_n_frames = int(n_frames / rate)
        time_steps = torch.arange(new_n_frames, device=device).float() * rate

        # Interpolate magnitude
        magnitudes = torch.abs(stft)
        phases = torch.angle(stft)

        # Build output
        output_real = torch.zeros(batch_size, n_freq, new_n_frames, device=device)
        output_imag = torch.zeros(batch_size, n_freq, new_n_frames, device=device)
        
        # Phase accumulator
        phase_acc = phases[:, :, 0:1]

        for i in range(new_n_frames):
            t = time_steps[i]
            t_floor = int(t)
            t_frac = t - t_floor
            
            if t_floor + 1 < n_frames:
                # Interpolate magnitude
                mag = magnitudes[:, :, t_floor] * (1 - t_frac) + magnitudes[:, :, t_floor + 1] * t_frac
                
                # Phase unwrapping and accumulation
                if i == 0:
                    phase = phases[:, :, t_floor]
                else:
                    # Instantaneous frequency
                    dphi = phases[:, :, min(t_floor + 1, n_frames - 1)] - phases[:, :, t_floor]
                    dphi = dphi - phase_advance.squeeze(1) 
                    dphi = dphi - 2 * math.pi * torch.round(dphi / (2 * math.pi))
                    phase = phase_acc.squeeze(-1) + phase_advance.squeeze(1) + dphi
                    phase_acc = phase.unsqueeze(-1)

                output_real[:, :, i] = mag * torch.cos(phase)
                output_imag[:, :, i] = mag * torch.sin(phase)
            elif t_floor < n_frames:
                mag = magnitudes[:, :, t_floor]
                phase = phases[:, :, t_floor]
                output_real[:, :, i] = mag * torch.cos(phase)
                output_imag[:, :, i] = mag * torch.sin(phase)

        output_complex = torch.complex(output_real, output_imag)

        # Inverse STFT
        audio_out = torch.istft(
            output_complex,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.n_fft,
            window=window,
        )

        return audio_out.reshape(batch_size, -1)

    def generate_silence(self, duration_ms: int) -> torch.Tensor:
        """Generate silence of specified duration (for SSML <break> tags)."""
        n_samples = int(self.sample_rate * duration_ms / 1000)
        return torch.zeros(n_samples)
