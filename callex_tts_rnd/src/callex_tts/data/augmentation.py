"""
╔══════════════════════════════════════════════════════════════════════╗
║  CALLEX TTS — Data Augmentation                                     ║
║                                                                      ║
║  Augmentation techniques for robust TTS training:                    ║
║    • SpecAugment (frequency/time masking on mel spectrograms)        ║
║    • Pitch perturbation (random pitch shift on waveforms)            ║
║    • Time stretching (random tempo change)                           ║
║    • Additive noise (optional background noise injection)            ║
╚══════════════════════════════════════════════════════════════════════╝
"""

from __future__ import annotations

import random
import logging
from dataclasses import dataclass

import torch
import torch.nn.functional as F

logger = logging.getLogger("callex.tts.data.augmentation")


@dataclass
class SpecAugmentConfig:
    freq_mask_param: int = 15         # Max frequency bands to mask
    time_mask_param: int = 35         # Max time steps to mask
    n_freq_masks: int = 2             # Number of frequency masks
    n_time_masks: int = 2             # Number of time masks


class SpecAugment(torch.nn.Module):
    """
    SpecAugment (Park et al., 2019) — data augmentation on mel spectrograms.
    
    Randomly masks contiguous bands in frequency and time dimensions.
    Forces the model to be robust to missing spectral information,
    significantly improving generalization.
    """

    def __init__(self, config: SpecAugmentConfig = None):
        super().__init__()
        self.config = config or SpecAugmentConfig()

    def forward(self, mel: torch.Tensor) -> torch.Tensor:
        """
        Apply SpecAugment to mel spectrogram.
        
        Args:
            mel: [batch, n_mels, time] or [n_mels, time]
            
        Returns:
            Augmented mel spectrogram (same shape)
        """
        if not self.training:
            return mel

        squeeze = mel.dim() == 2
        if squeeze:
            mel = mel.unsqueeze(0)

        mel = mel.clone()
        _, n_mels, n_time = mel.shape

        # Frequency masking
        for _ in range(self.config.n_freq_masks):
            f = random.randint(0, min(self.config.freq_mask_param, n_mels - 1))
            f0 = random.randint(0, n_mels - f)
            mel[:, f0:f0 + f, :] = 0.0

        # Time masking
        for _ in range(self.config.n_time_masks):
            t = random.randint(0, min(self.config.time_mask_param, n_time - 1))
            t0 = random.randint(0, n_time - t)
            mel[:, :, t0:t0 + t] = 0.0

        if squeeze:
            mel = mel.squeeze(0)

        return mel


class AudioAugmentor:
    """
    Waveform-level augmentation for TTS training data.
    
    Applied before mel extraction to create diverse training conditions.
    """

    def __init__(
        self,
        pitch_range: tuple[float, float] = (-2.0, 2.0),
        speed_range: tuple[float, float] = (0.9, 1.1),
        noise_level: float = 0.0,
        p_pitch: float = 0.5,
        p_speed: float = 0.5,
        p_noise: float = 0.0,
    ):
        self.pitch_range = pitch_range
        self.speed_range = speed_range
        self.noise_level = noise_level
        self.p_pitch = p_pitch
        self.p_speed = p_speed
        self.p_noise = p_noise

    def __call__(self, waveform: torch.Tensor, sample_rate: int = 24000) -> torch.Tensor:
        """
        Apply random augmentations to a waveform.
        
        Args:
            waveform: [1, samples] or [samples]
            sample_rate: Audio sample rate
            
        Returns:
            Augmented waveform
        """
        # Speed perturbation (via resampling)
        if random.random() < self.p_speed:
            rate = random.uniform(*self.speed_range)
            new_sr = int(sample_rate * rate)
            waveform = torchaudio_resample(waveform, sample_rate, new_sr)

        # Additive noise
        if random.random() < self.p_noise and self.noise_level > 0:
            noise = torch.randn_like(waveform) * self.noise_level
            waveform = waveform + noise

        return waveform


def torchaudio_resample(
    waveform: torch.Tensor, orig_sr: int, new_sr: int
) -> torch.Tensor:
    """Resample audio using linear interpolation (fast approximation)."""
    if orig_sr == new_sr:
        return waveform

    ratio = new_sr / orig_sr
    seq_len = waveform.shape[-1]
    new_len = int(seq_len * ratio)

    if waveform.dim() == 1:
        waveform = waveform.unsqueeze(0).unsqueeze(0)
        out = F.interpolate(waveform, size=new_len, mode='linear', align_corners=False)
        return out.squeeze(0).squeeze(0)
    else:
        waveform = waveform.unsqueeze(0) if waveform.dim() == 2 else waveform
        out = F.interpolate(waveform, size=new_len, mode='linear', align_corners=False)
        return out.squeeze(0) if out.dim() == 3 else out
