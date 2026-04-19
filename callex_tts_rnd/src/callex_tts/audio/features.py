"""
╔══════════════════════════════════════════════════════════════════════╗
║  CALLEX TTS — Mel Spectrogram Feature Extraction                   ║
║                                                                      ║
║  Centralized audio feature computation used by both training and     ║
║  inference pipelines. All parameters are config-driven.              ║
║                                                                      ║
║  Supports:                                                           ║
║    • Linear spectrogram                                              ║
║    • Log-mel spectrogram                                             ║
║    • MFCC extraction                                                 ║
║    • Griffin-Lim reconstruction (debug/fallback)                     ║
║    • Dynamic range compression                                       ║
╚══════════════════════════════════════════════════════════════════════╝
"""

from __future__ import annotations

import math
import logging
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F
import torchaudio

logger = logging.getLogger("callex.tts.audio.features")


@dataclass
class AudioConfig:
    """Audio feature extraction configuration."""
    sample_rate: int = 24000
    n_fft: int = 1024
    hop_length: int = 256
    win_length: int = 1024
    n_mels: int = 80
    fmin: float = 0.0
    fmax: float = 8000.0
    clip_val: float = 1e-5
    mel_scale: str = "slaney"
    norm: str = "slaney"
    center: bool = True
    pad_mode: str = "reflect"


class MelSpectrogramExtractor:
    """
    Production mel spectrogram extractor with configurable parameters.
    
    Provides consistent feature extraction across training and inference,
    ensuring the same audio processing pipeline is used everywhere.
    
    Usage:
        extractor = MelSpectrogramExtractor(AudioConfig(sample_rate=24000))
        mel = extractor.mel_spectrogram(waveform)     # [1, n_mels, time]
        linear = extractor.linear_spectrogram(waveform)
        audio = extractor.griffin_lim(mel)             # Reconstruction
    """

    def __init__(self, config: Optional[AudioConfig] = None):
        self.config = config or AudioConfig()
        
        # Build mel filterbank
        self._mel_basis = self._build_mel_basis()
        
        # Hann window (cached)
        self._window = torch.hann_window(self.config.win_length)
        
        logger.info(
            "MelSpectrogramExtractor: sr=%d, n_fft=%d, hop=%d, n_mels=%d",
            self.config.sample_rate, self.config.n_fft,
            self.config.hop_length, self.config.n_mels
        )

    def _build_mel_basis(self) -> torch.Tensor:
        """Build mel filterbank matrix using torchaudio."""
        mel_fb = torchaudio.functional.melscale_fbanks(
            n_freqs=self.config.n_fft // 2 + 1,
            f_min=self.config.fmin,
            f_max=self.config.fmax,
            n_mels=self.config.n_mels,
            sample_rate=self.config.sample_rate,
            norm=self.config.norm if self.config.norm != "none" else None,
            mel_scale=self.config.mel_scale,
        )
        return mel_fb  # [n_freqs, n_mels]

    def _stft(self, audio: torch.Tensor) -> torch.Tensor:
        """
        Compute STFT magnitude.
        
        Args:
            audio: [batch, samples] or [samples]
            
        Returns:
            Magnitude spectrogram [batch, n_fft//2+1, time]
        """
        if audio.dim() == 1:
            audio = audio.unsqueeze(0)

        window = self._window.to(audio.device)

        # Pad to ensure consistent output length
        pad_amount = (self.config.n_fft - self.config.hop_length) // 2
        audio = F.pad(audio, (pad_amount, pad_amount), mode=self.config.pad_mode)

        stft = torch.stft(
            audio,
            n_fft=self.config.n_fft,
            hop_length=self.config.hop_length,
            win_length=self.config.win_length,
            window=window,
            center=False,  # We've already padded
            return_complex=True,
        )
        return torch.abs(stft)  # [batch, n_fft//2+1, time]

    def linear_spectrogram(self, audio: torch.Tensor) -> torch.Tensor:
        """
        Compute linear (magnitude) spectrogram.
        
        Args:
            audio: [batch, samples] or [samples]
            
        Returns:
            Linear spectrogram [batch, n_fft//2+1, time]
        """
        return self._stft(audio)

    def mel_spectrogram(self, audio: torch.Tensor) -> torch.Tensor:
        """
        Compute log-mel spectrogram.
        
        Args:
            audio: [batch, samples] or [samples]
            
        Returns:
            Log-mel spectrogram [batch, n_mels, time]
        """
        magnitudes = self._stft(audio)  # [batch, n_fft//2+1, time]
        
        mel_basis = self._mel_basis.to(audio.device)
        # mel_basis is [n_freqs, n_mels], magnitudes is [batch, n_freqs, time]
        mel = torch.matmul(mel_basis.T, magnitudes)  # [batch, n_mels, time]
        
        # Log-scale with clamping for numerical stability
        mel = torch.log(torch.clamp(mel, min=self.config.clip_val))
        
        return mel

    def mel_to_linear(self, mel: torch.Tensor) -> torch.Tensor:
        """
        Approximate inverse mel → linear spectrogram.
        Uses pseudo-inverse of the mel filterbank.
        """
        mel_basis = self._mel_basis.to(mel.device)
        mel_linear = torch.exp(mel)  # Undo log
        inv_mel = torch.linalg.pinv(mel_basis.T)
        linear = torch.matmul(inv_mel, mel_linear)
        return torch.clamp(linear, min=0)

    def griffin_lim(
        self,
        mel: torch.Tensor,
        n_iterations: int = 32,
    ) -> torch.Tensor:
        """
        Griffin-Lim reconstruction from mel spectrogram.
        
        This is a debug/fallback tool — production should use the
        neural vocoder. But it's invaluable for quick sanity checks
        during development.
        
        Args:
            mel: Log-mel spectrogram [batch, n_mels, time]
            n_iterations: Number of Griffin-Lim iterations
            
        Returns:
            Reconstructed waveform [batch, samples]
        """
        linear = self.mel_to_linear(mel)  # [batch, n_fft//2+1, time]
        
        window = self._window.to(mel.device)
        
        # Initialize with random phase
        angles = torch.randn_like(linear) * 2 * math.pi
        
        for _ in range(n_iterations):
            complex_spec = linear * torch.exp(1j * angles)
            
            # iSTFT
            audio = torch.istft(
                complex_spec,
                n_fft=self.config.n_fft,
                hop_length=self.config.hop_length,
                win_length=self.config.win_length,
                window=window,
            )
            
            # Re-STFT to update phase
            stft = torch.stft(
                audio,
                n_fft=self.config.n_fft,
                hop_length=self.config.hop_length,
                win_length=self.config.win_length,
                window=window,
                return_complex=True,
            )
            angles = torch.angle(stft)
        
        # Final iSTFT
        complex_spec = linear * torch.exp(1j * angles)
        audio = torch.istft(
            complex_spec,
            n_fft=self.config.n_fft,
            hop_length=self.config.hop_length,
            win_length=self.config.win_length,
            window=window,
        )
        
        return audio

    @staticmethod
    def dynamic_range_compression(x: torch.Tensor, C: float = 1.0, clip_val: float = 1e-5) -> torch.Tensor:
        """Apply dynamic range compression to a spectrogram."""
        return torch.log(torch.clamp(x, min=clip_val) * C)

    @staticmethod
    def dynamic_range_decompression(x: torch.Tensor, C: float = 1.0) -> torch.Tensor:
        """Inverse of dynamic_range_compression."""
        return torch.exp(x) / C
