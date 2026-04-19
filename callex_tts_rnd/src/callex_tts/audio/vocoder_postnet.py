"""
╔══════════════════════════════════════════════════════════════════════╗
║  CALLEX TTS — Convolutional Post-Net                                ║
║                                                                      ║
║  Learned residual correction network applied to predicted mel         ║
║  spectrograms before vocoder input. Improves spectral detail         ║
║  and reduces artifacts at the boundary between synthesis and         ║
║  waveform generation.                                                ║
║                                                                      ║
║  Architecture: 5-layer 1D convolution with batch norm and tanh       ║
║  Non-linearity, following the Tacotron 2 post-net design.            ║
╚══════════════════════════════════════════════════════════════════════╝
"""

from __future__ import annotations

import torch
import torch.nn as nn


class ConvPostNet(nn.Module):
    """
    5-layer convolutional post-net for mel spectrogram refinement.
    
    Takes a predicted mel spectrogram and applies learned residual
    corrections to improve spectral detail. The output is added
    to the input (residual connection) so the network only needs
    to learn the error, not the entire spectrogram.
    
    Architecture:
        Conv1d → BatchNorm1d → Tanh (×4)
        Conv1d → BatchNorm1d → Dropout (×1, final)
    
    Args:
        n_mel_channels: Number of mel frequency channels
        postnet_embedding_dim: Hidden dimension of conv layers
        postnet_kernel_size: Kernel size for all conv layers
        postnet_n_layers: Number of conv layers
        postnet_dropout: Dropout rate
    """

    def __init__(
        self,
        n_mel_channels: int = 80,
        postnet_embedding_dim: int = 512,
        postnet_kernel_size: int = 5,
        postnet_n_layers: int = 5,
        postnet_dropout: float = 0.5,
    ):
        super().__init__()

        self.layers = nn.ModuleList()

        # First layer: n_mels → embedding_dim
        self.layers.append(
            nn.Sequential(
                nn.Conv1d(
                    n_mel_channels, postnet_embedding_dim,
                    kernel_size=postnet_kernel_size,
                    padding=(postnet_kernel_size - 1) // 2,
                ),
                nn.BatchNorm1d(postnet_embedding_dim),
            )
        )

        # Middle layers: embedding_dim → embedding_dim
        for _ in range(1, postnet_n_layers - 1):
            self.layers.append(
                nn.Sequential(
                    nn.Conv1d(
                        postnet_embedding_dim, postnet_embedding_dim,
                        kernel_size=postnet_kernel_size,
                        padding=(postnet_kernel_size - 1) // 2,
                    ),
                    nn.BatchNorm1d(postnet_embedding_dim),
                )
            )

        # Final layer: embedding_dim → n_mels
        self.layers.append(
            nn.Sequential(
                nn.Conv1d(
                    postnet_embedding_dim, n_mel_channels,
                    kernel_size=postnet_kernel_size,
                    padding=(postnet_kernel_size - 1) // 2,
                ),
                nn.BatchNorm1d(n_mel_channels),
            )
        )

        self.dropout = nn.Dropout(postnet_dropout)
        self.n_layers = postnet_n_layers

    def forward(self, mel: torch.Tensor) -> torch.Tensor:
        """
        Apply post-net correction to mel spectrogram.
        
        Args:
            mel: Predicted mel spectrogram [batch, n_mels, time]
            
        Returns:
            Refined mel spectrogram [batch, n_mels, time]
            (input + learned residual)
        """
        residual = mel

        for i, layer in enumerate(self.layers):
            x = layer(mel if i == 0 else x)
            if i < self.n_layers - 1:
                x = torch.tanh(x)
            x = self.dropout(x)

        # Residual connection: output = input + correction
        return residual + x
