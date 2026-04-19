"""
╔══════════════════════════════════════════════════════════════════════╗
║  CALLEX TTS — Learning Rate Schedulers                              ║
╚══════════════════════════════════════════════════════════════════════╝
"""

from __future__ import annotations

import math
import torch
from torch.optim.lr_scheduler import _LRScheduler


class WarmupCosineScheduler(_LRScheduler):
    """
    Cosine annealing with linear warmup.
    
    Schedule:
      1. Linear warmup from 0 → base_lr over warmup_steps
      2. Cosine decay from base_lr → min_lr over remaining steps
    
    This is the gold standard for training transformers and has
    been shown to outperform exponential decay for VITS models.
    """

    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        warmup_steps: int = 2000,
        max_steps: int = 500000,
        min_lr: float = 1e-6,
        last_epoch: int = -1,
    ):
        self.warmup_steps = warmup_steps
        self.max_steps = max_steps
        self.min_lr = min_lr
        super().__init__(optimizer, last_epoch)

    def get_lr(self) -> list[float]:
        step = self.last_epoch

        if step < self.warmup_steps:
            # Linear warmup
            scale = step / max(1, self.warmup_steps)
        else:
            # Cosine decay
            progress = (step - self.warmup_steps) / max(1, self.max_steps - self.warmup_steps)
            progress = min(progress, 1.0)
            scale = 0.5 * (1.0 + math.cos(math.pi * progress))

        return [
            self.min_lr + (base_lr - self.min_lr) * scale
            for base_lr in self.base_lrs
        ]


class ExponentialWarmupScheduler(_LRScheduler):
    """
    Exponential decay with linear warmup.
    
    Alternative to cosine — decays more aggressively early on.
    Good for fine-tuning pre-trained models.
    """

    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        warmup_steps: int = 2000,
        gamma: float = 0.9998,
        min_lr: float = 1e-6,
        last_epoch: int = -1,
    ):
        self.warmup_steps = warmup_steps
        self.gamma = gamma
        self.min_lr = min_lr
        super().__init__(optimizer, last_epoch)

    def get_lr(self) -> list[float]:
        step = self.last_epoch

        if step < self.warmup_steps:
            scale = step / max(1, self.warmup_steps)
            return [base_lr * scale for base_lr in self.base_lrs]

        decay_steps = step - self.warmup_steps
        return [
            max(self.min_lr, base_lr * (self.gamma ** decay_steps))
            for base_lr in self.base_lrs
        ]
