"""
╔══════════════════════════════════════════════════════════════════════╗
║  CALLEX TTS — Bucket Batch Sampler                                  ║
║                                                                      ║
║  Groups samples by similar length to minimize padding waste.         ║
║  Critical for efficient GPU utilization in TTS training where        ║
║  utterance lengths vary wildly (0.5s – 15s).                        ║
╚══════════════════════════════════════════════════════════════════════╝
"""

from __future__ import annotations

import math
import random
import logging
from typing import Iterator

from torch.utils.data import Sampler

logger = logging.getLogger("callex.tts.data.sampler")


class BucketBatchSampler(Sampler[list[int]]):
    """
    Groups dataset indices into buckets of similar length,
    then yields batches from within each bucket.
    
    This dramatically reduces padding waste compared to random sampling.
    For a dataset with 0.5s–15s utterances, this can improve
    GPU utilization by 30-50%.
    
    Usage:
        sampler = BucketBatchSampler(
            lengths=[len(sample) for sample in dataset],
            batch_size=32,
            n_buckets=10,
        )
        loader = DataLoader(dataset, batch_sampler=sampler)
    """

    def __init__(
        self,
        lengths: list[int],
        batch_size: int,
        n_buckets: int = 10,
        shuffle: bool = True,
        drop_last: bool = False,
        seed: int = 42,
    ):
        self.lengths = lengths
        self.batch_size = batch_size
        self.n_buckets = n_buckets
        self.shuffle = shuffle
        self.drop_last = drop_last
        self.seed = seed
        self.epoch = 0

        # Sort indices by length
        sorted_indices = sorted(range(len(lengths)), key=lambda i: lengths[i])

        # Divide into buckets
        bucket_size = math.ceil(len(sorted_indices) / n_buckets)
        self.buckets = [
            sorted_indices[i:i + bucket_size]
            for i in range(0, len(sorted_indices), bucket_size)
        ]

        logger.info(
            "BucketBatchSampler: %d samples, %d buckets, batch_size=%d",
            len(lengths), len(self.buckets), batch_size,
        )

    def __iter__(self) -> Iterator[list[int]]:
        rng = random.Random(self.seed + self.epoch)

        # Shuffle within each bucket
        all_batches: list[list[int]] = []
        for bucket in self.buckets:
            indices = list(bucket)
            if self.shuffle:
                rng.shuffle(indices)

            # Form batches within bucket
            for i in range(0, len(indices), self.batch_size):
                batch = indices[i:i + self.batch_size]
                if self.drop_last and len(batch) < self.batch_size:
                    continue
                all_batches.append(batch)

        # Shuffle batches across buckets
        if self.shuffle:
            rng.shuffle(all_batches)

        yield from all_batches

    def __len__(self) -> int:
        total = sum(
            math.ceil(len(b) / self.batch_size)
            if not self.drop_last
            else len(b) // self.batch_size
            for b in self.buckets
        )
        return total

    def set_epoch(self, epoch: int) -> None:
        """Set epoch for reproducible shuffling in distributed training."""
        self.epoch = epoch
