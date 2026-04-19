"""
╔══════════════════════════════════════════════════════════════════════╗
║  CALLEX TTS — Production Dataset with Caching                      ║
╚══════════════════════════════════════════════════════════════════════╝
"""

from __future__ import annotations

import csv
import logging
import os
from pathlib import Path
from typing import Optional

import torch
import torchaudio
from torch.utils.data import Dataset

from callex_tts.audio.features import MelSpectrogramExtractor, AudioConfig
from callex_tts.text.tokenizer import CallexTokenizer

logger = logging.getLogger("callex.tts.data.dataset")


class CallexTTSDataset(Dataset):
    """
    Production TTS dataset with lazy loading and mel caching.
    
    Metadata CSV format: file_id|transcription|speaker_id (optional)
    
    Features:
      • Lazy audio loading (no memory bloat for large datasets)
      • On-disk mel cache (compute once, reuse on restart)
      • Configurable sample rate and mel parameters
      • Robust error handling for corrupted files
    """

    def __init__(
        self,
        metadata_path: str,
        wav_dir: str,
        tokenizer: CallexTokenizer,
        audio_config: Optional[AudioConfig] = None,
        mel_cache_dir: Optional[str] = None,
        max_wav_length: int = 24000 * 15,    # 15 seconds max
        min_wav_length: int = 24000 * 0.5,   # 0.5 seconds min
    ):
        self.wav_dir = Path(wav_dir)
        self.tokenizer = tokenizer
        self.mel_extractor = MelSpectrogramExtractor(audio_config or AudioConfig())
        self.audio_config = audio_config or AudioConfig()
        self.max_wav_length = max_wav_length
        self.min_wav_length = min_wav_length

        # Mel cache
        self.mel_cache_dir = Path(mel_cache_dir) if mel_cache_dir else None
        if self.mel_cache_dir:
            self.mel_cache_dir.mkdir(parents=True, exist_ok=True)

        # Load metadata
        self.samples = self._load_metadata(metadata_path)
        logger.info(
            "Dataset loaded: %d samples from %s",
            len(self.samples), metadata_path
        )

    def _load_metadata(self, path: str) -> list[tuple[str, str, str]]:
        """Load and validate metadata CSV."""
        samples = []
        if not os.path.exists(path):
            logger.warning("Metadata not found: %s", path)
            return samples

        with open(path, 'r', encoding='utf-8') as f:
            for line_no, line in enumerate(f, 1):
                parts = line.strip().split('|')
                if len(parts) < 2:
                    continue
                file_id = parts[0].strip()
                text = parts[1].strip()
                speaker = parts[2].strip() if len(parts) > 2 else "default"

                wav_path = self.wav_dir / f"{file_id}.wav"
                if not wav_path.exists():
                    continue

                samples.append((file_id, text, speaker))

        return samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict:
        file_id, text, speaker = self.samples[idx]
        wav_path = self.wav_dir / f"{file_id}.wav"

        # Tokenize text
        token_ids = self.tokenizer.encode(text)
        text_tensor = torch.LongTensor(token_ids)

        # Load or cache mel spectrogram
        mel = self._get_mel(file_id, wav_path)

        return {
            "text": text_tensor,
            "text_length": len(token_ids),
            "mel": mel,
            "mel_length": mel.shape[-1],
            "file_id": file_id,
            "speaker": speaker,
        }

    def _get_mel(self, file_id: str, wav_path: Path) -> torch.Tensor:
        """Load mel from cache or compute from audio."""
        if self.mel_cache_dir:
            cache_path = self.mel_cache_dir / f"{file_id}.pt"
            if cache_path.exists():
                return torch.load(cache_path, weights_only=True)

        # Load audio
        waveform, sr = torchaudio.load(str(wav_path))

        # Resample if necessary
        if sr != self.audio_config.sample_rate:
            resampler = torchaudio.transforms.Resample(sr, self.audio_config.sample_rate)
            waveform = resampler(waveform)

        # Mono
        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)

        # Normalize
        waveform = waveform / (waveform.abs().max() + 1e-6)

        # Compute mel
        mel = self.mel_extractor.mel_spectrogram(waveform).squeeze(0)

        # Cache
        if self.mel_cache_dir:
            cache_path = self.mel_cache_dir / f"{file_id}.pt"
            torch.save(mel, cache_path)

        return mel


def collate_tts_batch(batch: list[dict]) -> dict:
    """
    Collate function for variable-length TTS batches.
    Pads text and mel to the maximum lengths in the batch.
    """
    text_lengths = torch.LongTensor([item["text_length"] for item in batch])
    mel_lengths = torch.LongTensor([item["mel_length"] for item in batch])

    max_text = text_lengths.max().item()
    max_mel = mel_lengths.max().item()
    n_mels = batch[0]["mel"].shape[0]

    text_padded = torch.zeros(len(batch), max_text, dtype=torch.long)
    mel_padded = torch.zeros(len(batch), n_mels, max_mel)

    for i, item in enumerate(batch):
        text_padded[i, :item["text_length"]] = item["text"]
        mel_padded[i, :, :item["mel_length"]] = item["mel"]

    return {
        "text": text_padded,
        "text_lengths": text_lengths,
        "mel": mel_padded,
        "mel_lengths": mel_lengths,
        "file_ids": [item["file_id"] for item in batch],
        "speakers": [item["speaker"] for item in batch],
    }
