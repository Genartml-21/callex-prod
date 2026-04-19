"""Tests for audio processing pipeline."""

import pytest
import numpy as np
import torch


class TestAudioEffectsChain:
    def test_effects_chain_passthrough(self):
        from callex_tts.audio.effects import AudioEffectsChain, EffectsChainConfig
        config = EffectsChainConfig(sample_rate=24000)
        chain = AudioEffectsChain(config)
        audio = np.random.randn(24000).astype(np.float32) * 0.5
        processed = chain.process(audio)
        assert processed.shape == audio.shape
        assert processed.dtype == np.float32

    def test_empty_audio(self):
        from callex_tts.audio.effects import AudioEffectsChain
        chain = AudioEffectsChain()
        result = chain.process(np.array([], dtype=np.float32))
        assert len(result) == 0

    def test_compressor_reduces_peaks(self):
        from callex_tts.audio.effects import DynamicRangeCompressor, CompressorConfig
        config = CompressorConfig(threshold_db=-20, ratio=4.0, makeup_gain_db=0)
        comp = DynamicRangeCompressor(config)
        # Create audio with a loud peak
        audio = np.zeros(1000, dtype=np.float64)
        audio[500] = 1.0  # Peak at full scale
        processed = comp.process(audio)
        assert abs(processed[500]) <= abs(audio[500])


class TestProsodyProcessor:
    def test_pitch_shift_identity(self):
        from callex_tts.audio.prosody import ProsodyProcessor
        proc = ProsodyProcessor(sample_rate=24000)
        audio = torch.randn(1, 24000)
        result = proc.pitch_shift(audio, semitones=0.0)
        assert result.shape == audio.shape

    def test_time_stretch_changes_length(self):
        from callex_tts.audio.prosody import ProsodyProcessor
        proc = ProsodyProcessor(sample_rate=24000)
        audio = torch.randn(1, 24000)
        # Slowing down should make it longer
        result = proc.time_stretch(audio, rate=0.5)
        assert result.shape[-1] > audio.shape[-1]

    def test_silence_generation(self):
        from callex_tts.audio.prosody import ProsodyProcessor
        proc = ProsodyProcessor(sample_rate=24000)
        silence = proc.generate_silence(1000)  # 1 second
        assert silence.shape[0] == 24000
        assert torch.all(silence == 0)


class TestMelExtractor:
    def test_mel_spectrogram_shape(self):
        from callex_tts.audio.features import MelSpectrogramExtractor, AudioConfig
        config = AudioConfig(sample_rate=24000, n_mels=80, hop_length=256)
        extractor = MelSpectrogramExtractor(config)
        audio = torch.randn(1, 24000)
        mel = extractor.mel_spectrogram(audio)
        assert mel.shape[1] == 80  # n_mels
        assert mel.dim() == 3  # [batch, n_mels, time]
