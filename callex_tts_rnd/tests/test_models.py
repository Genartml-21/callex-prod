"""Tests for model architecture (import and shape validation)."""

import pytest
import torch


class TestModelImports:
    """Verify all model components import and construct without errors."""

    def test_text_encoder(self):
        from callex_tts.models.encoder import TextEncoder
        enc = TextEncoder(vocab_size=100, hidden_channels=64, n_layers=2, n_heads=2)
        x = torch.randint(0, 100, (2, 10))
        lengths = torch.tensor([10, 8])
        out, m, logs, mask = enc(x, lengths)
        assert out.shape == (2, 64, 10)

    def test_posterior_encoder(self):
        from callex_tts.models.flow import PosteriorEncoder
        pe = PosteriorEncoder(in_channels=80, hidden_channels=64, out_channels=64, n_layers=2)
        mel = torch.randn(2, 80, 20)
        mask = torch.ones(2, 1, 20)
        z, m, logs = pe(mel, mask)
        assert z.shape == (2, 64, 20)

    def test_hifi_gan_generator(self):
        from callex_tts.models.vocoder import HiFiGANGenerator
        gen = HiFiGANGenerator(
            initial_channels=64,
            upsample_rates=(4, 4),
            upsample_initial_channel=128,
            upsample_kernel_sizes=(8, 8),
            resblock_kernel_sizes=(3, 7),
            resblock_dilation_sizes=((1, 3), (1, 3)),
            use_snake=False,
        )
        z = torch.randn(1, 64, 10)
        audio = gen(z)
        assert audio.shape[0] == 1
        assert audio.shape[1] == 1
        assert audio.shape[2] == 10 * 4 * 4  # upsampled

    def test_discriminator_mpd(self):
        from callex_tts.models.discriminator import MultiPeriodDiscriminator
        mpd = MultiPeriodDiscriminator(periods=(2, 3))
        y = torch.randn(1, 1, 1024)
        y_hat = torch.randn(1, 1, 1024)
        y_dr, y_dg, fmap_r, fmap_g = mpd(y, y_hat)
        assert len(y_dr) == 2

    def test_full_synthesis_network(self):
        from callex_tts.models.generator import CallexSynthesisNetwork
        # Small config for testing
        net = CallexSynthesisNetwork(
            vocab_size=50,
            hidden_channels=32,
            filter_channels=64,
            n_heads=2,
            n_layers=1,
            n_layers_flow=1,
            n_flows=1,
            n_layers_posterior=2,
            upsample_rates=(4, 4),
            upsample_initial_channel=64,
            upsample_kernel_sizes=(8, 8),
            resblock_kernel_sizes=(3,),
            resblock_dilation_sizes=((1, 3),),
            segment_size=256,
            use_snake=False,
        )
        assert net is not None
        # Just verify construction succeeds
        params = sum(p.numel() for p in net.parameters())
        assert params > 0
