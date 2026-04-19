"""Tests for Hindi text normalizer."""

import pytest
from callex_tts.text.normalizer import HindiTextNormalizer, NormalizerConfig


class TestHindiTextNormalizer:
    def setup_method(self):
        self.normalizer = HindiTextNormalizer()

    def test_empty_input(self):
        assert self.normalizer.normalize("") == ""
        assert self.normalizer.normalize("   ") == ""

    def test_unicode_normalization(self):
        # NFC normalization should be idempotent on already-normalized text
        text = "नमस्ते"
        result = self.normalizer.normalize(text)
        assert "नमस्ते" in result

    def test_currency_expansion(self):
        result = self.normalizer.normalize("₹500 देने हैं")
        assert "₹" not in result
        assert "रुपये" in result

    def test_abbreviation_expansion(self):
        result = self.normalizer.normalize("EMI भरना है")
        assert "ई एम आई" in result

    def test_whitespace_cleanup(self):
        result = self.normalizer.normalize("नमस्ते    आप    कैसे     हैं")
        assert "  " not in result

    def test_punctuation_normalization(self):
        result = self.normalizer.normalize("नमस्ते। कैसे हैं॥")
        assert "।" not in result
        assert "॥" not in result

    def test_config_disabled_features(self):
        config = NormalizerConfig(
            expand_numbers=False,
            expand_currency=False,
        )
        normalizer = HindiTextNormalizer(config)
        result = normalizer.normalize("₹500")
        assert "₹" in result  # Currency expansion disabled
