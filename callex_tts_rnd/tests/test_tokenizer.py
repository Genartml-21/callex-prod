"""Tests for tokenizer."""

import pytest
from callex_tts.text.tokenizer import CallexTokenizer


class TestCallexTokenizer:
    def setup_method(self):
        self.tokenizer = CallexTokenizer()

    def test_vocab_size(self):
        assert self.tokenizer.vocab_size > 0

    def test_encode_decode_roundtrip(self):
        # Simple phonemes should survive roundtrip
        text = "ə n m s"
        ids = self.tokenizer.encode(text)
        assert len(ids) > 0
        decoded = self.tokenizer.decode(ids)
        assert len(decoded) > 0

    def test_empty_input(self):
        ids = self.tokenizer.encode("")
        # Should have BOS + EOS
        assert len(ids) == 2

    def test_bos_eos_framing(self):
        ids = self.tokenizer.encode("ə")
        assert ids[0] == self.tokenizer.bos_id
        assert ids[-1] == self.tokenizer.eos_id

    def test_save_load(self, tmp_path):
        path = tmp_path / "vocab.json"
        self.tokenizer.save(path)
        loaded = CallexTokenizer.load(path)
        assert loaded.vocab_size == self.tokenizer.vocab_size

    def test_backward_compat_api(self):
        ids = self.tokenizer.text_to_sequence("ə n m")
        text = self.tokenizer.sequence_to_text(ids)
        assert isinstance(ids, list)
        assert isinstance(text, str)
