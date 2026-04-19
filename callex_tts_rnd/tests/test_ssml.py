"""Tests for SSML parser."""

import pytest
from callex_tts.text.ssml import SSMLParser, EmphasisLevel


class TestSSMLParser:
    def setup_method(self):
        self.parser = SSMLParser()

    def test_plain_text(self):
        segments = self.parser.parse("Hello world")
        assert len(segments) == 1
        assert segments[0].text == "Hello world"
        assert segments[0].prosody.rate == 1.0

    def test_prosody_rate(self):
        ssml = '<speak><prosody rate="fast">Hello</prosody></speak>'
        segments = self.parser.parse(ssml)
        fast_seg = [s for s in segments if s.text]
        assert len(fast_seg) > 0
        assert fast_seg[0].prosody.rate == 1.25

    def test_break_tag(self):
        ssml = '<speak>Hello<break time="500ms"/>World</speak>'
        segments = self.parser.parse(ssml)
        breaks = [s for s in segments if s.is_break]
        assert len(breaks) == 1
        assert breaks[0].break_ms == 500

    def test_max_break_limit(self):
        parser = SSMLParser(max_break_ms=1000)
        ssml = '<speak><break time="5000ms"/></speak>'
        segments = parser.parse(ssml)
        breaks = [s for s in segments if s.is_break]
        assert breaks[0].break_ms == 1000

    def test_invalid_ssml_fallback(self):
        segments = self.parser.parse("<invalid><unclosed>text")
        assert len(segments) >= 1
