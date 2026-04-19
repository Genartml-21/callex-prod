"""
╔══════════════════════════════════════════════════════════════════════╗
║  CALLEX TTS — SSML Parser                                          ║
║                                                                      ║
║  Parses Speech Synthesis Markup Language (SSML) tags to extract      ║
║  prosody control parameters for the synthesis pipeline.              ║
║                                                                      ║
║  Supported Tags:                                                     ║
║    <prosody rate="..." pitch="..." volume="...">text</prosody>       ║
║    <break time="500ms"/>                                             ║
║    <emphasis level="strong">text</emphasis>                          ║
║    <say-as interpret-as="digits">12345</say-as>                     ║
╚══════════════════════════════════════════════════════════════════════╝
"""

from __future__ import annotations

import re
import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional
from xml.etree import ElementTree as ET

logger = logging.getLogger("callex.tts.text.ssml")


class EmphasisLevel(str, Enum):
    NONE     = "none"
    REDUCED  = "reduced"
    MODERATE = "moderate"
    STRONG   = "strong"


@dataclass
class ProsodyParams:
    """Prosody parameters extracted from SSML tags."""
    rate: float = 1.0        # Speaking rate multiplier (0.5 = half speed, 2.0 = double)
    pitch: float = 0.0       # Pitch shift in semitones (-12 to +12)
    volume: float = 0.0      # Volume adjustment in dB (-20 to +20)
    emphasis: EmphasisLevel = EmphasisLevel.NONE

    def merge(self, other: "ProsodyParams") -> "ProsodyParams":
        """Merge two prosody params (nested tags accumulate)."""
        return ProsodyParams(
            rate=self.rate * other.rate,
            pitch=self.pitch + other.pitch,
            volume=self.volume + other.volume,
            emphasis=other.emphasis if other.emphasis != EmphasisLevel.NONE else self.emphasis,
        )


@dataclass
class SSMLSegment:
    """A segment of text with its prosody parameters."""
    text: str
    prosody: ProsodyParams = field(default_factory=ProsodyParams)
    break_ms: int = 0        # Silence duration after this segment (milliseconds)
    is_break: bool = False   # True if this segment is a pure break/silence


class SSMLParser:
    """
    Parse SSML input and extract text segments with prosody controls.
    
    If the input is not valid SSML (no <speak> root), it's treated
    as plain text with default prosody.
    
    Usage:
        parser = SSMLParser(max_break_ms=2000)
        segments = parser.parse('''
            <speak>
                <prosody rate="fast" pitch="+2st">
                    नमस्ते, आप कैसे हैं?
                </prosody>
                <break time="500ms"/>
                मैं ठीक हूँ।
            </speak>
        ''')
        for seg in segments:
            print(f"'{seg.text}' rate={seg.prosody.rate} pitch={seg.prosody.pitch}")
    """

    # Rate keyword mappings
    RATE_MAP: dict[str, float] = {
        "x-slow":  0.5,
        "slow":    0.75,
        "medium":  1.0,
        "fast":    1.25,
        "x-fast":  1.5,
    }

    # Pitch keyword mappings
    PITCH_MAP: dict[str, float] = {
        "x-low":   -6.0,
        "low":     -3.0,
        "medium":   0.0,
        "high":    +3.0,
        "x-high":  +6.0,
    }

    # Volume keyword mappings (dB)
    VOLUME_MAP: dict[str, float] = {
        "silent":  -96.0,
        "x-soft":  -12.0,
        "soft":    -6.0,
        "medium":   0.0,
        "loud":    +6.0,
        "x-loud":  +12.0,
    }

    # Emphasis level → prosody adjustment
    EMPHASIS_PROSODY: dict[EmphasisLevel, ProsodyParams] = {
        EmphasisLevel.REDUCED:  ProsodyParams(rate=1.1, pitch=-1.0, volume=-2.0),
        EmphasisLevel.MODERATE: ProsodyParams(rate=0.95, pitch=+1.0, volume=+2.0),
        EmphasisLevel.STRONG:   ProsodyParams(rate=0.85, pitch=+3.0, volume=+4.0),
    }

    def __init__(self, max_break_ms: int = 2000):
        self.max_break_ms = max_break_ms

    def parse(self, text: str) -> list[SSMLSegment]:
        """
        Parse SSML text into a list of segments with prosody params.
        
        If input is not valid SSML, returns a single segment with
        the raw text and default prosody.
        """
        text = text.strip()

        # Check if input is SSML
        if not self._is_ssml(text):
            return [SSMLSegment(text=text)]

        try:
            # Wrap in <speak> if not already
            if not text.startswith('<speak'):
                text = f'<speak>{text}</speak>'
            root = ET.fromstring(text)
            return self._parse_element(root, ProsodyParams())
        except ET.ParseError as e:
            logger.warning("SSML parse error: %s — treating as plain text", e)
            # Strip all XML tags and return as plain text
            plain = re.sub(r'<[^>]+>', '', text).strip()
            return [SSMLSegment(text=plain)]

    def _is_ssml(self, text: str) -> bool:
        """Check if text appears to be SSML."""
        ssml_tags = ['<speak', '<prosody', '<break', '<emphasis', '<say-as']
        return any(tag in text.lower() for tag in ssml_tags)

    def _parse_element(
        self, element: ET.Element, parent_prosody: ProsodyParams
    ) -> list[SSMLSegment]:
        """Recursively parse an XML element tree into segments."""
        segments: list[SSMLSegment] = []
        tag = self._clean_tag(element.tag)

        # Compute prosody for this element
        current_prosody = self._compute_prosody(element, tag, parent_prosody)

        # Handle text content before first child
        if element.text and element.text.strip():
            segments.append(SSMLSegment(
                text=element.text.strip(),
                prosody=current_prosody,
            ))

        # Process child elements
        for child in element:
            child_tag = self._clean_tag(child.tag)

            if child_tag == 'break':
                # Break element → silence segment
                break_ms = self._parse_break_time(child.get('time', '0ms'))
                break_ms = min(break_ms, self.max_break_ms)
                segments.append(SSMLSegment(
                    text='', is_break=True, break_ms=break_ms,
                    prosody=current_prosody,
                ))

            elif child_tag == 'say-as':
                # say-as → interpret content specially
                interpret = child.get('interpret-as', '')
                content = (child.text or '').strip()
                if interpret == 'digits' and content:
                    # Spell out each digit
                    content = ' '.join(content)
                segments.append(SSMLSegment(text=content, prosody=current_prosody))

            else:
                # Recursively parse prosody, emphasis, and other elements
                segments.extend(self._parse_element(child, current_prosody))

            # Handle tail text (text after a child element)
            if child.tail and child.tail.strip():
                segments.append(SSMLSegment(
                    text=child.tail.strip(),
                    prosody=current_prosody,
                ))

        return segments

    def _compute_prosody(
        self, element: ET.Element, tag: str, parent: ProsodyParams
    ) -> ProsodyParams:
        """Extract prosody params from element attributes."""
        if tag == 'prosody':
            rate = self._parse_rate(element.get('rate', ''))
            pitch = self._parse_pitch(element.get('pitch', ''))
            volume = self._parse_volume(element.get('volume', ''))
            local = ProsodyParams(rate=rate, pitch=pitch, volume=volume)
            return parent.merge(local)

        elif tag == 'emphasis':
            level = EmphasisLevel(element.get('level', 'moderate'))
            if level in self.EMPHASIS_PROSODY:
                local = self.EMPHASIS_PROSODY[level]
                return parent.merge(local)

        return parent

    def _parse_rate(self, value: str) -> float:
        """Parse rate attribute: keyword, percentage, or absolute."""
        if not value:
            return 1.0
        value = value.strip().lower()
        if value in self.RATE_MAP:
            return self.RATE_MAP[value]
        if value.endswith('%'):
            return float(value[:-1]) / 100.0
        try:
            return float(value)
        except ValueError:
            return 1.0

    def _parse_pitch(self, value: str) -> float:
        """Parse pitch attribute: keyword or semitone offset."""
        if not value:
            return 0.0
        value = value.strip().lower()
        if value in self.PITCH_MAP:
            return self.PITCH_MAP[value]
        # Parse "+3st", "-2st", "+3", "-2"
        match = re.match(r'([+-]?\d+(?:\.\d+)?)\s*(?:st|semitones?)?', value)
        if match:
            return float(match.group(1))
        return 0.0

    def _parse_volume(self, value: str) -> float:
        """Parse volume attribute: keyword or dB offset."""
        if not value:
            return 0.0
        value = value.strip().lower()
        if value in self.VOLUME_MAP:
            return self.VOLUME_MAP[value]
        match = re.match(r'([+-]?\d+(?:\.\d+)?)\s*(?:db)?', value)
        if match:
            return float(match.group(1))
        return 0.0

    def _parse_break_time(self, value: str) -> int:
        """Parse break time: '500ms' or '1s' → milliseconds."""
        value = value.strip().lower()
        if value.endswith('ms'):
            return int(float(value[:-2]))
        elif value.endswith('s'):
            return int(float(value[:-1]) * 1000)
        try:
            return int(float(value))
        except ValueError:
            return 0

    @staticmethod
    def _clean_tag(tag: str) -> str:
        """Remove XML namespace prefix from tag name."""
        if '}' in tag:
            return tag.split('}', 1)[1]
        return tag
