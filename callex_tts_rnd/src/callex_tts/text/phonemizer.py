"""
╔══════════════════════════════════════════════════════════════════════╗
║  CALLEX TTS — Grapheme-to-Phoneme (G2P) Engine                     ║
║                                                                      ║
║  Converts normalized Hindi text into International Phonetic          ║
║  Alphabet (IPA) sequences for the neural encoder.                    ║
║                                                                      ║
║  Pipeline:                                                           ║
║    Raw Text → Normalizer → Phonemizer → IPA Sequence → Tokenizer    ║
║                                                                      ║
║  Backends:                                                           ║
║    • epitran  — Rule-based Devanagari→IPA via Unicode transduction   ║
║    • espeak   — eSpeak NG phonemizer (fallback)                      ║
╚══════════════════════════════════════════════════════════════════════╝
"""

from __future__ import annotations

import logging
import re
from enum import Enum
from dataclasses import dataclass
from typing import Optional

logger = logging.getLogger("callex.tts.text.phonemizer")


class PhonemizationBackend(str, Enum):
    """Available G2P backends."""
    EPITRAN = "epitran"
    ESPEAK  = "espeak"


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  IPA Post-Processing Rules
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

# Hindi-specific IPA corrections that improve synthesis quality
IPA_CORRECTIONS: list[tuple[re.Pattern, str]] = [
    # Retroflex flap: ड़ should map to ɽ, not ɖ
    (re.compile(r'ɖ̤'), 'ɽ'),
    # Gemination: double consonants should be explicitly long
    (re.compile(r'(.)\1'), r'\1ː'),
    # Chandrabindu nasalization: ensure tilde on preceding vowel
    (re.compile(r'([aeioəɛɔʊɪ])̃'), r'\1̃'),
    # Clean up spurious whitespace in IPA output
    (re.compile(r'\s+'), ' '),
]

# Punctuation → prosodic boundary markers
PROSODY_MARKERS: dict[str, str] = {
    '.': '‖',     # Major prosodic boundary (sentence end)
    ',': '|',     # Minor prosodic boundary (clause)
    '?': '‖↗',   # Question intonation
    '!': '‖↑',   # Exclamation intonation
    ';': '|',     # Semicolon → minor boundary
    ':': '|',     # Colon → minor boundary
    '—': '|',     # Em-dash → minor boundary
}


@dataclass
class PhonemizationResult:
    """Result of phonemization with metadata."""
    phonemes: str                # IPA string
    original_text: str           # Input text
    n_phonemes: int              # Number of phoneme characters
    backend: str                 # Which backend was used
    word_boundaries: list[int]   # Indices where word boundaries occur in phoneme string


class CallexPhonemizer:
    """
    Production G2P engine for Hindi text.
    
    Converts Unicode Devanagari text into IPA phoneme sequences
    that the neural text encoder can process.
    
    Usage:
        phonemizer = CallexPhonemizer(backend="epitran")
        result = phonemizer.phonemize("नमस्ते आप कैसे हैं")
        print(result.phonemes)  # "nəməsteː aːp kɛːseː hɛ̃ː"
    """

    def __init__(
        self,
        backend: str = "epitran",
        language_code: str = "hin-Deva",
        preserve_punctuation: bool = True,
        with_stress: bool = False,
    ):
        self.backend_name = backend
        self.language_code = language_code
        self.preserve_punctuation = preserve_punctuation
        self.with_stress = with_stress

        # Initialize the selected backend
        self._backend = self._init_backend(backend, language_code)

        logger.info(
            "CallexPhonemizer initialized: backend=%s, lang=%s",
            backend, language_code
        )

    def _init_backend(self, backend: str, lang: str):
        """Initialize the G2P backend engine."""
        if backend == PhonemizationBackend.EPITRAN:
            try:
                import epitran
                return epitran.Epitran(lang)
            except ImportError:
                logger.error(
                    "epitran not installed. Install with: pip install epitran"
                )
                raise
            except Exception as e:
                logger.warning(
                    "Failed to load epitran with lang '%s': %s. "
                    "Will use character-level fallback.", lang, e
                )
                return None

        elif backend == PhonemizationBackend.ESPEAK:
            try:
                import subprocess
                # Verify eSpeak NG is installed
                subprocess.run(
                    ["espeak-ng", "--version"],
                    capture_output=True, check=True
                )
                return "espeak-ng"
            except (FileNotFoundError, subprocess.CalledProcessError):
                logger.warning("eSpeak NG not found, falling back to character mapping")
                return None

        else:
            logger.warning("Unknown backend '%s', using character mapping", backend)
            return None

    def phonemize(self, text: str) -> PhonemizationResult:
        """
        Convert text to IPA phonemes.
        
        Args:
            text: Normalized Hindi text (output from HindiTextNormalizer)
            
        Returns:
            PhonemizationResult with IPA string and metadata
        """
        if not text or not text.strip():
            return PhonemizationResult(
                phonemes="", original_text=text,
                n_phonemes=0, backend=self.backend_name,
                word_boundaries=[]
            )

        # Split into words, preserving punctuation positions
        words = text.split()
        phoneme_words: list[str] = []
        word_boundaries: list[int] = []
        current_pos = 0

        for word in words:
            # Check if word is pure punctuation
            if all(c in PROSODY_MARKERS for c in word):
                if self.preserve_punctuation:
                    marker = PROSODY_MARKERS.get(word, word)
                    phoneme_words.append(marker)
                    current_pos += len(marker) + 1
                continue

            # Separate trailing punctuation
            trailing_punct = ""
            while word and word[-1] in PROSODY_MARKERS:
                trailing_punct = word[-1] + trailing_punct
                word = word[:-1]

            if word:
                ipa = self._word_to_ipa(word)
                phoneme_words.append(ipa)
                word_boundaries.append(current_pos)
                current_pos += len(ipa)

                if trailing_punct and self.preserve_punctuation:
                    for p in trailing_punct:
                        marker = PROSODY_MARKERS.get(p, p)
                        phoneme_words.append(marker)
                        current_pos += len(marker)

                current_pos += 1  # space

        phoneme_string = ' '.join(phoneme_words)

        # Apply IPA post-processing corrections
        phoneme_string = self._apply_corrections(phoneme_string)

        return PhonemizationResult(
            phonemes=phoneme_string,
            original_text=text,
            n_phonemes=len(phoneme_string.replace(' ', '')),
            backend=self.backend_name,
            word_boundaries=word_boundaries,
        )

    def _word_to_ipa(self, word: str) -> str:
        """Convert a single word to IPA using the loaded backend."""
        if self._backend is None:
            return self._fallback_char_map(word)

        if self.backend_name == PhonemizationBackend.EPITRAN:
            try:
                return self._backend.transliterate(word)
            except Exception as e:
                logger.debug("Epitran failed on '%s': %s", word, e)
                return self._fallback_char_map(word)

        elif self.backend_name == PhonemizationBackend.ESPEAK:
            return self._espeak_phonemize(word)

        return self._fallback_char_map(word)

    def _espeak_phonemize(self, word: str) -> str:
        """Use eSpeak NG as G2P backend."""
        import subprocess
        try:
            result = subprocess.run(
                ["espeak-ng", "-v", "hi", "-q", "--ipa", word],
                capture_output=True, text=True, timeout=5
            )
            return result.stdout.strip()
        except Exception as e:
            logger.debug("eSpeak failed on '%s': %s", word, e)
            return self._fallback_char_map(word)

    def _fallback_char_map(self, word: str) -> str:
        """
        Character-level Devanagari → IPA mapping.
        Used when no G2P backend is available.
        Covers the most common Hindi consonants and vowels.
        """
        CHAR_IPA: dict[str, str] = {
            # Vowels
            'अ': 'ə', 'आ': 'aː', 'इ': 'ɪ', 'ई': 'iː',
            'उ': 'ʊ', 'ऊ': 'uː', 'ए': 'eː', 'ऐ': 'ɛː',
            'ओ': 'oː', 'औ': 'ɔː', 'ऋ': 'rɪ',
            # Matras (dependent vowels)
            'ा': 'aː', 'ि': 'ɪ', 'ी': 'iː', 'ु': 'ʊ',
            'ू': 'uː', 'े': 'eː', 'ै': 'ɛː', 'ो': 'oː',
            'ौ': 'ɔː', 'ृ': 'rɪ',
            # Consonants (with inherent schwa)
            'क': 'kə', 'ख': 'kʰə', 'ग': 'ɡə', 'घ': 'ɡʱə', 'ङ': 'ŋə',
            'च': 'tʃə', 'छ': 'tʃʰə', 'ज': 'dʒə', 'झ': 'dʒʱə', 'ञ': 'ɲə',
            'ट': 'ʈə', 'ठ': 'ʈʰə', 'ड': 'ɖə', 'ढ': 'ɖʱə', 'ण': 'ɳə',
            'त': 't̪ə', 'थ': 't̪ʰə', 'द': 'd̪ə', 'ध': 'd̪ʱə', 'न': 'nə',
            'प': 'pə', 'फ': 'pʰə', 'ब': 'bə', 'भ': 'bʱə', 'म': 'mə',
            'य': 'jə', 'र': 'rə', 'ल': 'lə', 'व': 'ʋə',
            'श': 'ʃə', 'ष': 'ʂə', 'स': 'sə', 'ह': 'ɦə',
            # Nukta consonants
            'क़': 'qə', 'ख़': 'xə', 'ग़': 'ɣə', 'ज़': 'zə',
            'ड़': 'ɽə', 'ढ़': 'ɽʱə', 'फ़': 'fə',
            # Special marks
            '्': '',       # Halant suppresses schwa
            'ं': 'n',      # Anusvara → nasal
            'ँ': '̃',       # Chandrabindu → nasalization
            'ः': 'h',      # Visarga → aspiration
        }

        result = []
        i = 0
        chars = list(word)
        while i < len(chars):
            # Try two-character sequences first
            if i + 1 < len(chars):
                digraph = chars[i] + chars[i + 1]
                if digraph in CHAR_IPA:
                    result.append(CHAR_IPA[digraph])
                    i += 2
                    continue

            char = chars[i]
            if char in CHAR_IPA:
                result.append(CHAR_IPA[char])
            # Skip unknown characters silently
            i += 1

        return ''.join(result)

    def _apply_corrections(self, ipa: str) -> str:
        """Apply Hindi-specific IPA post-processing corrections."""
        for pattern, replacement in IPA_CORRECTIONS:
            ipa = pattern.sub(replacement, ipa)
        return ipa.strip()

    @property
    def symbols(self) -> list[str]:
        """Return the complete phoneme symbol inventory."""
        # IPA symbols used in Hindi
        vowels = list("aeioəɛɔʊɪ")
        long_vowels = [v + 'ː' for v in vowels]
        nasalized = [v + '̃' for v in vowels]
        consonants = list("bdfghjklmnprstvwzʃʂɕʈɖɳɲŋɽɾɦʋ")
        aspirated = [c + 'ʰ' for c in "kptʈtʃ"]
        breathy = [c + 'ʱ' for c in "gbdɖdʒ"]
        affricates = ['tʃ', 'dʒ', 'tʃʰ', 'dʒʱ']
        specials = ['ː', '̃', '‖', '|', '↗', '↑', ' ']

        all_symbols = (
            vowels + long_vowels + nasalized +
            consonants + aspirated + breathy + affricates +
            specials
        )
        return sorted(set(all_symbols))
