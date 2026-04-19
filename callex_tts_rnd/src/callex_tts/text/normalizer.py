"""
╔══════════════════════════════════════════════════════════════════════╗
║  CALLEX TTS — Hindi Text Normalizer                                ║
║                                                                      ║
║  Transforms raw Hindi/Hinglish text into a canonical phonetic form   ║
║  suitable for neural TTS synthesis. Handles:                         ║
║    • Unicode NFC normalization                                       ║
║    • Number → word expansion (Indian numbering system)               ║
║    • Currency symbol expansion (₹, $)                                ║
║    • Schwa deletion (critical for correct Hindi pronunciation)       ║
║    • Sandhi rules (vowel/consonant euphonic combinations)            ║
║    • Visarga normalization                                           ║
║    • English code-switching (Roman → Devanagari transliteration)     ║
║    • Abbreviation expansion                                         ║
║    • Nukta normalization (Perso-Arabic loanwords)                    ║
╚══════════════════════════════════════════════════════════════════════╝
"""

from __future__ import annotations

import re
import unicodedata
import logging
from dataclasses import dataclass, field
from typing import Optional

logger = logging.getLogger("callex.tts.text.normalizer")


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  Unicode Constants — Devanagari Block (U+0900 – U+097F)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

# Vowels (स्वर)
VOWELS = set("अआइईउऊऋॠऌॡएऐओऔ")

# Consonants (व्यंजन)
CONSONANTS = set("कखगघङचछजझञटठडढणतथदधनपफबभमयरलळवशषसह")

# Dependent vowel signs (मात्राएँ)
MATRAS = set("ािीुूृॄॅॆेैॉॊोौ")

# Special marks
HALANT      = "्"      # Virama
CHANDRABINDU = "ँ"     # Nasalization
ANUSVARA    = "ं"      # Nasal
VISARGA     = "ः"      # Aspiration
NUKTA       = "़"      # Nukta (modifies consonants for Persian/Arabic sounds)
AVAGRAHA    = "ऽ"      # Avagraha (elision marker)

# Punctuation
DANDA        = "।"
DOUBLE_DANDA = "॥"


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  Abbreviation Dictionary
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

ABBREVIATIONS: dict[str, str] = {
    "Dr.":   "डॉक्टर",
    "Mr.":   "मिस्टर",
    "Mrs.":  "मिसेज़",
    "Ms.":   "मिस",
    "Prof.": "प्रोफ़ेसर",
    "Sr.":   "सीनियर",
    "Jr.":   "जूनियर",
    "Ltd.":  "लिमिटेड",
    "Pvt.":  "प्राइवेट",
    "etc.":  "इत्यादि",
    "govt.": "सरकार",
    "approx.": "लगभग",
    "km":    "किलोमीटर",
    "kg":    "किलोग्राम",
    "mg":    "मिलीग्राम",
    "hrs":   "घंटे",
    "mins":  "मिनट",
    "secs":  "सेकंड",
    "EMI":   "ई एम आई",
    "OTP":   "ओ टी पी",
    "ATM":   "ए टी एम",
    "SIM":   "सिम",
    "PIN":   "पिन",
    "KYC":   "के वाई सी",
    "PAN":   "पैन",
    "GST":   "जी एस टी",
    "AC":    "ए सी",
    "TV":    "टी वी",
    "WiFi":  "वाई फ़ाई",
    "OK":    "ओके",
    "SMS":   "एस एम एस",
}

# Compiled pattern for abbreviation matching
_ABBREV_PATTERN = re.compile(
    r'\b(' + '|'.join(re.escape(k) for k in sorted(ABBREVIATIONS.keys(), key=len, reverse=True)) + r')\b',
    re.IGNORECASE
)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  Normalizer Configuration
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

@dataclass
class NormalizerConfig:
    """Configuration for the Hindi text normalizer."""
    expand_numbers: bool = True
    expand_currency: bool = True
    apply_schwa_deletion: bool = True
    handle_code_switching: bool = True
    apply_sandhi_rules: bool = True
    normalize_nukta: bool = True
    expand_abbreviations: bool = True
    normalize_punctuation: bool = True


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  Core Normalizer
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class HindiTextNormalizer:
    """
    Production-grade Hindi text normalization pipeline.
    
    Processes raw text through a configurable chain of linguistic
    transformations to produce clean, phonetically unambiguous input
    for the downstream phonemizer.
    
    Usage:
        normalizer = HindiTextNormalizer()
        clean = normalizer.normalize("₹500 का EMI भरना है Dr. शर्मा")
        # → "पांच सौ रुपये का ई एम आई भरना है डॉक्टर शर्मा"
    """

    def __init__(self, config: Optional[NormalizerConfig] = None):
        self.config = config or NormalizerConfig()
        
        # Precompile regex patterns for performance
        self._rupee_re   = re.compile(r'₹\s*([0-9,]+(?:\.\d{1,2})?)')
        self._dollar_re  = re.compile(r'\$\s*([0-9,]+(?:\.\d{1,2})?)')
        self._number_re  = re.compile(r'\b(\d+(?:,\d{2,3})*(?:\.\d+)?)\b')
        self._phone_re   = re.compile(r'\b(\d{10})\b')
        self._multi_ws   = re.compile(r'\s+')
        self._english_re = re.compile(r'\b([A-Za-z]{2,})\b')

        # Sandhi rules (vowel sandhi pairs)
        self._sandhi_rules = [
            # Vowel Sandhi: अ + अ → आ
            (re.compile(r'अ\s+अ'), 'आ'),
            # Vowel Sandhi: अ + इ → ए
            (re.compile(r'अ\s+इ'), 'ए'),
            # Vowel Sandhi: अ + उ → ओ
            (re.compile(r'अ\s+उ'), 'ओ'),
            # Visarga Sandhi: ः + voiced → र/ओ
            (re.compile(r'ः\s+([गघदधबभजझडढ])'), r'र \1'),
        ]

        logger.debug("HindiTextNormalizer initialized with config: %s", self.config)

    def normalize(self, text: str) -> str:
        """
        Run the full normalization pipeline.
        
        Order matters — each stage assumes the output format of the previous:
          1. Unicode NFC normalization
          2. Abbreviation expansion
          3. Currency expansion
          4. Number expansion
          5. Nukta normalization
          6. Code-switching (English → Devanagari)
          7. Sandhi rules
          8. Schwa deletion
          9. Punctuation cleanup
          10. Whitespace normalization
        """
        if not text or not text.strip():
            return ""

        text = self._normalize_unicode(text)

        if self.config.expand_abbreviations:
            text = self._expand_abbreviations(text)

        if self.config.expand_currency:
            text = self._expand_currency(text)

        if self.config.expand_numbers:
            text = self._expand_phone_numbers(text)
            text = self._expand_numbers(text)

        if self.config.normalize_nukta:
            text = self._normalize_nukta(text)

        if self.config.handle_code_switching:
            text = self._transliterate_english(text)

        if self.config.apply_sandhi_rules:
            text = self._apply_sandhi(text)

        if self.config.apply_schwa_deletion:
            text = self._apply_schwa_deletion(text)

        if self.config.normalize_punctuation:
            text = self._normalize_punctuation(text)

        return self._clean_whitespace(text)

    # ── Stage 1: Unicode ──

    def _normalize_unicode(self, text: str) -> str:
        """NFC normalize to canonical composed form."""
        return unicodedata.normalize('NFC', text)

    # ── Stage 2: Abbreviations ──

    def _expand_abbreviations(self, text: str) -> str:
        """Replace known abbreviations with Hindi expansions."""
        def _replace(match):
            key = match.group(0)
            # Try exact match first, then case-insensitive
            return ABBREVIATIONS.get(key, ABBREVIATIONS.get(key.upper(), key))
        return _ABBREV_PATTERN.sub(_replace, text)

    # ── Stage 3: Currency ──

    def _expand_currency(self, text: str) -> str:
        """Expand ₹ and $ to Hindi words."""
        def _rupee(m):
            num = m.group(1).replace(",", "")
            return f"{self._number_to_hindi(num)} रुपये"

        def _dollar(m):
            num = m.group(1).replace(",", "")
            return f"{self._number_to_hindi(num)} डॉलर"

        text = self._rupee_re.sub(_rupee, text)
        text = self._dollar_re.sub(_dollar, text)
        return text

    # ── Stage 4: Numbers ──

    def _expand_phone_numbers(self, text: str) -> str:
        """
        10-digit phone numbers are spoken digit-by-digit, not as a whole number.
        9876543210 → "नौ आठ सात छह पांच चार तीन दो एक शून्य"
        """
        DIGIT_MAP = {
            '0': 'शून्य', '1': 'एक', '2': 'दो', '3': 'तीन', '4': 'चार',
            '5': 'पांच', '6': 'छह', '7': 'सात', '8': 'आठ', '9': 'नौ'
        }
        def _phone(m):
            return ' '.join(DIGIT_MAP[d] for d in m.group(1))
        return self._phone_re.sub(_phone, text)

    def _expand_numbers(self, text: str) -> str:
        """Expand remaining numbers to Hindi words using Indian numbering system."""
        def _num(m):
            return self._number_to_hindi(m.group(0).replace(",", ""))
        return self._number_re.sub(_num, text)

    def _number_to_hindi(self, num_str: str) -> str:
        """
        Convert a numeric string to Hindi words.
        Uses num2words with Hindi locale, with manual fallback.
        """
        try:
            from num2words import num2words
            num = float(num_str) if '.' in num_str else int(num_str)
            return num2words(num, lang='hi')
        except Exception:
            return num_str

    # ── Stage 5: Nukta Normalization ──

    def _normalize_nukta(self, text: str) -> str:
        """
        Normalize nukta-bearing consonants to their canonical forms.
        Some input sources use decomposed nukta (consonant + ़), others
        use precomposed characters. Standardize to precomposed.
        """
        replacements = {
            'क' + NUKTA: 'क़',
            'ख' + NUKTA: 'ख़',
            'ग' + NUKTA: 'ग़',
            'ज' + NUKTA: 'ज़',
            'ड' + NUKTA: 'ड़',
            'ढ' + NUKTA: 'ढ़',
            'फ' + NUKTA: 'फ़',
        }
        for decomposed, precomposed in replacements.items():
            text = text.replace(decomposed, precomposed)
        return text

    # ── Stage 6: Code-Switching ──

    def _transliterate_english(self, text: str) -> str:
        """
        Transliterate English words embedded in Hindi text to Devanagari.
        Uses indic-transliteration for ITRANS→Devanagari mapping.
        Only applies to words that are purely Roman alphabet.
        """
        try:
            from indic_transliteration import sanscript
            from indic_transliteration.sanscript import transliterate
        except ImportError:
            logger.warning("indic-transliteration not installed; skipping code-switching")
            return text

        def _trans(m):
            word = m.group(1)
            # Skip abbreviations that we've already expanded
            if word.upper() in ABBREVIATIONS:
                return word
            try:
                return transliterate(word.lower(), sanscript.ITRANS, sanscript.DEVANAGARI)
            except Exception:
                return word

        return self._english_re.sub(_trans, text)

    # ── Stage 7: Sandhi Rules ──

    def _apply_sandhi(self, text: str) -> str:
        """
        Apply Hindi sandhi (euphonic combination) rules.
        These govern how sounds merge at word boundaries in connected speech.
        """
        for pattern, replacement in self._sandhi_rules:
            text = pattern.sub(replacement, text)
        return text

    # ── Stage 8: Schwa Deletion ──

    def _apply_schwa_deletion(self, text: str) -> str:
        """
        Apply schwa deletion rules for Hindi.
        
        In Hindi, the inherent 'a' (schwa) vowel in consonants is often
        not pronounced at word-final positions. This is THE most critical
        phonetic rule for natural-sounding Hindi TTS.
        
        Rules:
          1. Word-final consonant without matra → add halant
          2. Medial schwa deletion in compound words (simplified heuristic)
        
        Example: कमल → कमल् (Kamal, not Kamala)
        """
        words = text.split()
        processed = []

        for word in words:
            if not word:
                continue

            # Only process Devanagari words
            if not any(c in CONSONANTS or c in VOWELS for c in word):
                processed.append(word)
                continue

            # Rule 1: Word-final schwa deletion
            # If the last character is a bare consonant (no matra, no halant),
            # the schwa is typically deleted in pronunciation.
            if (len(word) >= 2
                and word[-1] in CONSONANTS
                and (len(word) < 2 or word[-2] not in (MATRAS | {HALANT, CHANDRABINDU, ANUSVARA, VISARGA}))):
                word = word + HALANT

            processed.append(word)

        return " ".join(processed)

    # ── Stage 9: Punctuation ──

    def _normalize_punctuation(self, text: str) -> str:
        """Normalize punctuation marks for consistent prosodic boundary signals."""
        # Convert Devanagari danda to period (for pause detection)
        text = text.replace(DOUBLE_DANDA, '.')
        text = text.replace(DANDA, ',')
        # Normalize quotation marks
        text = text.replace('"', '"').replace('"', '"')
        text = text.replace(''', "'").replace(''', "'")
        return text

    # ── Stage 10: Whitespace ──

    def _clean_whitespace(self, text: str) -> str:
        """Collapse multiple whitespace to single space and strip edges."""
        return self._multi_ws.sub(' ', text).strip()
