"""
╔══════════════════════════════════════════════════════════════════════╗
║  CALLEX TTS — Symbol Tokenizer                                     ║
║                                                                      ║
║  Maps IPA phoneme strings to integer sequences for neural network    ║
║  input. Maintains a fixed symbol table for model compatibility.      ║
╚══════════════════════════════════════════════════════════════════════╝
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Optional, Union

logger = logging.getLogger("callex.tts.text.tokenizer")


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  Default Symbol Inventory
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

# Special tokens
PAD_TOKEN = "<pad>"
UNK_TOKEN = "<unk>"
BOS_TOKEN = "<bos>"
EOS_TOKEN = "<eos>"
WORD_SEP  = "<ws>"     # Word separator
SPECIAL_TOKENS = [PAD_TOKEN, UNK_TOKEN, BOS_TOKEN, EOS_TOKEN, WORD_SEP]

# IPA symbol inventory for Hindi
IPA_VOWELS = [
    'ə', 'a', 'e', 'i', 'o', 'u',
    'ɛ', 'ɔ', 'ʊ', 'ɪ',
]

IPA_LONG_VOWELS = [v + 'ː' for v in IPA_VOWELS]
IPA_NASALIZED = [v + '̃' for v in IPA_VOWELS]

IPA_CONSONANTS = [
    # Plosives
    'p', 'b', 't', 'd', 'k', 'g', 'q',
    # Retroflex plosives
    'ʈ', 'ɖ',
    # Affricates
    'tʃ', 'dʒ',
    # Aspirated
    'pʰ', 'bʱ', 'tʰ', 'dʱ', 'kʰ', 'gʱ',
    'ʈʰ', 'ɖʱ', 'tʃʰ', 'dʒʱ',
    # Dental (special diacritics)
    't̪', 'd̪', 't̪ʰ', 'd̪ʱ',
    # Nasals
    'n', 'm', 'ŋ', 'ɲ', 'ɳ',
    # Fricatives
    'f', 'v', 's', 'z', 'ʃ', 'ʂ', 'x', 'ɣ', 'h', 'ɦ',
    # Approximants
    'j', 'ʋ', 'l', 'ɭ',
    # Flaps/Trills
    'r', 'ɾ', 'ɽ', 'ɽʱ',
]

IPA_SUPRASEGMENTAL = [
    'ː',     # Length
    '̃',      # Nasalization
    'ˈ',     # Primary stress
    'ˌ',     # Secondary stress
]

PROSODIC_MARKERS = [
    '‖',     # Major boundary
    '|',     # Minor boundary
    '↗',     # Rising intonation
    '↑',     # High intonation
]

PUNCTUATION = list(".,?!;:-–—'\"() ")

# Build the default symbol table
DEFAULT_SYMBOLS = (
    SPECIAL_TOKENS +
    IPA_VOWELS + IPA_LONG_VOWELS + IPA_NASALIZED +
    IPA_CONSONANTS +
    IPA_SUPRASEGMENTAL + PROSODIC_MARKERS +
    PUNCTUATION
)


class CallexTokenizer:
    """
    Maps IPA phoneme strings ↔ integer sequences.
    
    Supports:
      • Fixed vocabulary for model checkpoint compatibility
      • Multi-character symbol matching (e.g., 'tʃʰ' as single token)
      • BOS/EOS framing
      • Vocabulary export/import from JSON
      
    Usage:
        tokenizer = CallexTokenizer()
        ids = tokenizer.encode("nəməsteː")
        text = tokenizer.decode(ids)
    """

    def __init__(
        self,
        symbols: Optional[list[str]] = None,
        add_bos_eos: bool = True,
    ):
        self.add_bos_eos = add_bos_eos
        
        # Build symbol ↔ ID mappings
        syms = symbols or DEFAULT_SYMBOLS
        # Deduplicate while preserving order
        seen = set()
        unique_syms = []
        for s in syms:
            if s not in seen:
                seen.add(s)
                unique_syms.append(s)
        
        self._sym2id: dict[str, int] = {s: i for i, s in enumerate(unique_syms)}
        self._id2sym: dict[int, str] = {i: s for i, s in enumerate(unique_syms)}
        
        # Sort multi-char symbols by length (longest first) for greedy matching
        self._sorted_symbols = sorted(
            [s for s in unique_syms if s not in SPECIAL_TOKENS and len(s) > 0],
            key=len, reverse=True
        )
        
        self.pad_id = self._sym2id[PAD_TOKEN]
        self.unk_id = self._sym2id[UNK_TOKEN]
        self.bos_id = self._sym2id[BOS_TOKEN]
        self.eos_id = self._sym2id[EOS_TOKEN]
        
        logger.info(
            "CallexTokenizer initialized: vocab_size=%d, bos_eos=%s",
            self.vocab_size, add_bos_eos
        )

    @property
    def vocab_size(self) -> int:
        """Total number of symbols in the vocabulary."""
        return len(self._sym2id)

    def encode(self, phoneme_string: str) -> list[int]:
        """
        Convert an IPA phoneme string to a list of integer IDs.
        
        Uses greedy longest-match tokenization to handle multi-character
        IPA symbols (e.g., 'tʃʰ' is one token, not three).
        
        Args:
            phoneme_string: IPA string from the phonemizer
            
        Returns:
            List of integer token IDs
        """
        if not phoneme_string:
            return [self.bos_id, self.eos_id] if self.add_bos_eos else []

        ids: list[int] = []
        if self.add_bos_eos:
            ids.append(self.bos_id)

        i = 0
        text = phoneme_string
        while i < len(text):
            matched = False
            for sym in self._sorted_symbols:
                if text[i:i + len(sym)] == sym:
                    ids.append(self._sym2id[sym])
                    i += len(sym)
                    matched = True
                    break
            if not matched:
                # Unknown character → UNK
                if text[i] != ' ':  # Don't log spaces as unknown
                    logger.debug("Unknown symbol: '%s' (U+%04X)", text[i], ord(text[i]))
                ids.append(self.unk_id)
                i += 1

        if self.add_bos_eos:
            ids.append(self.eos_id)

        return ids

    def decode(self, ids: list[int]) -> str:
        """
        Convert a list of integer IDs back to a phoneme string.
        
        Args:
            ids: List of integer token IDs
            
        Returns:
            IPA phoneme string
        """
        symbols = []
        for token_id in ids:
            sym = self._id2sym.get(token_id, UNK_TOKEN)
            if sym in SPECIAL_TOKENS:
                continue
            symbols.append(sym)
        return ''.join(symbols)

    def text_to_sequence(self, text: str) -> list[int]:
        """Alias for encode() — backward compatibility with old API."""
        return self.encode(text)

    def sequence_to_text(self, sequence: list[int]) -> str:
        """Alias for decode() — backward compatibility with old API."""
        return self.decode(sequence)

    def save(self, path: Union[str, Path]) -> None:
        """Export vocabulary to JSON for checkpoint reproducibility."""
        path = Path(path)
        data = {
            "version": "2.0",
            "symbols": list(self._sym2id.keys()),
            "add_bos_eos": self.add_bos_eos,
        }
        path.write_text(json.dumps(data, ensure_ascii=False, indent=2))
        logger.info("Vocabulary saved to %s (%d symbols)", path, self.vocab_size)

    @classmethod
    def load(cls, path: Union[str, Path]) -> "CallexTokenizer":
        """Load vocabulary from a previously saved JSON file."""
        path = Path(path)
        data = json.loads(path.read_text())
        return cls(
            symbols=data["symbols"],
            add_bos_eos=data.get("add_bos_eos", True),
        )
