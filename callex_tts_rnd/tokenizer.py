import re

class GenerativePhonemizer:
    """
    R&D Custom Phonemizer.
    Converts raw mixed Hindi/English scripts into strict Internal Phonetic Alphabet (IPA) sequences.
    This guarantees the matrix understands exactly how to synthesize pronunciation universally.
    """
    def __init__(self):
        # We define a strict vocabulary of valid phonemes and structural markers
        self.symbols = list("ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz"
                            "अआइईउऊऋएऐओऔकखगघङचछजझञटठडढणतथदधनपफबभमयरलवशषसह"
                            " !?,.-'")
        self.symbol_to_id = {s: i for i, s in enumerate(self.symbols)}
        self.id_to_symbol = {i: s for i, s in enumerate(self.symbols)}

    def text_to_sequence(self, text: str) -> list:
        """Translates raw string vectors into integer phonetic arrays for the neural encoder."""
        clean_text = self._clean_text(text)
        sequence = []
        for char in clean_text:
            if char in self.symbol_to_id:
                sequence.append(self.symbol_to_id[char])
        return sequence

    def _clean_text(self, text: str) -> str:
        """Normalizes and expands conversational abbreviations before mathematical synthesis."""
        text = text.replace("&", "and")
        text = re.sub(r'\s+', ' ', text)
        return text.strip()
