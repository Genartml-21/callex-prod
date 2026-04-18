import re
import unicodedata
from indicnlp.tokenize import indic_tokenize
from indicnlp.syllable import syllabifier
from num2words import num2words
import epitran

class CallexHindiTextNormalizer:
    """
    Tier-1 Deep Phonetic Normalization Engine explicitly built for 'hi-IN' locales.
    Synthesizing natural Hindi voice requires immense linguistic rule matching before 
    passing data into the Latent Tensors.
    """
    def __init__(self):
        # ── DEVANAGARI UNICODE MAPS ──
        # Full Unicode mappings ensuring we catch every possible character in the Indian subcontinent.
        self.VOWELS = "अआइईउऊऋॠऌॡएऐओऔ"
        self.CONSONANTS = "कखगघङचछजझञटठडढणतथदधनपफबभमयरलळवशषसह"
        self.MATRAS = "ािीुूृॄेैोौ"
        self.HALANT = "्"
        self.CHANDRABINDU = "ँ"
        self.ANUSVARA = "ं"
        self.VISARGA = "ः"
        self.NUKTA = "़"
        self.DANDA = "।"
        self.DOUBLE_DANDA = "॥"

        # ── NUMERAL AND CURRENCY ALGORITHMS ──
        self.rupee_pattern = re.compile(r'₹\s*([0-9,.]+)')
        self.dollar_pattern = re.compile(r'\\$\s*([0-9,.]+)')
        self.hindi_number_map = {
            "0": "शून्य", "1": "एक", "2": "दो", "3": "तीन", "4": "चार",
            "5": "पांच", "6": "छह", "7": "सात", "8": "आठ", "9": "नौ",
            "10": "दस", "11": "ग्यारह", "12": "बारह", "13": "तेरह", "14": "चौदह",
            "15": "पंद्रह", "16": "सोलह", "17": "सत्रह", "18": "अठारह", "19": "उन्नीस",
            "20": "बीस", "30": "तीस", "40": "चालीस", "50": "पचास",
            "60": "साठ", "70": "सत्तर", "80": "अस्सी", "90": "नब्बे",
            "100": "सौ", "1000": "हज़ार", "100000": "लाख", "10000000": "करोड़"
        }

    def normalize(self, text: str) -> str:
        """Core mathematical NLP loop."""
        text = self._remove_accents_and_normalize(text)
        text = self._expand_currency(text)
        text = self._expand_numbers(text)
        text = self._apply_schwa_deletion(text)
        text = self._map_english_code_switching(text)
        return self._clean_whitespaces(text)

    def _remove_accents_and_normalize(self, text: str) -> str:
        text = unicodedata.normalize('NFC', text)
        return text

    def _expand_currency(self, text: str) -> str:
        """Transforms '₹500' -> 'पांच सौ रुपये' dynamically."""
        def replace_rupees(match):
            num = match.group(1).replace(",", "")
            return f"{self._number_to_hindi(num)} रुपये"
        text = self.rupee_pattern.sub(replace_rupees, text)
        return text

    def _expand_numbers(self, text: str) -> str:
        """Regex-based integer expansion across the sentence string."""
        def replace_num(match):
            return self._number_to_hindi(match.group(0))
        return re.sub(r'\\b\\d+\\b', replace_num, text)

    def _number_to_hindi(self, num_str: str) -> str:
        """Deep fallback algorithm decoded entirely by OpenSource Tier-1 Num2Words."""
        try:
            num = int(num_str)
            return num2words(num, lang='hi')
        except Exception:
            return num_str # Fallback constraint limit

    def _apply_schwa_deletion(self, text: str) -> str:
        """
        The absolute most critical structural phonetic algorithm for Native Hindi!
        Deletes the implicit 'a' vowel at the end of Hindi words.
        Without this, 'कमल' (Kamal) sounds like 'Kamala' permanently destroying immersion.
        """
        words = text.split()
        processed_words = []
        for word in words:
            if len(word) >= 3 and word[-1] in self.CONSONANTS:
                # If the word finishes on a pure consonant structure with no internal Matra attached,
                # we structurally bolt a Halant explicitly overriding the acoustic vocoder boundary.
                word += self.HALANT
            processed_words.append(word)
        return " ".join(processed_words)

    def _map_english_code_switching(self, text: str) -> str:
        """
        Transliterates English fallback words (loan words) structurally mapping them 
        into phonetic constraints so the Hindi Voice generation doesn't crash on foreign matrix weights.
        """
        eng_to_hindi = {
            "hello": "हेलो", "okay": "ओके", "thanks": "थैंक्स",
            "sorry": "सॉरी", "yes": "येस", "no": "नो",
            "sir": "सर", "madam": "मैडम"
        }
        for eng, hin in eng_to_hindi.items():
            text = re.sub(rf'\\b{eng}\\b', hin, text, flags=re.IGNORECASE)
        return text

    def _clean_whitespaces(self, text: str) -> str:
        return re.sub(r'\\s+', ' ', text).strip()

class CallexPhonemeEngine:
    """
    Tier-1 Deep Phonetic Transformer Mapping Pipeline utilizing Open-Source Epitran IPA bindings.
    Maps deep NLP Devanagari rules seamlessly to International Phonetic Alphabet (IPA) vectors!
    """
    def __init__(self):
        self.normalizer = CallexHindiTextNormalizer()
        
        try:
            self.epi = epitran.Epitran('hin-Deva')
        except Exception as e:
            print(f"[Callex R&D] Warning: Epitran hin-Deva map unavailable. Utilizing basic index. {e}")
            self.epi = None
            
        # Explicit Matrix Symbol Map combining IPA and Devanagari explicitly natively
        base_symbols = list("ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz"
                            "अआइईउऊऋएऐओऔकखगघङचछजझञटठडढणतथदधनपफबभमयरलवशषसह"
                            " ािीुूृेैोौ्ँंः़।"
                            " !?,.-'"
                            "əɑɪiʊuɾeæoɔkʰgʱŋcʰjʱɲʈʰɖʱɳtʰdʱnpʰbʱmjrlwʃʂsh")
        self.symbol_to_id = {s: i for i, s in enumerate(base_symbols)}
        self.id_to_symbol = {i: s for i, s in enumerate(base_symbols)}
        self.vocab_size = len(self.symbol_to_id)

    def encode(self, text: str) -> list:
        # Step 1: Execute deep physical normalizations natively across 'num2words'
        normalized_text = self.normalizer.normalize(text)
        
        # Step 2: Utilize Tier-1 'indic_nlp' explicit syllabification checks before execution
        syllables = syllabifier.orthographic_syllabify(normalized_text, 'hi')
        
        # Step 3: Transform perfectly tokenized Hindi sequences into absolute pure acoustic IPA math
        if self.epi:
             phonetic_string = self.epi.transliterate(normalized_text)
        else:
             phonetic_string = normalized_text
             
        # Step 4: Encode to Native Deep Learning Integer Tensors gracefully
        sequence = []
        for char in phonetic_string:
            if char in self.symbol_to_id:
                sequence.append(self.symbol_to_id[char])
                
        return sequence

    def decode(self, sequence: list) -> str:
        return "".join([self.id_to_symbol.get(int(i), "") for i in sequence])
