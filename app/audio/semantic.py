"""
Semantic Speech Filter
======================

Filters out non-meaningful utterances that shouldn't trigger barge-in:
- Filler words (हाँ, ओह, um, uh)
- Very short transcriptions (< 3 chars)
- Repetitive words
- Low-confidence transcriptions
"""

import re
from typing import Optional, List


class SemanticFilter:
    """
    Determines if transcribed text is meaningful enough to warrant a response.
    
    Prevents barge-in from:
    - Filler words and interjections
    - Noise transcribed as gibberish
    - Very short utterances
    """
    
    def __init__(self, language: str = 'hi', min_length: int = 3):
        """
        Initialize semantic filter.
        
        Args:
            language: Primary language ('hi' for Hindi, 'en' for English)
            min_length: Minimum character length for meaningful text
        """
        self.language = language
        self.min_length = min_length
        
        # Filler words to ignore (case-insensitive)
        self.hindi_fillers = {
            'हाँ', 'हां', 'ना', 'नहीं',  # Yes/No (alone)
            'अच्छा', 'अरे', 'ओह', 'आह', 'ए', 'ओ',
            'हम', 'हम्म', 'उम्म', 'एम',
            'क्या', 'हुह', 'अहा', 'ओके',
            '...', '..', 'ह', 'न'
        }
        
        self.english_fillers = {
            'um', 'uh', 'umm', 'uhh', 'er', 'ah', 'oh',
            'hmm', 'hm', 'mhm', 'mm', 'mmm',
            'yeah', 'yep', 'yup', 'nah', 'nope',
            'ok', 'okay', 'k',
            'huh', 'eh', 'what', 'haha', 'ha',
            '...', '..', '.', 'a', 'i'
        }
        
        # Combined blocklist
        self.filler_words = self.hindi_fillers | self.english_fillers
        
        print(f"[Semantic Filter] Initialized with {len(self.filler_words)} filler words")
    
    def is_meaningful(self, text: str) -> bool:
        """
        Check if transcribed text is meaningful enough for processing.
        
        Args:
            text: Transcribed text from ASR
            
        Returns:
            True if meaningful, False if filler/noise
        """
        if not text:
            return False
        
        # Clean text
        text_clean = text.strip().lower()
        
        # Rule 1: Length check
        if len(text_clean) < self.min_length:
            return False
        
        # Rule 2: Exact filler word match
        if text_clean in self.filler_words:
            return False
        
        # Rule 3: Check if only filler words (space-separated)
        words = text_clean.split()
        if all(word in self.filler_words for word in words):
            return False
        
        # Rule 4: Repetition detection (same word repeated > 2 times)
        if len(words) >= 3:
            if self._is_repetitive(words):
                return False
        
        # Rule 5: Too many punctuation marks (gibberish indicator)
        punct_count = sum(1 for c in text if c in '.,!?;:-')
        if punct_count > len(text) * 0.3:  # >30% punctuation
            return False
        
        # Rule 6: Check for common noise patterns
        # e.g., "क क क", "ह ह ह", "a a a"
        single_char_pattern = re.match(r'^(\S)\s+\1(\s+\1)*$', text_clean)
        if single_char_pattern:
            return False
        
        # Passed all filters - this is meaningful
        return True
    
    def _is_repetitive(self, words: List[str]) -> bool:
        """Check if words list contains excessive repetition"""
        if len(words) < 3:
            return False
        
        # Count consecutive identical words
        max_consecutive = 1
        current_consecutive = 1
        
        for i in range(1, len(words)):
            if words[i] == words[i-1]:
                current_consecutive += 1
                max_consecutive = max(max_consecutive, current_consecutive)
            else:
                current_consecutive = 1
        
        # If same word repeated 3+ times, it's repetitive
        return max_consecutive >= 3
    
    def get_rejection_reason(self, text: str) -> Optional[str]:
        """
        Get reason why text was rejected (for debugging).
        
        Returns:
            Reason string if rejected, None if accepted
        """
        if not text:
            return "Empty text"
        
        text_clean = text.strip().lower()
        
        if len(text_clean) < self.min_length:
            return f"Too short ({len(text_clean)} chars)"
        
        if text_clean in self.filler_words:
            return f"Filler word: '{text_clean}'"
        
        words = text_clean.split()
        if all(word in self.filler_words for word in words):
            return f"Only filler words: {words}"
        
        if self._is_repetitive(words):
            return f"Repetitive: {words}"
        
        punct_count = sum(1 for c in text if c in '.,!?;:-')
        if punct_count > len(text) * 0.3:
            return f"Too much punctuation ({punct_count}/{len(text)})"
        
        if re.match(r'^(\S)\s+\1(\s+\1)*$', text_clean):
            return "Single character repetition"
        
        return None  # Accepted


# Test function
if __name__ == "__main__":
    print("Testing Semantic Filter...\n")
    
    filter = SemanticFilter(language='hi', min_length=3)
    
    test_cases = [
        ("हाँ", False, "Filler word"),
        ("ओह", False, "Filler word"),
        ("um", False, "English filler"),
        ("हम्म", False, "Hindi filler"),
        ("मुझे रिचार्ज नहीं चाहिए", True, "Meaningful sentence"),
        ("कल करूँगा", True, "Short but meaningful"),
        ("हाँ हाँ ठीक है", False, "Multiple fillers"),
        ("क्या क्या क्या", False, "Repetitive"),
        ("a", False, "Too short"),
        ("ह", False, "Single char"),
        ("आज नहीं", True, "Meaningful short phrase"),
        ("...", False, "Only punctuation"),
        ("okay okay okay", False, "English repetition"),
        ("मैं आज रिचार्ज करूँगा", True, "Full sentence"),
        ("", False, "Empty"),
        ("हाँ ठीक है समझ गया", True, "Filler + meaningful"),
    ]
    
    passed = 0
    failed = 0
    
    for text, expected, description in test_cases:
        result = filter.is_meaningful(text)
        status = "✅" if result == expected else "❌"
        
        if result != expected:
            reason = filter.get_rejection_reason(text)
            print(f"{status} '{text}' -> {result} (expected {expected}) [{description}] Reason: {reason}")
            failed += 1
        else:
            print(f"{status} '{text}' -> {result} [{description}]")
            passed += 1
    
    print(f"\n{'='*50}")
    print(f"Results: {passed} passed, {failed} failed")
    print(f"{'='*50}")
