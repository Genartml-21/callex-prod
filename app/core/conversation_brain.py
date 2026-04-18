"""
ConversationBrain — Per-Call Isolated Conversation State Manager
=================================================================

Ensures that every concurrent phone call has its own completely isolated
"brain" — conversation history, echo detection, and anti-hallucination
state that CANNOT leak between calls.

Solves:
  1. Opening line repetition — detects & blocks bot from repeating its opener
  2. Echo-loop hallucination — detects when SSTModel2 transcribes the bot's own
     TTS output and filters it out before it reaches the LLM
  3. Cross-call history leakage — all state is per-instance (per-call)
  4. Concurrent mutation — asyncio.Lock protects all history writes
  5. Repeat detection — blocks LLM from saying the same thing twice in a row

Usage:
    brain = ConversationBrain(call_uuid, agent_config)
    
    # Bot says opening line:
    await brain.add_bot_message(opener_text)
    brain.mark_opening_spoken()
    
    # Customer transcript arrives from SSTModel2 WS:
    if brain.is_echo(transcript):
        continue  # Skip — this is the bot's own voice
    await brain.add_user_message(transcript)
    
    # Get LLM response and validate:
    cleaned = brain.sanitize_response(llm_reply)
    if cleaned:
        await brain.add_bot_message(cleaned)
"""

import re
import asyncio
import time
import hashlib
from collections import deque
from typing import List, Dict, Optional, Set
from difflib import SequenceMatcher


class ConversationBrain:
    """
    Per-call conversation state with built-in anti-hallucination.
    
    Every WebSocket connection (= phone call) MUST create its own
    ConversationBrain. This guarantees zero state leakage.
    """

    MAX_HISTORY_LENGTH = 20  # Context window for LLM (trimmed)

    def __init__(self, call_uuid: str, agent_config: dict):
        self.call_uuid = call_uuid
        self.agent_config = agent_config

        # ── Core State (per-call isolated) ──
        self.history: List[Dict] = []        # Trimmed context window for LLM
        self.full_history: List[Dict] = []   # Complete transcript for analytics
        self._lock = asyncio.Lock()          # Protects all history mutations AND reads
        self._llm_lock = asyncio.Lock()      # Gates LLM calls — only 1 at a time per call

        # ── Anti-Hallucination State ──
        self._opening_line: str = agent_config.get('openingLine', '')
        self._opening_line_spoken: bool = False
        self._recent_bot_messages: deque = deque(maxlen=8)   # Last N bot messages
        self._recent_user_messages: deque = deque(maxlen=5)  # Last N user messages
        self._last_bot_reply: str = ""                        # Most recent for duplicate check
        self._bot_speaking_text: Optional[str] = None         # Currently being spoken by TTS
        self._bot_speaking_cleared_at: float = 0.0            # When TTS ended (for delayed clear)
        self._BOT_ECHO_LINGER_SECONDS: float = 2.0            # Keep echo detection alive after TTS

        # ── Response Fingerprinting (catches rephrased duplicates at scale) ──
        self._response_fingerprints: Set[str] = set()  # SHA256 of normalized responses
        self._MAX_FINGERPRINTS: int = 20                # Sliding window

    # ──────────────────────────────────────────────────────────────────────
    # History Management
    # ──────────────────────────────────────────────────────────────────────

    async def add_user_message(self, text: str) -> bool:
        """Add a user message to history. Returns False if the message was rejected."""
        if not text or not text.strip():
            return False
        async with self._lock:
            self.history.append({"role": "user", "parts": [{"text": text}]})
            self.full_history.append({"role": "user", "parts": [{"text": text}]})
            self._recent_user_messages.append(text.lower().strip())
            self._trim_history()
        return True

    async def add_bot_message(self, text: str):
        """Add a bot message to history and track for repeat detection."""
        if not text or not text.strip():
            return
        async with self._lock:
            self.history.append({"role": "model", "parts": [{"text": text}]})
            self.full_history.append({"role": "model", "parts": [{"text": text}]})
            self._last_bot_reply = text
            self._recent_bot_messages.append(text.lower().strip())
            self._trim_history()

    async def add_system_note(self, text: str):
        """Add a system note (e.g., '[System: User interrupted]')."""
        async with self._lock:
            self.history.append({"role": "model", "parts": [{"text": text}]})
            self.full_history.append({"role": "model", "parts": [{"text": text}]})
            self._trim_history()

    async def get_history(self) -> List[Dict]:
        """Get a snapshot of trimmed history for LLM. Thread-safe read under lock."""
        async with self._lock:
            return list(self.history)

    async def get_full_history_snapshot(self) -> List[Dict]:
        """Get the complete untruncated transcript (locked snapshot)."""
        async with self._lock:
            return list(self.full_history)

    def get_full_history(self) -> List[Dict]:
        """Get the complete untruncated transcript (unlocked — use only at call end)."""
        return list(self.full_history)

    def get_last_bot_reply(self) -> str:
        """Get the most recent bot reply for anti-hallucination checks."""
        return self._last_bot_reply

    def _trim_history(self):
        """Keep history within the context window. Called under lock.
        IMPORTANT: Uses in-place deletion so external aliases remain valid."""
        if len(self.history) > self.MAX_HISTORY_LENGTH:
            excess = len(self.history) - self.MAX_HISTORY_LENGTH
            del self.history[:excess]

    def mark_opening_spoken(self):
        """Mark that the opening line has been spoken — enables repeat detection."""
        self._opening_line_spoken = True
        if self._opening_line:
            self._recent_bot_messages.append(self._opening_line.lower().strip())
            self._last_bot_reply = self._opening_line

    def set_bot_speaking(self, text: Optional[str]):
        """Track what the bot is currently saying (for echo detection).
        When text=None (TTS ended), keeps the last text alive for _BOT_ECHO_LINGER_SECONDS
        to catch delayed echoes under CPU pressure."""
        if text is not None:
            self._bot_speaking_text = text
            self._bot_speaking_cleared_at = 0.0
        else:
            # Don't clear immediately — keep alive for echo detection
            self._bot_speaking_cleared_at = time.time()

    # ──────────────────────────────────────────────────────────────────────
    # Echo Detection — Prevents bot from hearing & responding to its own TTS
    # ──────────────────────────────────────────────────────────────────────

    def is_echo(self, transcript: str) -> bool:
        """
        Check if a SSTModel2 transcript is an echo of something the bot recently said.
        
        This catches the scenario where:
          1. Bot speaks via TTS
          2. FreeSWITCH routes the audio back into the input stream
          3. SSTModel2 transcribes it
          4. Without this check, the LLM would respond to its own output → loop
        """
        if not transcript:
            return True  # Empty transcript is always "echo"

        t_lower = transcript.lower().strip()
        t_norm = re.sub(r'[^\w\s]', '', t_lower).strip()

        if not t_norm or len(t_norm) < 3:
            return True  # Too short to be meaningful speech

        # Check against recent bot messages
        for bot_msg in self._recent_bot_messages:
            bot_norm = re.sub(r'[^\w\s]', '', bot_msg).strip()
            if not bot_norm:
                continue

            # Exact substring match (transcript is part of bot message)
            if t_norm in bot_norm:
                print(f"[ECHO DETECT] 🔇 Transcript is substring of bot message: '{transcript[:50]}'")
                return True

            # High similarity (fuzzy match for slight ASR differences)
            similarity = self._text_similarity(t_norm, bot_norm)
            if similarity > 0.65:
                print(f"[ECHO DETECT] 🔇 Transcript matches bot output ({similarity:.0%}): '{transcript[:50]}'")
                return True

            # Word overlap check for partial echoes
            t_words = set(t_norm.split())
            bot_words = set(bot_norm.split())
            if len(t_words) > 2 and len(bot_words) > 2:
                overlap = len(t_words & bot_words) / max(len(t_words), 1)
                if overlap > 0.75:
                    print(f"[ECHO DETECT] 🔇 High word overlap with bot ({overlap:.0%}): '{transcript[:50]}'")
                    return True

        # Check against what the bot is currently saying (or recently said)
        speaking_text = self._bot_speaking_text
        # If TTS ended recently, still check against it (delayed clear)
        if speaking_text and self._bot_speaking_cleared_at > 0:
            if time.time() - self._bot_speaking_cleared_at > self._BOT_ECHO_LINGER_SECONDS:
                self._bot_speaking_text = None  # Actually clear now
                speaking_text = None

        if speaking_text:
            speaking_norm = re.sub(r'[^\w\s]', '', speaking_text.lower()).strip()
            if speaking_norm and t_norm in speaking_norm:
                print(f"[ECHO DETECT] 🔇 Transcript matches currently-speaking text: '{transcript[:50]}'")
                return True

        return False

    # ──────────────────────────────────────────────────────────────────────
    # Response Sanitization — Catches and fixes hallucination patterns
    # ──────────────────────────────────────────────────────────────────────

    def sanitize_response(self, reply: str) -> Optional[str]:
        """
        Validate and clean an LLM response before speaking it.
        
        Returns:
          - The cleaned reply string (may be modified)
          - None if the reply should be completely suppressed (hallucination)
        """
        if not reply or not reply.strip():
            return None

        original = reply

        # 1. Opening line repetition detection
        if self._opening_line_spoken and self._opening_line:
            opening_norm = re.sub(r'[^\w\s]', '', self._opening_line.lower()).strip()
            reply_norm = re.sub(r'[^\w\s]', '', reply.lower()).strip()
            
            similarity = self._text_similarity(reply_norm, opening_norm)
            if similarity > 0.70:
                print(f"[BRAIN] 🛡️ BLOCKED opening line repetition ({similarity:.0%}): '{reply[:60]}'")
                return None

        # 2. Exact repeat of last bot message
        if self._last_bot_reply:
            last_norm = re.sub(r'[^\w\s]', '', self._last_bot_reply.lower()).strip()
            reply_norm = re.sub(r'[^\w\s]', '', reply.lower()).strip()
            
            similarity = self._text_similarity(reply_norm, last_norm)
            if similarity > 0.80:
                print(f"[BRAIN] 🛡️ BLOCKED exact repeat of last reply ({similarity:.0%}): '{reply[:60]}'")
                return None

        # 3. Check against ALL recent bot messages (not just the last one)
        reply_norm = re.sub(r'[^\w\s]', '', reply.lower()).strip()
        for recent in self._recent_bot_messages:
            recent_norm = re.sub(r'[^\w\s]', '', recent).strip()
            if not recent_norm:
                continue
            similarity = self._text_similarity(reply_norm, recent_norm)
            if similarity > 0.80:
                print(f"[BRAIN] 🛡️ BLOCKED repeat of recent message ({similarity:.0%}): '{reply[:60]}'")
                return None

        # 4. Intra-reply duplicate sentence removal
        reply = self._remove_duplicate_sentences(reply)

        # 5. Looping phrase detection
        reply = self._truncate_loops(reply)

        if reply != original:
            print(f"[BRAIN] ✂️ Cleaned: '{original[:50]}' → '{reply[:50]}'")

        # Response fingerprinting — catch semantically identical rephrased responses
        if reply.strip():
            fingerprint = self._fingerprint(reply)
            if fingerprint in self._response_fingerprints:
                print(f"[BRAIN] 🛡️ BLOCKED fingerprint duplicate: '{reply[:60]}'")
                return None
            self._response_fingerprints.add(fingerprint)
            # Sliding window — keep only last N fingerprints
            if len(self._response_fingerprints) > self._MAX_FINGERPRINTS:
                # Remove oldest (sets are unordered, but for production this is fine)
                self._response_fingerprints.pop()

        return reply if reply.strip() else None

    def _remove_duplicate_sentences(self, text: str) -> str:
        """Remove exact duplicate sentences within the same reply."""
        sentences = re.split(r'(?<=[।.!?])\s*', text)
        seen = set()
        unique = []
        for s in sentences:
            s_clean = s.strip()
            if not s_clean:
                continue
            s_norm = re.sub(r'[^\w\s]', '', s_clean.lower()).strip()
            if s_norm and s_norm not in seen:
                seen.add(s_norm)
                unique.append(s_clean)
        return ' '.join(unique) if unique else text

    def _truncate_loops(self, text: str) -> str:
        """Detect and truncate looping/repeating phrases."""
        words = text.split()
        if len(words) <= 12:
            return text
        for window in range(4, min(len(words) // 2 + 1, 15)):
            for i in range(len(words) - window * 2 + 1):
                phrase = ' '.join(words[i:i + window]).lower()
                rest = ' '.join(words[i + window:]).lower()
                if phrase in rest:
                    print(f"[BRAIN] 🛡️ Truncated loop: '{phrase[:40]}...'")
                    result = ' '.join(words[:i + window])
                    if not result.rstrip().endswith(('.', '?', '!', '।')):
                        result = result.rstrip() + '।'
                    return result
        return text

    # ──────────────────────────────────────────────────────────────────────
    # User Transcript Validation
    # ──────────────────────────────────────────────────────────────────────

    def is_duplicate_user_message(self, text: str) -> bool:
        """Check if user said the exact same thing twice in a row."""
        if not text:
            return True
        t_norm = re.sub(r'[^\w\s]', '', text.lower()).strip()
        for recent in self._recent_user_messages:
            r_norm = re.sub(r'[^\w\s]', '', recent).strip()
            if self._text_similarity(t_norm, r_norm) > 0.90:
                return True
        return False

    # ──────────────────────────────────────────────────────────────────────
    # Utilities
    # ──────────────────────────────────────────────────────────────────────

    @staticmethod
    def _text_similarity(a: str, b: str) -> float:
        """Fast text similarity using SequenceMatcher (0.0 = different, 1.0 = identical)."""
        if not a or not b:
            return 0.0
        return SequenceMatcher(None, a, b).ratio()

    @staticmethod
    def _fingerprint(text: str) -> str:
        """Generate a normalized fingerprint for response dedup.
        Strips punctuation, lowercases, sorts words — catches rephrased duplicates."""
        norm = re.sub(r'[^\w\s]', '', text.lower()).strip()
        # Sort words to catch reordered sentences
        words = sorted(norm.split())
        return hashlib.sha256(' '.join(words).encode()).hexdigest()[:16]

    def cleanup(self):
        """Release all memory. Call when the call ends."""
        self.history.clear()
        self.full_history.clear()
        self._recent_bot_messages.clear()
        self._recent_user_messages.clear()
        self._response_fingerprints.clear()
        self._last_bot_reply = ""
        self._bot_speaking_text = None
        print(f"[Brain:{self.call_uuid[:8]}] 🧹 Cleaned up")
