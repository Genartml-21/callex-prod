"""
FastReplyCache — Ultra-Low Latency FAQ Fast-Path for Voice AI
=============================================================

Provides instant (~5ms) replies for questions whose answers already exist
in the agent's systemPrompt or knowledgeBase, completely bypassing the LLM.

Architecture:
  1. At call startup, the agent's prompt + KB are parsed into FAQ pairs
  2. Each customer utterance is checked against cached Q&A via fuzzy matching
  3. High-confidence matches return instantly; low-confidence falls through to LLM

Cache is per-agent with configurable TTL (default 60s) so dashboard edits
propagate quickly while avoiding redundant extraction on every call.
"""

import re
import time
from typing import Optional, Dict, List, Tuple
from difflib import SequenceMatcher


# ── Global per-agent cache with TTL ──
_agent_faq_cache: Dict[str, Tuple[float, 'FastReplyCache']] = {}
FAQ_CACHE_TTL = 60  # seconds — agent prompt changes reflect within 60s


def get_or_create_cache(agent_id: str, system_prompt: str, knowledge_base: str = "", language: str = "hi-IN") -> 'FastReplyCache':
    """Get cached FastReplyCache for agent, or create a new one."""
    now = time.time()
    if agent_id in _agent_faq_cache:
        cached_time, cached_instance = _agent_faq_cache[agent_id]
        if now - cached_time < FAQ_CACHE_TTL:
            return cached_instance

    # Build new cache
    cache = FastReplyCache(language=language)
    cache.build_from_prompt(system_prompt, knowledge_base)
    _agent_faq_cache[agent_id] = (now, cache)
    print(f"[FAST-PATH] ✅ Built FAQ cache for agent {agent_id[:8]}... ({len(cache._faq_pairs)} Q&A pairs, {len(cache._ack_patterns)} ack patterns)")
    return cache


class FastReplyCache:
    """
    Zero-latency reply cache that pattern-matches customer utterances
    against pre-extracted FAQ pairs from the agent's system prompt.
    """

    # ── Common acknowledgment patterns (language-aware) ──
    _ACK_PATTERNS_HI = {
        # Affirmative
        r'^(haan?|ha+n?|ji|ji haan?|theek hai|thik hai|ok|okay|accha|acha|sahi hai|bilkul|zaroor|haa ji)$': None,  # None = let LLM handle (context-dependent)
        # Negative
        r'^(nahi|nai|na|no|nope|mat|nahi ji|nahi nahi|bilkul nahi)$': None,
        # Greetings (these should be instant)
        r'^(hello|hi|namaste|namaskar|namasте)$': None,
    }

    _ACK_PATTERNS_GU = {
        r'^(haa|ha+|ji|bhai|saru|barabar|theek chhe|ok|okay|accha)$': None,
        r'^(na|nai|nathi|nako|na bhai)$': None,
        r'^(hello|hi|kem cho|namaste)$': None,
    }

    _ACK_PATTERNS_EN = {
        r'^(yes|yeah|yep|sure|ok|okay|right|correct|exactly|absolutely)$': None,
        r'^(no|nope|not really|nah)$': None,
        r'^(hello|hi|hey)$': None,
    }

    # Match confidence threshold — only return cached reply if above this
    MATCH_THRESHOLD = 0.72

    def __init__(self, language: str = "hi-IN"):
        self._language = language
        self._faq_pairs: List[Tuple[List[str], str]] = []  # [(trigger_phrases, reply), ...]
        self._ack_patterns: Dict[str, Optional[str]] = {}
        self._raw_prompt = ""
        self._raw_kb = ""

        # Load language-appropriate acknowledgment patterns
        if "gu" in language.lower():
            self._ack_patterns = dict(self._ACK_PATTERNS_GU)
        elif "hi" in language.lower():
            self._ack_patterns = dict(self._ACK_PATTERNS_HI)
        else:
            self._ack_patterns = dict(self._ACK_PATTERNS_EN)

    def build_from_prompt(self, system_prompt: str, knowledge_base: str = ""):
        """Extract FAQ pairs from system prompt and knowledge base text."""
        self._raw_prompt = system_prompt
        self._raw_kb = knowledge_base

        combined = f"{system_prompt}\n\n{knowledge_base}".strip()
        if not combined:
            return

        # ── Strategy 1: Extract explicit Q&A patterns ──
        # Matches patterns like "Q: ... A: ..." or "Question: ... Answer: ..."
        qa_pattern = re.compile(
            r'(?:Q|Question|सवाल|प्रश्न)\s*[:\-]?\s*(.+?)\s*(?:A|Answer|जवाब|उत्तर)\s*[:\-]?\s*(.+?)(?=(?:Q|Question|सवाल|प्रश्न)\s*[:\-]|$)',
            re.IGNORECASE | re.DOTALL
        )
        for match in qa_pattern.finditer(combined):
            question = match.group(1).strip()
            answer = match.group(2).strip()
            if question and answer and len(answer) < 500:
                triggers = self._generate_trigger_variants(question)
                self._faq_pairs.append((triggers, answer))

        # ── Strategy 2: Extract "If customer asks X, say Y" patterns ──
        if_asks_pattern = re.compile(
            r'(?:if|agar|jab)\s+(?:customer|caller|user|koi)\s+(?:asks?|puche|bole|kahe)\s+(?:about\s+)?["\']?(.+?)["\']?\s*(?:,|—|->|→|then|to|toh)\s*(?:say|reply|bolo|kaho|respond)\s*[:\-]?\s*["\']?(.+?)["\']?\s*(?:\.|$)',
            re.IGNORECASE
        )
        for match in if_asks_pattern.finditer(combined):
            question = match.group(1).strip()
            answer = match.group(2).strip()
            if question and answer and len(answer) < 500:
                triggers = self._generate_trigger_variants(question)
                self._faq_pairs.append((triggers, answer))

        # ── Strategy 3: Extract key-value facts ──
        # Patterns like "Office address: 123 Main St" or "Price: 5000 rupees"
        fact_pattern = re.compile(
            r'(?:address|pata|location|jagah|price|keemat|rate|daam|timing|samay|hours|phone|number|email|website|discount|offer)\s*[:\-]\s*(.+?)(?:\n|$)',
            re.IGNORECASE
        )
        for match in fact_pattern.finditer(combined):
            fact_value = match.group(1).strip()
            # Extract what kind of fact this is from the match
            fact_key = match.group(0).split(':')[0].strip().lower() if ':' in match.group(0) else ""
            if fact_value and fact_key:
                # Generate question triggers for this fact
                triggers = self._fact_triggers(fact_key)
                if triggers:
                    self._faq_pairs.append((triggers, fact_value))

    def _generate_trigger_variants(self, question: str) -> List[str]:
        """Generate normalized trigger phrases from a question."""
        q = question.lower().strip()
        q = re.sub(r'[?.!,;:"\']', '', q)
        q = re.sub(r'\s+', ' ', q).strip()
        variants = [q]

        # Add without common filler words
        for filler in ['kya', 'ki', 'ka', 'ke', 'hai', 'hain', 'please', 'bataiye', 'batao', 'what is', 'what are']:
            cleaned = q.replace(filler, '').strip()
            cleaned = re.sub(r'\s+', ' ', cleaned)
            if cleaned and cleaned != q and len(cleaned) > 3:
                variants.append(cleaned)

        return variants

    def _fact_triggers(self, fact_key: str) -> List[str]:
        """Generate common question triggers for a fact type."""
        triggers_map = {
            'address': ['address', 'pata', 'kahan', 'location', 'jagah', 'office kahan', 'where'],
            'price': ['price', 'keemat', 'rate', 'kitna', 'kitne', 'cost', 'daam', 'charge'],
            'timing': ['timing', 'samay', 'kab', 'hours', 'time', 'open', 'band'],
            'phone': ['phone', 'number', 'contact', 'call'],
            'email': ['email', 'mail'],
            'website': ['website', 'site', 'url'],
            'discount': ['discount', 'offer', 'choot', 'deal'],
        }
        return triggers_map.get(fact_key, [])

    def match(self, user_text: str) -> Optional[str]:
        """
        Try to match user text against cached FAQ pairs.
        Returns the cached reply if confidence > threshold, else None.

        This runs in ~5ms — zero network calls.
        """
        if not user_text or len(user_text.strip()) < 2:
            return None

        normalized = user_text.lower().strip()
        normalized = re.sub(r'[?.!,;:"\']', '', normalized)
        normalized = re.sub(r'\s+', ' ', normalized).strip()

        # ── Check acknowledgment patterns first (regex, instant) ──
        for pattern, reply in self._ack_patterns.items():
            if re.match(pattern, normalized, re.IGNORECASE):
                # None means "let LLM handle" — these are context-dependent
                return reply  # Returns None for acks, which means fall through to LLM

        # ── Check FAQ pairs via fuzzy matching ──
        best_score = 0.0
        best_reply = None

        for triggers, reply in self._faq_pairs:
            for trigger in triggers:
                # Quick length check to skip obviously different strings
                len_ratio = len(normalized) / max(len(trigger), 1)
                if len_ratio < 0.3 or len_ratio > 3.0:
                    continue

                score = SequenceMatcher(None, normalized, trigger).ratio()

                # Boost score if key words overlap
                user_words = set(normalized.split())
                trigger_words = set(trigger.split())
                if trigger_words:
                    word_overlap = len(user_words & trigger_words) / len(trigger_words)
                    score = score * 0.6 + word_overlap * 0.4  # Weighted blend

                if score > best_score:
                    best_score = score
                    best_reply = reply

        if best_score >= self.MATCH_THRESHOLD and best_reply:
            print(f"[FAST-PATH] 🎯 Match score: {best_score:.2f} for '{user_text[:40]}' → '{best_reply[:50]}'")
            return best_reply

        return None
