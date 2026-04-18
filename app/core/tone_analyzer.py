"""
ToneAnalyzer — Real-Time Customer Emotion Detection & Adaptive Response
=======================================================================

Zero-latency NLP module that detects the customer's emotional state from
their transcript and generates dynamic LLM instructions for tone mirroring.

Architecture:
  - Pure keyword/pattern-based detection (~0.1ms per call, no API overhead)
  - Maintains rolling emotional state across the entire conversation
  - Outputs dynamic prompt injection for tone-matched responses
  - Also outputs TTS parameter hints for voice expressiveness tuning

Supported Emotions:
  angry, frustrated, confused, sad, happy, polite, neutral, rushed, skeptical
"""

import re
from typing import Dict, List, Optional, Tuple
from collections import deque


class ToneAnalyzer:
    """
    Production NLP tone analyzer for real-time voice conversations.
    
    Uses multi-language keyword matching (Hindi, Hinglish, English, Gujarati)
    with confidence scoring and rolling state management.
    """

    # ── Emotion Patterns: (regex_pattern, weight) ──
    # Higher weight = stronger signal. Patterns are case-insensitive.
    EMOTION_PATTERNS = {
        "angry": [
            # Hindi/Hinglish anger markers
            (r'\b(kya bakwas|bakwas|pagal|bewakoof|chutiya|gadha|haramkhor)\b', 1.0),
            (r'\b(gussa|naraz|irritate|frustrate|pareshan|tang)\b', 0.8),
            (r'\b(band karo|chup|bas karo|ruk|hat|chhod do|mat karo)\b', 0.7),
            (r'\b(kya hai ye|ye kya|kaisa service|bekar|wahiyat|ghatiya)\b', 0.8),
            (r'\b(complaint|complain|shikayat|manager se baat|senior)\b', 0.6),
            # English anger
            (r'\b(terrible|horrible|worst|useless|pathetic|disgusting|ridiculous)\b', 0.9),
            (r'\b(shut up|stop calling|dont call|fed up|sick of|enough)\b', 0.8),
            (r'\b(scam|fraud|cheat|loot|dhoka)\b', 0.9),
            # Gujarati anger
            (r'\b(bakvaas|gadhedu|pagal|shu chhe aa|band kar)\b', 0.8),
        ],
        "frustrated": [
            (r'\b(phir se|fir se|dobara|baar baar|kitni baar|har baar)\b', 0.8),
            (r'\b(samajh nahi|pata nahi|kuch nahi hota|koi fayda nahi)\b', 0.7),
            (r'\b(thak gaya|thak gayi|bore|pak gaya|tang aa gaya)\b', 0.8),
            (r'\b(already told|told you|again and again|how many times)\b', 0.8),
            (r'\b(kab tak|kitna time|jaldi karo|der ho rahi)\b', 0.6),
        ],
        "confused": [
            (r'\b(samajh nahi aaya|samajh nahi aa raha|kya matlab|matlab kya)\b', 0.8),
            (r'\b(confused|confuse|clear nahi|pata nahi|sure nahi)\b', 0.7),
            (r'\b(kaise|kaun sa|konsa|kya karu|what do you mean)\b', 0.5),
            (r'\b(dobara batao|phir se batao|repeat|explain)\b', 0.6),
            (r'\b(hain\?|kya\?|acha\?|really\?|sachi\?)\b', 0.4),
        ],
        "sad": [
            (r'\b(dukhi|udaas|pareshaan|mushkil|takleef|problem)\b', 0.7),
            (r'\b(afford nahi|paisa nahi|paise nahi|mehnga|expensive)\b', 0.6),
            (r'\b(koi madad nahi|helpless|majboor|lachar)\b', 0.8),
            (r'\b(unfortunately|sadly|sorry to say)\b', 0.5),
        ],
        "happy": [
            (r'\b(bahut accha|bahut badhiya|excellent|wonderful|amazing|great)\b', 0.8),
            (r'\b(khush|khushi|happy|glad|nice|perfect|superb)\b', 0.7),
            (r'\b(thank you|shukriya|dhanyavaad|thanks a lot|meherbani)\b', 0.6),
            (r'\b(haan bilkul|zaroor|of course|definitely|pakka)\b', 0.5),
            # Gujarati happy
            (r'\b(saru|barabar|maja|saras|bahu saru)\b', 0.7),
        ],
        "polite": [
            (r'\b(please|kripya|meherbani|request|anurodh)\b', 0.6),
            (r'\b(ji|ji haan|haan ji|aapka|aapke|aap)\b', 0.4),
            (r'\b(could you|would you|can you please|if possible)\b', 0.5),
            (r'\b(sorry|maaf|kshama|galti)\b', 0.5),
        ],
        "rushed": [
            (r'\b(jaldi|quick|fast|hurry|abhi|turant|fataafat)\b', 0.7),
            (r'\b(time nahi|busy|meeting|kaam hai|baad mein)\b', 0.6),
            (r'\b(haan haan|ok ok|theek theek|bas bas|jaldi bolo)\b', 0.7),
            (r'\b(short mein|briefly|quickly)\b', 0.6),
        ],
        "skeptical": [
            (r'\b(sach mein|pakka|sure|guarantee|promise|bharosa)\b', 0.5),
            (r'\b(jhooth|fake|scam|trust nahi|vishwas nahi)\b', 0.8),
            (r'\b(really\?|seriously\?|are you sure|is this real)\b', 0.6),
            (r'\b(proof|saboot|dikhao|batao kaise)\b', 0.6),
            (r'\b(pehle bhi suna|ye sab suna hai|marketing)\b', 0.7),
        ],
    }

    # ── Dynamic LLM Tone Instructions ──
    TONE_INSTRUCTIONS = {
        "angry": (
            "🔴 CUSTOMER IS ANGRY/UPSET. CRITICAL ADAPTATION REQUIRED:\n"
            "- FIRST: Validate their frustration. Say 'Main samajh sakta hun aap pareshan hain' or similar.\n"
            "- Be extremely empathetic, patient, and calm. NEVER match their anger.\n"
            "- Keep your response SHORT — angry customers don't want long explanations.\n"
            "- Offer an immediate solution or escalation path.\n"
            "- Use a soft, reassuring tone. No cheerfulness — it will feel dismissive.\n"
            "- If they're using harsh language, do NOT react. Stay professional and caring.\n"
        ),
        "frustrated": (
            "🟠 CUSTOMER IS FRUSTRATED. TONE ADAPTATION:\n"
            "- Acknowledge their frustration directly: 'Main jaanta hun ye pareshan karne wali baat hai.'\n"
            "- Be concise and solution-oriented. Don't repeat information they already know.\n"
            "- Show urgency — they want this resolved NOW, not later.\n"
            "- If they say 'phir se', apologize and give a direct answer.\n"
        ),
        "confused": (
            "🟡 CUSTOMER IS CONFUSED. TONE ADAPTATION:\n"
            "- Speak SLOWLY and CLEARLY. Use simple words.\n"
            "- Break complex information into small steps.\n"
            "- After explaining, ask 'Kya aapko samajh aa gaya?' to confirm.\n"
            "- Use examples and analogies to make things clearer.\n"
            "- Be patient — never make them feel stupid.\n"
        ),
        "sad": (
            "🔵 CUSTOMER SOUNDS SAD/TROUBLED. TONE ADAPTATION:\n"
            "- Be warm and compassionate. Show genuine concern.\n"
            "- If they mention financial difficulties, be sensitive and understanding.\n"
            "- Offer flexible options if available.\n"
            "- Use a gentle, caring tone. Don't be overly cheerful.\n"
        ),
        "happy": (
            "🟢 CUSTOMER IS HAPPY/POSITIVE. TONE ADAPTATION:\n"
            "- Match their energy! Be enthusiastic and warm.\n"
            "- Use positive language and build on their enthusiasm.\n"
            "- This is the best time to offer upgrades or additional benefits.\n"
            "- Thank them for their positivity.\n"
        ),
        "polite": (
            "🟢 CUSTOMER IS BEING POLITE. TONE ADAPTATION:\n"
            "- Mirror their politeness with equal courtesy.\n"
            "- Use respectful language: 'ji', 'aapka', formal Hindi.\n"
            "- Take extra care to be thorough and helpful.\n"
        ),
        "rushed": (
            "⚡ CUSTOMER IS IN A HURRY. TONE ADAPTATION:\n"
            "- Be EXTREMELY concise. Get to the point immediately.\n"
            "- Skip pleasantries and unnecessary filler.\n"
            "- Give the most important information first.\n"
            "- If they say 'busy', offer to call back later.\n"
            "- Maximum 1-2 sentences per reply.\n"
        ),
        "skeptical": (
            "🟤 CUSTOMER IS SKEPTICAL/DOUBTFUL. TONE ADAPTATION:\n"
            "- Build trust with facts and specifics, not vague promises.\n"
            "- Offer proof points: 'Aap humari website pe check kar sakte hain.'\n"
            "- Be transparent about terms and conditions.\n"
            "- Don't oversell — it will increase their suspicion.\n"
            "- Acknowledge their caution: 'Aapka savdhan rehna sahi hai.'\n"
        ),
        "neutral": "",  # No special instruction needed
    }

    # ── TTS Voice Parameter Hints ──
    TTS_HINTS = {
        "angry":      {"stability": 0.90, "style": 0.0},   # Very calm, stable voice
        "frustrated": {"stability": 0.88, "style": 0.0},   # Calm and steady
        "confused":   {"stability": 0.85, "style": 0.05},  # Clear and reassuring
        "sad":        {"stability": 0.80, "style": 0.10},  # Warm, gentle
        "happy":      {"stability": 0.75, "style": 0.15},  # Slightly expressive
        "polite":     {"stability": 0.85, "style": 0.05},  # Professional
        "rushed":     {"stability": 0.85, "style": 0.0},   # Direct and clear
        "skeptical":  {"stability": 0.88, "style": 0.0},   # Trustworthy, steady
        "neutral":    {"stability": 0.85, "style": 0.0},   # Default
    }

    def __init__(self, window_size: int = 5):
        """
        Args:
            window_size: Number of recent messages to consider for rolling emotion state
        """
        self._history: deque = deque(maxlen=window_size)
        self._current_emotion: str = "neutral"
        self._current_confidence: float = 0.0
        self._emotion_counts: Dict[str, int] = {}

    @property
    def current_emotion(self) -> str:
        return self._current_emotion

    @property
    def current_confidence(self) -> float:
        return self._current_confidence

    def analyze(self, text: str) -> Tuple[str, float]:
        """
        Analyze a customer utterance and update the rolling emotional state.
        
        Returns:
            (emotion, confidence) — e.g., ("angry", 0.85)
        
        Performance: ~0.1ms per call (pure regex, no API)
        """
        if not text or len(text.strip()) < 2:
            return self._current_emotion, self._current_confidence

        text_lower = text.lower().strip()
        scores: Dict[str, float] = {}

        for emotion, patterns in self.EMOTION_PATTERNS.items():
            total_score = 0.0
            matches = 0
            for pattern, weight in patterns:
                if re.search(pattern, text_lower, re.IGNORECASE):
                    total_score += weight
                    matches += 1
            if matches > 0:
                # Normalize: more matches = higher confidence, cap at 1.0
                scores[emotion] = min(1.0, total_score / max(matches, 1) * min(matches, 3) / 2)

        # Determine dominant emotion for this utterance
        if scores:
            dominant = max(scores, key=scores.get)
            confidence = scores[dominant]
        else:
            dominant = "neutral"
            confidence = 0.3

        # Add to rolling history
        self._history.append((dominant, confidence))

        # Calculate rolling emotion state (weighted recent messages more)
        emotion_weights: Dict[str, float] = {}
        for i, (emo, conf) in enumerate(self._history):
            recency_weight = (i + 1) / len(self._history)  # More recent = higher weight
            emotion_weights[emo] = emotion_weights.get(emo, 0) + conf * recency_weight

        # Find dominant rolling emotion
        if emotion_weights:
            rolling_dominant = max(emotion_weights, key=emotion_weights.get)
            rolling_confidence = min(1.0, emotion_weights[rolling_dominant] / len(self._history))
            
            # Only switch emotion if new one is significantly stronger
            if rolling_dominant != self._current_emotion:
                if rolling_confidence > 0.4:  # Minimum threshold to switch
                    self._current_emotion = rolling_dominant
                    self._current_confidence = rolling_confidence
            else:
                self._current_confidence = rolling_confidence
        
        if dominant != "neutral":
            print(f"[NLP TONE] 🎭 Detected: {dominant} ({confidence:.0%}) | Rolling: {self._current_emotion} ({self._current_confidence:.0%})")

        return self._current_emotion, self._current_confidence

    def get_tone_instruction(self) -> str:
        """Get the dynamic LLM prompt injection for the current emotional state."""
        if self._current_confidence < 0.35:
            return ""  # Not confident enough to inject tone instructions
        return self.TONE_INSTRUCTIONS.get(self._current_emotion, "")

    def get_tts_hints(self) -> Dict[str, float]:
        """Get TTS voice parameter adjustments for the current emotional state."""
        return self.TTS_HINTS.get(self._current_emotion, self.TTS_HINTS["neutral"])

    def reset(self):
        """Reset analyzer state (call this at call end)."""
        self._history.clear()
        self._current_emotion = "neutral"
        self._current_confidence = 0.0
        self._emotion_counts.clear()
