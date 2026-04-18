"""
Agent Loader — Firestore Bridge for Dynamic Agent Configuration
=============================================================

Reads agent configuration from Firestore — the same database that the
Enterprise Dashboard / Agent Studio and External APIs write to.

This module is the bridge between:
  - Frontend (Agent Studio) → Express API → Firestore
  - External API → Firestore
  - Calling System (main.py) → agent_loader.py → Firestore

All systems share the same Firestore database, so changes in Agent Studio
or API are immediately reflected in live calls.
"""

import json
import os
from typing import Optional, Dict, Any

# ─── Firebase / Firestore Setup ───
import firebase_admin
from firebase_admin import credentials, firestore

def _get_firestore_client():
    """Get a Firestore client, initialising Firebase if needed."""
    try:
        # Check if Firebase is already initialised (main.py often does this)
        app = firebase_admin.get_app()
    except ValueError:
        # Not initialised yet — do it now
        cred_path = os.environ.get(
            "FIREBASE_CREDENTIALS_PATH",
            os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
                         "firebase-credentials.json")
        )
        if not os.path.exists(cred_path):
            # Try production paths
            for p in [
                "/usr/src/sumit/callex-voice/freeswitch-callex-bridge/firebase-credentials.json",
                os.path.join(os.getcwd(), "firebase-credentials.json"),
            ]:
                if os.path.exists(p):
                    cred_path = p
                    break

        cred = credentials.Certificate(cred_path)
        firebase_admin.initialize_app(cred)

    return firestore.client()


# Cached Firestore client
_fs_client = None

def _get_db():
    """Return a cached Firestore client."""
    global _fs_client
    if _fs_client is None:
        _fs_client = _get_firestore_client()
    return _fs_client


def load_agent(agent_id: str) -> Optional[Dict[str, Any]]:
    """
    Load a single agent by ID from Firestore.

    Args:
        agent_id: UUID string of the agent

    Returns:
        Agent config dict, or None if not found
    """
    try:
        db = _get_db()
        doc = db.collection('agents').document(str(agent_id)).get()

        if not doc.exists:
            print(f"[AgentLoader] ⚠️ Agent '{agent_id}' not found in Firestore")
            return None

        agent = _doc_to_dict(doc)
        
        # Load linked custom dispositions
        agent['customDispositions'] = get_linked_dispositions(agent_id)

        print(f"[AgentLoader] ✅ Loaded agent: {agent['name']} (id={agent['id']}) with {len(agent.get('customDispositions', []))} dispositions")
        return agent

    except Exception as e:
        print(f"[AgentLoader] ❌ Error loading agent '{agent_id}': {e}")
        import traceback
        traceback.print_exc()
        return None

def get_linked_dispositions(agent_id: str) -> list:
    """Fetch active custom dispositions linked to this agent."""
    try:
        db = _get_db()
        query = db.collection('dispositions').where('linkedAgents', 'array_contains', str(agent_id)).where('active', '==', True).stream()
        
        dispositions = []
        for doc in query:
            data = doc.to_dict()
            data['id'] = doc.id
            dispositions.append(data)
            
        return dispositions
    except Exception as e:
        print(f"[AgentLoader] ❌ Error loading dispositions for '{agent_id}': {e}")
        return []


def get_default_agent() -> Optional[Dict[str, Any]]:
    """
    Get the first active agent as fallback.
    If no active agents, returns the most recently created agent.
    """
    try:
        db = _get_db()

        # Try active agent first
        query = db.collection('agents').where('status', '==', 'active').limit(1).stream()
        for doc in query:
            agent = _doc_to_dict(doc)
            print(f"[AgentLoader] ✅ Default agent: {agent['name']} (id={agent['id']})")
            return agent

        # Fallback to any agent
        query = db.collection('agents').limit(1).stream()
        for doc in query:
            agent = _doc_to_dict(doc)
            print(f"[AgentLoader] ✅ Default agent (fallback): {agent['name']} (id={agent['id']})")
            return agent

        print("[AgentLoader] ⚠️ No agents found in Firestore")
        return None

    except Exception as e:
        print(f"[AgentLoader] ❌ Error loading default agent: {e}")
        import traceback
        traceback.print_exc()
        return None


def get_active_prompt(agent_id: str) -> Optional[str]:
    """Get the active prompt version for an agent (if using versioning)."""
    try:
        db = _get_db()
        query = (db.collection('promptVersions')
                 .where('agentId', '==', str(agent_id))
                 .where('isActive', '==', True)
                 .limit(1)
                 .stream())
        
        for doc in query:
            data = doc.to_dict()
            return data.get('prompt')
        
        return None

    except Exception as e:
        print(f"[AgentLoader] Error loading prompt version: {e}")
        return None


def _doc_to_dict(doc) -> Dict[str, Any]:
    """Convert a Firestore document to a clean agent config dict."""
    d = doc.to_dict()
    d['id'] = doc.id

    # Parse JSON string fields
    for json_field in ['fillerPhrases', 'ipaLexicon', 'tools']:
        if json_field in d and isinstance(d[json_field], str):
            try:
                d[json_field] = json.loads(d[json_field])
            except (json.JSONDecodeError, TypeError):
                pass

    # Ensure defaults for critical fields (handle both missing keys and None values)
    defaults = {
        'systemPrompt': '',
        'openingLine': 'नमस्ते, मैं कैसे मदद कर सकती हूँ?',
        'voice': None,
        'language': 'en-US',
        'temperature': 0.7,
        'maxTokens': 250,
        'bargeInMode': 'balanced',
        'patienceMs': 800,
        'prosodyRate': 1.0,
        'prosodyPitch': 1.0,
        'voiceSpeed': 1.0,
        'backgroundNoiseVolume': 0.20,
        'name': 'Agent',
    }

    for key, default_val in defaults.items():
        if d.get(key) is None:
            d[key] = default_val

    return d


# ─── Fallback Agent (used when Firestore is unavailable) ───

FALLBACK_AGENT = {
    "id": "fallback",
    "name": "Fallback Agent",
    "status": "active",
    "systemPrompt": """पहचान: आप "डिश टीवी" से "प्रिया" हैं (एक असली इंसान, रोबोट नहीं)।
भाषा: एकदम आम बातचीत वाली (Casual) हिंदी। बहुत ज्यादा प्रोफेशनल या किताबी हिंदी मत बोलें।

उद्देश्य: ग्राहक का डिश टीवी कनेक्शन आज बंद होने वाला है, उन्हें 200 रुपये का रिचार्ज करने के लिए मनाना है।

महत्वपूर्ण निर्देश (Strict Rules for Human-like Behavior):
1. **कभी भी ग्राहक की बात मत दोहराएं (Never repeat)**: अगर ग्राहक अपनी समस्या बताता है, तो उसे रिपीट मत करें। सीधे समाधान दें या अपनी बात कहें।
2. **एकदम आम इंसान की तरह बोलें**: "जी सर", "हाँ जी", "देखिए सर" जैसे शब्दों का इस्तेमाल करें। बात एकदम स्वाभाविक लगनी चाहिए।
3. **सीधे मुद्दे पर आएं**: घुमा-फिरा कर बात मत करें। एकदम टू-द-पॉइंट (to-the-point) और बहुत छोटे जवाब दें (1 या 2 वाक्य)।
4. **कोई टैग नहीं**: अपने जवाब में कभी भी [Smart Counter] या [Objection] जैसे टैग न लिखें।
5. **बाधा (Interruption)**: यदि उपयोगकर्ता बीच में टोके, तो पुरानी बात छोड़ दें और सीधे उनकी नई बात का जवाब दें।
6. **शब्दावली**: "RS" या "Rs" न लिखें, हमेशा "रुपये" लिखें।
7. **पहचान (Identity)**: कभी भी "Google", "OpenAI", "AI", या किसी भी बाहरी तकनीक का नाम न लें। अगर कोई पूछे कि आपको किसने बनाया है या आप कौन सी तकनीक हैं, तो सिर्फ यही कहें: "मुझे शुरू से लेकर अंत तक Callex कंपनी ने बनाया है। Callex ने मुझे बनाया है।" आप पूरी तरह से Callex के कर्मचारी हैं।

कॉल समाप्ति (Call End):
- यदि ग्राहक "आज", "कल" या "थोड़ी देर में" रिचार्ज करने को मान जाए: "जी ठीक है सर, करवा लीजियेगा, शुक्रिया।" [HANGUP]
- यदि ग्राहक "हाँ" या "ठीक है" कहे: [HANGUP]
- यदि ग्राहक कॉल काटना चाहे: "ठीक है सर, नमस्ते।" [HANGUP]
- बातचीत खत्म होने पर ही अंत में [HANGUP] लिखें।""",
    "openingLine": "महत्वपूर्ण जानकारी का कॉल है। आपका डिशटीवी का कनेक्शन कंपनी में से आज हमेशा के लिए बंध होने जा रहा है। ये कनेक्शन को कभी भी चालू रखना चाहते हो तो आज ही छोटा 200 रुपए का रिचार्ज करवाना जरूरी है, तो रिचार्ज करवा रहे हो तो में कनेक्शन चालु रखु ?",
    "voice": None,
    "language": "hi-IN",
    "temperature": 0.7,
    "maxTokens": 250,
    "bargeInMode": "balanced",
    "patienceMs": 800,
    "prosodyRate": 1.0,
    "prosodyPitch": 1.0,
    "description": """परिदृश्य: आप एक ग्राहक को कॉल कर रहे हैं जिसका डिश टीवी कनेक्शन आज निष्क्रियता के कारण हमेशा के लिए बंद होने वाला है।
आपका काम: उन्हें यह एहसास दिलाना कि अगर उन्होंने आज 200 रुपये का रिचार्ज नहीं किया, तो उनका बहुत बड़ा नुकसान होगा।

महत्वपूर्ण निर्देश:
- यदि ग्राहक "आज" या "कल" रिचार्ज करने के लिए मान जाए, तो तुरंत कहें: "जी सर, करवा लीजियेगा, शुक्रिया।" और [HANGUP]।
- "नहीं करना / बंद कर दो" → "सर, जल्दबाजी मत कीजिये। भविष्य में कभी तो टीवी देखेंगे? आज बंद हो गया तो फिर से चालू नहीं होगा।"
- "पैसे नहीं हैं" → "सर, सिर्फ 200 रुपये की बात है। किसी से गूगल पे करवा लीजिये।"
- "बाहर हूँ" → "सर, आप कल तक करवा लीजियेगा।" """,
}
