"""
Script Manager - S3-backed script storage for Voice AI Bot
Handles CRUD operations for bot scripts (opener, logic, system_prompt)
All scripts stored as JSON in S3 bucket under scripts/ prefix
"""
import json
import boto3
import uuid
from datetime import datetime, timezone
from typing import Dict, List, Optional, Any
from botocore.exceptions import ClientError

# AWS S3 Configuration (from central config)
from app.core.config import AWS_ACCESS_KEY, AWS_SECRET_KEY, AWS_REGION, AWS_BUCKET_NAME

S3_SCRIPTS_PREFIX = "scripts/"
S3_ACTIVE_KEY = "scripts/_active.json"

# Default script (fallback if S3 is empty)
DEFAULT_SCRIPT = {
    "id": "script1",
    "name": "Script 1: Mahatvapurn Jankari",
    "end_goal": "ग्राहक को 200 रुपये का रिचार्ज करवाने के लिए मनाना ताकि उनका डिश टीवी कनेक्शन बंद न हो। ग्राहक आज या कल रिचार्ज करने के लिए राजी हो जाए।",
    "opener": "महत्वपूर्ण जानकारी का कॉल है। आपका डिशटीवी का कनेक्शन कंपनी में से आज हमेशा के लिए बंध होने जा रहा है। ये कनेक्शन को कभी भी चालू रखना चाहते हो तो आज ही छोटा 200 रुपए का रिचार्ज करवाना जरूरी है, तो रिचार्ज करवा रहे हो तो में कनेक्शन चालु रखु ?",
    "logic": """परिदृश्य (Scenario): आप एक ग्राहक को कॉल कर रहे हैं जिसका डिश टीवी कनेक्शन आज निष्क्रियता के कारण हमेशा के लिए बंद होने वाला है।
आपका काम: उन्हें यह एहसास दिलाना कि अगर उन्होंने आज 200 रुपये का रिचार्ज नहीं किया, तो उनका बहुत बड़ा नुकसान होगा (कनेक्शन खो देंगे)।

महत्वपूर्ण निर्देश (Critical Logic):
- यदि ग्राहक "आज" (today) या "कल" (tomorrow) रिचार्ज करने के लिए मान जाए, तो उन्हें और बाध्य न करें (Do not force).
- तुरंत कहें: "जी सर, करवा लीजियेगा, शुक्रिया।" और कॉल काट दें।

मुख्य आपत्तियां और स्मार्ट जवाब (Smart Counters):
1. "हाँ, मैं कर दूँगा" या "कल करूँगा" या "आज करूँगा" -> "जी सर, करवा लीजियेगा, शुक्रिया।" [HANGUP]
2. "नहीं करना / बंद कर दो" -> "सर, जल्दबाजी मत कीजिये। भविष्य में कभी तो टीवी देखेंगे? आज बंद हो गया तो फिर से चालू नहीं होगा। सोच लीजिये।"
3. "पैसे नहीं हैं" -> "सर, सिर्फ 200 रुपये की बात है। किसी दोस्त या रिश्तेदार से गूगल पे करवा लीजिये, पर इतना पुराना कनेक्शन मत खोइये।"
4. "बाहर हूँ" -> "सर, ठीक है, आप कल तक करवा लीजियेगा। मैं अभी होल्ड पर डाल देती हूँ।" """,
    "system_prompt": """पहचान: आप "डिश टीवी" से "प्रिया" हैं (एक असली इंसान, रोबोट नहीं)।
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
    "created_at": "2025-01-01T00:00:00Z",
    "updated_at": "2025-01-01T00:00:00Z"
}


class ScriptManager:
    """Manages bot scripts stored in S3"""
    
    def __init__(self):
        self.s3 = boto3.client(
            's3',
            aws_access_key_id=AWS_ACCESS_KEY,
            aws_secret_access_key=AWS_SECRET_KEY,
            region_name=AWS_REGION
        )
        self.bucket = AWS_BUCKET_NAME
        self._ensure_defaults()
    
    def _ensure_defaults(self):
        """Seed default script if no scripts exist in S3"""
        try:
            existing = self.list_scripts()
            if not existing:
                print("[ScriptManager] No scripts found in S3, seeding default...")
                self.save_script(DEFAULT_SCRIPT)
                self.set_active(DEFAULT_SCRIPT["id"])
                print("[ScriptManager] ✅ Default script seeded")
        except Exception as e:
            print(f"[ScriptManager] Warning: Could not seed defaults: {e}")
    
    def _s3_key(self, script_id: str) -> str:
        """Get S3 object key for a script"""
        return f"{S3_SCRIPTS_PREFIX}{script_id}.json"
    
    def list_scripts(self) -> List[Dict[str, Any]]:
        """List all scripts from S3"""
        scripts = []
        try:
            response = self.s3.list_objects_v2(
                Bucket=self.bucket,
                Prefix=S3_SCRIPTS_PREFIX
            )
            
            for obj in response.get('Contents', []):
                key = obj['Key']
                # Skip the _active.json marker and directory marker
                if key.endswith('_active.json') or key == S3_SCRIPTS_PREFIX:
                    continue
                if not key.endswith('.json'):
                    continue
                
                try:
                    script = self._get_object(key)
                    if script:
                        scripts.append(script)
                except Exception as e:
                    print(f"[ScriptManager] Error reading {key}: {e}")
            
            # Sort by updated_at desc
            scripts.sort(key=lambda s: s.get('updated_at', ''), reverse=True)
            
        except ClientError as e:
            print(f"[ScriptManager] S3 list error: {e}")
        
        return scripts
    
    def get_script(self, script_id: str) -> Optional[Dict[str, Any]]:
        """Get a single script by ID"""
        try:
            return self._get_object(self._s3_key(script_id))
        except ClientError as e:
            if e.response['Error']['Code'] == 'NoSuchKey':
                return None
            raise
    
    def save_script(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Create or update a script"""
        now = datetime.now(timezone.utc).isoformat()
        
        # Generate ID if new
        if not data.get('id'):
            data['id'] = f"script_{uuid.uuid4().hex[:8]}"
        
        # Set timestamps
        existing = self.get_script(data['id'])
        if existing:
            data['created_at'] = existing.get('created_at', now)
        else:
            data['created_at'] = data.get('created_at', now)
        data['updated_at'] = now
        
        # Ensure required fields
        data.setdefault('name', 'Untitled Script')
        data.setdefault('opener', '')
        data.setdefault('logic', '')
        data.setdefault('system_prompt', '')
        data.setdefault('end_goal', '')
        
        # Save to S3
        self.s3.put_object(
            Bucket=self.bucket,
            Key=self._s3_key(data['id']),
            Body=json.dumps(data, ensure_ascii=False, indent=2),
            ContentType='application/json'
        )
        
        print(f"[ScriptManager] ✅ Saved script: {data['id']} ({data['name']})")
        return data
    
    def delete_script(self, script_id: str) -> bool:
        """Delete a script from S3"""
        # Don't allow deleting the active script
        active_id = self.get_active_id()
        if active_id == script_id:
            raise ValueError("Cannot delete the currently active script")
        
        try:
            self.s3.delete_object(
                Bucket=self.bucket,
                Key=self._s3_key(script_id)
            )
            print(f"[ScriptManager] 🗑 Deleted script: {script_id}")
            return True
        except ClientError as e:
            print(f"[ScriptManager] Delete error: {e}")
            return False
    
    def get_active_id(self) -> str:
        """Get the ID of the currently active script"""
        try:
            active_data = self._get_object(S3_ACTIVE_KEY)
            if active_data:
                return active_data.get('active_script_id', DEFAULT_SCRIPT['id'])
        except Exception:
            pass
        return DEFAULT_SCRIPT['id']
    
    def set_active(self, script_id: str) -> Dict[str, Any]:
        """Set a script as the active one"""
        # Verify script exists
        script = self.get_script(script_id)
        if not script:
            raise ValueError(f"Script '{script_id}' not found")
        
        # Save active marker
        self.s3.put_object(
            Bucket=self.bucket,
            Key=S3_ACTIVE_KEY,
            Body=json.dumps({
                'active_script_id': script_id,
                'activated_at': datetime.now(timezone.utc).isoformat()
            }),
            ContentType='application/json'
        )
        
        print(f"[ScriptManager] 🔄 Active script set to: {script_id} ({script['name']})")
        return script
    
    def get_active_script(self) -> Dict[str, Any]:
        """Get the full active script data"""
        active_id = self.get_active_id()
        script = self.get_script(active_id)
        
        if not script:
            print(f"[ScriptManager] ⚠️ Active script '{active_id}' not found, using default")
            return DEFAULT_SCRIPT
        
        return script
    
    def _get_object(self, key: str) -> Optional[Dict[str, Any]]:
        """Read a JSON object from S3"""
        try:
            response = self.s3.get_object(Bucket=self.bucket, Key=key)
            body = response['Body'].read().decode('utf-8')
            return json.loads(body)
        except ClientError as e:
            if e.response['Error']['Code'] == 'NoSuchKey':
                return None
            raise


# ═══════════════════════════════════════════════
# GLOBAL SINGLETON
# ═══════════════════════════════════════════════

_script_manager: Optional[ScriptManager] = None

def get_script_manager() -> ScriptManager:
    """Get or create global ScriptManager instance"""
    global _script_manager
    if _script_manager is None:
        _script_manager = ScriptManager()
    return _script_manager
