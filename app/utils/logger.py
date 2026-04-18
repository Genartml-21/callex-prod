"""
Logging Utilities for Voice Bot
Handles call tracking and outcome detection for test.py
"""
import threading
from datetime import datetime, timedelta
import os
from typing import Optional, List, Dict
from app.core.database import get_db_session, Call, CallOutcome, Recording

# IST Offset
IST_OFFSET = timedelta(hours=5, minutes=30)

def get_ist_time():
    return datetime.utcnow() + IST_OFFSET

class CallTracker:
    """Tracks active calls and their database records (Thread-Safe)"""
    
    def __init__(self):
        self._lock = threading.Lock()
        self.active_calls = {}  # Map websocket_id/uuid -> call_data
    
    def start_call(self, call_uuid: str, phone_number: Optional[str] = None) -> int:
        """Create database record for new call"""
        db = get_db_session()
        try:
            # Generate UUID if not provided
            if not call_uuid:
                import uuid
                call_uuid = str(uuid.uuid4())
            
            call = Call(
                call_uuid=call_uuid,
                phone_number=phone_number,
                start_time=get_ist_time(),
                status="in_progress"
            )
            db.add(call)
            db.commit()
            db.refresh(call)
            
            with self._lock:
                self.active_calls[call_uuid] = {
                    "call_id": call.id,
                    "call_uuid": call_uuid,
                    "start_time": call.start_time,
                    "conversation": []
                }
            
            print(f"[DB] Call started: {call_uuid} (ID: {call.id})")
            return call.id
            
        except Exception as e:
            print(f"[DB Error] Failed to create call record: {e}")
            db.rollback()
            return None
        finally:
            db.close()
    
    def log_message(self, call_uuid: str, role: str, text: str):
        """Log a message to the conversation history"""
        with self._lock:
            if call_uuid and call_uuid in self.active_calls:
                self.active_calls[call_uuid]["conversation"].append({
                    "role": role,
                    "text": text,
                    "timestamp": get_ist_time()
                })

    def end_call(self, call_uuid: str, status: str = "completed", recording_filename: str = None, outcome_override: dict = None):
        """Update database record when call ends"""
        call_data = None
        with self._lock:
            if call_uuid in self.active_calls:
                call_data = self.active_calls[call_uuid]
                del self.active_calls[call_uuid]
        
        if not call_data:
            return
        
        db = get_db_session()
        try:
            call = db.query(Call).filter(Call.id == call_data["call_id"]).first()
            if call:
                end_time = get_ist_time()
                duration = (end_time - call_data["start_time"]).total_seconds()
                
                call.end_time = end_time
                call.duration_seconds = duration
                call.status = status
                
                # Detect outcome from conversation (or use override)
                outcome = outcome_override if outcome_override else self._detect_outcome(call_data["conversation"])
                
                if outcome:
                    # Build transcript from conversation
                    transcript_text = ""
                    for msg in call_data.get("conversation", []):
                        role = "User" if msg["role"] == "user" else "Bot"
                        transcript_text += f"{role}: {msg['text']}\n"
                    
                    # Parse commitment_date from string to datetime.date to prevent SQLite TypeError
                    parsed_commitment_date = outcome.get("commitment_date")
                    if parsed_commitment_date and isinstance(parsed_commitment_date, str):
                        try:
                            from datetime import datetime
                            # Only parse the first 10 chars (YYYY-MM-DD)
                            parsed_commitment_date = datetime.strptime(parsed_commitment_date[:10], "%Y-%m-%d").date()
                        except ValueError:
                            parsed_commitment_date = None

                    call_outcome = CallOutcome(
                        call_id=call.id,
                        customer_agreed=outcome.get("agreed"),
                        commitment_date=parsed_commitment_date,
                        unclear_response=outcome.get("unclear") or outcome.get("unclear_response"),
                        disposition=outcome.get("disposition"),
                        notes=outcome.get("notes"),
                        transcript=transcript_text.strip() if transcript_text else None,
                        summary=outcome.get("summary"),
                        sentiment=outcome.get("sentiment"),
                        structured_data=outcome.get("structuredData"),
                    )
                    db.add(call_outcome)
                
                # Add recording record if file exists
                if recording_filename:
                    size = 0
                    # Only check size for local files
                    if recording_filename and not recording_filename.startswith("http") and os.path.exists(recording_filename):
                        try:
                            size = os.path.getsize(recording_filename)
                        except:
                            pass
                            
                    recording = Recording(
                        call_id=call.id,
                        file_path=recording_filename,
                        file_size_bytes=size
                    )
                    db.add(recording)

                db.commit()
                print(f"[DB] Call ended: {call.call_uuid} ({duration:.1f}s, Agreed={outcome.get('agreed') if outcome else None})")
            
        except Exception as e:
            print(f"[DB Error] Failed to end call: {e}")
            db.rollback()
        finally:
            db.close()
    
    def _detect_outcome(self, conversation: list) -> Optional[dict]:
        """Analyze conversation to detect customer agreement"""
        if not conversation:
            return None
        
        # Keywords for detection (Hindi/English mixed)
        agreement_keywords = [
            "हाँ", "हा", "han", "haan", "yes", "ji haan", "ji han",
            "ठीक है", "thik hai", "theek hai", "okay", "ok",
            "कर दूंगा", "kar dunga", "karunga", "billing",
            "आज", "aaj", "today", 
            "कल", "kal", "tomorrow",
            "शाम तक", "shaam tak", "evening",
            "रात तक", "raat tak", "night",
            "अभी", "abhi", "now"
        ]
        
        decline_keywords = [
            "नहीं", "nahi", "no", 
            "बंद कर दो", "band kar do", "band kardo",
            "मत करो", "mat karo", 
            "नहीं करना", "nahi karna",
            "hata do", "cancel"
        ]
        
        # Analyze ONLY user messages
        user_messages = [msg["text"] for msg in conversation if msg["role"] == "user"]
        
        # If conversation is very short, it's likely unclear/hangup
        if not user_messages:
            return None

        # Look at the last few interactions as they are most decisive
        recent_messages = user_messages[-3:]
        
        agreed = False
        declined = False
        unclear = False
        commitment_date = None
        notes = ""
        
        # Helper to normalize text
        def normalize(text):
            return text.lower().strip()

        for msg in recent_messages:
            msg_lower = normalize(msg)
            
            # Check for decline first (explicit refusal)
            if any(k in msg_lower for k in decline_keywords):
                declined = True
                notes = "Customer declined recharge"
                # If they said no, we stop looking - refusal overrides previous weak agreements
                agreed = False 
                break
            
            # Check for agreement
            if any(k in msg_lower for k in agreement_keywords):
                agreed = True
                
                # Check for commitment timing
                if any(k in msg_lower for k in ["कल", "kal", "tomorrow"]):
                    commitment_date = get_ist_time().date() + timedelta(days=1)
                    notes = "Customer agreed to recharge tomorrow"
                elif any(k in msg_lower for k in ["आज", "aaj", "today", "अभी", "abhi", "now"]):
                    commitment_date = get_ist_time().date()
                    notes = "Customer agreed to recharge today"
                else:
                    notes = "Customer agreed to recharge"
                
                # Keep looking in case they decline in a later message, but for now we have a match
        
        # Logic refinement: If neither explicitly agreed nor declined, it's unclear
        if not agreed and not declined:
            unclear = True
            notes = "Response unclear or call dropped"
        
        return {
            "agreed": agreed if not unclear else None,
            "commitment_date": commitment_date,
            "unclear": unclear,
            "notes": notes
        }

# Global tracker instance
tracker = CallTracker()
