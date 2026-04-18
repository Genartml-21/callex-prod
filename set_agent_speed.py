"""
One-time script to set voiceSpeed for a specific agent in Firestore.
Usage: python3 set_agent_speed.py <agent_id> <speed>
"""
import sys
import os

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from app.core.agent_loader import _get_db

def main():
    agent_id = sys.argv[1] if len(sys.argv) > 1 else "X1rVIyKcE61qjXHbsg0W"
    speed = float(sys.argv[2]) if len(sys.argv) > 2 else 1.15
    
    db = _get_db()
    doc = db.collection('agents').document(agent_id).get()
    
    if not doc.exists:
        print(f"❌ Agent {agent_id} not found")
        return
    
    name = doc.to_dict().get('name', 'Unknown')
    old_speed = doc.to_dict().get('voiceSpeed', 1.0)
    
    db.collection('agents').document(agent_id).update({
        'voiceSpeed': speed
    })
    
    print(f"✅ Agent '{name}' (ID: {agent_id})")
    print(f"   voiceSpeed: {old_speed} → {speed}")

if __name__ == "__main__":
    main()
