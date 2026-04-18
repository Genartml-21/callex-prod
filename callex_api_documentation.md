# Callex Voice AI — Developer Integration Guide

Welcome to the Callex Voice AI integration documentation. This document covers everything you need to connect your frontend dashboard to the Callex backend AI engine. 

The backend handles ultra-low latency real-time voice synthesis, dynamic AI manipulation, and AI-driven call analytics. Your job as a frontend/integration developer is to interface with our REST API, construct Agent configurations in Firebase/Firestore, and (optionally) bind to our WebSocket server for real-time live monitoring.

**Important Note:** The Callex engine utilizes proprietary streaming models and custom GenArtML TTS models. All vendor infrastructure is heavily abstracted behind the `callex` engine.


---

## 1. Agent Configuration Schema (Firestore)

When creating or updating an "Agent" from your dashboard, you must save an `agent_config` document to Firebase/Firestore. The backend constantly reads these configurations heavily to power the AI calls.

**Collection Location:** `agents/{agent_id}`
**Data Structure:**

```json
{
  "id": "agent-123",                    // (String) Unique identifier for the agent
  "name": "Sales Executive",            // (String) Display name
  "language": "hi-IN",                  // (String) Supported: "hi-IN", "gu-IN", "en-US"
  "voice": "MF4J4IDTRo0AxOO4dpFR",      // (String) The Callex Voice ID to use
  
  // -- ADVANCED BEHAVIOR --
  "voiceSpeed": 1.25,                   // (Float) Controls how fast the bot speaks (default: 1.25)
  "enableNLP": true,                    // (Boolean) Master switch for Real-Time Emotion Adaptation (ToneAnalyzer)
  
  // -- BOT BRAIN --
  "systemPrompt": "You are a sales agent...",  // (String) The base personality and rules of the bot
  "knowledgeBase": "Company is established in..." // (String) FAQ structure loaded into the Fast-Path zero-latency cache
}
```

### 1.1 What `enableNLP` Does (Emotion Mirroring)
Setting `"enableNLP": true` activates a zero-latency NLP pipeline. 
If the customer is angry, the AI automatically detects it in `<1ms`, drops its voice pitch, and injects calming prompts into the LLM logic without you doing anything. If the customer is happy, it raises its expressiveness. **Highly recommended for production.**

### 1.2 What `voiceSpeed` Does
Overrides the default speed of the speaker. A value of `1.0` is standard. A value of `1.25` is highly recommended for fast-paced Hindi telecalling.


---

## 2. REST API Documentation

The REST API allows you to fetch real-time analytics, AI disposition breakdowns, and historic call logs.

**Base URL:** `http(s)://<your-backend-ip>:<port>/api`

### 2.1 Fetch All Calls
**Endpoint:** `GET /calls`
**Description:** Returns a paginated list of all calls, including their AI Analysis (Sentiment, Summary, Structured Data) exactly as the NLP generated it at the end of the call.

**Query Parameters:**
- `skip` (int, default: 0) — Offset for pagination
- `limit` (int, default: 100) — Number of records to return
- `status` (string, optional) — Filter by `completed` or `in_progress`

**Response (`200 OK`):**
```json
[
  {
    "id": 1042,
    "call_uuid": "fb8a932b-45ab-4c22-...",
    "phone_number": "9199XXXXX123",
    "start_time": "2026-04-12T10:15:30",
    "end_time": "2026-04-12T10:17:45",
    "duration_seconds": 135.2,
    "status": "completed",
    "has_recording": true,

    // -- AI GENERATED ANALYSIS --
    "customer_agreed": true,             // Did the customer agree to the goal?
    "commitment_date": "2026-12-01",     // Nullable, extracted if mentioned
    "unclear_response": false,           // True if the AI was confused by the caller
    "disposition": "Interested",         // High-level call outcome (Interested, Not Interested, Do Not Call)
    "notes": "Follow up tomorrow",       // AI's internal scratchpad thoughts
    "transcript": "Bot: Hello... \nUser: Yes...", // Full call transcript
    "summary": "Customer liked the product.", // AI-generated TL;DR
    "sentiment": "positive",             // "positive", "negative", "neutral"
    "structured_data": "{\"custom_key\": \"custom_value\"}" // Extracted JSON data
  }
]
```

### 2.2 Fetch Call by UUID
**Endpoint:** `GET /calls/{call_uuid}`
**Description:** Fetch a single call's deeply-detailed analytics and transcript.

**Response (`200 OK`):**
Same structure as `GET /calls`, but returns a single Object.

**Errors:**
- `404 Not Found`: Emitted if the UUID does not exist.


---

## 3. Real-Time Telemetry via WebSocket (Advanced)

If you are building a "Live Call Dashboard", you can tap into the backend's WebSocket to receive live telemetry about the call *as it happens*.

**WebSocket URL:** `ws://<your-backend-ip>:<port>/ws/{call_uuid}`

### 3.1 Live Telemetry Events
When listening to this server, you will receive real-time JSON payloads mirroring the progression of the call.

**Bot is Speaking:**
```json
{
  "type": "bot_speaking",
  "text": "Hello! Am I speaking with Rahul?"
}
```

**Bot Stopped Speaking:**
```json
{
  "type": "bot_stopped"
}
```

**Customer Sent Transcription:**
```json
{
  "type": "message",
  "source": "user",
  "text": "Yes, tell me who is calling?"
}
```

**Call Ended / System Updates:**
```json
{
    "type": "system_note",
    "note": "[Analysis completed successfully]"
}
```

### 3.2 Live Supervisor Override commands (Dashboard to Bot)
You can send commands directly heavily manipulating a live call. Sent via JSON over the WebSocket:

1. **Hangup Call Immediately:**
```json
{
  "type": "HANGUP_CALL"
}
```

2. **Whisper To Bot (Ghosting):**
Directly inject a hidden text prompt into the AI's "ear" without the customer hearing it. The AI will adapt immediately in its next sentence.
```json
{
  "type": "whisper",
  "message": "Offer them a 20% discount if they buy right now."
}
```

3. **Human Barge-In / Takeover:**
Immediately stops the AI logic and transitions the call.
```json
{
  "type": "barge"
}
```

---

## Technical Restrictions & Hard Limits
- **File System:** All audio recordings are passed to the `upload_to_firebase` task. Ensure Firebase storage configurations are kept intact.
- **Outcomes:** Call analytics (`duration_seconds`, `summary`) are completely finalized **only when the caller hangs up using the FreeSWITCH hook**. Do not pull analysis for `in_progress` calls as the data will be heavily incomplete.
