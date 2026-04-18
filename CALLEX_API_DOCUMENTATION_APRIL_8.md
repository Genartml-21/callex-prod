# Callex AI — Complete Developer API Documentation (April 8th Update)

> **Version:** 3.1 (Gujarati Model Update) &nbsp;|&nbsp; **API Base URL:** `http://62.171.170.48:4500` &nbsp;|&nbsp; **Last Updated:** April 8, 2026

---

## ⚠️ Important: URL Guide

| Service | URL | Purpose |
|---------|-----|---------|
| **Admin Dashboard (GUI)** | `http://62.171.170.48:5173` | Human-facing web panel |
| **API Server** | `http://62.171.170.48:4500` | All REST API calls from apps/Postman |
| **AI Voice Engine (WebSocket)** | `ws://62.171.170.48:8085` | FreeSWITCH / live call WebSocket |

---

## Table of Contents

1. [Admin Panel](#admin-panel)
2. [Getting Your API Key](#getting-your-api-key)
3. [Authentication](#authentication)
4. [Quick Start — Postman](#quick-start--postman)
5. [External API — API Key Auth](#external-api--api-key-auth-apiv1)
   - [Agents](#agents-api)
   - [Calls & Transcripts](#calls--transcripts-api)
   - [Voices](#voices-api)
   - [Supervisor — Live Calls](#supervisor-api-live-calls)
   - [Dashboard KPIs](#dashboard-kpis)
   - [Debug Utilities](#debug-utilities)
6. [Internal Dashboard API — JWT Auth](#internal-dashboard-api--jwt-auth-api)
   - [Auth (Login/Register)](#auth)
   - [Agents (Internal)](#agents-internal)
   - [Analytics & Call Logs](#analytics--call-logs)
   - [Settings — API Keys & Webhooks](#settings--api-keys--webhooks)
   - [Dialer & Campaigns](#dialer--campaigns)
   - [Follow-Ups](#follow-ups)
   - [Knowledge Base](#knowledge-base)
   - [Routing Rules](#routing-rules)
7. [WebSocket — AI Voice Engine](#websocket--ai-voice-engine)
8. [Available Voices Reference](#available-voices-reference)
9. [Agent Configuration Reference](#agent-configuration-reference)
10. [Error Reference](#error-reference)
11. [Best Practices](#best-practices)

---

## Admin Panel

Everything below is also available via the **Callex Dashboard UI**:

🔗 **Dashboard URL:** [http://62.171.170.48:5173](http://62.171.170.48:5173)

| Feature | Dashboard Location |
|---------|--------------------|
| Create/Edit/Delete Agents | Sidebar → **Agent Studio** |
| Generate/Revoke API Keys | Sidebar → **Settings** → API Keys |
| Live Call Monitoring | Sidebar → **Live Supervisor** |
| Call Analytics & Full Logs | Sidebar → **Analytics** |
| Campaign / Dialer Management | Sidebar → **Dialer** |
| Knowledge Base | Sidebar → **Knowledge Base** |
| Call Routing Rules | Sidebar → **Routing** |
| Webhooks | Sidebar → **Settings** → Webhooks |
| Reports & Exports | Sidebar → **Reports** |
| Billing & Usage | Sidebar → **Billing** |
| Agent Simulation | Sidebar → **Simulation** |

---

## Getting Your API Key

1. Open the Dashboard → [http://62.171.170.48:5173](http://62.171.170.48:5173) and log in.
2. Click **Settings** in the left sidebar.
3. Enter a **Key Name** (e.g., `Production App`).
4. Select the **Environment**: **Test** (`ck_test_`) or **Live** (`ck_live_`).
5. Click **Generate** and **copy the key immediately** — it is shown only once.

> ⚠️ The full key is displayed **only once at creation time**.

---

## Authentication

### External API (API Key)

All `/api/v1/*` endpoints require an API key in the `Authorization` header:

```
Authorization: Bearer <YOUR_API_KEY>
```

| Environment | Prefix | Example |
|-------------|--------|---------|
| Test | `ck_test_` | `ck_test_a1b2c3d4_e5f6g7h8i9j0...` |
| Live | `ck_live_` | `ck_live_f8e7d6c5_b4a3z2y1x0w9...` |

### Internal Dashboard API (JWT)

All `/api/*` (non-v1) endpoints use a **JWT Bearer token** obtained from the login endpoint.

```
Authorization: Bearer <JWT_TOKEN>
```

---

## Quick Start — Postman

1. Get an API Key from the Dashboard.
2. Open Postman → New Request → `GET`
3. URL: `http://62.171.170.48:4500/api/v1/calls`
4. Headers tab → Add:
   - **Key:** `Authorization`
   - **Value:** `Bearer ck_live_YOUR_KEY_HERE`
5. Click **Send** — you should see your call history.

---

---

# External API — API Key Auth (`/api/v1`)

All endpoints below use `Authorization: Bearer <API_KEY>` and are at base URL `http://62.171.170.48:4500`.

---

## Agents API

### 1. List All Agents

```
GET /api/v1/agents
```

**Query Parameters:**

| Param | Type | Default | Description |
|-------|------|---------|-------------|
| `page` | integer | `1` | Page number |
| `limit` | integer | `10` | Items per page (max 100) |
| `status` | string | — | Filter: `draft`, `active`, `paused` |

```bash
curl -X GET "http://62.171.170.48:4500/api/v1/agents?page=1&limit=10&status=active" \
  -H "Authorization: Bearer ck_live_YOUR_KEY"
```

**Response — `200 OK`**

```json
{
  "agents": [
    {
      "id": "5fa23d1b-722a-4318-aecc-6e6ad9d1a8e1",
      "name": "Priya - Sales",
      "status": "active",
      "language": "hi-IN",
      "voice": "MF4J4IDTRo0AxOO4dpFR",
      "createdAt": "2026-03-12T10:00:00.000Z"
    }
  ],
  "pagination": { "page": 1, "limit": 10, "total": 12, "totalPages": 2 }
}
```

---

### 2. Create a New Agent

```
POST /api/v1/agents
```

Only `name` is required. All other fields have smart defaults.

**Body (JSON):**

| Field | Type | Required | Default | Description |
|-------|------|----------|---------|-------------|
| `name` | string | **Yes** | — | Agent display name |
| `description` | string | No | `""` | Internal description |
| `systemPrompt` | string | No | `""` | LLM instructions & personality |
| `openingLine` | string | No | `""` | First sentence AI speaks |
| `voice` | string | No | `"MF4J4IDTRo0AxOO4dpFR"` | Callex Voice ID |
| `language` | string | No | `"en-US"` | Language code (`en-US`, `hi-IN`, `gu-IN`) |
| `temperature` | float | No | `0.7` | Creativity 0.0–1.0 |
| `maxTokens` | integer | No | `250` | Max LLM tokens per reply |
| `maxDuration` | integer | No | `30` | Max call duration (minutes) |
| `bargeInMode` | string | No | `"balanced"` | `aggressive`, `balanced`, `polite`, `disabled` |
| `voicemailLogic` | string | No | `"hangup"` | `hangup`, `leave_message`, `human_escalate` |
| `webhookUrl` | string | No | `null` | Post-call webhook URL |
| `recordCall` | boolean | No | `true` | Record the call |
| `piiRedaction` | boolean | No | `true` | Auto-redact sensitive info |
| `autoSummary` | boolean | No | `true` | Generate AI call summary |
| `advancedNlpEnabled` | boolean | No | `false` | Enable psychological sales and persuasion framework |

> 📘 See full [Agent Configuration Reference](#agent-configuration-reference) for all 50+ fields.

```bash
curl -X POST http://62.171.170.48:4500/api/v1/agents \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer ck_live_YOUR_KEY" \
  -d '{
    "name": "Sales Agent - Premium",
    "systemPrompt": "You are a professional sales agent. Be persuasive but polite.",
    "openingLine": "કેમ છો! હું એક ખાસ ઑફર માટે કૉલ કરી રહ્યો છું.",
    "voice": "MF4J4IDTRo0AxOO4dpFR",
    "language": "gu-IN",
    "temperature": 0.6,
    "maxDuration": 15,
    "dynamicCodeSwitching": true
  }'

> **📝 Note on Gujarati (`gu-IN`) Integration:**
> When you pass `"language": "gu-IN"`, the Callex backend automatically:
> 1. Configures the CX STT Engine streaming and fallback ASR pipelines to properly identify spoken Gujarati.
> 2. Sets the LLM's primary script processing to native Gujarati format (`હા`, `ના`).
> 3. Triggers dynamic code switching so your agent will smoothly transition between English, Hindi, and pure Gujarati depending on the customer's response.
> 4. Applies localized conversational check-ins.
```

**Response — `201 Created`**

```json
{
  "message": "Agent successfully created.",
  "agentId": "5fa23d1b-722a-4318-aecc-6e6ad9d1a8e1",
  "agent": {
    "id": "5fa23d1b-722a-4318-aecc-6e6ad9d1a8e1",
    "name": "Sales Agent - Premium",
    "status": "draft",
    "createdAt": "2026-03-17T10:00:00.000Z"
  }
}
```

> 💡 New agents start as `draft`. Activate with `PUT /api/v1/agents/{id}` → `"status": "active"`.

---

### 3. Get Agent Details

```
GET /api/v1/agents/{agentId}
```

```bash
curl -X GET http://62.171.170.48:4500/api/v1/agents/5fa23d1b-722a-4318-aecc-6e6ad9d1a8e1 \
  -H "Authorization: Bearer ck_live_YOUR_KEY"
```

**Response — `200 OK`**

```json
{
  "id": "5fa23d1b-722a-4318-aecc-6e6ad9d1a8e1",
  "name": "Sales Agent - Premium",
  "systemPrompt": "You are a professional sales agent.",
  "voice": "MF4J4IDTRo0AxOO4dpFR",
  "language": "hi-IN",
  "temperature": 0.6,
  "status": "active",
  "PromptVersion": [
    { "version": 2, "prompt": "Updated prompt...", "isActive": true },
    { "version": 1, "prompt": "Original prompt...", "isActive": false }
  ]
}
```

---

### 4. Update an Agent

Send only the fields you want to change.

```
PUT /api/v1/agents/{agentId}
```

```bash
curl -X PUT http://62.171.170.48:4500/api/v1/agents/5fa23d1b-722a-4318-aecc-6e6ad9d1a8e1 \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer ck_live_YOUR_KEY" \
  -d '{
    "name": "Sales Agent - Enterprise",
    "temperature": 0.5,
    "status": "active"
  }'
```

**Response — `200 OK`**

```json
{
  "message": "Agent updated successfully.",
  "agentId": "5fa23d1b-722a-4318-aecc-6e6ad9d1a8e1",
  "agent": { "id": "5fa23d1b-722a-4318-aecc-6e6ad9d1a8e1", "name": "Sales Agent - Enterprise", "status": "active" }
}
```

---

### 5. Delete an Agent

> ⚠️ **Irreversible.** Permanently deletes the agent, all prompt versions, and follow-ups linked to it.

```
DELETE /api/v1/agents/{agentId}
```

```bash
curl -X DELETE http://62.171.170.48:4500/api/v1/agents/5fa23d1b-722a-4318-aecc-6e6ad9d1a8e1 \
  -H "Authorization: Bearer ck_live_YOUR_KEY"
```

**Response — `200 OK`**

```json
{ "message": "Agent deleted successfully.", "agentId": "5fa23d1b-722a-4318-aecc-6e6ad9d1a8e1" }
```

---

## Calls & Transcripts API

### 6. List All Calls

Retrieve a paginated list of all calls made through your agents (active + completed).

```
GET /api/v1/calls
```

**Query Parameters:**

| Param | Type | Default | Description |
|-------|------|---------|-------------|
| `page` | integer | `1` | Page number |
| `limit` | integer | `20` | Items per page (max 100) |
| `status` | string | — | Filter: `active`, `completed`, `failed` |
| `agentId` | string | — | Filter by specific agent |

```bash
curl -X GET "http://62.171.170.48:4500/api/v1/calls?page=1&limit=20&status=completed" \
  -H "Authorization: Bearer ck_live_YOUR_KEY"
```

**Response — `200 OK`**

```json
{
  "calls": [
    {
      "id": "d9e4ff50-eeba-465d-a68a-fca353fafcf7",
      "phoneNumber": "+919876543210",
      "agentId": "1a81e4bb-f51d-4f22-abc1-c7b9686e3019",
      "agentName": "Priya - Sales",
      "status": "completed",
      "duration": 145,
      "sentiment": "positive",
      "hasTranscript": true,
      "hasRecording": true,
      "recordingUrl": "https://storage.callex.ai/...",
      "startedAt": "2026-03-17T12:30:00.000Z",
      "endedAt": "2026-03-17T12:32:25.000Z"
    }
  ],
  "pagination": { "page": 1, "limit": 20, "total": 47, "totalPages": 3 }
}
```

---

### 7. Get Call Details

Full call details including the complete transcript, recording URL, and AI summary.

```
GET /api/v1/calls/{callId}
```

```bash
curl -X GET http://62.171.170.48:4500/api/v1/calls/d9e4ff50-eeba-465d-a68a-fca353fafcf7 \
  -H "Authorization: Bearer ck_live_YOUR_KEY"
```

**Response — `200 OK`**

```json
{
  "id": "d9e4ff50-eeba-465d-a68a-fca353fafcf7",
  "phoneNumber": "+919876543210",
  "agentId": "1a81e4bb-f51d-4f22-abc1-c7b9686e3019",
  "agentName": "Priya - Sales",
  "status": "completed",
  "duration": 145,
  "sentiment": "positive",
  "transcript": "AI: नमस्ते...\nCustomer: हाँ बोलिए...",
  "transcriptMessages": [
    { "role": "ai", "text": "नमस्ते, मैं Callex से बोल रही हूँ...", "timestamp": 1710340200 },
    { "role": "customer", "text": "हाँ बोलिए...", "timestamp": 1710340205 }
  ],
  "recordingUrl": "https://storage.callex.ai/recordings/...",
  "summary": "Customer agreed to recharge within 24 hours.",
  "outcome": null,
  "startedAt": "2026-03-17T12:30:00.000Z",
  "endedAt": "2026-03-17T12:32:25.000Z"
}
```

---

### 8. Get Call Transcript Only

```
GET /api/v1/calls/{callId}/transcript
```

```bash
curl -X GET http://62.171.170.48:4500/api/v1/calls/d9e4ff50-eeba-465d-a68a-fca353fafcf7/transcript \
  -H "Authorization: Bearer ck_live_YOUR_KEY"
```

**Response — `200 OK`**

```json
{
  "callId": "d9e4ff50-eeba-465d-a68a-fca353fafcf7",
  "transcript": "AI: नमस्ते...\nCustomer: हाँ बोलिए...",
  "messages": [
    { "role": "ai", "text": "नमस्ते...", "timestamp": 1710340200 },
    { "role": "customer", "text": "हाँ बोलिए...", "timestamp": 1710340205 }
  ],
  "messageCount": 2
}
```

---

## Voices API

### 9. List Available Voices

```
GET /api/v1/voices
```

```bash
curl -X GET http://62.171.170.48:4500/api/v1/voices \
  -H "Authorization: Bearer ck_live_YOUR_KEY"
```

**Response — `200 OK`**

```json
{
  "voices": [
    { "id": "MF4J4IDTRo0AxOO4dpFR", "name": "Devi", "description": "Clear Hindi female voice", "language": "hi-IN", "gender": "female", "style": "professional", "isDefault": true },
    { "id": "1qEiC6qsybMkmnNdVMbK", "name": "Monika", "description": "Modulated professional female voice", "language": "hi-IN", "gender": "female", "style": "professional", "isDefault": false },
    { "id": "qDuRKMlYmrm8trt5QyBn", "name": "Taksh", "description": "Powerful and commanding male voice", "language": "hi-IN", "gender": "male", "style": "authoritative", "isDefault": false },
    { "id": "LQ2auZHpAQ9h4azztqMT", "name": "Parveen", "description": "Confident male voice — warm and persuasive", "language": "hi-IN", "gender": "male", "style": "confident", "isDefault": false },
    { "id": "s6cZdgI3j07hf4frz4Q8", "name": "Arvi", "description": "Desi conversational female voice", "language": "hi-IN", "gender": "female", "style": "conversational", "isDefault": false }
  ],
  "total": 5
}
```

> 💡 Use the `id` value as the `voice` field when creating or updating agents.

---

## Supervisor API (Live Calls)

### 10. Get All Active Calls

Returns all currently active (in-progress) calls for your account.

```
GET /api/v1/supervisor/calls
```

```bash
curl -X GET http://62.171.170.48:4500/api/v1/supervisor/calls \
  -H "Authorization: Bearer ck_live_YOUR_KEY"
```

**Response — `200 OK`**

```json
[
  {
    "id": "ac544763-dda7-474e-87e2-926789874ff1",
    "phoneNumber": "+919876543210",
    "agentId": "Z7z52IQOsdUgG5ivAiB5",
    "agentName": "Sales Agent - Premium",
    "status": "active",
    "sentiment": "neutral",
    "startedAt": "2026-03-17T12:00:00.000Z"
  }
]
```

> Returns an **array** (not an object). Empty array `[]` means no active calls right now.

---

### 11. Whisper to AI Agent

Send a hidden instruction to the AI mid-call. The caller does **not** hear this. The AI incorporates it naturally into its next responses.

```
POST /api/v1/supervisor/calls/{callId}/whisper
```

**Body (JSON):**

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `message` | string | **Yes** | Instruction to send to the AI |

```bash
curl -X POST http://62.171.170.48:4500/api/v1/supervisor/calls/ac544763/whisper \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer ck_live_YOUR_KEY" \
  -d '{ "message": "Offer the customer a 10% discount to close the deal." }'
```

**Response — `200 OK`**

```json
{ "success": true, "message": "Offer the customer a 10% discount to close the deal." }
```

---

### 12. Barge Into a Call

Take over an active AI call. The AI stops immediately and call status changes to `transferred`.

```
POST /api/v1/supervisor/calls/{callId}/barge
```

```bash
curl -X POST http://62.171.170.48:4500/api/v1/supervisor/calls/ac544763/barge \
  -H "Authorization: Bearer ck_live_YOUR_KEY"
```

**Response — `200 OK`**

```json
{ "success": true }
```

---

## Dashboard KPIs

### 13. Get Platform KPIs

Real-time statistics including active calls, completed today, queue depth.

```
GET /api/v1/dashboard/kpis
```

```bash
curl -X GET http://62.171.170.48:4500/api/v1/dashboard/kpis \
  -H "Authorization: Bearer ck_live_YOUR_KEY"
```

**Response — `200 OK`**

```json
{
  "activeCalls": 2,
  "completedToday": 145,
  "avgMOS": 4.12,
  "slaPercent": 95,
  "apiFallbackRate": 0.5,
  "aiAgentsAvailable": 8,
  "humanAgentsAvailable": 2,
  "queueDepth": 0
}
```

---

## Debug Utilities

### 14. Check My Identity (Debug)

Verify which `userId` your API key is linked to. Useful for debugging empty call/agent responses.

```
GET /api/v1/debug/my-identity
```

```bash
curl -X GET http://62.171.170.48:4500/api/v1/debug/my-identity \
  -H "Authorization: Bearer ck_live_YOUR_KEY"
```

**Response — `200 OK`**

```json
{
  "userId": "abc123xyz",
  "env": "live",
  "keyId": "key-doc-id",
  "ownedAgents": 3,
  "agentIds": ["agent1", "agent2", "agent3"],
  "callsWithUserId": 47,
  "message": "If ownedAgents is 0, your API key may be linked to the wrong user account."
}
```

> 💡 If `ownedAgents` is `0` and you have created agents, your API key may belong to a different user account in the system.

---

---

# Internal Dashboard API — JWT Auth (`/api`)

These endpoints are used by the Callex web dashboard. They require a **JWT token** from the login endpoint (stored in browser `localStorage`). For external integrations, prefer the `/api/v1/` endpoints above.

**Base URL:** `http://62.171.170.48:4500`

---

## Auth

### Login

```
POST /api/auth/login
```

**Body (JSON):**

```json
{ "email": "your@email.com", "password": "yourpassword" }
```

**Response — `200 OK`**

```json
{
  "token": "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9...",
  "user": { "id": "abc123", "email": "your@email.com", "name": "Daksh", "role": "user" }
}
```

> Use the `token` value as `Bearer <token>` in all subsequent dashboard API calls.

### Register

```
POST /api/auth/register
```

```json
{ "email": "new@email.com", "password": "securepassword", "name": "Your Name" }
```

---

## Agents (Internal)

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/api/agents` | List your agents (newest first) |
| `GET` | `/api/agents/{id}` | Get agent + all prompt versions |
| `POST` | `/api/agents` | Create agent |
| `PATCH` | `/api/agents/{id}` | Update agent fields |
| `DELETE` | `/api/agents/{id}` | Delete agent |
| `PATCH` | `/api/agents/{id}/status` | Quick status toggle |
| `POST` | `/api/agents/{id}/prompt-version` | Save new prompt version |
| `GET` | `/api/agents/{id}/prompt-versions` | List all prompt versions |
| `POST` | `/api/agents/tts-preview` | Preview a voice (returns audio stream) |
| `POST` | `/api/agents/clone-voice` | Clone a custom voice (multipart audio upload) |

---

## Analytics & Call Logs

### List Call Logs ✅ (Fixed March 17, 2026)

```
GET /api/analytics/calls
```

Returns all calls for the authenticated user — both calls directly linked by `userId` and calls linked through agent ownership. Sorted newest first.

**Query Parameters:**

| Param | Type | Default | Description |
|-------|------|---------|-------------|
| `page` | integer | `1` | Page number |
| `limit` | integer | `50` | Items per page (max 200) |
| `status` | string | — | Filter: `active`, `completed` |
| `sentiment` | string | — | Filter: `positive`, `neutral`, `negative`, `angry` |
| `minDuration` | integer | — | Minimum call duration in seconds |

```bash
# With JWT token
curl -X GET "http://62.171.170.48:4500/api/analytics/calls?page=1&limit=50" \
  -H "Authorization: Bearer YOUR_JWT_TOKEN"
```

**Response — `200 OK`**

```json
{
  "calls": [
    {
      "id": "d9e4ff50-eeba-465d-a68a-fca353fafcf7",
      "phoneNumber": "+919876543210",
      "agentId": "1a81e4bb-f51d-4f22-abc1-c7b9686e3019",
      "agentName": "Priya - Sales",
      "status": "completed",
      "duration": 145,
      "sentiment": "positive",
      "transcript": "AI: नमस्ते...",
      "startedAt": "2026-03-17T12:30:00.000Z",
      "endedAt": "2026-03-17T12:32:25.000Z"
    }
  ],
  "total": 47,
  "pagination": { "page": 1, "limit": 50, "totalPages": 1 }
}
```

### Get Call Detail

```
GET /api/analytics/calls/{callId}
```

Returns full call record including transcript, recording URL, summary.

### Trigger AI Summary (ACW)

```
POST /api/analytics/calls/{callId}/acw
```

Triggers CX-Model AI to generate a summary of the call transcript.

```json
{
  "summary": "Customer agreed to recharge. High intent.",
  "structuredData": "{\"intent\": \"subscription renewal\", \"agreed\": true, \"followUpRequired\": false}",
  "redactedTranscript": "AI: Hello...\nCustomer: [REDACTED PHONE]..."
}
```

### Analytics Stats (Global)

```
GET /api/analytics/stats
```

```json
{
  "total": 523,
  "completed": 498,
  "sentiment": { "positive": 210, "neutral": 195, "negative": 58, "angry": 35 }
}
```

---

## Settings — API Keys & Webhooks

### API Keys

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/api/settings/api-keys` | List all active API keys |
| `POST` | `/api/settings/api-keys` | Create a new API key |
| `DELETE` | `/api/settings/api-keys/{id}` | Revoke an API key |

**Create API Key — Body:**

```json
{ "name": "Production App", "env": "live" }
```

**Response includes `fullKey`** — save it, it is shown only once:

```json
{
  "id": "key-doc-id",
  "name": "Production App",
  "prefix": "ck_live_a1b2c3d4",
  "env": "live",
  "fullKey": "ck_live_a1b2c3d4_e5f6g7h8i9j0k1l2m3n4o5p6q7r8s9"
}
```

### Webhooks

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/api/settings/webhooks` | List webhooks |
| `POST` | `/api/settings/webhooks` | Create webhook |
| `PATCH` | `/api/settings/webhooks/{id}` | Update webhook |
| `DELETE` | `/api/settings/webhooks/{id}` | Delete webhook |
| `POST` | `/api/settings/webhooks/{id}/test` | Send a test event |

**Create Webhook — Body:**

```json
{
  "url": "https://yourdomain.com/webhook",
  "events": ["call.completed", "call.started"],
  "secret": "your-webhook-secret"
}
```

---

## Dialer & Campaigns

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/api/dialer/campaigns` | List campaigns |
| `POST` | `/api/dialer/campaigns` | Create campaign |
| `PATCH` | `/api/dialer/campaigns/{id}` | Update campaign |
| `PATCH` | `/api/dialer/campaigns/{id}/status` | Start/pause campaign |
| `DELETE` | `/api/dialer/campaigns/{id}` | Delete campaign |

**Create Campaign — Body:**

```json
{
  "name": "March Recharge Drive",
  "agentId": "5fa23d1b-...",
  "contacts": ["+919876543210", "+919812345678"],
  "scheduledAt": "2026-03-18T09:00:00Z",
  "maxConcurrent": 5
}
```

---

## Follow-Ups

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/api/followups` | List due follow-ups |
| `POST` | `/api/followups` | Create follow-up |
| `PATCH` | `/api/followups/{id}/status` | Mark done/pending |

---

## Knowledge Base

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/api/knowledge` | List knowledge documents |
| `POST` | `/api/knowledge` | Upload document (multipart/form-data) |
| `DELETE` | `/api/knowledge/{id}` | Delete document |
| `POST` | `/api/knowledge/{id}/resync` | Re-process document for embeddings |

---

## Routing Rules

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/api/routing/rules` | List routing rules |
| `POST` | `/api/routing/rules` | Create rule |
| `PATCH` | `/api/routing/rules/{id}` | Update rule |
| `DELETE` | `/api/routing/rules/{id}` | Delete rule |

---

---

# WebSocket — AI Voice Engine

The Python AI engine is accessible at **port 8085** via WebSocket.

## Agent-Specific WebSocket

```
ws://62.171.170.48:8085/agent/{agentId}
```

FreeSWITCH connects to this URL and the AI loads the specific agent config from Firestore.

## Default WebSocket

```
ws://62.171.170.48:8085/
```

Uses the first `active` agent as fallback.

## List All Agents with WebSocket URLs

```
GET http://62.171.170.48:8085/agents
```

```json
{
  "agents": [
    {
      "id": "Z7z52IQOsdUgG5ivAiB5",
      "name": "Recharge Assistant",
      "status": "active",
      "websocket_url": "ws://62.171.170.48:8085/agent/Z7z52IQOsdUgG5ivAiB5"
    }
  ],
  "total": 1
}
```

## Headers Passed by FreeSWITCH

The AI engine reads these headers to identify each call:

| Header | Description |
|--------|-------------|
| `x-call-id` | Unique call UUID |
| `x-agent-id` | Agent to load |
| `x-phone-number` | Caller's phone number |
| `Caller-Caller-ID-Number` | FreeSWITCH caller ID |
| `variable_sip_from_user` | SIP from user |

## URL Routing Patterns

| Pattern | Behavior |
|---------|----------|
| `ws://server:8085/agent/{id}` | Load specific agent |
| `ws://server:8085/?agent_id={id}` | Query param alternative |
| `ws://server:8085/` | Default/fallback agent |

## FreeSWITCH Integration Example

```
execute_on_answer='socket 127.0.0.1:8085/agent/Z7z52IQOsdUgG5ivAiB5 async full'
```

## Live Transcript WebSocket (Supervisor)

```
ws://62.171.170.48:4500/?type=supervisor&callId={callId}
```

Stream real-time transcript as a call progresses.

**Messages from server:**

```json
{ "type": "transcript_line", "line": "Bot: Hello! How can I help?" }
{ "type": "sentiment_update", "sentiment": "positive" }
{ "type": "call_ended", "callId": "abc123" }
```

---

---

# Available Voices Reference

| Voice ID | Name | Gender | Style | Default |
|----------|------|--------|-------|---------|
| `MF4J4IDTRo0AxOO4dpFR` | **Devi** | Female | Professional, Clear Hindi | ✅ Yes |
| `1qEiC6qsybMkmnNdVMbK` | **Monika** | Female | Modulated, Professional | No |
| `qDuRKMlYmrm8trt5QyBn` | **Taksh** | Male | Powerful, Commanding | No |
| `LQ2auZHpAQ9h4azztqMT` | **Parveen** | Male | Confident, Warm | No |
| `s6cZdgI3j07hf4frz4Q8` | **Arvi** | Female | Conversational, Friendly | No |

---

# Agent Configuration Reference

All fields are optional on `PUT`/`PATCH`. Only `name` is required on `POST`.

### Core Fields

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `name` | string | — | Agent name |
| `description` | string | `""` | Internal description |
| `systemPrompt` | string | `""` | LLM personality & instructions |
| `openingLine` | string | `""` | First words spoken on call |
| `voice` | string | `"MF4J4IDTRo0AxOO4dpFR"` | Callex Voice ID |
| `language` | string | `"en-US"` | Language code (`en-US`, `en-GB`, `hi-IN`, `gu-IN`, `es-ES`, `fr-FR`, `de-DE`) |
| `status` | string | `"draft"` | `draft`, `active`, `paused` |

### LLM Settings

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `temperature` | float | `0.7` | Creativity 0.0–1.0 |
| `maxTokens` | integer | `250` | Max tokens per response |
| `sttEngine` | string | `"callex-1.1"` | Speech-to-text engine |
| `llmModel` | string | `"callex-1.3"` | LLM model identifier |
| `strictToolCalling` | boolean | `true` | Enforce tool calling schema |
| `topK` | integer | `5` | Knowledge base top-K results |
| `similarityThresh` | float | `0.75` | Knowledge base similarity threshold |

### Voice & TTS

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `prosodyRate` | float | `1.0` | Speech speed (0.5–2.0) |
| `prosodyPitch` | float | `1.0` | Voice pitch |
| `fillerPhrases` | array | `["Let me check..."]` | Mid-sentence filler phrases |
| `speakingStyle` | string | `"professional"` | `professional`, `friendly`, `urgent`, `empathetic` |
| `backgroundAmbience` | string | `"none"` | `none`, `office`, `call_center` |

### Call Behavior

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `bargeInMode` | string | `"balanced"` | `aggressive`, `balanced`, `polite`, `disabled` |
| `patienceMs` | integer | `800` | End-of-turn silence timeout (ms) |
| `maxDuration` | integer | `30` | Max call duration (minutes) |
| `ringTimeout` | integer | `30` | Ring timeout before hangup (seconds) |
| `voicemailLogic` | string | `"hangup"` | `hangup`, `leave_message`, `human_escalate` |
| `voicemailDropAudio` | string | `null` | Audio URL to play when voicemail detected |
| `fallbackMessage` | string | `"I'm sorry..."` | Response when AI is unsure |
| `processDtmf` | boolean | `true` | Process DTMF keypress tones |
| `amdPrecision` | string | `"balanced"` | Answering Machine Detection: `fast`, `balanced`, `precise` |

### Compliance & Safety

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `profanityFilter` | string | `"redact"` | `redact`, `allow`, `hangup` |
| `topicRestriction` | boolean | `false` | Restrict to agent's topic only |
| `piiRedaction` | boolean | `true` | Auto-redact credit cards, phone numbers |
| `dncLitigatorScrub` | boolean | `true` | Skip known litigators |
| `complianceScript` | string | `null` | Mandatory compliance intro text |

### Integrations

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `webhookUrl` | string | `null` | URL to POST call result after hanging up |
| `postCallSms` | string | `null` | SMS template to send post-call |
| `tools` | array | `[]` | Tool definitions for function calling |
| `ipaLexicon` | object | `{}` | Custom pronunciation dictionary |

### Intelligence

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `sentimentRouting` | boolean | `false` | Transfer to human if caller is angry |
| `emotionalMirroring` | boolean | `true` | Match caller energy/tone |
| `dynamicCodeSwitching` | boolean | `true` | Auto switch languages mid-call |
| `multiAgentHandoff` | boolean | `false` | Hand off to another agent |
| `objectionHandling` | string | `"standard"` | `standard` or `persistent` |
| `competitorAlerts` | string | `""` | Comma-separated competitor names to flag |
| `supervisorWhisper` | boolean | `true` | Allow supervisor whisper mid-call |

### Post-Call Automation

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `autoSummary` | boolean | `true` | Generate AI summary after call |
| `autoSentiment` | boolean | `true` | Analyze sentiment automatically |
| `recordCall` | boolean | `true` | Record and store the call audio |
| `autoFollowUp` | boolean | `true` | Schedule follow-up automatically |
| `followUpDefaultDays` | integer | `1` | Days after call to follow up |
| `followUpDefaultTime` | string | `"10:00"` | Time of day for follow-up |

### Billing

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `costCapTokens` | integer | `5000` | Max tokens before billing cutoff |
| `callBlending` | boolean | `false` | Blend inbound + outbound calls |
| `geoCallerId` | boolean | `false` | Use local caller ID based on geography |

---

# Error Reference

| Status | Code | Meaning |
|--------|------|---------|
| `200` | OK | Success |
| `201` | Created | Resource created |
| `204` | No Content | Deleted successfully |
| `400` | Bad Request | Missing required field — `{ "error": "Agent 'name' is required." }` |
| `401` | Unauthorized | Missing or malformed Authorization header |
| `403` | Forbidden | Invalid or revoked API key / expired JWT |
| `404` | Not Found | Resource doesn't exist or doesn't belong to you |
| `409` | Conflict | Duplicate resource (e.g., email already registered) |
| `500` | Server Error | Internal error — check server logs |

---

# Best Practices

1. **Store API keys securely** — never expose in frontend code or public repos.
2. **Use Test keys** (`ck_test_`) for development, **Live keys** (`ck_live_`) for production.
3. **Paginate** all list calls — always use `page` and `limit` params.
4. **Partial updates** — on PUT/PATCH, send only changed fields.
5. **Revoke compromised keys** immediately from Dashboard → Settings.
6. **Use `/api/v1/debug/my-identity`** to troubleshoot empty call/agent responses.
7. **Call `GET /api/v1/voices`** to get the latest voice IDs before creating agents.
8. **JWT tokens expire** — re-login or refresh on 403 response from dashboard APIs.

---

## Quick Reference Table

| Action | Method | Endpoint |
|--------|--------|----------|
| **EXTERNAL API** | | |
| List agents | `GET` | `/api/v1/agents?page=1&limit=10` |
| Create agent | `POST` | `/api/v1/agents` |
| Get agent | `GET` | `/api/v1/agents/{id}` |
| Edit agent | `PUT` | `/api/v1/agents/{id}` |
| Delete agent | `DELETE` | `/api/v1/agents/{id}` |
| List all calls (history) | `GET` | `/api/v1/calls?page=1&limit=20` |
| Get call details | `GET` | `/api/v1/calls/{id}` |
| Get transcript | `GET` | `/api/v1/calls/{id}/transcript` |
| List voices | `GET` | `/api/v1/voices` |
| Active calls (live) | `GET` | `/api/v1/supervisor/calls` |
| Whisper to AI | `POST` | `/api/v1/supervisor/calls/{id}/whisper` |
| Barge into call | `POST` | `/api/v1/supervisor/calls/{id}/barge` |
| Dashboard KPIs | `GET` | `/api/v1/dashboard/kpis` |
| Debug my identity | `GET` | `/api/v1/debug/my-identity` |
| **DASHBOARD API** | | |
| Login | `POST` | `/api/auth/login` |
| Call logs (full history) | `GET` | `/api/analytics/calls` |
| Call detail | `GET` | `/api/analytics/calls/{id}` |
| AI call summary | `POST` | `/api/analytics/calls/{id}/acw` |
| List API keys | `GET` | `/api/settings/api-keys` |
| Create API key | `POST` | `/api/settings/api-keys` |
| Revoke API key | `DELETE` | `/api/settings/api-keys/{id}` |
| List webhooks | `GET` | `/api/settings/webhooks` |
| Create webhook | `POST` | `/api/settings/webhooks` |
| List campaigns | `GET` | `/api/dialer/campaigns` |

---

> 📌 **Remember:** Everything available via API is also accessible in the **[Admin Dashboard](http://62.171.170.48:5173)** with a visual interface.
