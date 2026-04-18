# Callex AI Voice Assistant - Developer API Documentation (v3.0)

Welcome to the Callex AI Engine v3 developer documentation. This document covers the comprehensive REST APIs for the Agent Studio, Knowledge Extraction, Shadow Sandbox Meta-Training, and the real-time Ultra-Low Latency WebSocket Streaming Engine.

## Table of Contents
1. [Base URL & Authentication](#1-base-url--authentication)
2. [Agent Management APIs](#2-agent-management-apis)
3. [Sandbox & Auto-Training (Meta-Agent)](#3-sandbox--auto-training-meta-agent)
4. [Live System Configuration Updates](#4-live-system-configuration-updates)
5. [Realtime Voice Streaming (WebSockets)](#5-realtime-voice-streaming-websockets)

---

## 1. Base URL & Authentication

**Base Endpoint:** `http(s)://<your-domain>/api` (or `:8085/api` natively)
**Voice Stream Endpoint:** `ws(s)://<your-domain>:8085/agent/:id`

**Authentication:** 
The Express.js Agent Dashboard relies on JWT-style Bearer tokens (or integrated Firebase Auth tokens) passed via the `Authorization: Bearer <TOKEN>` header.

---

## 2. Agent Management APIs

### 2.1 List All Agents
`GET /api/agents`

Retrieves all agents linked to the authenticated user. Agents are sorted by most recent (`createdAt`).

**Query Parameters:**
- `page` (number): Page number (default: 1)
- `limit` (number): Items per page (default: 50)
- `pagination` (boolean): Set to `false` to retrieve all concurrently.

**Response:**
```json
{
  "agents": [
    {
      "id": "A1B2C3D4",
      "name": "Support Agent",
      "isTrainingSandbox": false,
      "parentAgentId": null,
      "status": "active"
    }
  ],
  "pagination": { "page": 1, "limit": 50, "total": 1, "totalPages": 1 }
}
```

### 2.2 Create a New Agent
`POST /api/agents`
**Content-Type:** `multipart/form-data` (Supports Knowledge File Uploads seamlessly)

Creates an agent. Behind the scenes, the system will **automatically generate a mirror "Training Sandbox" agent** (Shadow Clone) alongside the production model.

**Payload:** (Form Data or JSON)
- `name` (string, Required)
- `systemPrompt` (string, Optional)
- `openingLine` (string)
- `file` (File, Optional - Auto-extracts Knowledge Base using CX-Model Flash via OCR/Text Analysis)

*(And 40+ configurable properties like `strictToolCalling`, `emotionalMirroring`, `maxTokens`, etc.)*

**Important Result:** Creating `Agent ID="111"` automatically generates a hidden `Agent ID="222"` where `isTrainingSandbox=true` and `parentAgentId="111"`.

### 2.3 Modify Agent Context & Parameters
`PATCH /api/agents/:id`
**Content-Type:** `multipart/form-data`

Updates the agent properties.
**Note:** Editing `systemPrompt` directly pushes a new immutable `promptVersions` historical schema version globally.

---

## 3. Sandbox & Auto-Training (Meta-Agent)

Callex v3 introduces **Concurrent Shadow Training Sandbox** architecture. This uses the CX-Model-Pro Meta-Agent to securely rewrite instructions based on real human trainer corrections entirely asynchronously without lagging production audio.

### 3.1 Filtering Sandbox vs Production
To build your UI dashboard logic:
- `Production Agents`: Filter where `isTrainingSandbox != true`
- `Sandbox Agents`: Filter where `isTrainingSandbox == true`

### 3.2 Push Sandbox to Production
When an Auto-Trained AI Sandbox performs correctly, deploy its logic out to the live caller-facing parent agent.

`POST /api/agents/:id/push-to-prod`
*(Where `:id` is the ID of the Sandbox Agent)*

**Architecture Workflow:**
1. Validates the Sandbox Agent has a `parentAgentId`.
2. Copies the current Sandbox `systemPrompt`.
3. Force-updates the Parent Agent `systemPrompt`.
4. Adds a version marker mapping origin `"vX — Pushed from Sandbox"`.
5. Re-invalidates native Live Voice Cache natively across all active processes.

**Returns:**
```json
{
  "success": true,
  "message": "Sandbox prompt securely deployed to Production Agent {parentAgentId}"
}
```

---

## 4. Live System Configuration Updates (Backend Natively)

With the **Threaded Execution Update (v3.1)**, Firestore synchronizations are now detached from Core Audio pipelines. 

**Database Flow:**
- Writes/Updates performed via the APIs (`/api/agents`) hit Google Firestore.
- During a live phone call, `backend/main.py` fetches configurations automatically.
- Caching: To guarantee **Ultra-Low Latency** (<1.5s TTS/VAD round-trips), the Engine fetches `systemPrompt` locally from RAM instead of checking Firestore on every sentence.
- **Cache Refresh**: Every 30 seconds, `db_get_doc()` executes unblocked across `ThreadPoolExecutor` nodes and immediately syncs API changes downstream to active phone calls.

---

## 5. Realtime Voice Streaming (WebSockets)

Connecting legacy soft-phones or modern VoIP dialers natively into Callex:

**WebSocket URL:** 
`ws://<your-server-ip>:8085/agent/<agentId>`

**Protocol Workflow:**
1. Open WebSocket connection.
2. The bot engine authenticates `agentId`. If valid, bot will speak `openingLine` instantly.
3. Your client must stream Raw Audio Binary (`S16LE`, `8000Hz/16000Hz`, `Mono`).
4. Server responds asynchronously with `bytes` of `audio/mpeg` (Callex TTS generated natively).
5. Ensure your system sends silent frames.

**Latency Optimization Hooks:**
- **Vad Timeouts:** Our VAD now considers a speech pause longer than `800ms` as a complete intent.
- **Microphone Barge-In:** The WebSocket engine features **Adaptive Echo Canceling & Barge-in**. If user speaks during AI playback, the backend will forcibly drop generation buffers natively to listen immediately.
