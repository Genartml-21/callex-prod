# Callex REST API & Developer Integration Guide

> **Version:** 3.1 | **Updated:** April 10, 2026

Welcome to the comprehensive Callex API documentation (v3.1). This guide covers the entirely new **Shadow Sandbox architecture**, the **Zero-Latency Threaded Audio Engine** details, and exactly how frontend developers & 3rd-party integrators must prepare their endpoints.


---

## 🛑 📂 Section 1: Developer Migration Guide (v3.1)

To fully support the new **Sandbox Auto-Trainer** feature natively, your UI dashboard and integrations require a few structural updates.

### 1. Dual-Tab Agent UI (Filtering)

Your React/Agent Dashboard must logically split agents into "Production" vs "Sandbox".

* **Production Tab:** When rendering the `GET /api/agents` payload, filter out Sandbox agents. Only render if `item.isTrainingSandbox === false` or `item.isTrainingSandbox === undefined`.
* **Sandbox Tab:** Render exclusively if `item.isTrainingSandbox === true`.

### 2. Implement the "Push to Production" Engine

Inside your new Sandbox agents view, implement a UI button (e.g., "Deploy to Live").

* **Action:** Trigger a request to `POST /api/agents/:id/push-to-prod`.
* **Flow:** This mathematical API forcibly copies the perfect Sandbox training and natively injects it into the Production Agent's active memory slot. Always use a confirmation modal to avoid accidental rewrites.

### 3. Deprecate Manual Cloning

When integrating, **do not manually create duplicate sandbox bots**.

* Whenever you ping `POST /api/agents` to create a standard bot, the Callex backend automatically calculates and spins up a hidden identical clone flagged with `isTrainingSandbox: true`.

---

## 🔌 Section 2: Agent Management APIs

* **Base URL:** `https://your-server.com/api` (or `:8085/api`)
* **Authentication:** `Authorization: Bearer <TOKEN>`

---

### 🤖 2.1 List All Agents

Retrieve a paginated list of all agents. Filter this list UI-side using `isTrainingSandbox` to separate your views.

* **Method:** `GET`
* **URL:** `/api/agents`
* **Query Parameters:**
  * `page` (default 1)
  * `limit` (default 50)
  * `pagination` (set to `false` to retrieve a flat un-paginated array)

**Response:**

```json
{
  "agents": [
    {
      "id": "A1B2C3-Prod",
      "name": "Customer Service",
      "isTrainingSandbox": false,
      "status": "active"
    },
    {
      "id": "X9Y8Z7-Sand",
      "name": "Customer Service - Training Sandbox",
      "isTrainingSandbox": true,
      "parentAgentId": "A1B2C3-Prod",
      "status": "draft"
    }
  ],
  "pagination": { "page": 1, "limit": 50, "total": 2, "totalPages": 1 }
}
```

### 🚀 2.2 Create a New Agent (With Knowledge Extraction)

Create a new AI agent and automatically generate its linked Training Sandbox.

* **Method:** `POST`
* **URL:** `/api/agents`
* **Content-Type:** `multipart/form-data` (Supports file uploads) or `application/json`

**Request Body:**

| Key              | Type   | Required | Description                                                                 |
| :--------------- | :----- | :------- | :-------------------------------------------------------------------------- |
| `name`         | String | Yes      | The internal name for the agent                                             |
| `systemPrompt` | String | No       | The core instruction prompt for LLM behavior                                |
| `file`         | File   | No       | Upload a PDF/Excel/CSV document. Handled automatically via CX-Model Flash OCR |
| `voice`        | String | No       | Callex Voice ID (e.g.,`alloy`)                                        |

**Response (200 OK):**

```json
{
  "id": "A1B2C3-Prod",
  "name": "Support Agent",
  "systemPrompt": "You are a support agent...",
  "status": "draft"
}
```

### 🔍 2.2b Fetch Specific Agent

Retrieves the exhaustive details of a single agent by ID.

* **Method:** `GET`
* **URL:** `/api/agents/:id`

**Response (200 OK):**
```json
{
  "id": "A1B2C3-Prod",
  "name": "Support Agent",
  "systemPrompt": "You are a support agent...",
  "status": "draft"
}
```

### ⚙️ 2.3 Update Agent Settings

Updates specific agent property fields via partial merging. Editing `systemPrompt` explicitly tracks a new prompt version organically.

* **Method:** `PATCH`
* **URL:** `/api/agents/:id`
* **Content-Type:** `multipart/form-data` or `application/json`

**Request Body:**

```json
{
  "temperature": 0.5,
  "maxTokens": 300,
  "systemPrompt": "You are a very helpful updated support agent."
}
```

**Response (200 OK):**

```json
{
  "id": "A1B2C3-Prod",
  "temperature": 0.5,
  "maxTokens": 300,
  "systemPrompt": "You are a very helpful updated support agent."
}
```

### 🧠 2.4 Upload Knowledge Base

Appends a PDF, CSV, Excel, or Text file into an existing agent's memory via vector parsing.

* **Method:** `POST`
* **URL:** `/api/agents/:id/knowledge`
* **Content-Type:** `multipart/form-data`

**Request Body:**

* `file` (File Blob) : The document to upload.

**Response (200 OK):**

```json
{
  "message": "Knowledge uploaded and processed successfully",
  "trainingSummary": {
    "totalFaqs": 45,
    "lastTrainedFile": "pricing_sheet.pdf"
  }
}
```

### 🔄 2.5 Push Sandbox to Production

Overwrites the Live Parent agent's logic natively with the verified Sandbox agent's logic.

* **Method:** `POST`
* **URL:** `/api/agents/:id/push-to-prod`
* *(Where `:id` is the ID of the Shadow Sandbox Clone)*

**Response (200 OK):**

```json
{
  "success": true,
  "message": "Sandbox prompt successfully deployed to Production Agent"
}
```

### 📜 2.6 Fetch Agent Prompt Versions

Retrieves the historical timeline of System Prompt changes for a specific agent.

* **Method:** `GET`
* **URL:** `/api/agents/:id/prompt-versions`

**Response (200 OK):**

```json
[
  {
    "version": 2,
    "prompt": "You are a very helpful updated support agent.",
    "isActive": true,
    "label": "v2 - Edit Settings Update"
  },
  {
    "version": 1,
    "prompt": "You are a support agent...",
    "isActive": false,
    "label": "v1 - Initial"
  }
]
```

### 🧠 2.7 Clear Knowledge Base

Deletes all uploaded PDF context linked to the agent.

* **Method:** `DELETE`
* **URL:** `/api/agents/:id/knowledge`

**Response (200 OK):**
```json
{
  "message": "Knowledge base cleared successfully"
}
```

### 🏷️ 2.8 Update Agent Status

Toggle an agent's active state (e.g., active, draft, paused).

* **Method:** `PATCH`
* **URL:** `/api/agents/:id/status`
* **Content-Type:** `application/json`

**Request Body:**
```json
{
  "status": "paused"
}
```

**Response (200 OK):**
```json
{
  "id": "A1B2C3-Prod",
  "status": "paused"
}
```

### 🗑️ 2.9 Delete Agent

Permanently deletes the agent alongside all its prompt histories and follow-ups securely.

* **Method:** `DELETE`
* **URL:** `/api/agents/:id`

**Response (200 OK):**

```json
{
  "success": true
}
```

---

## 🎧 Section 3: Ultra-Low Latency Voice Streams (v3.1)

The v3 architecture features completely threaded asynchronous caching explicitly to prevent audio dropping and latency freezing when multiple callers hang up simultaneously.

### 📡 3.1 Establishing WebSocket Connections

Connect your legacy soft-phones or modern VoIP dialers natively into Callex.

* **Protocol URL:** `ws(s)://your-server:8085/agent/:id`
* **Behavior:**
  * Once the connection stabilizes, the AI actively dictates the `openingLine`.
  * Expect Raw Binary audio (`S16LE`, `80k/160k Hz`)
  * Ensure the WebRTC/VoIP client provides absolute *silent frames*.

### ⏱️ 3.2 Real-Time Configuration Updates

* To guarantee sub-1.5s real-time latency over WebSockets, **System Prompts are cached in local server memory RAM**.
* When you update an agent or trigger `push-to-prod`, understand that the active backend ThreadPools will take **up to 30 seconds** to validate their schema dynamically and push the new settings into live active phone calls automatically.
