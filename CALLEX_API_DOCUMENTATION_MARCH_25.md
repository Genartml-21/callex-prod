# Callex REST API Documentation

> **Version:** 2.2 | **Updated:** March 25, 2026

Welcome to the comprehensive Callex API documentation. This guide covers how frontend developers can upload PDF/Excel knowledge bases, as well as providing the full schema for all external 3rd-party integration APIs.

---

## 📂 Section 1: PDF Knowledge Upload (Frontend Guide)

The backend handles automatic parsing and vectorization of PDF, DOC, CSV, and Excel files.

### 1. Create an Agent with a Document

Create a new AI agent and immediately feed it a knowledge base document in one request.

* **Method:** `POST`
* **URL:** `/api/agents`
* **Auth:** Bearer Token (JWT User Session)
* **Content-Type:** `multipart/form-data`

**Request Body (FormData):**

| Key              | Type   | Required | Description                                                                 |
| ---------------- | ------ | -------- | --------------------------------------------------------------------------- |
| `name`         | String | Yes      | The internal name for the agent                                             |
| `systemPrompt` | String | No       | The core behavior instructions. Optional if relying solely on the document. |
| `file`         | File   | No       | The PDF/Excel/Word document. Max size 20MB.                                 |
| `voice`        | String | No       | Callex voice ID (e.g.,`MF4J4IDTRo0AxOO4dpFR`)                         |
| `dispositions` | JSON   | No       | Stringified array of objects: `[{"name": "Sold", "category": "Sales"}]` |

**Success Response (201 Created):**

```json
{
  "id": "agent-uuid-string",
  "name": "Support Agent",
  "hasKnowledgeBase": true,
  "createdAt": "2026-03-23T10:00:00.000Z"
}
```

### 2. Append Document to Existing Agent

Upload a document to an agent that has already been created.

* **Method:** `POST`
* **URL:** `/api/agents/:id/knowledge`
* **Auth:** Bearer Token (JWT User Session)
* **Content-Type:** `multipart/form-data`

**Request Body (FormData):**

| Key      | Type | Required | Description                                             |
| -------- | ---- | -------- | ------------------------------------------------------- |
| `file` | File | Yes      | The PDF/Excel document to append to the agent's memory. |

**Success Response (200 OK):**

```json
{
  "message": "Knowledge base successfully processed and embedded."
}
```

---

## 🔌 Section 2: External Developer APIs (API Key Auth)

These APIs are meant for server-to-server integration by developers embedding Callex into their own platforms (CRMs, dashboards).

* **Base URL:** `https://your-server.com/api/v1`
* **Authentication:** `Authorization: Bearer <API_KEY>`
* **Content-Type:** `application/json`

---

### 📞 2.1 Call History & Management

#### Get All Calls

Retrieve a paginated list of all calls (active and completed) associated with the API Key's account.

* **Method:** `GET`
* **URL:** `/v1/calls`
* **Query Parameters:** 
  * `page` (default 1)
  * `limit` (default 50)
  * `status` (active/completed)
  * `startDate` (Format: YYYY-MM-DD or ISO — Inclusive from start of day)
  * `endDate` (Format: YYYY-MM-DD or ISO — Inclusive to end of day)

**Response:**

```json
{
  "calls": [
    {
      "id": "call-uuid",
      "phoneNumber": "+1234567890",
      "agentId": "agent-uuid",
      "status": "completed",
      "duration": 45,
      "disposition": "Interested"
    }
  ],
  "pagination": { "page": 1, "limit": 50, "total": 120, "totalPages": 3 }
}
```

#### Get Specific Call Details

Get comprehensive analytics, full transcript, and summary for a single call.

* **Method:** `GET`
* **URL:** `/v1/calls/:id`

**Response:**

```json
{
  "id": "call-uuid",
  "status": "completed",
  "duration": 120,
  "sentiment": "positive",
  "transcript": "Agent: Hello...\nUser: Hi there...",
  "disposition": "Follow Up",
  "recordingUrl": "https://storage.../recording.wav",
  "summary": "The customer is interested in buying the premium plan next week.",
  "startedAt": "2026-03-23T10:00:00.000Z"
}
```

#### Update Call Disposition (Outcome)

Update the sales disposition or outcome tag of a specific call remotely.

* **Method:** `PATCH`
* **URL:** `/v1/calls/:id/disposition`

**Request Body:**

```json
{
  "disposition": "Call Back Later"
}
```

**Response:**

```json
{
  "message": "Disposition updated successfully",
  "id": "call-uuid",
  "disposition": "Call Back Later"
}
```

#### Get Only Transcript

Fetch solely the raw text conversation history for ingestion into a CRM.

* **Method:** `GET`
* **URL:** `/v1/calls/:id/transcript`

**Response:**

```json
{
  "transcript": "Agent: Hello...\nUser: Hi there..."
}
```

---

### 🕵️‍♂️ 2.2 Live Supervisor APIs

APIs to monitor and intervene in phone queries as they happen in absolute real-time.

#### List Active Live Calls

Get a list of calls that are currently ongoing / active right now.

* **Method:** `GET`
* **URL:** `/v1/supervisor/calls`

**Response:**

```json
[
  {
    "id": "active-call-uuid",
    "phoneNumber": "+1234567890",
    "agentName": "Sales Rep",
    "status": "active"
  }
]
```

#### Whisper (Private Message to AI)

Send a hidden system instruction to the AI while it is speaking to the customer. The customer will not hear this instruction, but the AI will act on it.

* **Method:** `POST`
* **URL:** `/v1/supervisor/calls/:id/whisper`

**Request Body:**

```json
{
  "message": "Offer them a 20% discount if they sign up today."
}
```

#### Barge (Takeover Call)

Immediately disconnect the AI and forward the live customer call to a human agent's SIP or phone number.

* **Method:** `POST`
* **URL:** `/v1/supervisor/calls/:id/barge`

**Request Body:**

```json
{
  "destinationNumber": "+19876543210"
}
```

---

### 🏷️ 2.3 Disposition Tags (CRUD)

Manage the dropdown list of possible call outcomes (e.g., "Not Interested", "Sale Made").

#### List All Dispositions

* **Method:** `GET`
* **URL:** `/v1/dispositions`
* **Query Parameters:**
  * `pagination` - Set to `false` to return a direct flat array of all dispositions (bypasses wrap).
  * `linkedAgent` - Pass an Agent UUID to filter and return ONLY dispositions active for that specific agent.

**Response (if pagination=false):**

```json
[
  {
    "id": "disp-uuid",
    "name": "Callback Requested",
    "category": "Lead",
    "tagline": "Use this when the customer asks us to call them back later",
    "requiresNote": true,
    "requiredFields": [
      {
        "name": "best_time_to_call",
        "type": "string",
        "description": "The specific time or day the customer wants to be called back"
      }
    ],
    "linkedAgents": ["agent-uuid-1"]
  }
]
```

#### Create a Disposition

* **Method:** `POST`
* **URL:** `/v1/dispositions`

**Request Body:**

```json
{
  "name": "Busy / No Answer",
  "category": "Unreachable",
  "requiresNote": false
}
```

**Response:** `201 Created` with full object tree.

#### Update a Disposition

* **Method:** `PUT`
* **URL:** `/v1/dispositions/:id`

**Request Body:** (All fields optional)

```json
{
  "active": false,
  "name": "Do Not Call (DNC)"
}
```

#### Delete a Disposition

* **Method:** `DELETE`
* **URL:** `/v1/dispositions/:id`

**Response:**

```json
{
  "message": "Disposition deleted successfully",
  "id": "disp-uuid"
}
```

---

### 🤖 2.4 Agent Management APIs

Programmatically create, update, and manage your AI voice bots.

#### List Agents

* **Method:** `GET`
* **URL:** `/v1/agents`

#### Create Agent (JSON alternative)

If you don't need to upload a PDF file, you can use pure JSON to create an agent.

* **Method:** `POST`
* **URL:** `/v1/agents`
* **Content-Type:** `application/json`

**Request Body (FormData OR JSON):**

```json
{
  "name": "API Agent",
  "systemPrompt": "You are a helpful assistant...",
  "voice": "MF4J4IDTRo0AxOO4dpFR",
  "dispositions": [
     {"name": "Callback Requests", "category": "Sales"},
     {"name": "Do Not Call", "category": "General"}
  ]
}
```

*Note: You can pass actual `File` blobs purely via `multipart/form-data`. If using `multipart/form-data`, all boolean (`"true"`, `"false"`) and numeric values (`"1.5"`) sent as strings are automatically cast to their strict types by the server. If you just want to pass dispositions and no file, standard JSON is completely fine.*

#### Get Specific Agent

* **Method:** `GET`
* **URL:** `/v1/agents/:id`

#### Update Agent Settings

* **Method:** `PUT`
* **URL:** `/v1/agents/:id`

**Request Body:**

```json
{
  "language": "es-ES",
  "temperature": 0.5
}
```

#### Delete Agent

* **Method:** `DELETE`
* **URL:** `/v1/agents/:id`

#### Delete Agent Knowledge Base

Clears all uploaded PDF context from the agent, resetting its memory.

* **Method:** `DELETE`
* **URL:** `/v1/agents/:id/knowledge`

---

### 📊 2.5 General & Debugging

#### Get KPIs and Stats

Fetch account statistics for custom dashboard rendering.

* **Method:** `GET`
* **URL:** `/v1/dashboard/kpis`

**Response:**

```json
{
  "totalCalls": 450,
  "totalMinutes": 1200,
  "averageCallLengthSeconds": 160,
  "topAgent": "Support Bot"
}
```

#### Check API Key Identity

If you are seeing missing data, check which user account is actually tied to your API Key.

* **Method:** `GET`
* **URL:** `/v1/debug/my-identity`

**Response:**

```json
{
  "userId": "user-uuid",
  "ownedAgents": 3,
  "callsWithUserId": 150
}
```
