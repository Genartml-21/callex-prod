# 👑 Callex Master API Reference (v3.2)

*Host Network Server:* `http://103.150.187.88:4500`
*WebSocket Engine:* `ws://103.150.187.88:8085`

This is the ultimate, exhaustive mapping of every single backend microservice, feature gateway, and system utility present in the Callex platform. It includes every internal dashboard requirement and external developer hook.

---

## 🛑 1. Core Integrator APIs (`/api/v1`)

The public-facing APIs for connecting Python scripts, CRON jobs, and external Webhooks.
*Auth Required:* `x-api-key: <token>`

### Agent Management

* `GET /api/v1/agents` - List all production agents
* `GET /api/v1/agents/:id` - Fetch specific agent details
* `POST /api/v1/agents` - Mathematically clone and deploy a new base agent
* `PUT /api/v1/agents/:id` - Full configuration overwrite for an agent
* `PATCH /api/v1/agents/:id/prompt` - Quickly update just the system prompt
* `DELETE /api/v1/agents/:id` - Liquidate an agent and its memory arrays

### Knowledge Management

* `POST /api/v1/agents/:id/knowledge` - Upload PDF/CSV vectorized knowledge
* `DELETE /api/v1/agents/:id/knowledge` - Wipe vector database for an agent

### Call Logs & Supervisor

* `GET /api/v1/calls` - Fetch historical call history data
* `GET /api/v1/calls/:id` - Fetch specific call metadata
* `GET /api/v1/calls/:id/transcript` - Retrieve completed transcript strings
* `GET /api/v1/calls/:id/transcript/live` - Stream active transcription arrays
* `PATCH /api/v1/calls/:id/disposition` - Re-categorize the CRM status of a call
* `GET /api/v1/supervisor/calls` - Fetch all concurrently active AI conversations
* `POST /api/v1/supervisor/calls/:id/whisper` - AI secretly speaks to the human operator
* `POST /api/v1/supervisor/calls/:id/barge` - Human overrides the AI natively

### Sub-Routines

* `GET /api/v1/voices` - Fetch available CX-Voice/CX-STT voices
* `GET /api/v1/dashboard/kpis` - Pull high-level analytics summary
* `GET /api/v1/dispositions` - Fetch all possible CRM call outcomes
* `POST /api/v1/dispositions` - Create a new CRM disposition rule
* `PUT /api/v1/dispositions/:id` - Edit a disposition rule
* `DELETE /api/v1/dispositions/:id` - Delete a disposition rule
* `GET /api/v1/debug/my-identity` - Check validity of your API Key

---

## 💻 2. Dashboard Internal APIs (`/api/...`)

These routes power your React/Vite web interface.
*Auth Required:* `Authorization: Bearer <jwt-token>`

### 👤 Authentication (`/api/auth`)

* `POST /api/auth/login` - Generate secure JWT payload
* `POST /api/auth/register` - Create new operator account
* `GET /api/auth/me` - Validate active token state

### 🤖 Agent Studio Engine (`/api/agents`)

* `GET /api/agents` - Fetch internal dashboard agent list
* `GET /api/agents/:id` - Specific agent mapping
* `POST /api/agents` - Internal agent visual generator
* `PATCH /api/agents/:id` - Update agent UI form
* `DELETE /api/agents/:id` - Agent removal
* `POST /api/agents/:id/knowledge` - Knowledge parser for dashboard
* `DELETE /api/agents/:id/knowledge` - Clear knowledge state
* `POST /api/agents/tts-preview` - CX-Voice real-time voice tester
* `POST /api/agents/clone-voice` - Voice morphing setup block
* `POST /api/agents/:id/prompt-version` - Create explicit A/B version of prompts
* `GET /api/agents/:id/prompt-versions` - List historical versions
* `POST /api/agents/:id/push-to-prod` - Overwrite LIVE with SANDBOX natively
* `PATCH /api/agents/:id/status` - Pause or active an agent status

### 📊 Dashboard & KPI (`/api/dashboard`)

* `GET /api/dashboard/kpis` - Financial and call volume metrics
* `GET /api/dashboard/ab-test` - Live A/B testing graph metrics
* `GET /api/dashboard/events` - Live system event feed
* `POST /api/dashboard/events` - Custom event dispatcher

### 👁️ Supervisor UI (`/api/supervisor`)

* `GET /api/supervisor/calls` - Internal active call polling
* `POST /api/supervisor/calls` - Launch UI-triggered call
* `PATCH /api/supervisor/calls/:id/end` - Kill call immediately natively
* `POST /api/supervisor/calls/:id/whisper` - Internal whisper inject
* `POST /api/supervisor/calls/:id/barge` - Internal barge inject
* `GET /api/supervisor/calls/:id/transcript` - Retrieve live arrays
* `PATCH /api/supervisor/calls/:id/sentiment` - Flag manual sentiment score

### 🧪 QA & Machine Learning Simulation (`/api/simulation`)

* `POST /api/simulation/batch` - Run 100x speed batch AI testing
* `POST /api/simulation/adversarial` - Launch "Angry Caller" red-team AI
* `GET /api/simulation/results/:jobId` - Fetch parsed test results
* `POST /api/simulation/agent-chat` - Sandbox playground socket connector
* `POST /api/simulation-stream/agent-chat-predictive` - Predictive chat flow
* `POST /api/simulation-stream/agent-chat-stream` - Fast stream visualizer

### 📞 Predictive Dialer Engine (`/api/dialer`)

* `GET /api/dialer/campaigns` - List running marketing campaigns
* `POST /api/dialer/campaigns` - Launch multi-threaded CSV outbound dialing
* `PATCH /api/dialer/campaigns/:id` - Edit campaign parameters
* `PATCH /api/dialer/campaigns/:id/status` - Pause/Resume campaigns
* `DELETE /api/dialer/campaigns/:id` - Destroy campaign queues

### 🎙️ Telecom & SIP Routing (`/api/telecom` & `/api/routing`)

* `GET /api/telecom/numbers` - List active Twilio/SignalWire inbound DIDs
* `POST /api/telecom/numbers` - Provision new phone number
* `DELETE /api/telecom/numbers/:id` - Destroy assigned number
* `GET /api/telecom/dnc` - "Do Not Call" registry lookup
* `POST /api/telecom/dnc` - Add number to blocklist
* `DELETE /api/telecom/dnc/:id` - Remove number from blocklist
* `GET /api/routing/rules` - View DID-to-Agent routing map
* `POST /api/routing/rules` - Create inbound queue routing logic
* `PATCH /api/routing/rules/:id` - Modify logic
* `DELETE /api/routing/rules/:id` - Delete tracking logic
* `POST /api/routing/evaluate` - Test dummy routing mathematically

### 🛡️ Security & PCI (`/api/security`)

* `GET /api/security/voice-signatures` - Voice biometric database
* `POST /api/security/voice-signatures` - Train new biometric block
* `DELETE /api/security/voice-signatures/:id` - Delete voice template
* `POST /api/security/pci` - Redaction verification hook
* `POST /api/security/hash-audio` - Deepfake hash verification

### 📈 Analytics & Quality Assurance (`/api/analytics` & `/api/qa`)

* `GET /api/analytics/` - Master analytics block
* `GET /api/analytics/calls` - Comprehensive CRM history view
* `GET /api/analytics/calls/:id` - Call breakdown (timing, cost, vectors)
* `POST /api/analytics/calls/:id/acw` - After-Call Work manual append
* `GET /api/analytics/stats` - Token cost generation mapping
* `GET /api/qa/scores/:callId` - LLM auto-judged call scores
* `POST /api/qa/scores` - Manual override QA score
* `GET /api/qa/dispositions` - See `api/v1/dispositions`
* `POST /api/qa/dispositions` - Create QA triggers
* `PUT /api/qa/dispositions/:id` - Edit QA triggers
* `DELETE /api/qa/dispositions/:id` - Delete QA triggers

### 💰 Billing & Admin Control (`/api/billing` & `/api/admin`)

* `GET /api/billing/stats` - Organization balance & limit metrics
* `POST /api/billing/increment` - Manual usage credit application
* `GET /api/admin/stats` - Superadmin omni-view of usage
* `GET /api/admin/users` - Global user registry
* `GET /api/admin/users/:id` - User specifics
* `POST /api/admin/users` - Create raw organization user
* `PATCH /api/admin/users/:id` - Modify user permissions/balance
* `DELETE /api/admin/users/:id` - Execute user obliteration
* `GET /api/admin/agents` - View ALL agents globally
* `PATCH /api/admin/agents/:id` - Modify any arbitrary agent
* `POST /api/admin/maintenance` - Master kill-switch (pause ALL traffic)
* `GET /api/admin/agents-by-user` - Segment agents geographically
* `GET /api/admin/users/:userId/export` - Export GDPR data structure
* `GET /api/admin/users/:userId/calls` - Isolate call data by user

### 🧩 Integrations & Knowledge (`/api/integrations` & `/api/knowledge`)

* `GET /api/integrations/` - List CRM adapters (Hubspot, Salesforce, etc)
* `PATCH /api/integrations/:id/connect` - Hook generic OAuth workflow
* `PATCH /api/integrations/:id/disconnect` - Sever API ties
* `GET /api/knowledge/` - Fetch all vectorized databases
* `POST /api/knowledge/` - Master multi-upload pipeline
* `DELETE /api/knowledge/:id` - Wipe knowledge index
* `POST /api/knowledge/:id/resync` - Force chunking re-build

### ⚙️ Settings, Reports & FollowUps

* `GET /api/settings/api-keys` - List Developer API limits
* `POST /api/settings/api-keys` - Provision new developer key
* `DELETE /api/settings/api-keys/:id` - Revoke authorization globally
* `GET /api/settings/webhooks` - List outbound call webhooks
* `POST /api/settings/webhooks` - Build webhook trigger
* `PATCH /api/settings/webhooks/:id` - Modify webhook trigger
* `DELETE /api/settings/webhooks/:id` - Remove webhook
* `POST /api/settings/webhooks/:id/test` - Fire dummy webhook
* `GET /api/reports/export` - CRM full CSV generation
* `GET /api/followups/` - Scheduled agent call-backs
* `POST /api/followups/` - Inject task explicitly
* `PATCH /api/followups/:id/status` - Mark task achieved

---

_Generated explicitly for Callex Platform architecture. Use this as your true-north map of operations._
