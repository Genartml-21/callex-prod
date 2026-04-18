# Callex API Integration & Testing Guide (v3.2)

> **Server IP Details**
> **Host IP:** `103.150.187.88`
> **Base URL:** `http://103.150.187.88:4500/api/v1`

This document has been updated following the latest migration to the new production server. It contains the exact URLs, authentication rules, and copy-paste cURL tests your developer needs to verify the new system is perfectly functional.

---

## 🔑 1. Authentication Updates (Important!)

The authentication gateway has been heavily upgraded to be universally compatible with your developer scripts and integrations.

You no longer need to strictly prefix your keys with `Bearer ` if you are passing raw API keys. 
You can use any of the following standard header formats to authenticate:

1. `Authorization: Bearer <token>` *(Standard JWT/Dashboard format)*
2. `Authorization: <token>` *(Raw API Key)*
3. `x-api-key: <token>` *(Alternative integration format)*
4. `api-key: <token>` 

> [!TIP]
> This resolves previous issues where external Python scripts passing raw API keys were blocked with "Missing or invalid Authorization header" errors.

---

## 🧪 2. System Verification (cURL Tests)

Your developer can run these direct terminal commands to visually verify the server routing, database queries, and data extraction pipelines are working.

*Note: Replace `<ACTUAL_API_KEY>` with your dashboard-generated Developer API Key or JWT Token.*

### Test A: List All Agents 
Verifies that the database is connected and endpoints are publicly accessible.

```bash
curl -X GET "http://103.150.187.88:4500/api/v1/agents" \
     -H "x-api-key: <ACTUAL_API_KEY>"
```

### Test B: Create a Testing Agent
Verifies that the `try/catch` and validation logic is functioning properly in the core pipeline, and that the Shadow Sandbox engine spins up the training agent automatically in the background.

```bash
curl -X POST "http://103.150.187.88:4500/api/v1/agents" \
     -H "Content-Type: application/json" \
     -H "x-api-key: <ACTUAL_API_KEY>" \
     -d '{
       "name": "Integration Test Agent",
       "systemPrompt": "You are a friendly customer service bot testing the API.",
       "language": "en-US",
       "voice": "alloy"
     }'
```

### Test C: Edit / Patch an Agent
To verify the authentication fix we just applied allows updating the system prompt gracefully.
*(Note: Replace `YOUR_AGENT_ID` with the ID returned by Test B)*

```bash
curl -X PUT "http://103.150.187.88:4500/api/v1/agents/YOUR_AGENT_ID" \
     -H "Content-Type: application/json" \
     -H "x-api-key: <ACTUAL_API_KEY>" \
     -d '{
       "systemPrompt": "Updated instructions applied perfectly via API.",
       "temperature": 0.8
     }'
```

---

## 🛑 3. Connecting the WebSockets (Real-Time Audio)

When connecting the real-time AI dialers into the application pipeline, developers should ensure their WebSocket URIs point to the actively port-forwarded audio bridge.

**New Server WS Node:** `ws://103.150.187.88:8085/v1/ws/call/:id`

> [!IMPORTANT]
> The backend automatically caches System Prompts into local RAM to handle simultaneous calls gracefully. Ensure you inform the developer that modifications to agent behavior via the REST API may take up to **30 seconds** to universally sync across massive inbound call structures. 
