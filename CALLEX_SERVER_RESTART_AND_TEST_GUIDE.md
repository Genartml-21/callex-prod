# Callex — Production QA & Server Restart Guide

> **Version:** 1.0 | **Date:** March 17, 2026 | **Prepared by:** Callex Engineering

---

## The Problem — Why APIs Return "Invalid or expired token"

The error `{"error": "Invalid or expired token"}` means the **server has not been restarted** after the latest code was pulled.

The new code fixes are on GitHub but the server is still running the **old code** in memory. You must restart the server every time code is pulled.

---

## Step 1 — SSH Into the Server

```bash
ssh root@62.171.170.48
```

Navigate to the project directory:

```bash
cd /usr/src/sumit/cx-voice/freeswitch-cx-voice-bridge
```

---

## Step 2 — Pull Latest Code (Do This Every Time)

```bash
# Discard any local server changes and pull fresh
git stash
git pull origin main
```

Expected output:
```
Already up to date.
```
or
```
Updating abc1234..def5678
 enterprise/backend/src/routes/analytics.js  | 99 ++++
 enterprise/backend/src/routes/external.js   | 23 ++
 app/main.py                                 | 76 +-
```

---

## Step 3 — Restart All Services

```bash
pm2 restart all
```

To verify they restarted:
```bash
pm2 list
```

You should see all processes with status `online` and a fresh uptime (seconds, not hours).

If you only want to restart the Node.js backend:
```bash
pm2 restart enterprise-backend
```

If you only want to restart the Python voice engine:
```bash
pm2 restart callex-python
```

> **Check `pm2 list` to see the exact process names on your server.**

---

## Step 4 — Verify the Server is Running

```bash
curl http://localhost:4500/api/health
```

Expected response:
```json
{"status": "ok", "ts": "2026-03-17T13:30:00.000Z"}
```

---

## Step 5 — Test Both APIs in Postman

### API Key to use (yours):
```
ck_live_d3188b50_94a44765acfc97d19ced4416
```

---

### TEST 1: Identity Check (Run This First)

`GET http://62.171.170.48:4500/api/v1/debug/my-identity`

**Header:**
```
Authorization: Bearer ck_live_d3188b50_94a44765acfc97d19ced4416
```

**What to check in the response:**

```json
{
  "userId": "abc123",
  "ownedAgents": 3,
  "agentIds": ["agent1", "agent2", "agent3"],
  "callsWithUserId": 5
}
```

| Field | What it Means |
|-------|---------------|
| `ownedAgents` | Number of AI agents linked to this API key. Must be > 0 |
| `callsWithUserId` | Calls directly tagged with this userId |

> ⚠️ **If `ownedAgents` is 0** — your API key belongs to a different user account than the one that has agents. This is a data mismatch and needs to be fixed in Firestore.

---

### TEST 2: Call History (All Calls)

`GET http://62.171.170.48:4500/api/v1/calls`

**Header:**
```
Authorization: Bearer ck_live_d3188b50_94a44765acfc97d19ced4416
```

**Expected Response:**
```json
{
  "calls": [...],
  "pagination": { "page": 1, "limit": 20, "total": 47, "totalPages": 3 }
}
```

**What it returns:** All completed + active calls for this account, sorted newest first.

> If `calls` is empty but `ownedAgents > 0` — calls exist but have no `userId` field. This is the older call data issue. New calls made after the restart will appear correctly.

---

### TEST 3: Active Calls (Live Right Now)

`GET http://62.171.170.48:4500/api/v1/supervisor/calls`

**Header:**
```
Authorization: Bearer ck_live_d3188b50_94a44765acfc97d19ced4416
```

**Expected Response (during an active call):**
```json
[
  {
    "id": "abc123-def456",
    "phoneNumber": "+919876543210",
    "agentId": "agent-id-here",
    "agentName": "Priya - Sales",
    "status": "active",
    "sentiment": "neutral",
    "startedAt": "2026-03-17T13:30:00.000Z"
  }
]
```

**Expected Response (no active calls):**
```json
[]
```

> Empty array `[]` means no one is currently on a live call. This is correct — not an error.

---

## Troubleshooting Guide

| Error | Cause | Fix |
|-------|-------|-----|
| `{"error": "Invalid or expired token"}` | Server not restarted | Run `git stash && git pull && pm2 restart all` |
| `{"error": "Missing or invalid Authorization header."}` | Wrong format | Header must be exactly: `Authorization: Bearer ck_live_...` |
| `{"error": "Invalid or revoked API Key."}` | API key is wrong or revoked | Generate a new key from Dashboard → Settings |
| `calls: []` (empty, no error) | No calls for this account yet | Make a test call, then check again |
| `ownedAgents: 0` in identity check | API key belongs to wrong user | Check which account created the API key |
| `{"error": "Not found"}` on `/api/v1/...` | Server running old code | Restart with `pm2 restart all` |
| `500 Internal Server Error` | Firestore connection issue | Check `pm2 logs` on server |

---

## What Was Fixed (March 17, 2026)

### Fix 1 — Call History List Was Returning 404
**File:** `enterprise/backend/src/routes/analytics.js`

The `GET /api/analytics/calls` endpoint (used by the dashboard UI) did not exist. Added it with:
- Direct userId match
- AgentId fallback for older calls missing userId
- Pagination support

### Fix 2 — Call Completion Not Saving Duration
**File:** `app/main.py`

When a call ended, the Firestore update was using a fragile query to find the document. Fixed to:
- Directly use call UUID as document ID
- Save `duration` field (seconds)
- Always save `userId` on call completion

### Fix 3 — Whisper/Barge Returning 404 for Older Calls
**File:** `enterprise/backend/src/routes/external.js`

Whisper and Barge endpoints were doing strict `call.userId === apiUserId` check. Fixed to also check if the call's `agentId` belongs to this user — so older calls without userId still work.

---

## Quick Command Reference (Server)

```bash
# Pull latest code
git stash && git pull origin main

# Restart everything
pm2 restart all

# Check all process status
pm2 list

# View live logs (Node.js backend)
pm2 logs enterprise-backend --lines 50

# View live logs (Python voice engine)
pm2 logs callex-python --lines 50

# Health check
curl http://localhost:4500/api/health
```

---

## Postman Quick Setup

1. Open Postman
2. New Collection → "Callex API"
3. Add a **Collection Variable**: `api_key` = `ck_live_d3188b50_94a44765acfc97d19ced4416`
4. In each request, Header: `Authorization` → `Bearer {{api_key}}`

| Request Name | Method | URL |
|-------------|--------|-----|
| Identity Check | GET | `http://62.171.170.48:4500/api/v1/debug/my-identity` |
| All Call History | GET | `http://62.171.170.48:4500/api/v1/calls` |
| Active Calls | GET | `http://62.171.170.48:4500/api/v1/supervisor/calls` |
| List Agents | GET | `http://62.171.170.48:4500/api/v1/agents` |

---

> **Remember:** Every time code is updated on GitHub, you must `git pull` + `pm2 restart all` on the server. Without restart, the old code keeps running.
