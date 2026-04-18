# Callex API — Quick URL Guide

This is a quick reference for the two main API endpoints you need to view your call data.

All requests require this Header:
```
Authorization: Bearer <YOUR_API_KEY>
```

---

## 1. View ALL Call History
This API returns a paginated list of all calls (both completed and active) for your account.

- **Method:** `GET`
- **URL:** `http://62.171.170.48:4500/api/v1/calls`
- **What it does:** Shows every call made or received.
- **Example Response:**
```json
{
  "calls": [ ... list of calls ... ],
  "pagination": { "page": 1, "limit": 20, "total": 150, "totalPages": 8 }
}
```

---

## 2. View LIVE / ACTIVE Calls
This API returns an array of calls that are currently happening *right now*. 

- **Method:** `GET`
- **URL:** `http://62.171.170.48:4500/api/v1/supervisor/calls`
- **What it does:** Shows live calls so you can monitor them or barge/whisper. 
- **Example Response:**
```json
[
  {
    "id": "abc1234",
    "status": "active",
    "agentName": "Sales Agent"
  }
]
```
*(Note: If no calls are currently live, this simply returns `[]`).*
