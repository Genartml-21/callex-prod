# Callex Voice AI — Custom Disposition API Protocol
> **Version:** 3.1 | **Engine:** Callex Voice AI Core

Welcome to the **Callex Voice AI Disposition Module**. A disposition is not just a tag indicating how a call ended; in the Callex ecosystem, a disposition dictates **exactly what data** the AI must extract from the customer interaction.

**Critical Workflow Requirement:** All custom dispositions *must* be linked to at least one active AI Agent upon creation. When configuring your application's UI, you should query the `/v1/agents` endpoint, display a list of available agents to the user, and pass their selected Agent IDs to the disposition creation payload.

---

## 🛠️ The Disposition Object Architecture

A customized disposition in Callex follows this exact JSON structure:

```json
{
  "id": "disp_8f72a9b3",
  "name": "Meeting Booked",
  "tagline": "The customer agreed to a follow-up meeting",
  "requiresNote": true,
  "category": "Sales",
  "requiredFields": [
    { "name": "Meeting Date", "type": "date", "description": "When the meeting is scheduled" },
    { "name": "Decision Maker Present", "type": "boolean", "description": "Did they confirm decision making power?" }
  ],
  "linkedAgents": ["agent_12345"],
  "linkedCampaigns": ["camp_98765"],
  "active": true
}
```

### Required Fields for Creation
* `name` *(String)*: The core title of the outcome (e.g., "Not Interested", "Sale Made").
* `linkedAgents` *(Array of Strings)*: An array of Agent IDs that will evaluate calls against this disposition. **You must pass at least one Agent ID.**

### Optional Context Fields
* `tagline` *(String)*: A brief sentence defining exactly what this disposition means, passed to the AI as context.
* `requiredFields` *(Array)*: An array of JSON objects. These define the "Schema" of information the agent is supposed to parse from the conversation. 
* `linkedCampaigns` *(Array of Strings)*: Connects this disposition to specific out-dialing Campaign IDs.
* `active` *(Boolean)*: Setting to `false` archives the disposition without deleting historical call data.

---

## 🚀 API Endpoints

**Base Endpoint URL:** `https://api.callex.ai/v1/dispositions`
**Authentication:** `Authorization: Bearer <YOUR_API_KEY>`

### 1. Create a New Custom Disposition
* **Method:** `POST`
* **URL:** `/v1/dispositions`

Creates a new intelligent disposition schema in your workspace and links it to the selected agents.

**Request Payload:**
```json
{
  "name": "High Value Lead",
  "tagline": "Customer has a budget over $10,000 and requires urgent sales followup",
  "category": "Hot Lead",
  "requiredFields": [
    { "name": "Estimated Budget", "type": "number" },
    { "name": "Timeline to Purchase", "type": "string" }
  ],
  "linkedAgents": ["agent_abc123", "agent_xyz789"]
}
```

**Success Response (201 Created):**
```json
{
  "message": "Disposition created successfully",
  "id": "disp_x92k3m1",
  ... (returns full object containing linkedAgents)
}
```
**Error Response (400 Bad Request):**
```json
{
  "error": "You must select at least one Agent to link this disposition to."
}
```

---

### 2. Update a Disposition & Manage Links
* **Method:** `PUT`
* **URL:** `/v1/dispositions/:id`

Modify an existing disposition. This is highly useful for dynamically **linking and unlinking** an existing disposition to newly created agents or campaigns.

**Request Payload:** *(All fields are optional, send only what you wish to update)*
```json
{
  "tagline": "Updated tagline for better AI comprehension",
  "linkedCampaigns": ["camp_123"],
  "linkedAgents": ["agent_abc123"]
}
```

---

### 3. Fetch All Available Agents (For UI linking)
* **Method:** `GET`
* **URL:** `/v1/agents`

Before creating a disposition, use this endpoint to populate a dropdown menu in your user interface, allowing the user to select which Agents the disposition applies to.

---

### 4. Fetch / List Your Dispositions
* **Method:** `GET`
* **URL:** `/v1/dispositions`
* **Query Params:** `?page=1&limit=50`

Returns a paginated array of all custom dispositions in your Callex workspace. Filter this list dynamically on your frontend by inspecting the `linkedAgents` arrays to display only relevant possibilities when rendering a specific Agent's settings module.

**Success Response (200 OK):**
```json
{
  "dispositions": [
    {
      "id": "disp_8f72a9b3",
      "name": "Meeting Booked",
      "linkedAgents": ["agent_12345"]
    }
  ],
  "pagination": {
    "page": 1,
    "limit": 50,
    "total": 12,
    "totalPages": 1
  }
}
```

---

### 5. Archive / Delete a Disposition
* **Method:** `DELETE`
* **URL:** `/v1/dispositions/:id`

Permanently deletes the custom disposition.
*(Note: If you wish to keep historical reporting intact, we strongly recommend using the `PUT` endpoint to set `"active": false` instead of outright deletion).*

---

## 🧠 How Callex Voice AI Uses This Data

When a live call concludes, the Callex Voice AI pipeline evaluates the transcript strictly against the **active dispositions** embedded inside the `linkedAgents` array for the AI Agent that handled the call. 

If the AI determines the outcome logically matches the `tagline` of a disposition, it will automatically attempt to extract every variable listed inside `requiredFields` directly from the conversational context, delivering a perfectly structured JSON object back to your webhook or API dashboard infrastructure. 
