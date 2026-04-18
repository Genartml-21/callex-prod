// API client — proxied through Vite to http://localhost:4000 (or backend in prod)
// Falls back to MOCK DATA when server is unreachable so the UI is fully testable.

const BASE = '/api';

// ─── Fetch wrapper ───────────────────────────────────────────────────────────
export async function apiFetch(path, options = {}) {
    try {
        // Get JWT token from localStorage (set during login)
        const token = localStorage.getItem('token');
        const authHeaders = token ? { Authorization: `Bearer ${token}` } : {};

        const res = await fetch(`${BASE}${path}`, {
            headers: { 'Content-Type': 'application/json', ...authHeaders, ...options.headers },
            ...options,
            body: options.body && typeof options.body !== 'string' ? JSON.stringify(options.body) : options.body,
        });

        if (!res.ok) {
            let errMsg = `API error ${res.status}`;
            try {
                const errData = await res.json();
                if (errData.error) errMsg = errData.error;
            } catch (e) { /* ignore parse error */ }
            throw new Error(errMsg);
        }

        // Return null/empty for 204 No Content
        if (res.status === 204) return null;
        return res.json();
    } catch (err) {
        console.error(`[API] ${path} failed:`, err.message);
        throw err; // Genuine production error throw
    }
}

export const api = {
    // Generic methods
    get: (path) => apiFetch(path),
    post: (path, data) => apiFetch(path, { method: 'POST', body: data }),
    patch: (path, data) => apiFetch(path, { method: 'PATCH', body: data }),
    delete: (path) => apiFetch(path, { method: 'DELETE' }),

    // Dashboard
    kpis: () => apiFetch('/dashboard/kpis'),
    abTest: () => apiFetch('/dashboard/ab-test'),
    events: () => apiFetch('/dashboard/events'),

    // Supervisor
    activeCalls: () => apiFetch('/supervisor/calls'),
    simulateCall: (data) => apiFetch('/supervisor/calls', { method: 'POST', body: data }),
    endCall: (id) => apiFetch(`/supervisor/calls/${id}/end`, { method: 'PATCH' }),
    whisper: (id, message) => apiFetch(`/supervisor/calls/${id}/whisper`, { method: 'POST', body: { message } }),
    exportReports: (type, range) => apiFetch(`/reports/export?type=${type}&range=${range}`, { method: 'GET' }),

    // Follow Ups
    followups: () => apiFetch('/followups'),
    createFollowUp: (data) => apiFetch('/followups', { method: 'POST', body: data }),
    setFollowUpStatus: (id, status) => apiFetch(`/followups/${id}/status`, { method: 'PATCH', body: { status } }),
    barge: (id) => apiFetch(`/supervisor/calls/${id}/barge`, { method: 'POST' }),
    transcript: (id) => apiFetch(`/supervisor/calls/${id}/transcript`),

    // Agents
    agents: () => apiFetch('/agents'),
    agent: (id) => apiFetch(`/agents/${id}`),
    createAgent: (data) => apiFetch('/agents', { method: 'POST', body: data }),
    updateAgent: (id, data) => apiFetch(`/agents/${id}`, { method: 'PATCH', body: data }),
    deleteAgent: (id) => apiFetch(`/agents/${id}`, { method: 'DELETE' }),
    agentPromptVersions: (id) => apiFetch(`/agents/${id}/prompt-versions`),
    savePromptVersion: (id, data) => apiFetch(`/agents/${id}/prompt-version`, { method: 'POST', body: data }),
    setAgentStatus: (id, status) => apiFetch(`/agents/${id}/status`, { method: 'PATCH', body: { status } }),
    ttsPreview: (data) => apiFetch('/agents/tts-preview', { method: 'POST', body: data }),
    cloneVoice: (formData) => fetch(`${BASE}/agents/clone-voice`, { method: 'POST', body: formData }).then(async r => { if (!r.ok) throw new Error(await r.text()); return r.json(); }),

    // Knowledge
    docs: () => apiFetch('/knowledge'),
    uploadDoc: (formData) => fetch(`${BASE}/knowledge`, { method: 'POST', body: formData }).then(r => r.json()).catch(() => ({ success: true, _mock: true })),
    deleteDoc: (id) => apiFetch(`/knowledge/${id}`, { method: 'DELETE' }),
    resyncDoc: (id) => apiFetch(`/knowledge/${id}/resync`, { method: 'POST' }),

    // Simulation
    runBatch: (data) => apiFetch('/simulation/batch', { method: 'POST', body: data }),
    runAdversarial: (data) => apiFetch('/simulation/adversarial', { method: 'POST', body: data }),
    agentChat: (data) => fetch(`${BASE}/simulation/agent-chat`, { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(data) }),
    agentChatStream: (data) => fetch(`${BASE}/simulation-stream/agent-chat-stream`, { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(data) }),
    agentChatPredictive: (data) => fetch(`${BASE}/simulation-stream/agent-chat-predictive`, { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(data) }).then(res => res.json()),

    // Dialer
    campaigns: () => apiFetch('/dialer/campaigns'),
    createCampaign: (data) => apiFetch('/dialer/campaigns', { method: 'POST', body: data }),
    updateCampaign: (id, data) => apiFetch(`/dialer/campaigns/${id}`, { method: 'PATCH', body: data }),
    setCampaignStatus: (id, status) => apiFetch(`/dialer/campaigns/${id}/status`, { method: 'PATCH', body: { status } }),
    deleteCampaign: (id) => apiFetch(`/dialer/campaigns/${id}`, { method: 'DELETE' }),

    // Analytics & Billing
    callLogs: (params = '') => apiFetch(`/analytics/calls${params}`),
    callDetail: (id) => apiFetch(`/analytics/calls/${id}`),
    triggerACW: (id) => apiFetch(`/analytics/calls/${id}/acw`, { method: 'POST' }),
    analyticsStats: () => apiFetch('/analytics/stats'),
    billingStats: () => apiFetch('/billing/stats'),

    // Routing
    routingRules: () => apiFetch('/routing/rules'),
    createRule: (data) => apiFetch('/routing/rules', { method: 'POST', body: data }),
    updateRule: (id, data) => apiFetch(`/routing/rules/${id}`, { method: 'PATCH', body: data }),
    deleteRule: (id) => apiFetch(`/routing/rules/${id}`, { method: 'DELETE' }),

    // Integrations
    integrations: () => apiFetch('/integrations'),
    connectIntegration: (id, config) => apiFetch(`/integrations/${id}/connect`, { method: 'PATCH', body: { config } }),
    disconnectIntegration: (id) => apiFetch(`/integrations/${id}/disconnect`, { method: 'PATCH' }),

    // Security
    voiceSignatures: () => apiFetch('/security/voice-signatures'),
    createVoiceSig: (data) => apiFetch('/security/voice-signatures', { method: 'POST', body: data }),
    deleteVoiceSig: (id) => apiFetch(`/security/voice-signatures/${id}`, { method: 'DELETE' }),

    // Settings
    apiKeys: () => apiFetch('/settings/api-keys'),
    createApiKey: (data) => apiFetch('/settings/api-keys', { method: 'POST', body: data }),
    deleteApiKey: (id) => apiFetch(`/settings/api-keys/${id}`, { method: 'DELETE' }),
    webhooks: () => apiFetch('/settings/webhooks'),
    createWebhook: (data) => apiFetch('/settings/webhooks', { method: 'POST', body: data }),
    updateWebhook: (id, data) => apiFetch(`/settings/webhooks/${id}`, { method: 'PATCH', body: data }),
    deleteWebhook: (id) => apiFetch(`/settings/webhooks/${id}`, { method: 'DELETE' }),
    testWebhook: (id) => apiFetch(`/settings/webhooks/${id}/test`, { method: 'POST' }),
};
