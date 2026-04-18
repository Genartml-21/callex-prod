import { create } from 'zustand';

export const useStore = create((set, get) => ({
    // UI
    sidebarOpen: true,
    toast: null,
    showToast: (msg, type = 'success') => {
        set({ toast: { msg, type } });
        setTimeout(() => set({ toast: null }), 3500);
    },

    // Calls
    activeCalls: [],
    setActiveCalls: (calls) => set({ activeCalls: calls }),

    // Agents
    agents: [],
    setAgents: (agents) => set({ agents }),
    selectedAgent: null,
    setSelectedAgent: (agent) => set({ selectedAgent: agent }),

    // Campaigns
    campaigns: [],
    setCampaigns: (campaigns) => set({ campaigns }),

    // KPIs (dashboard)
    kpis: null,
    setKPIs: (kpis) => set({ kpis }),

    // Events
    events: [],
    addEvent: (event) => set(s => ({ events: [event, ...s.events].slice(0, 50) })),
}));
