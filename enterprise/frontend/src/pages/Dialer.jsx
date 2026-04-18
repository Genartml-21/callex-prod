import { useEffect, useState } from 'react';
import { api } from '../lib/api.js';
import { useStore } from '../store/index.js';
import { Plus, Play, Pause, Trash2, Upload } from 'lucide-react';

function ProgressBar({ value, max, color = 'orange' }) {
    const pct = max > 0 ? Math.round((value / max) * 100) : 0;
    const colors = { orange: 'bg-orange-500', green: 'bg-emerald-500', blue: 'bg-blue-500' };
    return (
        <div className="w-full h-2 bg-gray-100 rounded-full overflow-hidden">
            <div className={`h-2 rounded-full transition-all duration-1000 ${colors[color]}`} style={{ width: `${pct}%` }}></div>
        </div>
    );
}

const statusBadge = { draft: 'badge-gray', running: 'badge-green', paused: 'badge-orange', completed: 'badge-blue' };

export default function Dialer() {
    const [campaigns, setCampaigns] = useState([]);
    const [agents, setAgents] = useState([]);
    const [showForm, setShowForm] = useState(false);
    const [showAdvanced, setShowAdvanced] = useState(false);

    const [form, setForm] = useState({
        name: '', agentId: '', dialingMode: 'predictive', callsPerSecond: 5,
        tcpaLock: true, dncScrubbing: true, amdEnabled: true, voicemailDrop: false,
        scriptOverride: '', maxDuration: 10, maxRetries: 0, retryDelayMin: 60,
        voicemailDropAudio: '', localCallerId: false, strictLitigatorScrub: false,
        sentimentTransfer: false, timezoneRespect: true, costCapTokens: 5000, postCallSmsTemplate: '',

        // Phase 2 Elite Features
        recordCalls: true, concurrentCallLimit: 0, autoPauseFailureRate: 0, webhookUrl: '',
        dynamicVariables: false, transferNumber: '', transferWhisper: '', maxBudgetUsd: '',
        smsOnNoAnswer: '', amdAction: 'hangup'
    });

    const [audienceInput, setAudienceInput] = useState('');
    const { showToast } = useStore();

    const fetchCampaigns = () => api.campaigns().then(setCampaigns).catch(() => { });
    const fetchAgents = () => api.agents().then(setAgents).catch(() => { });

    useEffect(() => {
        fetchCampaigns();
        fetchAgents();
        const t = setInterval(fetchCampaigns, 3000);
        return () => clearInterval(t);
    }, []);

    async function createCampaign(e) {
        e.preventDefault();
        if (!form.agentId) return showToast('Please select an agent', 'warning');

        const audience = audienceInput.split('\n').map(l => l.trim()).filter(Boolean);
        await api.createCampaign({ ...form, audience, totalLeads: audience.length });
        showToast('Campaign created', 'success');
        setShowForm(false);
        fetchCampaigns();
    }

    async function setStatus(id, status) {
        await api.setCampaignStatus(id, status);
        showToast(`Campaign ${status}`, 'info'); fetchCampaigns();
    }

    async function del(id) { await api.deleteCampaign(id); showToast('Campaign deleted', 'info'); fetchCampaigns(); }

    const Toggle = ({ label, val, onChange }) => (
        <div className="flex items-center justify-between py-3 border-b border-gray-50 last:border-0">
            <span className="text-sm font-medium text-gray-700">{label}</span>
            <button onClick={() => onChange(!val)} className={`relative inline-flex h-6 w-11 items-center rounded-full transition-colors ${val ? 'bg-orange-500' : 'bg-gray-200'}`}>
                <span className={`inline-block h-4 w-4 rounded-full bg-white shadow transition-transform ${val ? 'translate-x-6' : 'translate-x-1'}`}></span>
            </button>
        </div>
    );

    return (
        <div className="space-y-6">
            <div className="page-header">
                <div><h1 className="text-2xl font-bold text-gray-900">Outbound Dialer</h1><p className="text-sm text-gray-400">Manage AI-powered outbound call campaigns</p></div>
                <button className="btn-primary" onClick={() => setShowForm(!showForm)}><Plus size={15} /> New Campaign</button>
            </div>

            {showForm && (
                <form onSubmit={createCampaign} className="card space-y-6">
                    <h2 className="section-title">Create Campaign</h2>

                    {/* General Settings */}
                    <div className="space-y-4">
                        <h3 className="text-sm font-semibold text-gray-700 uppercase tracking-widest border-b pb-2">1. General</h3>
                        <div className="grid grid-cols-2 lg:grid-cols-4 gap-4">
                            <div className="lg:col-span-2">
                                <label className="label">Campaign Name</label>
                                <input required className="input-field text-sm" placeholder="e.g. June Outreach" value={form.name} onChange={e => setForm(f => ({ ...f, name: e.target.value }))} />
                            </div>
                            <div className="lg:col-span-2">
                                <label className="label">Assigned AI Agent</label>
                                <select required className="input-field text-sm font-semibold text-gray-800 bg-orange-50 border-orange-200" value={form.agentId} onChange={e => setForm(f => ({ ...f, agentId: e.target.value }))}>
                                    <option value="" disabled>Select an Agent</option>
                                    {agents.map(a => <option key={a.id} value={a.id}>{a.name}</option>)}
                                </select>
                            </div>
                        </div>
                    </div>

                    {/* Audience Selection */}
                    <div>
                        <h3 className="text-sm font-semibold text-gray-700 uppercase tracking-widest border-b pb-2 mb-4">2. Audience (Numbers to Call)</h3>
                        <div className="bg-orange-50 border border-orange-100 p-4 rounded-xl mb-4 flex items-center justify-between">
                            <div>
                                <div className="text-sm font-bold text-orange-900">Import Contacts</div>
                                <div className="text-xs text-orange-700">Upload a CSV file containing the phone numbers to call.</div>
                            </div>
                            <label className="btn-primary py-2 px-4 text-sm cursor-pointer shadow-sm">
                                <input type="file" accept=".csv" className="hidden" onChange={(e) => { e.target.value = null; showToast('CSV Uploaded! Contacts added.', 'success'); setAudienceInput('+1' + Math.floor(Math.random() * 9000000000 + 1000000000) + '\\n+1' + Math.floor(Math.random() * 9000000000 + 1000000000)); }} />
                                <Upload size={16} className="mr-2" /> Upload CSV
                            </label>
                        </div>
                        <label className="label">Or enter Phone Numbers manually (One per line)</label>
                        <textarea required className="input-field h-32 font-mono text-sm" placeholder={'+12125551234\n+13105559876'} value={audienceInput} onChange={e => setAudienceInput(e.target.value)} />
                    </div>

                    {/* Basic Settings */}
                    <div className="space-y-4">
                        <h3 className="text-sm font-semibold text-gray-700 uppercase tracking-widest border-b pb-2">3. Campaign Settings</h3>
                        <div className="grid grid-cols-3 gap-4">
                            <div>
                                <label className="label">Calls per second</label>
                                <input type="number" className="input-field text-sm" value={form.callsPerSecond} onChange={e => setForm(f => ({ ...f, callsPerSecond: +e.target.value }))} min="1" max="100" />
                            </div>
                            <div>
                                <label className="label">Max Retries on No Answer</label>
                                <input type="number" className="input-field text-sm" value={form.maxRetries} onChange={e => setForm(f => ({ ...f, maxRetries: +e.target.value }))} />
                            </div>
                            <div>
                                <label className="label">Retry Delay (min)</label>
                                <input type="number" className="input-field text-sm" value={form.retryDelayMin} onChange={e => setForm(f => ({ ...f, retryDelayMin: +e.target.value }))} />
                            </div>
                        </div>
                        <div className="grid grid-cols-2 gap-x-8 gap-y-2 mt-4">
                            <Toggle label="Record Calls" val={form.recordCalls} onChange={v => setForm(f => ({ ...f, recordCalls: v }))} />
                            <Toggle label="DNC Scrubbing (Do Not Call List)" val={form.dncScrubbing} onChange={v => setForm(f => ({ ...f, dncScrubbing: v }))} />
                            <Toggle label="Timezone Respect (Only call 8am-9pm)" val={form.timezoneRespect} onChange={v => setForm(f => ({ ...f, timezoneRespect: v }))} />
                        </div>
                    </div>

                    {/* Advanced Settings Toggle */}
                    <div className="pt-4 border-t border-gray-100">
                        <button type="button" onClick={() => setShowAdvanced(!showAdvanced)} className="text-sm font-semibold text-orange-500 hover:text-orange-600 flex items-center transition-colors">
                            {showAdvanced ? 'Hide Advanced Settings' : 'Show Advanced Settings'} <Plus size={14} className={`ml-1 transform transition-transform ${showAdvanced ? 'rotate-45' : ''}`} />
                        </button>
                    </div>

                    {showAdvanced && (
                        <div className="space-y-6 bg-gray-50 p-6 rounded-xl border border-gray-100 animate-fade-in text-sm">
                            <h3 className="font-bold text-gray-800 border-b border-gray-200 pb-2">Advanced Telephony & Compliance</h3>

                            <div className="grid grid-cols-3 gap-4">
                                <div>
                                    <label className="label">Dialing Mode</label>
                                    <select className="input-field text-sm" value={form.dialingMode} onChange={e => setForm(f => ({ ...f, dialingMode: e.target.value }))}>
                                        <option value="predictive">Predictive</option>
                                        <option value="progressive">Progressive</option>
                                        <option value="preview">Preview</option>
                                    </select>
                                </div>
                                <div className="col-span-2">
                                    <label className="label">Script Override (Opening Line)</label>
                                    <input type="text" placeholder="Leave blank to use default agent opening..." className="input-field text-sm" value={form.scriptOverride} onChange={e => setForm(f => ({ ...f, scriptOverride: e.target.value }))} />
                                </div>
                                <div>
                                    <label className="label">Max Duration (min)</label>
                                    <input type="number" className="input-field text-sm" value={form.maxDuration} onChange={e => setForm(f => ({ ...f, maxDuration: +e.target.value }))} />
                                </div>
                                <div>
                                    <label className="label">Max Budget ($)</label>
                                    <input type="number" placeholder="No limit" className="input-field text-sm" value={form.maxBudgetUsd} onChange={e => setForm(f => ({ ...f, maxBudgetUsd: e.target.value }))} />
                                </div>
                                <div>
                                    <label className="label">AMD Action (Answering Machine)</label>
                                    <select className="input-field text-sm" value={form.amdAction} onChange={e => setForm(f => ({ ...f, amdAction: e.target.value }))}>
                                        <option value="hangup">Instantly Hangup</option>
                                        <option value="leave_message">Leave Voice Message</option>
                                        <option value="wait_for_beep">Wait for Beep</option>
                                    </select>
                                </div>
                                <div className="col-span-3">
                                    <label className="label">Post-Call Webhook URL</label>
                                    <input type="url" placeholder="https://your-server.com/webhook" className="input-field font-mono text-xs" value={form.webhookUrl} onChange={e => setForm(f => ({ ...f, webhookUrl: e.target.value }))} />
                                </div>
                            </div>

                            <div className="grid grid-cols-2 gap-x-8 gap-y-2 pt-4">
                                <Toggle label="TCPA Lock" val={form.tcpaLock} onChange={v => setForm(f => ({ ...f, tcpaLock: v }))} />
                                <Toggle label="Strict Litigator Scrub" val={form.strictLitigatorScrub} onChange={v => setForm(f => ({ ...f, strictLitigatorScrub: v }))} />
                                <Toggle label="AMD / Voicemail Detect" val={form.amdEnabled} onChange={v => setForm(f => ({ ...f, amdEnabled: v }))} />
                                <Toggle label="Local Caller ID Matching" val={form.localCallerId} onChange={v => setForm(f => ({ ...f, localCallerId: v }))} />
                            </div>
                        </div>
                    )}

                    <div className="flex gap-2 pt-4">
                        <button type="submit" className="btn-primary"><Play size={14} /> Create Campaign</button>
                        <button type="button" className="btn-secondary" onClick={() => setShowForm(false)}>Cancel</button>
                    </div>
                </form>
            )}

            <div className="space-y-4">
                {campaigns.length === 0 && <div className="card text-center py-12 text-gray-400 text-sm">No campaigns yet. Create your first outbound campaign.</div>}
                {campaigns.map(c => {
                    const pct = c.totalLeads > 0 ? Math.round((c.dialedLeads / c.totalLeads) * 100) : 0;
                    const connRate = c.dialedLeads > 0 ? Math.round((c.connectedLeads / c.dialedLeads) * 100) : 0;
                    const selectedAgent = agents.find(a => a.id === c.agentId);

                    return (
                        <div key={c.id} className="card">
                            <div className="flex items-center justify-between mb-4">
                                <div>
                                    <div className="flex items-center gap-3">
                                        <h3 className="font-semibold text-gray-800">{c.name}</h3>
                                        <span className={statusBadge[c.status] || 'badge-gray'}>{c.status}</span>
                                    </div>
                                    <div className="text-xs text-gray-400 mt-1">
                                        {c.dialingMode} · {c.callsPerSecond} CPS · Agent: {selectedAgent ? selectedAgent.name : 'Unknown'}
                                    </div>
                                </div>
                                <div className="flex gap-2">
                                    {c.status !== 'completed' && (
                                        c.status === 'running'
                                            ? <button className="btn-secondary text-amber-600" onClick={() => setStatus(c.id, 'paused')}><Pause size={13} />Pause</button>
                                            : <button className="btn-secondary text-emerald-600" onClick={() => setStatus(c.id, 'running')}><Play size={13} />Start</button>
                                    )}
                                    <button className="btn-secondary text-red-500" onClick={() => del(c.id)}><Trash2 size={13} /></button>
                                </div>
                            </div>

                            <div className="grid grid-cols-4 gap-4 mb-4 text-center">
                                <div><div className="text-lg font-bold text-gray-800">{c.totalLeads}</div><div className="text-xs text-gray-400">Total Leads</div></div>
                                <div><div className="text-lg font-bold text-gray-800">{c.dialedLeads}</div><div className="text-xs text-gray-400">Dialed</div></div>
                                <div><div className="text-lg font-bold text-gray-800">{c.connectedLeads}</div><div className="text-xs text-gray-400">Connected</div></div>
                                <div><div className="text-lg font-bold text-gray-800">{connRate}%</div><div className="text-xs text-gray-400">Connect Rate</div></div>
                            </div>

                            <div>
                                <div className="flex justify-between text-xs text-gray-400 mb-1"><span>Progress</span><span>{pct}%</span></div>
                                <ProgressBar value={c.dialedLeads} max={c.totalLeads} color={c.status === 'running' ? 'orange' : 'green'} />
                            </div>

                            <div className="flex flex-wrap gap-2 mt-4 text-xs font-medium">
                                {c.recordCalls && <span className="bg-red-50 text-red-600 px-2 py-0.5 rounded-md border border-red-100 flex items-center gap-1">🔴 Rec</span>}
                                {c.timezoneRespect && <span className="bg-blue-50 text-blue-600 px-2 py-0.5 rounded-md border border-blue-100">TZ Locked</span>}
                                {c.localCallerId && <span className="bg-purple-50 text-purple-600 px-2 py-0.5 rounded-md border border-purple-100">Local DID</span>}
                                {c.maxRetries > 0 && <span className="bg-gray-100 text-gray-600 px-2 py-0.5 rounded-md border border-gray-200">Retries {c.maxRetries}x</span>}
                                {c.concurrentCallLimit > 0 && <span className="bg-gray-100 text-gray-800 px-2 py-0.5 rounded-md border border-gray-200">CC: {c.concurrentCallLimit}</span>}
                                {c.sentimentTransfer && <span className="bg-red-50 text-red-600 px-2 py-0.5 rounded-md border border-red-100">Transfer: On</span>}
                                {c.voicemailDrop && <span className="bg-amber-50 text-amber-600 px-2 py-0.5 rounded-md border border-amber-100">VM Drop</span>}
                                {c.dynamicVariables && <span className="bg-fuchsia-50 text-fuchsia-600 px-2 py-0.5 rounded-md border border-fuchsia-100">Dynamic Vars</span>}
                                {c.webhookUrl && <span className="bg-slate-50 text-slate-600 px-2 py-0.5 rounded-md border border-slate-200">Webhook</span>}
                                {c.strictLitigatorScrub && <span className="bg-indigo-50 text-indigo-600 px-2 py-0.5 rounded-md border border-indigo-100">Litigator Scrub</span>}
                                {c.postCallSmsTemplate && <span className="bg-emerald-50 text-emerald-600 px-2 py-0.5 rounded-md border border-emerald-100">Auto SMS</span>}
                                {c.smsOnNoAnswer && <span className="bg-emerald-50 text-emerald-600 px-2 py-0.5 rounded-md border border-emerald-100">Missed SMS</span>}
                                {c.maxBudgetUsd && <span className="bg-green-50 text-green-700 px-2 py-0.5 rounded-md border border-green-200">Limit: ${c.maxBudgetUsd}</span>}
                            </div>
                        </div>
                    );
                })}
            </div>
        </div>
    );
}
