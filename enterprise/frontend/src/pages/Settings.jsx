import { useEffect, useState } from 'react';
import { api } from '../lib/api.js';
import { useStore } from '../store/index.js';
import { Plus, Trash2, Key, Webhook, Copy, Eye, EyeOff, Send } from 'lucide-react';

const EVENT_OPTIONS = ['call.started', 'call.completed', 'call.transferred', 'call.failed', 'campaign.started', 'campaign.completed', 'agent.error'];

export default function Settings() {
    const [apiKeys, setApiKeys] = useState([]);
    const [webhooks, setWebhooks] = useState([]);
    const [newKeyName, setNewKeyName] = useState('');
    const [newKeyEnv, setNewKeyEnv] = useState('test');
    const [createdKey, setCreatedKey] = useState(null);
    const [whForm, setWhForm] = useState({ url: '', events: [], secret: '' });
    const [showWHForm, setShowWHForm] = useState(false);
    const { showToast } = useStore();

    const fetchApiKeys = () => api.apiKeys().then(setApiKeys).catch(() => { });
    const fetchWebhooks = () => api.webhooks().then(setWebhooks).catch(() => { });

    useEffect(() => { fetchApiKeys(); fetchWebhooks(); }, []);

    async function createKey(e) {
        e.preventDefault();
        const key = await api.createApiKey({ name: newKeyName, env: newKeyEnv });
        setCreatedKey(key); setNewKeyName(''); fetchApiKeys();
        showToast(`API key "${key.name}" created — copy it now, it won't be shown again!`, 'info');
    }

    async function deleteKey(id) {
        await api.deleteApiKey(id);
        showToast('API key revoked', 'info'); fetchApiKeys();
    }

    async function createWebhook(e) {
        e.preventDefault();
        await api.createWebhook(whForm);
        showToast('Webhook saved', 'success'); setShowWHForm(false); setWhForm({ url: '', events: [], secret: '' }); fetchWebhooks();
    }

    async function deleteWebhook(id) { await api.deleteWebhook(id); fetchWebhooks(); showToast('Webhook deleted', 'info'); }

    async function testWebhook(id) {
        const res = await api.testWebhook(id);
        showToast(res.message, 'success');
    }

    function toggleEvent(ev) {
        setWhForm(f => ({ ...f, events: f.events.includes(ev) ? f.events.filter(e => e !== ev) : [...f.events, ev] }));
    }

    function copyToClipboard(text) { navigator.clipboard.writeText(text); showToast('Copied to clipboard', 'success'); }

    return (
        <div className="space-y-8">
            <div className="page-header">
                <div><h1 className="text-2xl font-bold text-gray-900">Enterprise Settings</h1><p className="text-sm text-gray-400">API keys, webhooks, and platform configuration</p></div>
            </div>

            {/* API Keys */}
            <div className="card space-y-4">
                <h2 className="section-title flex items-center gap-2"><Key size={16} className="text-orange-500" />API Keys</h2>
                <form onSubmit={createKey} className="flex gap-3">
                    <input required className="input-field" placeholder="Key name (e.g. Production App)" value={newKeyName} onChange={e => setNewKeyName(e.target.value)} />
                    <select className="input-field w-36" value={newKeyEnv} onChange={e => setNewKeyEnv(e.target.value)}>
                        <option value="test">Test</option><option value="live">Live</option>
                    </select>
                    <button type="submit" className="btn-primary shrink-0"><Plus size={14} />Generate</button>
                </form>

                {createdKey && (
                    <div className="p-4 bg-orange-50 border border-orange-200 rounded-xl">
                        <div className="text-xs font-semibold text-orange-600 mb-2 flex items-center gap-1"><Key size={12} />New Key — Copy it now, shown only once</div>
                        <div className="flex items-center gap-2">
                            <code className="text-xs font-mono bg-white border border-orange-100 px-3 py-2 rounded-lg flex-1 truncate">{createdKey.fullKey}</code>
                            <button className="btn-primary" onClick={() => copyToClipboard(createdKey.fullKey)}><Copy size={13} />Copy</button>
                        </div>
                    </div>
                )}

                <div className="divide-y divide-gray-50">
                    {apiKeys.length === 0 && <p className="text-sm text-gray-400 text-center py-6">No API keys yet.</p>}
                    {apiKeys.map(k => (
                        <div key={k.id} className="flex items-center gap-4 py-3">
                            <span className={k.env === 'live' ? 'badge-orange' : 'badge-gray'}>{k.env}</span>
                            <div className="flex-1"><div className="text-sm font-medium text-gray-800">{k.name}</div><div className="text-xs font-mono text-gray-400">{k.prefix}••••••••</div></div>
                            <div className="text-xs text-gray-300">{k.lastUsed ? `Used ${new Date(k.lastUsed).toLocaleDateString()}` : 'Never used'}</div>
                            <button className="p-1.5 hover:bg-red-50 rounded-lg text-red-400" onClick={() => deleteKey(k.id)}><Trash2 size={13} /></button>
                        </div>
                    ))}
                </div>
            </div>

            {/* Webhooks */}
            <div className="card space-y-4">
                <div className="flex items-center justify-between">
                    <h2 className="section-title flex items-center gap-2"><Webhook size={16} className="text-orange-500" />Webhooks</h2>
                    <button className="btn-primary" onClick={() => setShowWHForm(!showWHForm)}><Plus size={14} />Add Webhook</button>
                </div>

                {showWHForm && (
                    <form onSubmit={createWebhook} className="p-4 bg-gray-50 rounded-xl space-y-3 border border-gray-100">
                        <div><label className="label">Endpoint URL</label><input required className="input-field" type="url" placeholder="https://your-server.com/webhook" value={whForm.url} onChange={e => setWhForm(f => ({ ...f, url: e.target.value }))} /></div>
                        <div><label className="label">Secret (for HMAC Signature)</label><input className="input-field" placeholder="Optional webhook secret" value={whForm.secret} onChange={e => setWhForm(f => ({ ...f, secret: e.target.value }))} /></div>
                        <div>
                            <label className="label">Subscribe to Events</label>
                            <div className="flex flex-wrap gap-2 mt-1">
                                {EVENT_OPTIONS.map(ev => (
                                    <button key={ev} type="button" onClick={() => toggleEvent(ev)} className={`text-xs px-3 py-1.5 rounded-lg border transition-all font-medium ${whForm.events.includes(ev) ? 'bg-orange-500 text-white border-orange-500' : 'border-gray-200 text-gray-500 hover:border-orange-300'}`}>{ev}</button>
                                ))}
                            </div>
                        </div>
                        <div className="flex gap-2"><button type="submit" className="btn-primary">Save Webhook</button><button type="button" className="btn-secondary" onClick={() => setShowWHForm(false)}>Cancel</button></div>
                    </form>
                )}

                <div className="divide-y divide-gray-50">
                    {webhooks.length === 0 && <p className="text-sm text-gray-400 text-center py-6">No webhooks configured.</p>}
                    {webhooks.map(wh => (
                        <div key={wh.id} className="py-3">
                            <div className="flex items-start gap-3">
                                <div className="flex-1 min-w-0">
                                    <div className="text-sm font-mono text-gray-700 truncate">{wh.url}</div>
                                    <div className="flex flex-wrap gap-1 mt-1.5">
                                        {(JSON.parse(wh.events || '[]')).map(ev => <span key={ev} className="badge-orange" style={{ fontSize: '10px' }}>{ev}</span>)}
                                    </div>
                                </div>
                                <div className="flex gap-1 shrink-0">
                                    <button className="btn-secondary py-1.5 text-xs" onClick={() => testWebhook(wh.id)}><Send size={12} />Test</button>
                                    <button className="p-1.5 hover:bg-red-50 rounded-lg text-red-400" onClick={() => deleteWebhook(wh.id)}><Trash2 size={13} /></button>
                                </div>
                            </div>
                        </div>
                    ))}
                </div>
            </div>
        </div>
    );
}
