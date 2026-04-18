import { useEffect, useState } from 'react';
import { api } from '../lib/api.js';
import { useStore } from '../store/index.js';
import { Plus, Trash2, Edit, MessageSquare } from 'lucide-react';

export default function Routing() {
    const [rules, setRules] = useState([]);
    const [showForm, setShowForm] = useState(false);
    const [editRule, setEditRule] = useState(null);
    const [form, setForm] = useState({ name: '', intentTag: '', destination: 'agent', priority: 0, whisperMsg: '', ctiDelay: 0, smsEscalation: false });
    const { showToast } = useStore();

    const fetchRules = () => api.routingRules().then(setRules).catch(() => { });
    useEffect(() => { fetchRules(); }, []);

    async function submit(e) {
        e.preventDefault();
        if (editRule) { await api.updateRule(editRule.id, form); showToast('Rule updated', 'success'); }
        else { await api.createRule(form); showToast('Rule created', 'success'); }
        setShowForm(false); setEditRule(null); setForm({ name: '', intentTag: '', destination: 'agent', priority: 0, whisperMsg: '', ctiDelay: 0, smsEscalation: false }); fetchRules();
    }

    function startEdit(rule) { setEditRule(rule); setForm(rule); setShowForm(true); }
    async function del(id) { await api.deleteRule(id); showToast('Rule deleted', 'info'); fetchRules(); }

    const Toggle = ({ label, hint, val, onChange }) => (
        <div className="border border-gray-100 rounded-xl p-3 flex items-center justify-between">
            <div><div className="text-sm font-medium text-gray-700">{label}</div>{hint && <div className="text-xs text-gray-400">{hint}</div>}</div>
            <button onClick={() => onChange(!val)} className={`relative inline-flex h-5 w-9 rounded-full transition-colors ${val ? 'bg-orange-500' : 'bg-gray-200'}`}>
                <span className={`inline-block h-3 w-3 mt-1 ml-1 rounded-full bg-white shadow transition-transform ${val ? 'translate-x-4' : ''}`}></span>
            </button>
        </div>
    );

    return (
        <div className="space-y-6">
            <div className="page-header">
                <div><h1 className="text-2xl font-bold text-gray-900">Routing & Human Handoffs</h1><p className="text-sm text-gray-400">Skills-based routing with sentiment-buffered handoffs</p></div>
                <button className="btn-primary" onClick={() => { setEditRule(null); setForm({ name: '', intentTag: '', destination: 'agent', priority: 0, whisperMsg: '', ctiDelay: 0, smsEscalation: false }); setShowForm(!showForm) }}><Plus size={15} />Add Rule</button>
            </div>

            {showForm && (
                <form onSubmit={submit} className="card space-y-4">
                    <h2 className="section-title">{editRule ? 'Edit' : 'New'} Routing Rule</h2>
                    <div className="grid grid-cols-3 gap-4">
                        <div><label className="label">Rule Name</label><input required className="input-field" value={form.name} onChange={e => setForm(f => ({ ...f, name: e.target.value }))} /></div>
                        <div><label className="label">Intent Tag</label><input required className="input-field" placeholder="e.g. cancel_plan" value={form.intentTag} onChange={e => setForm(f => ({ ...f, intentTag: e.target.value }))} /></div>
                        <div><label className="label">Priority</label><input type="number" className="input-field" value={form.priority} onChange={e => setForm(f => ({ ...f, priority: +e.target.value }))} /></div>
                    </div>
                    <div className="grid grid-cols-2 gap-4">
                        <div><label className="label">Destination</label>
                            <select className="input-field" value={form.destination} onChange={e => setForm(f => ({ ...f, destination: e.target.value }))}>
                                <option value="agent">AI Agent</option><option value="human">Human Agent Queue</option><option value="voicemail">Voicemail</option>
                            </select>
                        </div>
                        <div><label className="label">CTI Screen-Pop Delay (sec)</label><input type="number" className="input-field" value={form.ctiDelay} onChange={e => setForm(f => ({ ...f, ctiDelay: +e.target.value }))} min="0" max="30" /></div>
                    </div>
                    <div><label className="label">Sentiment-Buffered Whisper Message</label><input className="input-field" placeholder="Message played to human agent before connecting..." value={form.whisperMsg || ''} onChange={e => setForm(f => ({ ...f, whisperMsg: e.target.value }))} /></div>
                    <Toggle label="SMS Escalation (PCI Compliance)" hint="Allow AI to send SMS mid-call for sensitive data collection" val={form.smsEscalation} onChange={v => setForm(f => ({ ...f, smsEscalation: v }))} />
                    <div className="flex gap-2"><button type="submit" className="btn-primary">Save Rule</button><button type="button" className="btn-secondary" onClick={() => setShowForm(false)}>Cancel</button></div>
                </form>
            )}

            <div className="card p-0 overflow-hidden">
                <table className="w-full">
                    <thead className="bg-gray-50 border-b border-gray-100">
                        <tr>
                            <th className="table-header">Priority</th><th className="table-header">Rule Name</th><th className="table-header">Intent Tag</th>
                            <th className="table-header">Destination</th><th className="table-header">CTI Delay</th><th className="table-header">SMS Esc.</th><th className="table-header">Actions</th>
                        </tr>
                    </thead>
                    <tbody>
                        {rules.length === 0 && <tr><td colSpan={7} className="text-center py-10 text-gray-400 text-sm">No routing rules. Add one to get started.</td></tr>}
                        {rules.map(r => (
                            <tr key={r.id} className="hover:bg-gray-50 transition-colors">
                                <td className="table-cell font-mono text-center">{r.priority}</td>
                                <td className="table-cell font-medium">{r.name}</td>
                                <td className="table-cell"><span className="font-mono text-xs bg-gray-100 px-2 py-1 rounded-lg">{r.intentTag}</span></td>
                                <td className="table-cell"><span className={r.destination === 'human' ? 'badge-blue' : r.destination === 'voicemail' ? 'badge-gray' : 'badge-green'}>{r.destination}</span></td>
                                <td className="table-cell">{r.ctiDelay}s</td>
                                <td className="table-cell">{r.smsEscalation ? <span className="badge-orange">On</span> : <span className="badge-gray">Off</span>}</td>
                                <td className="table-cell flex gap-1">
                                    <button className="p-1.5 hover:bg-orange-50 rounded-lg text-orange-400" onClick={() => startEdit(r)}><Edit size={13} /></button>
                                    <button className="p-1.5 hover:bg-red-50 rounded-lg text-red-400" onClick={() => del(r.id)}><Trash2 size={13} /></button>
                                </td>
                            </tr>
                        ))}
                    </tbody>
                </table>
            </div>
        </div>
    );
}
