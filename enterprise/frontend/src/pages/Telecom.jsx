import { useState, useEffect } from 'react';
import { api } from '../lib/api.js';
import { useStore } from '../store/index.js';
import { Phone, ShieldAlert, Plus, Trash2, MapPin, Loader2, GitBranch } from 'lucide-react';

export default function Telecom() {
    const [numbers, setNumbers] = useState([]);
    const [dnc, setDnc] = useState([]);
    const [loading, setLoading] = useState(true);
    const { showToast } = useStore();

    const [newNum, setNewNum] = useState({ number: '', friendlyName: '', provider: 'twilio' });
    const [newDnc, setNewDnc] = useState({ number: '', reason: '' });

    useEffect(() => {
        loadData();
    }, []);

    async function loadData() {
        setLoading(true);
        try {
            const [nums, dncs] = await Promise.all([
                api.get('/telecom/numbers'),
                api.get('/telecom/dnc')
            ]);
            setNumbers(nums);
            setDnc(dncs);
        } catch (err) {
            showToast('error', 'Failed to load telecom data');
        } finally {
            setLoading(false);
        }
    }

    async function handleAddNumber(e) {
        e.preventDefault();
        if (!newNum.number) return;
        try {
            await api.post('/telecom/numbers', newNum);
            setNewNum({ number: '', friendlyName: '', provider: 'twilio' });
            loadData();
            showToast('success', 'Phone number added');
        } catch (err) {
            showToast('error', 'Failed to add number');
        }
    }

    async function handleAddDnc(e) {
        e.preventDefault();
        if (!newDnc.number) return;
        try {
            await api.post('/telecom/dnc', { ...newDnc, addedBy: 'Admin' });
            setNewDnc({ number: '', reason: '' });
            loadData();
            showToast('success', 'Added to Do Not Call registry');
        } catch (err) {
            showToast('error', 'Failed to add to DNC');
        }
    }

    async function handleDeleteDnc(id) {
        try {
            await api.delete(`/telecom/dnc/${id}`);
            loadData();
            showToast('success', 'Removed from DNC list');
        } catch (err) {
            showToast('error', 'Failed to remove from DNC list');
        }
    }

    // Format phone number nicer
    const formatPhone = (phone) => {
        if (!phone) return '';
        const cleaned = ('' + phone).replace(/\D/g, '');
        const match = cleaned.match(/^(\d{1,3})(\d{3})(\d{3})(\d{4})$/);
        if (match) {
            return `+${match[1]} (${match[2]}) ${match[3]}-${match[4]}`;
        }
        return phone;
    };

    return (
        <div className="space-y-8">
            {/* Header */}
            <div className="page-header">
                <div>
                    <h1 className="text-2xl font-bold text-gray-900">Phone Numbers & DNC</h1>
                    <p className="text-sm text-gray-400 mt-0.5">Manage inbound Caller IDs and the global Do Not Call registry</p>
                </div>
            </div>

            <div className="grid grid-cols-1 lg:grid-cols-2 gap-8">

                {/* Left Column: Phone Numbers */}
                <div className="space-y-4">
                    <div className="flex items-center gap-2 mb-2">
                        <Phone size={18} className="text-blue-500" />
                        <h2 className="text-lg font-bold text-gray-800">Your Numbers</h2>
                    </div>

                    <form onSubmit={handleAddNumber} className="card p-4 flex gap-3 bg-blue-50/30 border-blue-100">
                        <input
                            type="text"
                            placeholder="+1234567890"
                            className="input-field w-36 font-mono"
                            value={newNum.number}
                            onChange={e => setNewNum({ ...newNum, number: e.target.value })}
                        />
                        <input
                            type="text"
                            placeholder="Friendly Name (e.g. Sales Line)"
                            className="input-field flex-1"
                            value={newNum.friendlyName}
                            onChange={e => setNewNum({ ...newNum, friendlyName: e.target.value })}
                        />
                        <button type="submit" className="btn-primary whitespace-nowrap px-4 py-2 bg-blue-600 hover:bg-blue-700 shadow-blue-500/30">
                            <Plus size={16} /> Buy Num
                        </button>
                    </form>

                    <div className="card overflow-hidden">
                        {loading ? (
                            <div className="p-8 flex justify-center text-gray-400"><Loader2 className="animate-spin" /></div>
                        ) : numbers.length === 0 ? (
                            <div className="p-8 text-center text-gray-400 text-sm">No phone numbers provisioned yet.</div>
                        ) : (
                            <div className="divide-y divide-gray-100">
                                {numbers.map(n => (
                                    <div key={n.id} className="p-4 hover:bg-gray-50 flex items-center justify-between">
                                        <div>
                                            <div className="flex items-center gap-2">
                                                <span className="font-mono font-bold text-gray-900">{formatPhone(n.number)}</span>
                                                {n.friendlyName && <span className="text-xs font-medium bg-gray-100 text-gray-600 px-2 py-0.5 rounded">{n.friendlyName}</span>}
                                            </div>
                                            <div className="text-xs text-gray-400 mt-1 flex items-center gap-1.5">
                                                <MapPin size={12} className="text-gray-300" /> Provider: <span className="capitalize">{n.provider}</span>
                                            </div>
                                        </div>
                                        <div className="text-right">
                                            {n.routingRule ? (
                                                <div className="flex items-center gap-1.5 text-xs text-emerald-600 font-medium bg-emerald-50 px-2 py-1 rounded-md border border-emerald-100">
                                                    <GitBranch size={12} /> {n.routingRule.name}
                                                </div>
                                            ) : (
                                                <button className="text-xs text-orange-500 hover:underline font-medium">Link Route</button>
                                            )}
                                        </div>
                                    </div>
                                ))}
                            </div>
                        )}
                    </div>
                </div>

                {/* Right Column: Global DNC List */}
                <div className="space-y-4">
                    <div className="flex items-center gap-2 mb-2">
                        <ShieldAlert size={18} className="text-rose-500" />
                        <h2 className="text-lg font-bold text-gray-800">Do Not Call (DNC) Registry</h2>
                    </div>
                    <p className="text-sm text-gray-500 mb-4">
                        Numbers listed here will be actively blocked from all outbound dialing campaigns instantly.
                    </p>

                    <form onSubmit={handleAddDnc} className="card p-4 flex gap-3 bg-rose-50/30 border-rose-100">
                        <input
                            type="text"
                            placeholder="+1234567890"
                            className="input-field w-36 font-mono"
                            value={newDnc.number}
                            onChange={e => setNewDnc({ ...newDnc, number: e.target.value })}
                        />
                        <input
                            type="text"
                            placeholder="Reason (e.g. Requested removal)"
                            className="input-field flex-1"
                            value={newDnc.reason}
                            onChange={e => setNewDnc({ ...newDnc, reason: e.target.value })}
                        />
                        <button type="submit" className="btn-primary whitespace-nowrap px-4 py-2 bg-rose-600 hover:bg-rose-700 shadow-rose-500/30">
                            <ShieldAlert size={16} /> Block Output
                        </button>
                    </form>

                    <div className="card overflow-hidden h-[400px] flex flex-col">
                        <div className="bg-gray-50 p-2 border-b border-gray-100 text-xs font-semibold text-gray-400 px-4 flex justify-between">
                            <span>BLOCKED NUMBER</span>
                            <span>DATE ADDED</span>
                        </div>
                        <div className="overflow-y-auto flex-1 p-0">
                            {loading ? (
                                <div className="p-8 flex justify-center text-gray-400"><Loader2 className="animate-spin" /></div>
                            ) : dnc.length === 0 ? (
                                <div className="p-8 text-center text-gray-400 text-sm">Registry is empty. Safe to dial.</div>
                            ) : (
                                <div className="divide-y divide-gray-50">
                                    {dnc.map(entry => (
                                        <div key={entry.id} className="p-3 px-4 hover:bg-rose-50/30 flex items-center justify-between group">
                                            <div>
                                                <div className="font-mono text-sm font-bold text-gray-700">{formatPhone(entry.number)}</div>
                                                <div className="text-xs text-gray-400 mt-0.5">{entry.reason || 'No reason provided'} ({entry.addedBy})</div>
                                            </div>
                                            <div className="flex items-center gap-4">
                                                <span className="text-xs text-gray-400">
                                                    {new Date(entry.createdAt).toLocaleDateString()}
                                                </span>
                                                <button
                                                    onClick={() => handleDeleteDnc(entry.id)}
                                                    className="text-gray-300 hover:text-red-500 transition-colors p-1 opacity-0 group-hover:opacity-100"
                                                    title="Remove from DNC"
                                                >
                                                    <Trash2 size={14} />
                                                </button>
                                            </div>
                                        </div>
                                    ))}
                                </div>
                            )}
                        </div>
                    </div>
                </div>

            </div>
        </div>
    );
}
