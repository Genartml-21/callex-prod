import { useState, useEffect } from 'react';
import { api } from '../lib/api.js';
import { Users, Clock, Coffee, ShieldAlert, PhoneCall, Loader2 } from 'lucide-react';

const AUX_STATES = [
    { id: 'available', label: 'Available', color: 'bg-emerald-500', icon: PhoneCall },
    { id: 'acw', label: 'ACW (Wrap-up)', color: 'bg-blue-500', icon: Clock },
    { id: 'break', label: 'On Break', color: 'bg-orange-500', icon: Coffee },
    { id: 'lunch', label: 'Lunch', color: 'bg-amber-500', icon: Coffee },
    { id: 'offline', label: 'Offline', color: 'bg-gray-400', icon: Users },
];

export default function WFM() {
    const [states, setStates] = useState([]);
    const [loading, setLoading] = useState(true);

    // Stats
    const availableCount = states.filter(s => s.state === 'available').length;
    const acwCount = states.filter(s => s.state === 'acw').length;
    const breakCount = states.filter(s => ['break', 'lunch'].includes(s.state)).length;

    useEffect(() => {
        loadStates();
        // In a real app this would be driven by websockets, polling for now
        const int = setInterval(loadStates, 10000);
        return () => clearInterval(int);
    }, []);

    async function loadStates() {
        try {
            const data = await api.get('/wfm/states');

            // Mock some data if empty to show the UI
            if (data.length === 0) {
                setStates([
                    { user: { id: '1', name: 'John Doe', role: 'agent' }, state: 'available', timestamp: new Date(Date.now() - 1000 * 60 * 5) },
                    { user: { id: '2', name: 'Sarah Smith', role: 'supervisor' }, state: 'acw', timestamp: new Date(Date.now() - 1000 * 45) },
                    { user: { id: '3', name: 'Mike Ross', role: 'agent' }, state: 'break', timestamp: new Date(Date.now() - 1000 * 60 * 12) },
                    { user: { id: '4', name: 'Amanda Jones', role: 'agent' }, state: 'available', timestamp: new Date(Date.now() - 1000 * 60 * 2) },
                    { user: { id: '5', name: 'Robert Chen', role: 'agent' }, state: 'offline', timestamp: new Date(Date.now() - 1000 * 60 * 60 * 2) },
                ]);
                setLoading(false);
                return;
            }

            setStates(data);
        } catch (err) {
            console.error(err);
        } finally {
            setLoading(false);
        }
    }

    const formatDuration = (dateStr) => {
        const diff = Math.floor((new Date() - new Date(dateStr)) / 1000);
        const m = Math.floor(diff / 60);
        const s = diff % 60;
        return `${m}m ${s}s`;
    };

    return (
        <div className="space-y-8">
            {/* Header */}
            <div className="page-header">
                <div>
                    <h1 className="text-2xl font-bold text-gray-900">Workforce Management</h1>
                    <p className="text-sm text-gray-400 mt-0.5">Live human agent status board and AUX states</p>
                </div>
            </div>

            {/* KPI Cards */}
            <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
                <div className="card p-6 border-l-4 border-l-emerald-500 flex items-center gap-4 bg-white/50">
                    <div className="w-12 h-12 rounded-xl bg-emerald-50 flex items-center justify-center text-emerald-600">
                        <PhoneCall size={24} />
                    </div>
                    <div>
                        <div className="text-2xl font-black text-gray-900">{availableCount}</div>
                        <div className="text-xs font-bold text-emerald-600 uppercase tracking-widest mt-1">Available</div>
                    </div>
                </div>

                <div className="card p-6 border-l-4 border-l-blue-500 flex items-center gap-4 bg-white/50">
                    <div className="w-12 h-12 rounded-xl bg-blue-50 flex items-center justify-center text-blue-600">
                        <Clock size={24} />
                    </div>
                    <div>
                        <div className="text-2xl font-black text-gray-900">{acwCount}</div>
                        <div className="text-xs font-bold text-blue-600 uppercase tracking-widest mt-1">In Wrap-up</div>
                    </div>
                </div>

                <div className="card p-6 border-l-4 border-l-orange-500 flex items-center gap-4 bg-white/50">
                    <div className="w-12 h-12 rounded-xl bg-orange-50 flex items-center justify-center text-orange-600">
                        <Coffee size={24} />
                    </div>
                    <div>
                        <div className="text-2xl font-black text-gray-900">{breakCount}</div>
                        <div className="text-xs font-bold text-orange-600 uppercase tracking-widest mt-1">On Break</div>
                    </div>
                </div>

                <div className="card p-6 border-l-4 border-l-gray-400 flex items-center gap-4 bg-white/50">
                    <div className="w-12 h-12 rounded-xl bg-gray-50 flex items-center justify-center text-gray-600">
                        <Users size={24} />
                    </div>
                    <div>
                        <div className="text-2xl font-black text-gray-900">{states.length}</div>
                        <div className="text-xs font-bold text-gray-500 uppercase tracking-widest mt-1">Total Staff</div>
                    </div>
                </div>
            </div>

            {/* Status Board */}
            <div className="card overflow-hidden">
                <div className="p-4 border-b border-gray-100 flex items-center justify-between bg-gray-50/50">
                    <h3 className="font-bold text-gray-800 flex items-center gap-2">
                        <ShieldAlert size={16} className="text-orange-500" /> Live Agent States
                    </h3>
                    <span className="text-xs text-gray-500 flex items-center gap-1.5">
                        <span className="relative flex h-2 w-2">
                            <span className="animate-ping absolute inline-flex h-full w-full rounded-full bg-emerald-400 opacity-75"></span>
                            <span className="relative inline-flex rounded-full h-2 w-2 bg-emerald-500"></span>
                        </span>
                        Live Refresh
                    </span>
                </div>

                <table className="w-full text-left border-collapse">
                    <thead>
                        <tr className="border-b border-gray-100/50 text-xs text-gray-400 uppercase tracking-wider bg-white">
                            <th className="p-4 font-semibold">Agent Name</th>
                            <th className="p-4 font-semibold">Role</th>
                            <th className="p-4 font-semibold">Current State</th>
                            <th className="p-4 font-semibold">Time in State</th>
                            <th className="p-4 text-right font-semibold">Actions</th>
                        </tr>
                    </thead>
                    <tbody className="divide-y divide-gray-50/50">
                        {loading ? (
                            <tr><td colSpan="5" className="p-8 text-center"><Loader2 className="animate-spin text-gray-400 mx-auto" /></td></tr>
                        ) : states.length === 0 ? (
                            <tr><td colSpan="5" className="p-8 text-center text-gray-400 text-sm">No agent states found.</td></tr>
                        ) : states.map(s => {
                            const stateInfo = AUX_STATES.find(x => x.id === s.state) || AUX_STATES[0];
                            const StateIcon = stateInfo.icon;

                            return (
                                <tr key={s.user.id} className="hover:bg-gray-50/50 transition-colors">
                                    <td className="p-4">
                                        <div className="flex items-center gap-3">
                                            <div className="w-8 h-8 rounded-full bg-orange-100 text-orange-600 flex items-center justify-center font-bold text-xs">
                                                {s.user.name.split(' ').map(n => n[0]).join('')}
                                            </div>
                                            <span className="font-semibold text-sm text-gray-800">{s.user.name}</span>
                                        </div>
                                    </td>
                                    <td className="p-4 text-sm text-gray-500 capitalize">{s.user.role}</td>
                                    <td className="p-4">
                                        <div className="flex items-center gap-2">
                                            <div className={`w-2.5 h-2.5 rounded-full ${stateInfo.color}`} />
                                            <span className="text-sm font-medium text-gray-700">{stateInfo.label}</span>
                                        </div>
                                    </td>
                                    <td className="p-4 font-mono text-sm text-gray-600">
                                        {formatDuration(s.timestamp)}
                                    </td>
                                    <td className="p-4 text-right">
                                        <button className="text-xs font-semibold text-orange-600 hover:text-orange-700 hover:bg-orange-50 px-3 py-1.5 rounded transition-colors">
                                            Force Logout
                                        </button>
                                    </td>
                                </tr>
                            )
                        })}
                    </tbody>
                </table>
            </div>

        </div>
    );
}
