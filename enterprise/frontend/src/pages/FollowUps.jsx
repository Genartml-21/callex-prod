import { useEffect, useState } from 'react';
import { api } from '../lib/api.js';
import { useStore } from '../store/index.js';
import { CalendarClock, PhoneOutgoing, UserCircle, Bot, CheckCircle2, XCircle, Clock } from 'lucide-react';

export default function FollowUps() {
    const [followups, setFollowups] = useState([]);
    const [loading, setLoading] = useState(true);
    const { showToast } = useStore();

    const fetchFollowUps = async () => {
        setLoading(true);
        try {
            const data = await api.followups();
            setFollowups(data);
        } catch (err) {
            console.error(err);
            showToast('error', 'Failed to load follow-ups');
        } finally {
            setLoading(false);
        }
    };

    useEffect(() => {
        fetchFollowUps();
        const interval = setInterval(fetchFollowUps, 30000);
        return () => clearInterval(interval);
    }, []);

    const updateStatus = async (id, status) => {
        await api.setFollowUpStatus(id, status);
        fetchFollowUps();
        showToast('success', `Follow-up marked as ${status}`);
    };

    const StatusBadge = ({ status }) => {
        switch (status) {
            case 'pending': return <span className="badge-orange text-xs"><Clock size={12} className="inline mr-1" /> Pending</span>;
            case 'completed': return <span className="badge-green text-xs"><CheckCircle2 size={12} className="inline mr-1" /> Completed</span>;
            case 'cancelled': return <span className="badge-red text-xs"><XCircle size={12} className="inline mr-1" /> Cancelled</span>;
            case 'failed': return <span className="badge-red text-xs">Failed</span>;
            default: return <span className="badge-gray text-xs">{status}</span>;
        }
    };

    return (
        <div className="space-y-6">
            <div className="page-header">
                <div>
                    <h1 className="text-2xl font-bold text-gray-900 flex items-center gap-3">
                        <CalendarClock className="text-orange-500" /> Auto-Scheduled Follow-Ups
                    </h1>
                    <p className="text-sm text-gray-400 mt-1">AI-extracted commitments to call back users automatically</p>
                </div>
                <button onClick={fetchFollowUps} className="btn-secondary">Refresh</button>
            </div>

            <div className="bg-white border text-left border-gray-100 rounded-2xl shadow-sm overflow-hidden">
                <table className="w-full text-sm text-left">
                    <thead className="bg-gray-50/50 text-gray-500 font-semibold border-b border-gray-100">
                        <tr>
                            <th className="px-6 py-4">Customer Phone</th>
                            <th className="px-6 py-4">Scheduled For</th>
                            <th className="px-6 py-4">AI Agent Handled</th>
                            <th className="px-6 py-4">Semantic Reason</th>
                            <th className="px-6 py-4">Status</th>
                            <th className="px-6 py-4 text-right">Actions</th>
                        </tr>
                    </thead>
                    <tbody className="divide-y divide-gray-50">
                        {loading && followups.length === 0 ? (
                            <tr><td colSpan="6" className="text-center py-12 text-gray-400">Loading follow-ups...</td></tr>
                        ) : followups.length === 0 ? (
                            <tr>
                                <td colSpan="6" className="text-center py-12">
                                    <div className="flex flex-col items-center justify-center text-gray-400">
                                        <CalendarClock size={40} className="mb-3 opacity-20 text-orange-500" />
                                        <p>No follow-ups scheduled yet.</p>
                                    </div>
                                </td>
                            </tr>
                        ) : (
                            followups.map(f => (
                                <tr key={f.id} className="hover:bg-gray-50/50 transition-colors">
                                    <td className="px-6 py-4 font-mono font-medium text-gray-800 flex items-center gap-2">
                                        <UserCircle size={16} className="text-gray-400" /> {f.phoneNumber}
                                    </td>
                                    <td className="px-6 py-4 text-gray-600 font-medium">
                                        {new Date(f.scheduledFor).toLocaleString(undefined, {
                                            weekday: 'short', month: 'short', day: 'numeric',
                                            hour: 'numeric', minute: '2-digit'
                                        })}
                                    </td>
                                    <td className="px-6 py-4">
                                        <div className="flex items-center gap-2 text-gray-600">
                                            <Bot size={14} className="text-orange-500" /> {f.agent?.name || 'Unknown Agent'}
                                        </div>
                                    </td>
                                    <td className="px-6 py-4 text-gray-500 italic text-xs max-w-xs truncate">
                                        "{f.reason || 'No specific reason provided'}"
                                    </td>
                                    <td className="px-6 py-4">
                                        <StatusBadge status={f.status} />
                                    </td>
                                    <td className="px-6 py-4 text-right space-x-2">
                                        {f.status === 'pending' && (
                                            <>
                                                <button onClick={() => updateStatus(f.id, 'completed')} className="text-green-600 bg-green-50 px-3 py-1.5 rounded-lg font-semibold hover:bg-green-100 transition-colors border border-green-200" title="Mark Completed Manually"><CheckCircle2 size={14} className="inline" /> Done</button>
                                                <button onClick={() => updateStatus(f.id, 'cancelled')} className="text-red-500 bg-red-50 px-3 py-1.5 rounded-lg font-semibold hover:bg-red-100 transition-colors border border-red-100" title="Cancel Follow-Up"><XCircle size={14} className="inline" /> Cancel</button>
                                            </>
                                        )}
                                        {f.status !== 'pending' && (
                                            <button onClick={() => updateStatus(f.id, 'pending')} className="text-orange-500 bg-orange-50 px-3 py-1.5 rounded-lg font-semibold hover:bg-orange-100 transition-colors border border-orange-100" title="Reactivate"><Clock size={14} className="inline" /> Requeue</button>
                                        )}
                                    </td>
                                </tr>
                            ))
                        )}
                    </tbody>
                </table>
            </div>
        </div>
    );
}
