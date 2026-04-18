import { useEffect, useState, useRef } from 'react';
import { api } from '../lib/api.js';
import { useStore } from '../store/index.js';
import { useAuth } from '../contexts/AuthContext.jsx';
import {
    TrendingUp, Phone, Zap, Shield, Users, Activity, RadioTower,
    Clock, Coffee, ShieldAlert, PhoneCall, Loader2, PhoneOff, PhoneForwarded,
    Crown, Bot, DollarSign, Eye, Trash2, Plus, X, Search, Key, FileText, AlertCircle
} from 'lucide-react';
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer } from 'recharts';

const WS_URL = `ws://${window.location.host.replace('3000', '4000')}`;

// ─── AUX States for Workforce ───
const AUX_STATES = [
    { id: 'available', label: 'Available', color: 'bg-emerald-500', icon: PhoneCall },
    { id: 'acw', label: 'ACW (Wrap-up)', color: 'bg-blue-500', icon: Clock },
    { id: 'break', label: 'On Break', color: 'bg-orange-500', icon: Coffee },
    { id: 'lunch', label: 'Lunch', color: 'bg-amber-500', icon: Coffee },
    { id: 'offline', label: 'Offline', color: 'bg-gray-400', icon: Users },
];

function KPICard({ icon: Icon, label, value, color = 'orange', sub }) {
    const colors = { orange: 'bg-orange-50 text-orange-600', green: 'bg-emerald-50 text-emerald-600', blue: 'bg-blue-50 text-blue-600', red: 'bg-red-50 text-red-600', purple: 'bg-purple-50 text-purple-600', amber: 'bg-amber-50 text-amber-600' };
    return (
        <div className="kpi-card hover:shadow-md transition-shadow">
            <div className={`w-10 h-10 rounded-xl ${colors[color]} flex items-center justify-center mb-3`}>
                <Icon size={18} />
            </div>
            <div className="text-2xl font-bold text-gray-900">{value ?? '—'}</div>
            <div className="text-sm text-gray-500 font-medium">{label}</div>
            {sub && <div className="text-xs text-gray-400 mt-0.5">{sub}</div>}
        </div>
    );
}

const buildChartPoint = (kpi) => ({
    time: new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' }),
    calls: kpi?.activeCalls ?? 0,
    mos: parseFloat(kpi?.avgMOS ?? 0),
});

const formatDuration = (dateStr) => {
    const diff = Math.floor((new Date() - new Date(dateStr)) / 1000);
    const m = Math.floor(diff / 60);
    const s = diff % 60;
    return `${m}m ${s}s`;
};

export default function Dashboard() {
    const { userRole } = useAuth();
    if (userRole === 'superadmin') return <AdminDashboard />;
    return <UserDashboard />;
}

// ═══════════════════════════════════════════════════════════
// ADMIN DASHBOARD — Platform-wide stats + user management
// ═══════════════════════════════════════════════════════════
function AdminDashboard() {
    const [stats, setStats] = useState(null);
    const [users, setUsers] = useState([]);
    const [loading, setLoading] = useState(true);
    const [search, setSearch] = useState('');
    const [selectedUser, setSelectedUser] = useState(null);
    const [userDetail, setUserDetail] = useState(null);
    const [userCalls, setUserCalls] = useState([]);
    const [detailLoading, setDetailLoading] = useState(false);
    const [detailTab, setDetailTab] = useState('overview');
    const [showCreate, setShowCreate] = useState(false);
    const [createForm, setCreateForm] = useState({ email: '', name: '', password: '', role: 'user' });
    const [error, setError] = useState('');
    const [exporting, setExporting] = useState(null);
    const { showToast } = useStore ? useStore() : { showToast: () => {} };

    useEffect(() => { loadData(); }, []);

    async function loadData() {
        setLoading(true);
        try {
            const [statsData, usersData] = await Promise.all([
                api.get('/admin/stats'),
                api.get('/admin/users'),
            ]);
            setStats(statsData);
            setUsers(usersData);
        } catch (e) { setError(e.message); }
        finally { setLoading(false); }
    }

    async function viewUser(userId) {
        setSelectedUser(userId);
        setDetailLoading(true);
        setDetailTab('overview');
        setUserCalls([]);
        try {
            const [detail, calls] = await Promise.all([
                api.get(`/admin/users/${userId}`),
                api.get(`/admin/users/${userId}/calls`).catch(() => []),
            ]);
            setUserDetail(detail);
            setUserCalls(calls);
        } catch (e) { setError(e.message); }
        finally { setDetailLoading(false); }
    }

    async function exportUserData(userId, email) {
        setExporting(userId);
        try {
            const data = await api.get(`/admin/users/${userId}/export`);
            const blob = new Blob([JSON.stringify(data, null, 2)], { type: 'application/json' });
            const url = URL.createObjectURL(blob);
            const a = document.createElement('a');
            a.href = url;
            a.download = `user_${email}_export.json`;
            document.body.appendChild(a);
            a.click();
            document.body.removeChild(a);
            URL.revokeObjectURL(url);
            if (typeof showToast === 'function') showToast(`Exported data for ${email}`, 'success');
        } catch (e) {
            if (typeof showToast === 'function') showToast(e.message, 'error');
        }
        finally { setExporting(null); }
    }

    async function createUser() {
        try {
            await api.post('/admin/users', createForm);
            setShowCreate(false);
            setCreateForm({ email: '', name: '', password: '', role: 'user' });
            loadData();
        } catch (e) { setError(e.message); }
    }

    async function deleteUser(id, email) {
        if (!confirm(`Delete user ${email} and ALL their data? This cannot be undone.`)) return;
        try {
            await api.delete(`/admin/users/${id}`);
            setSelectedUser(null);
            setUserDetail(null);
            loadData();
        } catch (e) { setError(e.message); }
    }

    const filtered = users.filter(u =>
        u.email.toLowerCase().includes(search.toLowerCase()) ||
        u.name.toLowerCase().includes(search.toLowerCase())
    );

    if (loading) return (
        <div className="flex items-center justify-center h-64">
            <Loader2 size={32} className="animate-spin text-orange-500" />
        </div>
    );

    return (
        <div className="space-y-8">
            {/* Header */}
            <div className="page-header">
                <div>
                    <h1 className="text-2xl font-bold text-gray-900 flex items-center gap-2">
                        <Crown size={24} className="text-orange-500" />
                        Admin Command Center
                    </h1>
                    <p className="text-sm text-gray-400 mt-0.5">Platform-wide monitoring & user management</p>
                </div>
                <button onClick={() => setShowCreate(true)} className="btn-primary flex items-center gap-2">
                    <Plus size={16} /> Create User
                </button>
            </div>

            {error && (
                <div className="flex items-center gap-2 p-3 bg-red-50 border border-red-100 rounded-xl text-sm text-red-600">
                    <AlertCircle size={15} />{error}
                    <button onClick={() => setError('')} className="ml-auto"><X size={14} /></button>
                </div>
            )}

            {/* ═══ Platform-wide KPIs ═══ */}
            {stats && (
                <div className="grid grid-cols-2 lg:grid-cols-4 gap-4">
                    <KPICard icon={Users} label="Total Users" value={stats.totalUsers} color="orange" sub="registered accounts" />
                    <KPICard icon={Bot} label="Total Agents" value={stats.totalAgents} color="blue" sub="across all users" />
                    <KPICard icon={DollarSign} label="Total Revenue" value="$0.00" color="green" sub="coming soon" />
                    <KPICard icon={Phone} label="Total Calls" value={stats.totalCalls} color="purple" sub="all time" />
                </div>
            )}

            {/* Secondary stats */}
            {stats && (
                <div className="grid grid-cols-2 md:grid-cols-4 gap-3">
                    {[
                        { label: 'Campaigns', value: stats.totalCampaigns, icon: PhoneForwarded, color: 'amber' },
                        { label: 'Active API Keys', value: stats.totalApiKeys, icon: Key, color: 'purple' },
                        { label: 'Active Calls', value: stats.activeCalls, icon: PhoneCall, color: 'green' },
                        { label: 'Documents', value: stats.totalDocs, icon: FileText, color: 'blue' },
                    ].map(s => (
                        <div key={s.label} className="glass-panel p-4 rounded-2xl flex items-center gap-3">
                            <div className={`w-10 h-10 rounded-xl bg-${s.color}-50 flex items-center justify-center`}>
                                <s.icon size={18} className={`text-${s.color}-500`} />
                            </div>
                            <div>
                                <div className="text-xl font-bold text-gray-900">{s.value}</div>
                                <div className="text-xs text-gray-500">{s.label}</div>
                            </div>
                        </div>
                    ))}
                </div>
            )}

            {/* ═══ User Management Table ═══ */}
            <div className="space-y-3">
                <h2 className="text-lg font-bold text-gray-900 flex items-center gap-2">
                    <Users size={20} className="text-orange-500" /> All Users
                </h2>
                <div className="relative">
                    <Search size={16} className="absolute left-4 top-1/2 -translate-y-1/2 text-gray-400" />
                    <input
                        value={search} onChange={e => setSearch(e.target.value)}
                        placeholder="Search users by email or name..."
                        className="w-full pl-10 pr-4 py-3 glass-panel rounded-xl text-sm border-none outline-none bg-white/60"
                    />
                </div>
                <div className="glass-panel rounded-2xl overflow-hidden">
                    <div className="overflow-x-auto">
                        <table className="w-full">
                            <thead>
                                <tr className="border-b border-gray-100 bg-gray-50/50">
                                    <th className="text-left px-5 py-3 text-xs font-bold text-gray-500 uppercase tracking-wider">User</th>
                                    <th className="text-center px-3 py-3 text-xs font-bold text-gray-500 uppercase tracking-wider">Role</th>
                                    <th className="text-center px-3 py-3 text-xs font-bold text-gray-500 uppercase tracking-wider">Agents</th>
                                    <th className="text-center px-3 py-3 text-xs font-bold text-gray-500 uppercase tracking-wider">Campaigns</th>
                                    <th className="text-center px-3 py-3 text-xs font-bold text-gray-500 uppercase tracking-wider">Calls</th>
                                    <th className="text-center px-3 py-3 text-xs font-bold text-gray-500 uppercase tracking-wider">API Keys</th>
                                    <th className="text-center px-3 py-3 text-xs font-bold text-gray-500 uppercase tracking-wider">Docs</th>
                                    <th className="text-left px-3 py-3 text-xs font-bold text-gray-500 uppercase tracking-wider">Created</th>
                                    <th className="text-center px-3 py-3 text-xs font-bold text-gray-500 uppercase tracking-wider">Actions</th>
                                </tr>
                            </thead>
                            <tbody>
                                {filtered.map(u => (
                                    <tr key={u.id} className="border-b border-gray-50 hover:bg-orange-50/30 transition-colors">
                                        <td className="px-5 py-3">
                                            <div className="flex items-center gap-3">
                                                <div className="w-8 h-8 rounded-full bg-gradient-to-br from-orange-400 to-orange-600 flex items-center justify-center text-white text-xs font-bold">
                                                    {u.name?.[0]?.toUpperCase() || u.email[0].toUpperCase()}
                                                </div>
                                                <div>
                                                    <div className="text-sm font-semibold text-gray-900">{u.name}</div>
                                                    <div className="text-xs text-gray-500">{u.email}</div>
                                                </div>
                                            </div>
                                        </td>
                                        <td className="px-3 py-3 text-center">
                                            <span className={`px-2 py-0.5 rounded-full text-xs font-semibold ${u.role === 'superadmin' ? 'bg-orange-100 text-orange-700' : u.role === 'admin' ? 'bg-blue-100 text-blue-700' : 'bg-gray-100 text-gray-600'}`}>{u.role}</span>
                                        </td>
                                        <td className="px-3 py-3 text-center text-sm font-semibold text-gray-700">{u.agents}</td>
                                        <td className="px-3 py-3 text-center text-sm font-semibold text-gray-700">{u.campaigns}</td>
                                        <td className="px-3 py-3 text-center text-sm font-semibold text-gray-700">{u.calls}</td>
                                        <td className="px-3 py-3 text-center text-sm font-semibold text-gray-700">{u.apiKeys}</td>
                                        <td className="px-3 py-3 text-center text-sm font-semibold text-gray-700">{u.knowledgeDocs || 0}</td>
                                        <td className="px-3 py-3 text-xs text-gray-500">{new Date(u.createdAt).toLocaleDateString()}</td>
                                        <td className="px-3 py-3 text-center">
                                            <div className="flex items-center justify-center gap-1">
                                                <button onClick={() => viewUser(u.id)} className="p-1.5 hover:bg-blue-50 rounded-lg text-gray-400 hover:text-blue-500 transition-colors" title="View Details">
                                                    <Eye size={14} />
                                                </button>
                                                <button onClick={() => exportUserData(u.id, u.email)} disabled={exporting === u.id} className="p-1.5 hover:bg-green-50 rounded-lg text-gray-400 hover:text-green-500 transition-colors" title="Export All Data">
                                                    {exporting === u.id ? <Loader2 size={14} className="animate-spin" /> : <FileText size={14} />}
                                                </button>
                                                {u.role !== 'superadmin' && (
                                                    <button onClick={() => deleteUser(u.id, u.email)} className="p-1.5 hover:bg-red-50 rounded-lg text-gray-400 hover:text-red-500 transition-colors" title="Delete User">
                                                        <Trash2 size={14} />
                                                    </button>
                                                )}
                                            </div>
                                        </td>
                                    </tr>
                                ))}
                                {filtered.length === 0 && (
                                    <tr><td colSpan={9} className="text-center py-12 text-gray-400 text-sm">No users found</td></tr>
                                )}
                            </tbody>
                        </table>
                    </div>
                </div>
            </div>

            {/* Create User Modal */}
            {showCreate && (
                <div className="fixed inset-0 z-50 bg-gray-900/40 backdrop-blur-sm flex items-center justify-center">
                    <div className="bg-white rounded-2xl shadow-2xl w-full max-w-md p-6 space-y-4">
                        <div className="flex items-center justify-between">
                            <h3 className="text-lg font-bold text-gray-900">Create New User</h3>
                            <button onClick={() => setShowCreate(false)} className="p-1 hover:bg-gray-100 rounded-lg"><X size={18} /></button>
                        </div>
                        <div className="space-y-3">
                            <div><label className="text-xs font-semibold text-gray-600 mb-1 block">Email</label><input value={createForm.email} onChange={e => setCreateForm(f => ({ ...f, email: e.target.value }))} className="input-field" placeholder="user@company.com" /></div>
                            <div><label className="text-xs font-semibold text-gray-600 mb-1 block">Full Name</label><input value={createForm.name} onChange={e => setCreateForm(f => ({ ...f, name: e.target.value }))} className="input-field" placeholder="John Smith" /></div>
                            <div><label className="text-xs font-semibold text-gray-600 mb-1 block">Password</label><input type="password" value={createForm.password} onChange={e => setCreateForm(f => ({ ...f, password: e.target.value }))} className="input-field" placeholder="••••••••" /></div>
                            <div><label className="text-xs font-semibold text-gray-600 mb-1 block">Role</label>
                                <select value={createForm.role} onChange={e => setCreateForm(f => ({ ...f, role: e.target.value }))} className="input-field">
                                    <option value="user">User</option><option value="admin">Admin</option>
                                </select>
                            </div>
                        </div>
                        <div className="flex justify-end gap-2 pt-2">
                            <button onClick={() => setShowCreate(false)} className="px-4 py-2 text-sm text-gray-500 hover:text-gray-700">Cancel</button>
                            <button onClick={createUser} className="btn-primary">Create User</button>
                        </div>
                    </div>
                </div>
            )}

            {/* ═══ Enhanced User Detail Modal ═══ */}
            {selectedUser && (
                <div className="fixed inset-0 z-50 bg-gray-900/40 backdrop-blur-sm flex items-center justify-center">
                    <div className="bg-white rounded-2xl shadow-2xl w-full max-w-4xl max-h-[90vh] overflow-hidden flex flex-col">
                        {/* Modal Header */}
                        <div className="p-5 border-b border-gray-100 flex items-center justify-between shrink-0">
                            <h3 className="text-lg font-bold text-gray-900">User Details</h3>
                            <div className="flex items-center gap-2">
                                {userDetail && (
                                    <button onClick={() => exportUserData(selectedUser, userDetail.user.email)} disabled={exporting === selectedUser} className="btn-secondary flex items-center gap-2 text-sm py-1.5">
                                        {exporting === selectedUser ? <Loader2 size={14} className="animate-spin" /> : <FileText size={14} />}
                                        Export All Data
                                    </button>
                                )}
                                <button onClick={() => { setSelectedUser(null); setUserDetail(null); setUserCalls([]); }} className="p-1.5 hover:bg-gray-100 rounded-lg"><X size={18} /></button>
                            </div>
                        </div>

                        {detailLoading ? (
                            <div className="flex justify-center py-16"><Loader2 size={24} className="animate-spin text-orange-500" /></div>
                        ) : userDetail ? (
                            <div className="flex-1 overflow-y-auto p-6 space-y-5">
                                {/* User Info Header */}
                                <div className="flex items-center gap-4 p-4 bg-gray-50 rounded-xl">
                                    <div className="w-14 h-14 rounded-full bg-gradient-to-br from-orange-400 to-orange-600 flex items-center justify-center text-white text-xl font-bold shadow-lg">
                                        {userDetail.user.name?.[0]?.toUpperCase() || 'U'}
                                    </div>
                                    <div className="flex-1">
                                        <div className="font-bold text-gray-900 text-lg">{userDetail.user.name}</div>
                                        <div className="text-sm text-gray-500">{userDetail.user.email}</div>
                                        <div className="text-xs text-gray-400 mt-0.5">Joined: {new Date(userDetail.user.createdAt).toLocaleDateString()} · ID: {userDetail.user.id?.slice(0,8)}...</div>
                                    </div>
                                    <span className={`px-3 py-1.5 rounded-full text-xs font-bold ${userDetail.user.role === 'superadmin' ? 'bg-orange-100 text-orange-700' : userDetail.user.role === 'admin' ? 'bg-blue-100 text-blue-700' : 'bg-gray-100 text-gray-600'}`}>{userDetail.user.role}</span>
                                </div>

                                {/* Usage Stats Summary */}
                                <div className="grid grid-cols-4 gap-3">
                                    {[
                                        { label: 'Agents', value: userDetail.agents?.length || 0, icon: Bot, color: 'blue' },
                                        { label: 'Calls', value: userCalls.length, icon: Phone, color: 'green' },
                                        { label: 'API Keys', value: userDetail.apiKeys || 0, icon: Key, color: 'purple' },
                                        { label: 'Documents', value: userDetail.knowledgeDocs || 0, icon: FileText, color: 'amber' },
                                    ].map(s => (
                                        <div key={s.label} className="bg-white p-3 rounded-xl border border-gray-100 text-center">
                                            <s.icon size={18} className={`text-${s.color}-500 mx-auto mb-1`} />
                                            <div className="text-lg font-bold text-gray-900">{s.value}</div>
                                            <div className="text-[10px] text-gray-500">{s.label}</div>
                                        </div>
                                    ))}
                                </div>

                                {/* Detail Tabs */}
                                <div className="flex gap-1 bg-gray-100 p-1 rounded-xl">
                                    {[
                                        { id: 'overview', label: 'Overview' },
                                        { id: 'calls', label: `Call Logs (${userCalls.length})` },
                                        { id: 'agents', label: `Agents (${userDetail.agents?.length || 0})` },
                                    ].map(tab => (
                                        <button key={tab.id} onClick={() => setDetailTab(tab.id)} className={`flex-1 py-2 text-xs font-bold rounded-lg transition-all ${detailTab === tab.id ? 'bg-white shadow-sm text-gray-900' : 'text-gray-500 hover:text-gray-700'}`}>
                                            {tab.label}
                                        </button>
                                    ))}
                                </div>

                                {/* Tab: Overview */}
                                {detailTab === 'overview' && (
                                    <div className="space-y-4">
                                        {userDetail.agents?.length > 0 && (
                                            <div>
                                                <div className="flex items-center gap-2 mb-2"><Bot size={16} className="text-orange-500" /><span className="text-sm font-bold text-gray-700">Agents</span></div>
                                                <div className="space-y-1 bg-gray-50 p-3 rounded-xl">{userDetail.agents.map(a => (
                                                    <div key={a.id} className="flex items-center justify-between p-2.5 bg-white rounded-lg border border-gray-100">
                                                        <div>
                                                            <span className="text-sm font-medium text-gray-800">{a.name}</span>
                                                            <span className="text-[10px] text-gray-400 ml-2">{a.llmModel || 'callex-1.3'}</span>
                                                        </div>
                                                        <span className={`px-2 py-0.5 rounded-full text-xs ${a.status === 'active' ? 'bg-green-100 text-green-700' : a.status === 'paused' ? 'bg-orange-100 text-orange-700' : 'bg-gray-100 text-gray-500'}`}>{a.status}</span>
                                                    </div>
                                                ))}</div>
                                            </div>
                                        )}
                                        {userCalls.length > 0 && (
                                            <div>
                                                <div className="flex items-center gap-2 mb-2"><Phone size={16} className="text-orange-500" /><span className="text-sm font-bold text-gray-700">Recent Calls</span></div>
                                                <div className="space-y-1 bg-gray-50 p-3 rounded-xl">{userCalls.slice(0, 5).map(c => (
                                                    <div key={c.id} className="flex items-center justify-between p-2.5 bg-white rounded-lg border border-gray-100">
                                                        <div>
                                                            <span className="text-sm font-medium text-gray-800">{c.phoneNumber || 'N/A'}</span>
                                                            <span className="text-xs text-gray-400 ml-2">{c.agentName || ''}</span>
                                                            <span className="text-xs text-gray-400 ml-2">{c.duration ? `${c.duration}s` : '-'}</span>
                                                        </div>
                                                        <span className={`px-2 py-0.5 rounded-full text-xs ${c.sentiment === 'positive' ? 'bg-green-100 text-green-700' : c.sentiment === 'negative' ? 'bg-red-100 text-red-700' : 'bg-gray-100 text-gray-500'}`}>{c.sentiment || c.status}</span>
                                                    </div>
                                                ))}</div>
                                                {userCalls.length > 5 && <button onClick={() => setDetailTab('calls')} className="text-xs text-orange-500 font-semibold mt-2 hover:underline">View all {userCalls.length} calls →</button>}
                                            </div>
                                        )}
                                        {userDetail.agents?.length === 0 && userCalls.length === 0 && (
                                            <div className="text-center py-8 text-gray-400 text-sm">This user has no activity yet.</div>
                                        )}
                                    </div>
                                )}

                                {/* Tab: Call Logs */}
                                {detailTab === 'calls' && (
                                    <div>
                                        {userCalls.length === 0 ? (
                                            <div className="text-center py-8 text-gray-400 text-sm">No call logs found for this user.</div>
                                        ) : (
                                            <div className="glass-panel rounded-xl overflow-hidden">
                                                <table className="w-full">
                                                    <thead>
                                                        <tr className="border-b border-gray-100 bg-gray-50/50">
                                                            <th className="text-left px-4 py-2.5 text-[10px] font-bold text-gray-500 uppercase">Phone</th>
                                                            <th className="text-left px-3 py-2.5 text-[10px] font-bold text-gray-500 uppercase">Agent</th>
                                                            <th className="text-center px-3 py-2.5 text-[10px] font-bold text-gray-500 uppercase">Duration</th>
                                                            <th className="text-center px-3 py-2.5 text-[10px] font-bold text-gray-500 uppercase">Sentiment</th>
                                                            <th className="text-center px-3 py-2.5 text-[10px] font-bold text-gray-500 uppercase">Status</th>
                                                            <th className="text-left px-3 py-2.5 text-[10px] font-bold text-gray-500 uppercase">Date</th>
                                                        </tr>
                                                    </thead>
                                                    <tbody>
                                                        {userCalls.map(c => (
                                                            <tr key={c.id} className="border-b border-gray-50 hover:bg-gray-50/50">
                                                                <td className="px-4 py-2.5 text-sm text-gray-800 font-medium">{c.phoneNumber || 'N/A'}</td>
                                                                <td className="px-3 py-2.5 text-xs text-gray-600">{c.agentName || '-'}</td>
                                                                <td className="px-3 py-2.5 text-center text-xs text-gray-600">{c.duration ? `${c.duration}s` : '-'}</td>
                                                                <td className="px-3 py-2.5 text-center">
                                                                    <span className={`px-2 py-0.5 rounded-full text-[10px] font-semibold ${c.sentiment === 'positive' ? 'bg-green-100 text-green-700' : c.sentiment === 'negative' ? 'bg-red-100 text-red-700' : c.sentiment === 'angry' ? 'bg-red-200 text-red-800' : 'bg-gray-100 text-gray-500'}`}>{c.sentiment || '-'}</span>
                                                                </td>
                                                                <td className="px-3 py-2.5 text-center">
                                                                    <span className={`px-2 py-0.5 rounded-full text-[10px] font-semibold ${c.status === 'completed' ? 'bg-green-100 text-green-700' : c.status === 'active' ? 'bg-blue-100 text-blue-700' : 'bg-gray-100 text-gray-500'}`}>{c.status}</span>
                                                                </td>
                                                                <td className="px-3 py-2.5 text-xs text-gray-500">{c.startedAt ? new Date(c.startedAt._seconds ? c.startedAt._seconds * 1000 : c.startedAt).toLocaleString() : '-'}</td>
                                                            </tr>
                                                        ))}
                                                    </tbody>
                                                </table>
                                            </div>
                                        )}
                                    </div>
                                )}

                                {/* Tab: Agents Detail */}
                                {detailTab === 'agents' && (
                                    <div>
                                        {(!userDetail.agents || userDetail.agents.length === 0) ? (
                                            <div className="text-center py-8 text-gray-400 text-sm">No agents created by this user.</div>
                                        ) : (
                                            <div className="space-y-3">
                                                {userDetail.agents.map(a => (
                                                    <div key={a.id} className="bg-white p-4 rounded-xl border border-gray-100 shadow-sm">
                                                        <div className="flex items-center justify-between mb-2">
                                                            <div className="flex items-center gap-2">
                                                                <Bot size={16} className="text-orange-500" />
                                                                <span className="font-bold text-gray-900">{a.name}</span>
                                                            </div>
                                                            <span className={`px-2 py-0.5 rounded-full text-xs font-semibold ${a.status === 'active' ? 'bg-green-100 text-green-700' : a.status === 'paused' ? 'bg-orange-100 text-orange-700' : 'bg-gray-100 text-gray-500'}`}>{a.status}</span>
                                                        </div>
                                                        <div className="grid grid-cols-4 gap-3 text-xs">
                                                            <div><span className="text-gray-400">Model:</span><span className="ml-1 font-semibold text-gray-700">{a.llmModel || 'callex-1.3'}</span></div>
                                                            <div><span className="text-gray-400">Language:</span><span className="ml-1 font-semibold text-gray-700">{a.language || 'en-US'}</span></div>
                                                            <div><span className="text-gray-400">Speed:</span><span className="ml-1 font-semibold text-gray-700">{a.prosodyRate || 1.0}x</span></div>
                                                            <div><span className="text-gray-400">Patience:</span><span className="ml-1 font-semibold text-gray-700">{a.patienceMs || 800}ms</span></div>
                                                        </div>
                                                        {a.description && <p className="text-xs text-gray-400 mt-2 line-clamp-2">{a.description}</p>}
                                                    </div>
                                                ))}
                                            </div>
                                        )}
                                    </div>
                                )}
                            </div>
                        ) : null}
                    </div>
                </div>
            )}
        </div>
    );
}

// ═══════════════════════════════════════════════════════════
// USER DASHBOARD — Original call center command center view
// ═══════════════════════════════════════════════════════════
function UserDashboard() {
    const [kpis, setKPIs] = useState(null);
    const [abTest, setABTest] = useState(null);
    const [events, setEvents] = useState([]);
    const [chartData, setChartData] = useState([]);
    const wsRef = useRef(null);
    const { showToast } = useStore();

    // ─── Workforce State ───
    const [wfmStates, setWfmStates] = useState([]);
    const [wfmLoading, setWfmLoading] = useState(true);

    const availableCount = wfmStates.filter(s => s.state === 'available').length;
    const acwCount = wfmStates.filter(s => s.state === 'acw').length;
    const dialingCount = wfmStates.filter(s => s.state === 'dialing').length;
    const idleCount = wfmStates.filter(s => s.state === 'offline').length;

    useEffect(() => {
        // Fetch KPIs, A/B, Events
        Promise.all([api.kpis(), api.abTest(), api.events()]).then(([k, ab, ev]) => {
            setKPIs(k); setABTest(ab); setEvents(ev);
            if (k) setChartData(prev => [...prev.slice(-11), buildChartPoint(k)]);
        }).catch(() => { });

        // WebSocket for live updates
        if (!WS_URL) return;
        try {
            const ws = new WebSocket(`${WS_URL}?type=dashboard`);
            wsRef.current = ws;
            ws.onmessage = (e) => {
                try {
                    const msg = JSON.parse(e.data);
                    if (msg.type === 'kpi') {
                        setKPIs(msg.data);
                        setChartData(prev => [...prev.slice(-11), buildChartPoint(msg.data)]);
                        if (msg.data.events?.length) setEvents(msg.data.events);
                    }
                } catch { }
            };
            ws.onerror = () => { };
            return () => ws.close();
        } catch { }
    }, []);

    // ─── Workforce data fetch ───
    useEffect(() => {
        loadWfmStates();
        const int = setInterval(loadWfmStates, 10000);
        return () => clearInterval(int);
    }, []);

    async function loadWfmStates() {
        try {
            const data = await api.get('/wfm/states');
            if (data.length === 0) {
                setWfmStates([
                    { user: { id: '1', name: 'John Doe', role: 'agent' }, state: 'available', timestamp: new Date(Date.now() - 1000 * 60 * 5) },
                    { user: { id: '2', name: 'Sarah Smith', role: 'supervisor' }, state: 'acw', timestamp: new Date(Date.now() - 1000 * 45) },
                    { user: { id: '3', name: 'Mike Ross', role: 'agent' }, state: 'break', timestamp: new Date(Date.now() - 1000 * 60 * 12) },
                    { user: { id: '4', name: 'Amanda Jones', role: 'agent' }, state: 'available', timestamp: new Date(Date.now() - 1000 * 60 * 2) },
                    { user: { id: '5', name: 'Robert Chen', role: 'agent' }, state: 'offline', timestamp: new Date(Date.now() - 1000 * 60 * 60 * 2) },
                ]);
                setWfmLoading(false);
                return;
            }
            setWfmStates(data);
        } catch (err) {
            console.error(err);
        } finally {
            setWfmLoading(false);
        }
    }

    return (
        <div className="space-y-8">
            {/* Header */}
            <div className="page-header">
                <div>
                    <h1 className="text-2xl font-bold text-gray-900">Command Center</h1>
                    <p className="text-sm text-gray-400 mt-0.5">Real-time AI ops & workforce overview</p>
                </div>
                <div className="flex items-center gap-2 px-3 py-1.5 bg-emerald-50 rounded-xl">
                    <span className="w-2 h-2 rounded-full bg-emerald-500 animate-pulse"></span>
                    <span className="text-xs font-semibold text-emerald-700">Live</span>
                </div>
            </div>

            {/* KPIs Row */}
            <div className="grid grid-cols-2 lg:grid-cols-4 gap-4">
                <KPICard icon={Phone} label="Total Calls" value={kpis?.totalCalls ?? 0} color="orange" sub="today" />
                <KPICard icon={Phone} label="Active Calls" value={kpis?.activeCalls ?? 0} color="blue" sub="right now" />
                <KPICard icon={PhoneOff} label="Calls Idle" value={kpis?.callsIdle ?? 0} color="amber" sub="waiting" />
                <KPICard icon={PhoneForwarded} label="Dialing & Ringing" value={kpis?.dialingRinging ?? 0} color="purple" sub="in progress" />
                <KPICard icon={Clock} label="Wrap Up" value={kpis?.wrapUp ?? 0} color="red" sub="post-call" />
                <KPICard icon={Shield} label="SLA Rate" value={kpis ? `${kpis.slaPercent}%` : '—'} color="green" sub="target: 95%" />
                <KPICard icon={Zap} label="API Fallback Rate" value={kpis ? `${kpis.apiFallbackRate}%` : '—'} color="red" sub="last 30m" />
            </div>

            {/* Chart + Workforce Summary Row */}
            <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">

                {/* Live chart */}
                <div className="card lg:col-span-2">
                    <h2 className="section-title mb-4">Live Call Volume & MOS</h2>
                    <ResponsiveContainer width="100%" height={180}>
                        <LineChart data={chartData}>
                            <CartesianGrid strokeDasharray="3 3" stroke="#f0f0f0" />
                            <XAxis dataKey="time" tick={{ fontSize: 11 }} />
                            <YAxis yAxisId="left" tick={{ fontSize: 11 }} />
                            <YAxis yAxisId="right" orientation="right" domain={[3, 5]} tick={{ fontSize: 11 }} />
                            <Tooltip />
                            <Line yAxisId="left" type="monotone" dataKey="calls" stroke="#f97316" strokeWidth={2} dot={false} />
                            <Line yAxisId="right" type="monotone" dataKey="mos" stroke="#10b981" strokeWidth={2} dot={false} />
                        </LineChart>
                    </ResponsiveContainer>
                </div>

                {/* Workforce Summary (merged from WFM) */}
                <div className="card">
                    <h2 className="section-title mb-4">Workforce</h2>
                    <div className="space-y-4">
                        <div>
                            <div className="flex justify-between text-sm mb-1">
                                <span className="text-gray-500">AI Agents</span>
                                <span className="font-semibold text-gray-800">{kpis?.aiAgentsAvailable ?? 8} available</span>
                            </div>
                            <div className="h-2.5 rounded-full bg-orange-100">
                                <div className="h-2.5 rounded-full bg-orange-500" style={{ width: `${(kpis?.aiAgentsAvailable ?? 8) / 10 * 100}%` }}></div>
                            </div>
                        </div>
                        <div>
                            <div className="flex justify-between text-sm mb-1">
                                <span className="text-gray-500">Human Agents</span>
                                <span className="font-semibold text-gray-800">{kpis?.humanAgentsAvailable ?? 3} available</span>
                            </div>
                            <div className="h-2.5 rounded-full bg-blue-100">
                                <div className="h-2.5 rounded-full bg-blue-500" style={{ width: `${(kpis?.humanAgentsAvailable ?? 3) / 5 * 100}%` }}></div>
                            </div>
                        </div>
                        <div className="pt-2 border-t border-gray-50">
                            <div className="text-sm text-gray-500">Queue Depth</div>
                            <div className="text-2xl font-bold text-gray-900">{kpis?.queueDepth ?? 0}</div>
                        </div>
                    </div>
                </div>
            </div>

            {/* ─── Workforce Status Board (merged from WFM page) ─── */}
            <div className="space-y-4">
                <h2 className="text-lg font-bold text-gray-900 flex items-center gap-2">
                    <Users size={20} className="text-orange-500" /> Live Agent Status
                </h2>

                {/* Workforce KPI pills */}
                <div className="grid grid-cols-2 md:grid-cols-4 gap-3">
                    <div className="card p-4 border-l-4 border-l-emerald-500 flex items-center gap-3 bg-white/50">
                        <div className="w-10 h-10 rounded-xl bg-emerald-50 flex items-center justify-center text-emerald-600"><PhoneCall size={20} /></div>
                        <div><div className="text-xl font-black text-gray-900">{availableCount}</div><div className="text-[10px] font-bold text-emerald-600 uppercase tracking-widest">Available</div></div>
                    </div>
                    <div className="card p-4 border-l-4 border-l-blue-500 flex items-center gap-3 bg-white/50">
                        <div className="w-10 h-10 rounded-xl bg-blue-50 flex items-center justify-center text-blue-600"><Clock size={20} /></div>
                        <div><div className="text-xl font-black text-gray-900">{acwCount}</div><div className="text-[10px] font-bold text-blue-600 uppercase tracking-widest">Wrap-up</div></div>
                    </div>
                    <div className="card p-4 border-l-4 border-l-purple-500 flex items-center gap-3 bg-white/50">
                        <div className="w-10 h-10 rounded-xl bg-purple-50 flex items-center justify-center text-purple-600"><PhoneForwarded size={20} /></div>
                        <div><div className="text-xl font-black text-gray-900">{dialingCount}</div><div className="text-[10px] font-bold text-purple-600 uppercase tracking-widest">Dialing</div></div>
                    </div>
                    <div className="card p-4 border-l-4 border-l-gray-400 flex items-center gap-3 bg-white/50">
                        <div className="w-10 h-10 rounded-xl bg-gray-50 flex items-center justify-center text-gray-600"><Users size={20} /></div>
                        <div><div className="text-xl font-black text-gray-900">{wfmStates.length}</div><div className="text-[10px] font-bold text-gray-500 uppercase tracking-widest">Total Staff</div></div>
                    </div>
                </div>

                {/* Agent Status Table */}
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
                            {wfmLoading ? (
                                <tr><td colSpan="5" className="p-8 text-center"><Loader2 className="animate-spin text-gray-400 mx-auto" /></td></tr>
                            ) : wfmStates.length === 0 ? (
                                <tr><td colSpan="5" className="p-8 text-center text-gray-400 text-sm">No agent states found.</td></tr>
                            ) : wfmStates.map(s => {
                                const stateInfo = AUX_STATES.find(x => x.id === s.state) || AUX_STATES[0];
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
                                );
                            })}
                        </tbody>
                    </table>
                </div>
            </div>

            {/* A/B Test + Event Log */}
            <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">

                {/* A/B Test */}
                <div className="card">
                    <h2 className="section-title mb-4 flex items-center gap-2">
                        <TrendingUp size={16} className="text-orange-500" /> A/B Model Test
                    </h2>
                    {abTest && (
                        <div className="space-y-4">
                            {['champion', 'challenger'].map(key => {
                                const m = abTest[key];
                                const isWinner = abTest.winner === key;
                                return (
                                    <div key={key} className={`p-4 rounded-xl border-2 ${isWinner ? 'border-orange-200 bg-orange-50' : 'border-gray-100'}`}>
                                        <div className="flex items-center justify-between mb-2">
                                            <span className="font-semibold text-gray-800 capitalize">{key}</span>
                                            {isWinner && <span className="badge-orange">Winner {abTest.confidence}% conf.</span>}
                                        </div>
                                        <div className="font-mono text-xs text-gray-400 mb-3">{m.model}</div>
                                        <div className="grid grid-cols-3 gap-3 text-center">
                                            <div><div className="text-lg font-bold text-gray-800">{m.csat}</div><div className="text-xs text-gray-400">CSAT</div></div>
                                            <div><div className="text-lg font-bold text-gray-800">{m.avgDuration}s</div><div className="text-xs text-gray-400">Avg Dur.</div></div>
                                            <div><div className="text-lg font-bold text-gray-800">{m.conversions}%</div><div className="text-xs text-gray-400">Conv.</div></div>
                                        </div>
                                    </div>
                                );
                            })}
                        </div>
                    )}
                </div>

                {/* Event Log */}
                <div className="card">
                    <h2 className="section-title mb-4 flex items-center gap-2">
                        <RadioTower size={16} className="text-orange-500" /> System Event Log
                    </h2>
                    <div className="space-y-2 max-h-64 overflow-y-auto">
                        {events.length === 0 && <p className="text-sm text-gray-400 text-center py-4">No events yet</p>}
                        {events.map(ev => (
                            <div key={ev.id} className="flex items-start gap-3 py-2 border-b border-gray-50 last:border-0">
                                <span className={`mt-0.5 ${ev.severity === 'error' ? 'text-red-500' : ev.severity === 'warning' ? 'text-amber-500' : 'text-emerald-500'}`}>
                                    <Activity size={13} />
                                </span>
                                <div className="flex-1 min-w-0">
                                    <div className="text-xs font-mono text-gray-500">{ev.type}</div>
                                    <div className="text-sm text-gray-700 truncate">{ev.message}</div>
                                </div>
                                <div className="text-xs text-gray-300 shrink-0">{new Date(ev.createdAt).toLocaleTimeString()}</div>
                            </div>
                        ))}
                    </div>
                </div>
            </div>
        </div>
    );
}
