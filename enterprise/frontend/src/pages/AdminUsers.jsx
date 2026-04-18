import { useState, useEffect } from 'react';
import { api } from '../lib/api.js';
import {
    Crown, Users, Bot, PhoneOutgoing, Key, FileText, Trash2, Plus,
    Eye, X, Loader2, Search, Calendar, Globe, Shield, BarChart3,
    Phone, GitBranch, CalendarClock, AlertCircle
} from 'lucide-react';

export default function AdminUsers() {
    const [users, setUsers] = useState([]);
    const [stats, setStats] = useState(null);
    const [loading, setLoading] = useState(true);
    const [selectedUser, setSelectedUser] = useState(null);
    const [userDetail, setUserDetail] = useState(null);
    const [detailLoading, setDetailLoading] = useState(false);
    const [showCreate, setShowCreate] = useState(false);
    const [createForm, setCreateForm] = useState({ email: '', name: '', password: '', role: 'user' });
    const [error, setError] = useState('');
    const [search, setSearch] = useState('');

    useEffect(() => { loadData(); }, []);

    async function loadData() {
        setLoading(true);
        try {
            const [usersData, statsData] = await Promise.all([
                api.get('/admin/users'),
                api.get('/admin/stats'),
            ]);
            setUsers(usersData);
            setStats(statsData);
        } catch (e) {
            setError(e.message);
        } finally {
            setLoading(false);
        }
    }

    async function viewUser(userId) {
        setSelectedUser(userId);
        setDetailLoading(true);
        try {
            const data = await api.get(`/admin/users/${userId}`);
            setUserDetail(data);
        } catch (e) {
            setError(e.message);
        } finally {
            setDetailLoading(false);
        }
    }

    async function createUser() {
        try {
            await api.post('/admin/users', createForm);
            setShowCreate(false);
            setCreateForm({ email: '', name: '', password: '', role: 'user' });
            loadData();
        } catch (e) {
            setError(e.message);
        }
    }

    async function deleteUser(id, email) {
        if (!confirm(`Delete user ${email} and ALL their data? This cannot be undone.`)) return;
        try {
            await api.delete(`/admin/users/${id}`);
            setSelectedUser(null);
            setUserDetail(null);
            loadData();
        } catch (e) {
            setError(e.message);
        }
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
        <div className="space-y-6">
            {/* Header */}
            <div className="flex items-center justify-between">
                <div>
                    <h1 className="text-2xl font-bold text-gray-900 flex items-center gap-2">
                        <Crown size={24} className="text-orange-500" />
                        Super Admin Panel
                    </h1>
                    <p className="text-sm text-gray-500 mt-1">Manage users, monitor activity, and control platform access</p>
                </div>
                <button
                    onClick={() => setShowCreate(true)}
                    className="btn-primary flex items-center gap-2"
                >
                    <Plus size={16} /> Create User
                </button>
            </div>

            {/* Error */}
            {error && (
                <div className="flex items-center gap-2 p-3 bg-red-50 border border-red-100 rounded-xl text-sm text-red-600">
                    <AlertCircle size={15} />{error}
                    <button onClick={() => setError('')} className="ml-auto"><X size={14} /></button>
                </div>
            )}

            {/* Platform Stats */}
            {stats && (
                <div className="grid grid-cols-2 md:grid-cols-4 lg:grid-cols-7 gap-3">
                    {[
                        { label: 'Total Users', value: stats.totalUsers, icon: Users, color: 'orange' },
                        { label: 'Total Agents', value: stats.totalAgents, icon: Bot, color: 'blue' },
                        { label: 'Total Calls', value: stats.totalCalls, icon: Phone, color: 'emerald' },
                        { label: 'Campaigns', value: stats.totalCampaigns, icon: PhoneOutgoing, color: 'purple' },
                        { label: 'API Keys', value: stats.totalApiKeys, icon: Key, color: 'amber' },
                        { label: 'Active Calls', value: stats.activeCalls, icon: BarChart3, color: 'green' },
                        { label: 'Documents', value: stats.totalDocs, icon: FileText, color: 'cyan' },
                    ].map(s => (
                        <div key={s.label} className="glass-panel p-4 rounded-2xl text-center">
                            <div className={`w-10 h-10 mx-auto rounded-xl bg-${s.color}-50 flex items-center justify-center mb-2`}>
                                <s.icon size={18} className={`text-${s.color}-500`} />
                            </div>
                            <div className="text-2xl font-bold text-gray-900">{s.value}</div>
                            <div className="text-xs text-gray-500 mt-0.5">{s.label}</div>
                        </div>
                    ))}
                </div>
            )}

            {/* Search */}
            <div className="relative">
                <Search size={16} className="absolute left-4 top-1/2 -translate-y-1/2 text-gray-400" />
                <input
                    value={search}
                    onChange={e => setSearch(e.target.value)}
                    placeholder="Search users by email or name..."
                    className="w-full pl-10 pr-4 py-3 glass-panel rounded-xl text-sm border-none outline-none bg-white/60"
                />
            </div>

            {/* Users Table */}
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
                                        <span className={`px-2 py-0.5 rounded-full text-xs font-semibold ${u.role === 'superadmin' ? 'bg-orange-100 text-orange-700' :
                                                u.role === 'admin' ? 'bg-blue-100 text-blue-700' :
                                                    'bg-gray-100 text-gray-600'
                                            }`}>{u.role}</span>
                                    </td>
                                    <td className="px-3 py-3 text-center text-sm font-semibold text-gray-700">{u.agents}</td>
                                    <td className="px-3 py-3 text-center text-sm font-semibold text-gray-700">{u.campaigns}</td>
                                    <td className="px-3 py-3 text-center text-sm font-semibold text-gray-700">{u.calls}</td>
                                    <td className="px-3 py-3 text-center text-sm font-semibold text-gray-700">{u.apiKeys}</td>
                                    <td className="px-3 py-3 text-center text-sm font-semibold text-gray-700">{u.knowledgeDocs}</td>
                                    <td className="px-3 py-3 text-xs text-gray-500">{new Date(u.createdAt).toLocaleDateString()}</td>
                                    <td className="px-3 py-3 text-center">
                                        <div className="flex items-center justify-center gap-1">
                                            <button
                                                onClick={() => viewUser(u.id)}
                                                className="p-1.5 hover:bg-blue-50 rounded-lg text-gray-400 hover:text-blue-500 transition-colors"
                                                title="View Details"
                                            >
                                                <Eye size={14} />
                                            </button>
                                            {u.role !== 'superadmin' && (
                                                <button
                                                    onClick={() => deleteUser(u.id, u.email)}
                                                    className="p-1.5 hover:bg-red-50 rounded-lg text-gray-400 hover:text-red-500 transition-colors"
                                                    title="Delete User"
                                                >
                                                    <Trash2 size={14} />
                                                </button>
                                            )}
                                        </div>
                                    </td>
                                </tr>
                            ))}
                            {filtered.length === 0 && (
                                <tr>
                                    <td colSpan={9} className="text-center py-12 text-gray-400 text-sm">
                                        No users found
                                    </td>
                                </tr>
                            )}
                        </tbody>
                    </table>
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
                            <div>
                                <label className="text-xs font-semibold text-gray-600 mb-1 block">Email</label>
                                <input
                                    value={createForm.email}
                                    onChange={e => setCreateForm(f => ({ ...f, email: e.target.value }))}
                                    className="input-field"
                                    placeholder="user@company.com"
                                />
                            </div>
                            <div>
                                <label className="text-xs font-semibold text-gray-600 mb-1 block">Full Name</label>
                                <input
                                    value={createForm.name}
                                    onChange={e => setCreateForm(f => ({ ...f, name: e.target.value }))}
                                    className="input-field"
                                    placeholder="John Smith"
                                />
                            </div>
                            <div>
                                <label className="text-xs font-semibold text-gray-600 mb-1 block">Password</label>
                                <input
                                    type="password"
                                    value={createForm.password}
                                    onChange={e => setCreateForm(f => ({ ...f, password: e.target.value }))}
                                    className="input-field"
                                    placeholder="••••••••"
                                />
                            </div>
                            <div>
                                <label className="text-xs font-semibold text-gray-600 mb-1 block">Role</label>
                                <select
                                    value={createForm.role}
                                    onChange={e => setCreateForm(f => ({ ...f, role: e.target.value }))}
                                    className="input-field"
                                >
                                    <option value="user">User</option>
                                    <option value="admin">Admin</option>
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

            {/* User Detail Modal */}
            {selectedUser && (
                <div className="fixed inset-0 z-50 bg-gray-900/40 backdrop-blur-sm flex items-center justify-center">
                    <div className="bg-white rounded-2xl shadow-2xl w-full max-w-3xl max-h-[85vh] overflow-y-auto p-6 space-y-5">
                        <div className="flex items-center justify-between">
                            <h3 className="text-lg font-bold text-gray-900">User Details</h3>
                            <button onClick={() => { setSelectedUser(null); setUserDetail(null); }} className="p-1 hover:bg-gray-100 rounded-lg"><X size={18} /></button>
                        </div>

                        {detailLoading ? (
                            <div className="flex justify-center py-12"><Loader2 size={24} className="animate-spin text-orange-500" /></div>
                        ) : userDetail ? (
                            <div className="space-y-5">
                                {/* User Info */}
                                <div className="flex items-center gap-4 p-4 bg-gray-50 rounded-xl">
                                    <div className="w-12 h-12 rounded-full bg-gradient-to-br from-orange-400 to-orange-600 flex items-center justify-center text-white text-lg font-bold">
                                        {userDetail.user.name?.[0]?.toUpperCase() || 'U'}
                                    </div>
                                    <div className="flex-1">
                                        <div className="font-bold text-gray-900">{userDetail.user.name}</div>
                                        <div className="text-sm text-gray-500">{userDetail.user.email}</div>
                                        <div className="text-xs text-gray-400 mt-0.5">Joined: {new Date(userDetail.user.createdAt).toLocaleDateString()}</div>
                                    </div>
                                    <span className={`px-3 py-1 rounded-full text-xs font-semibold ${userDetail.user.role === 'superadmin' ? 'bg-orange-100 text-orange-700' :
                                            userDetail.user.role === 'admin' ? 'bg-blue-100 text-blue-700' :
                                                'bg-gray-100 text-gray-600'
                                        }`}>{userDetail.user.role}</span>
                                </div>

                                {/* Agents */}
                                <DetailSection title="Agents" icon={Bot} count={userDetail.agents.length}>
                                    {userDetail.agents.map(a => (
                                        <div key={a.id} className="flex items-center justify-between p-2 bg-white rounded-lg border border-gray-100">
                                            <span className="text-sm font-medium text-gray-800">{a.name}</span>
                                            <span className={`px-2 py-0.5 rounded-full text-xs ${a.status === 'active' ? 'bg-green-100 text-green-700' : 'bg-gray-100 text-gray-500'}`}>{a.status}</span>
                                        </div>
                                    ))}
                                </DetailSection>

                                {/* API Keys */}
                                <DetailSection title="API Keys" icon={Key} count={userDetail.apiKeys.length}>
                                    {userDetail.apiKeys.map(k => (
                                        <div key={k.id} className="flex items-center justify-between p-2 bg-white rounded-lg border border-gray-100">
                                            <div>
                                                <span className="text-sm font-medium text-gray-800">{k.name}</span>
                                                <span className="text-xs text-gray-400 ml-2">{k.prefix}</span>
                                            </div>
                                            <div className="flex items-center gap-2">
                                                <span className={`px-2 py-0.5 rounded-full text-xs ${k.env === 'live' ? 'bg-green-100 text-green-700' : 'bg-yellow-100 text-yellow-700'}`}>{k.env}</span>
                                                <span className={`px-2 py-0.5 rounded-full text-xs ${k.active ? 'bg-green-100 text-green-700' : 'bg-red-100 text-red-700'}`}>{k.active ? 'active' : 'revoked'}</span>
                                            </div>
                                        </div>
                                    ))}
                                </DetailSection>

                                {/* Campaigns */}
                                <DetailSection title="Campaigns" icon={PhoneOutgoing} count={userDetail.campaigns.length}>
                                    {userDetail.campaigns.map(c => (
                                        <div key={c.id} className="flex items-center justify-between p-2 bg-white rounded-lg border border-gray-100">
                                            <span className="text-sm font-medium text-gray-800">{c.name}</span>
                                            <div className="flex items-center gap-2">
                                                <span className="text-xs text-gray-400">{c.totalLeads} leads</span>
                                                <span className={`px-2 py-0.5 rounded-full text-xs ${c.status === 'running' ? 'bg-green-100 text-green-700' : c.status === 'completed' ? 'bg-blue-100 text-blue-700' : 'bg-gray-100 text-gray-500'}`}>{c.status}</span>
                                            </div>
                                        </div>
                                    ))}
                                </DetailSection>

                                {/* Recent Calls */}
                                <DetailSection title="Recent Calls" icon={Phone} count={userDetail.recentCalls.length}>
                                    {userDetail.recentCalls.slice(0, 10).map(c => (
                                        <div key={c.id} className="flex items-center justify-between p-2 bg-white rounded-lg border border-gray-100">
                                            <div>
                                                <span className="text-sm font-medium text-gray-800">{c.phoneNumber || 'N/A'}</span>
                                                <span className="text-xs text-gray-400 ml-2">{c.duration ? `${c.duration}s` : '-'}</span>
                                            </div>
                                            <span className={`px-2 py-0.5 rounded-full text-xs ${c.sentiment === 'positive' ? 'bg-green-100 text-green-700' : c.sentiment === 'negative' ? 'bg-red-100 text-red-700' : 'bg-gray-100 text-gray-500'}`}>{c.sentiment || c.status}</span>
                                        </div>
                                    ))}
                                </DetailSection>

                                {/* Webhooks */}
                                <DetailSection title="Webhooks" icon={Globe} count={userDetail.webhooks.length}>
                                    {userDetail.webhooks.map(w => (
                                        <div key={w.id} className="flex items-center justify-between p-2 bg-white rounded-lg border border-gray-100">
                                            <span className="text-sm text-gray-800 truncate max-w-xs">{w.url}</span>
                                            <span className={`px-2 py-0.5 rounded-full text-xs ${w.active ? 'bg-green-100 text-green-700' : 'bg-gray-100 text-gray-500'}`}>{w.active !== false ? 'active' : 'inactive'}</span>
                                        </div>
                                    ))}
                                </DetailSection>

                                {/* Knowledge Docs */}
                                <DetailSection title="Knowledge Docs" icon={FileText} count={userDetail.knowledgeDocs.length}>
                                    {userDetail.knowledgeDocs.map(d => (
                                        <div key={d.id} className="flex items-center justify-between p-2 bg-white rounded-lg border border-gray-100">
                                            <span className="text-sm font-medium text-gray-800">{d.name}</span>
                                            <span className={`px-2 py-0.5 rounded-full text-xs ${d.status === 'synced' ? 'bg-green-100 text-green-700' : d.status === 'error' ? 'bg-red-100 text-red-700' : 'bg-yellow-100 text-yellow-700'}`}>{d.status}</span>
                                        </div>
                                    ))}
                                </DetailSection>

                                {/* Follow-Ups */}
                                <DetailSection title="Follow-Ups" icon={CalendarClock} count={userDetail.followups.length}>
                                    {userDetail.followups.map(f => (
                                        <div key={f.id} className="flex items-center justify-between p-2 bg-white rounded-lg border border-gray-100">
                                            <div>
                                                <span className="text-sm font-medium text-gray-800">{f.phoneNumber}</span>
                                                <span className="text-xs text-gray-400 ml-2">{f.reason}</span>
                                            </div>
                                            <span className={`px-2 py-0.5 rounded-full text-xs ${f.status === 'completed' ? 'bg-green-100 text-green-700' : 'bg-yellow-100 text-yellow-700'}`}>{f.status}</span>
                                        </div>
                                    ))}
                                </DetailSection>
                            </div>
                        ) : null}
                    </div>
                </div>
            )}
        </div>
    );
}

function DetailSection({ title, icon: Icon, count, children }) {
    if (count === 0) return null;
    return (
        <div>
            <div className="flex items-center gap-2 mb-2">
                <Icon size={16} className="text-orange-500" />
                <span className="text-sm font-bold text-gray-700">{title}</span>
                <span className="text-xs bg-gray-100 text-gray-500 px-1.5 py-0.5 rounded-full">{count}</span>
            </div>
            <div className="space-y-1 bg-gray-50 p-3 rounded-xl">
                {children}
            </div>
        </div>
    );
}
