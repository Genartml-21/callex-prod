import { useState, useEffect } from 'react';
import { NavLink, Outlet, useLocation } from 'react-router-dom';
import { useAuth } from '../contexts/AuthContext.jsx';
import { useStore } from '../store/index.js';
import { LogoImage, APP_NAME } from '../lib/logo.jsx';
import AIAssistant from './AIAssistant.jsx';
import {
    LayoutDashboard, Headphones, Bot, BookOpen, FlaskConical,
    PhoneOutgoing, BarChart3, GitBranch, Plug, ShieldCheck, Settings, LogOut, Activity,
    Users, Search, Command, FileAudio, CheckCircle2, Phone, CreditCard, Download, CalendarClock, Crown
} from 'lucide-react';
// Enterprise navigation grouped by category
const NAV_SECTIONS = [
    {
        title: 'Operations',
        items: [
            { to: '/dashboard', icon: LayoutDashboard, label: 'Command Center' },
            { to: '/supervisor', icon: Headphones, label: 'Live Supervisor' },
        ]
    },
    {
        title: 'Analytics & QA',
        items: [
            { to: '/analytics', icon: FileAudio, label: 'Logs & Recordings' },
            { to: '/qa', icon: CheckCircle2, label: 'QA & Dispositions' },
            { to: '/reports', icon: Download, label: 'Reports & Exports' },
        ]
    },
    {
        title: 'Automation',
        items: [
            { to: '/agents', icon: Bot, label: 'Agent Studio' },
            { to: '/dialer', icon: PhoneOutgoing, label: 'Outbound Dialer' },
            { to: '/simulation', icon: FlaskConical, label: 'Simulation & QA' },
        ]
    },
    {
        title: 'Routing & Telecom',
        items: [
            { to: '/routing', icon: GitBranch, label: 'Routing Rules' },
            { to: '/followups', icon: CalendarClock, label: 'Auto Follow-Ups' },
            { to: '/telecom', icon: Phone, label: 'Phone Numbers & DNC' },
        ]
    },
    {
        title: 'System Admin',
        items: [
            { to: '/knowledge', icon: BookOpen, label: 'Knowledge Base' },
            { to: '/integrations', icon: Plug, label: 'Integrations' },
            { to: '/security', icon: ShieldCheck, label: 'Security & Legal' },
            { to: '/settings', icon: Settings, label: 'Settings' },
            { to: '/billing', icon: CreditCard, label: 'Billing & Usage' },
        ]
    }
];

// Combine all routes for Cmd+K search
const ALL_ROUTES = NAV_SECTIONS.flatMap(s => s.items);

export default function Layout() {
    const { user, userRole, logout } = useAuth();
    const { showToast } = useStore();
    const location = useLocation();

    // ═══ SUPERADMIN: monitoring-only sidebar ═══
    const activeSections = userRole === 'superadmin' ? [
        {
            title: 'Platform',
            items: [
                { to: '/dashboard', icon: LayoutDashboard, label: 'Command Center' },
                { to: '/supervisor', icon: Headphones, label: 'Live Supervisor' },
            ]
        },
        {
            title: 'Analytics',
            items: [
                { to: '/analytics', icon: FileAudio, label: 'Logs & Recordings' },
                { to: '/reports', icon: Download, label: 'Reports & Exports' },
            ]
        },
        {
            title: 'Agents',
            items: [
                { to: '/agents', icon: Bot, label: 'Agent Studio' },
            ]
        },
        {
            title: 'Admin',
            items: [
                { to: '/admin/users', icon: Crown, label: 'Users Management' },
                { to: '/settings', icon: Settings, label: 'Settings' },
                { to: '/billing', icon: CreditCard, label: 'Billing & Usage' },
            ]
        },
    ] : userRole === 'admin' ? NAV_SECTIONS : NAV_SECTIONS.map(s => {
        const allowed = ['/dashboard', '/agents', '/billing', '/analytics', '/followups', '/routing', '/integrations', '/settings', '/dialer'];
        const items = s.items.filter(i => allowed.includes(i.to));
        return items.length > 0 ? { ...s, items } : null;
    }).filter(Boolean);

    const activeRoutes = activeSections.flatMap(s => s.items);

    async function handleLogout() {
        await logout();
        showToast('Signed out successfully', 'info');
    }

    const initials = user?.displayName
        ? user.displayName.split(' ').map(n => n[0]).join('').toUpperCase().slice(0, 2)
        : user?.email?.[0]?.toUpperCase() || 'U';

    return (
        <div className="flex h-screen bg-transparent overflow-hidden">
            {/* Sidebar */}
            <aside className="w-64 glass-panel flex flex-col shrink-0 z-20">
                {/* Logo */}
                <div className="px-5 py-5 border-b border-white/50 bg-white/40 backdrop-blur-md">
                    <div className="flex items-center gap-2.5">
                        <LogoImage size={32} />
                        <div>
                            <div className="font-bold text-gray-900 text-sm leading-none">{APP_NAME.split(' ')[0]}</div>
                            <div className="text-xs text-gray-400 mt-0.5">Enterprise Platform</div>
                        </div>
                    </div>
                </div>

                {/* Nav */}
                <nav className="flex-1 py-4 overflow-y-auto w-full">
                    {activeSections.map((section, idx) => (
                        <div key={idx} className="mb-6 last:mb-0">
                            <div className="px-6 text-[10px] font-bold tracking-wider text-gray-400 uppercase mb-2">
                                {section.title}
                            </div>
                            <div className="space-y-0.5">
                                {section.items.map(({ to, icon: Icon, label }) => (
                                    <NavLink
                                        key={to}
                                        to={to}
                                        className={({ isActive }) =>
                                            `flex items-center gap-3 px-4 py-2 mx-3 rounded-xl text-sm font-medium transition-all duration-300 ${isActive
                                                ? 'bg-orange-500/10 text-orange-600 shadow-sm shadow-orange-500/10 border border-orange-500/20 backdrop-blur-sm shadow-inner'
                                                : 'text-gray-500 hover:text-gray-900 hover:bg-white/60 hover:shadow-sm'
                                            }`
                                        }
                                    >
                                        <Icon size={16} className="shrink-0" />
                                        <span className="truncate">{label}</span>
                                    </NavLink>
                                ))}
                            </div>
                        </div>
                    ))}
                </nav>

                {/* User footer */}
                <div className="p-4 border-t border-white/50 bg-white/40 backdrop-blur-md">
                    <div className="flex items-center gap-3">
                        {user?.photoURL ? (
                            <img src={user.photoURL} alt="" className="w-8 h-8 rounded-full object-cover" />
                        ) : (
                            <div className="w-8 h-8 rounded-full bg-orange-100 flex items-center justify-center text-orange-600 font-bold text-xs">
                                {initials}
                            </div>
                        )}
                        <div className="flex-1 min-w-0">
                            <div className="text-xs font-semibold text-gray-700 truncate">{user?.displayName || 'User'}</div>
                            <div className="text-xs text-gray-400 truncate">{user?.email}</div>
                        </div>
                        <button
                            onClick={handleLogout}
                            title="Sign out"
                            className="p-1.5 hover:bg-red-50 rounded-lg text-gray-400 hover:text-red-500 transition-colors"
                        >
                            <LogOut size={14} />
                        </button>
                    </div>
                </div>
            </aside>

            {/* Main content */}
            <main className="flex-1 overflow-y-auto flex flex-col relative w-full z-10 transition-all duration-300">
                {/* Command Palette trigger in status bar */}
                <LiveStatusBar />

                <div className="p-8 max-w-screen-xl w-full mx-auto flex-1 relative" key={location.pathname}>
                    <div className="animate-fade-in w-full h-full">
                        <Outlet />
                    </div>
                </div>
            </main>

            {/* Floating AI Assistant */}
            {userRole === 'user' && <AIAssistant />}

            {/* Global Command Palette */}
            <CommandPalette activeRoutes={activeRoutes} />
        </div>
    );
}

// Global Command Palette Modal
import { useNavigate } from 'react-router-dom';

function CommandPalette({ activeRoutes }) {
    const [open, setOpen] = useState(false);
    const [query, setQuery] = useState('');
    const navigate = useNavigate();

    useEffect(() => {
        const down = (e) => {
            if (e.key === 'k' && (e.metaKey || e.ctrlKey)) {
                e.preventDefault();
                setOpen((o) => !o);
            }
            if (e.key === 'Escape') setOpen(false);
        };
        document.addEventListener('keydown', down);
        return () => document.removeEventListener('keydown', down);
    }, []);

    if (!open) return null;

    const filtered = activeRoutes.filter(r =>
        r.label.toLowerCase().includes(query.toLowerCase())
    );

    return (
        <div className="fixed inset-0 z-50 bg-gray-900/40 backdrop-blur-sm flex items-start justify-center pt-[15vh]">
            <div className="bg-white w-full max-w-lg rounded-2xl shadow-2xl overflow-hidden border border-gray-100 flex flex-col" onClick={e => e.stopPropagation()}>
                <div className="flex items-center gap-3 p-4 border-b border-gray-100">
                    <Search size={18} className="text-gray-400" />
                    <input
                        autoFocus
                        value={query}
                        onChange={e => setQuery(e.target.value)}
                        placeholder="Search pages, settings, analytics... (Jump to)"
                        className="flex-1 bg-transparent border-none outline-none text-gray-900 placeholder:text-gray-400"
                    />
                    <div className="flex items-center gap-1">
                        <kbd className="px-1.5 py-0.5 bg-gray-100 border border-gray-200 rounded text-[10px] text-gray-500 font-mono">esc</kbd>
                    </div>
                </div>

                <div className="max-h-[60vh] overflow-y-auto p-2">
                    {filtered.length === 0 ? (
                        <div className="text-center py-8 text-gray-400 text-sm">No results found for "{query}"</div>
                    ) : (
                        filtered.map((r, i) => (
                            <button
                                key={r.to}
                                onClick={() => { navigate(r.to); setOpen(false); setQuery(''); }}
                                className="w-full flex items-center gap-3 p-3 text-left rounded-xl hover:bg-gray-50 text-gray-700 hover:text-orange-600 group outline-none focus:bg-orange-50 focus:text-orange-600 transition-colors"
                                autoFocus={i === 0 && query.length > 0}
                            >
                                <div className="w-8 h-8 rounded-lg bg-gray-100 group-hover:bg-orange-100 flex items-center justify-center group-focus:bg-orange-100">
                                    <r.icon size={16} className="text-gray-500 group-hover:text-orange-600 group-focus:text-orange-600" />
                                </div>
                                <div className="flex-1 font-medium text-sm">{r.label}</div>
                            </button>
                        ))
                    )}
                </div>
            </div>
            <div className="fixed inset-0 -z-10" onClick={() => setOpen(false)} />
        </div>
    );
}

function LiveStatusBar() {
    const [stats, setStats] = useState({ activeCalls: 0, mos: '4.20', sla: 98 });
    useEffect(() => {
        const fetchKPIs = async () => {
            try {
                const token = localStorage.getItem('token');
                const headers = token ? { Authorization: `Bearer ${token}` } : {};
                const res = await fetch('/api/dashboard/kpis', { headers });
                if (res.ok) {
                    const data = await res.json();
                    setStats({
                        activeCalls: data.activeCalls || 0,
                        mos: data.avgMOS ? parseFloat(data.avgMOS).toFixed(2) : '0.00',
                        sla: data.slaPercent != null ? Math.round(data.slaPercent) : 0,
                    });
                }
            } catch (e) { /* Backend offline, keep last known values */ }
        };
        fetchKPIs();
        const t = setInterval(fetchKPIs, 8000);
        return () => clearInterval(t);
    }, []);
    return (
        <div className="sticky top-0 z-30 glass-nav px-8 py-2.5 flex items-center gap-6 shadow-sm mb-2">
            <div className="flex items-center gap-2">
                <span className="relative flex h-2.5 w-2.5">
                    <span className="animate-ping absolute inline-flex h-full w-full rounded-full bg-emerald-400 opacity-75"></span>
                    <span className="relative inline-flex rounded-full h-2.5 w-2.5 bg-emerald-500"></span>
                </span>
                <span className="text-xs font-bold text-emerald-600 uppercase tracking-widest">Live</span>
            </div>
            <div className="h-4 w-px bg-gray-100" />
            <div className="flex items-center gap-1.5">
                <PhoneOutgoing size={12} className="text-orange-500" />
                <span className="text-xs font-semibold text-gray-700">{stats.activeCalls}</span>
                <span className="text-xs text-gray-400">active calls</span>
            </div>
            <div className="flex items-center gap-1.5">
                <Activity size={12} className="text-blue-500" />
                <span className="text-xs font-semibold text-gray-700">{stats.mos}</span>
                <span className="text-xs text-gray-400">MOS</span>
            </div>
            <div className="flex items-center gap-1.5">
                <ShieldCheck size={12} className="text-emerald-500" />
                <span className="text-xs font-semibold text-gray-700">{stats.sla}%</span>
                <span className="text-xs text-gray-400">SLA</span>
            </div>
            <div className="ml-auto text-xs text-gray-300">{new Date().toLocaleTimeString()}</div>
        </div>
    );
}
