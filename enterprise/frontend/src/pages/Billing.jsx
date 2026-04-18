import { useState, useEffect } from 'react';
import { api } from '../lib/api.js';
import { CreditCard, Activity, Cpu, PhoneCall, DownloadCloud, DollarSign, Wallet, FileText } from 'lucide-react';
import { AreaChart, Area, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer } from 'recharts';

export default function Billing() {
    const [view, setView] = useState('usage');
    const [stats, setStats] = useState(null);
    const [loading, setLoading] = useState(true);

    // Mock historical data for the chart
    const historyData = [
        { date: '1st', cost: 120 }, { date: '5th', cost: 350 }, { date: '10th', cost: 480 },
        { date: '15th', cost: 920 }, { date: '20th', cost: 1450 }, { date: '25th', cost: 1890 },
        { date: 'Today', cost: 2345 }
    ];

    useEffect(() => {
        loadStats();
    }, []);

    async function loadStats() {
        setLoading(true);
        try {
            const data = await api.billingStats();
            setStats(data || {
                month: new Date().toISOString().substring(0, 7),
                telecomMins: 0,
                llmTokens: 0,
                sttMinutes: 0,
                totalCostUsd: 0
            });
        } catch (err) {
            console.error(err);
        } finally {
            setLoading(false);
        }
    }

    // Cost estimates based on generic enterprise pricing
    const telecomCost = stats ? (stats.telecomMins * 0.013).toFixed(2) : 0;
    const llmCost = stats ? (stats.llmTokens / 1000 * 0.01).toFixed(2) : 0;
    const sttCost = stats ? (stats.sttMinutes * 0.005).toFixed(2) : 0;

    return (
        <div className="space-y-8">
            {/* Header */}
            <div className="flex justify-between items-end">
                <div className="page-header mb-0">
                    <div>
                        <h1 className="text-2xl font-bold text-gray-900">Billing & Subscription</h1>
                        <p className="text-sm text-gray-400 mt-0.5">Manage your plans, track telecom minutes, and infrastructure costs</p>
                    </div>
                </div>
                <div className="flex gap-2">
                    <div className="flex bg-gray-100 p-1 rounded-xl">
                        <button onClick={() => setView('usage')} className={`px-4 py-2 text-sm font-semibold rounded-lg transition-colors ${view === 'usage' ? 'bg-white shadow-sm text-gray-900' : 'text-gray-500 hover:text-gray-700'}`}>Usage & Cost</button>
                        <button onClick={() => setView('plans')} className={`px-4 py-2 text-sm font-semibold rounded-lg transition-colors ${view === 'plans' ? 'bg-white shadow-sm text-gray-900' : 'text-gray-500 hover:text-gray-700'}`}>Subscription Plans</button>
                    </div>
                    {view === 'usage' && (
                        <button className="btn-secondary text-sm h-[40px] mt-[4px]">
                            <DownloadCloud size={16} className="mr-2" /> Download Invoice
                        </button>
                    )}
                </div>
            </div>

            {view === 'plans' ? (
                <div className="animate-fade-in space-y-8">
                    <div className="card p-6 border-l-4 border-l-orange-500 bg-orange-50/30 flex justify-between items-center shadow-sm">
                        <div>
                            <div className="flex items-center gap-3 mb-1">
                                <h3 className="font-bold text-gray-900 text-lg">Current Plan: Enterprise (Usage-Based)</h3>
                                <span className="px-2 py-0.5 bg-emerald-100 text-emerald-700 text-xs font-bold rounded-full">Active</span>
                            </div>
                            <p className="text-sm text-gray-500">You are currently on the pay-as-you-go Enterprise tier. No monthly flat fee, pay only for what you use.</p>
                        </div>
                        <button className="btn-primary py-2 px-5 shadow-md bg-orange-500 hover:bg-orange-600 transition-all">Manage Payment Methods</button>
                    </div>

                    <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
                        {[
                            { name: 'Starter', price: '$49', desc: 'Perfect for small teams and startups testing voice agents.', callPrice: '$0.05/min', sttPrice: '$0.01/min', features: ['1 Voice Agent', 'Standard TTS Output', 'Basic Analytics', 'Email Support'] },
                            { name: 'Pro', price: '$199', desc: 'Scale up your operations with multiple agents and API access.', callPrice: '$0.03/min', sttPrice: '$0.008/min', features: ['5 Voice Agents', 'Ultra-realistic TTS', 'Advanced RAG Access', 'Priority Support'] },
                            { name: 'Enterprise', price: 'Custom', current: true, desc: 'High-volume routing, dedicated infrastructure, and SLA.', callPrice: 'Discounted', sttPrice: 'Discounted', features: ['Unlimited Agents', 'Custom Voice Clones', 'Dedicated Server', '24/7 Phone Support'] }
                        ].map(plan => (
                            <div key={plan.name} className={`relative flex flex-col card p-6 ${plan.current ? 'border-2 border-orange-500 shadow-lg' : 'border border-gray-100'}`}>
                                {plan.current && <div className="absolute top-0 right-0 bg-orange-500 text-white text-[10px] font-bold px-3 py-1 rounded-bl-lg rounded-tr-xl uppercase tracking-wider">Current Plan</div>}
                                <h3 className="text-xl font-bold text-gray-900 mb-2">{plan.name}</h3>
                                <div className="flex items-end gap-1 mb-4">
                                    <span className="text-3xl font-black text-gray-900">{plan.price}</span>
                                    {plan.price !== 'Custom' && <span className="text-gray-400 mb-1">/ month</span>}
                                </div>
                                <p className="text-sm text-gray-500 h-10 mb-6">{plan.desc}</p>

                                <div className="space-y-4 flex-1">
                                    <div className="bg-gray-50 p-3 rounded-lg flex justify-between text-sm">
                                        <span className="text-gray-500">Telecom</span><span className="font-semibold text-gray-900">{plan.callPrice}</span>
                                    </div>
                                    <div className="bg-gray-50 p-3 rounded-lg flex justify-between text-sm">
                                        <span className="text-gray-500">AI Inference</span><span className="font-semibold text-gray-900">{plan.sttPrice}</span>
                                    </div>

                                    <div className="pt-4 space-y-2">
                                        {plan.features.map(f => (
                                            <div key={f} className="flex items-center gap-2 text-sm text-gray-600">
                                                <div className="w-1.5 h-1.5 rounded-full bg-orange-500"></div> {f}
                                            </div>
                                        ))}
                                    </div>
                                </div>

                                <button className={`mt-8 w-full py-3 rounded-xl font-bold transition-colors ${plan.current ? 'bg-orange-100 text-orange-700' : 'bg-gray-900 text-white hover:bg-gray-800'}`}>
                                    {plan.current ? 'Your Current Plan' : 'Upgrade to ' + plan.name}
                                </button>
                            </div>
                        ))}
                    </div>
                </div>
            ) : loading || !stats ? (
                <div className="p-12 flex justify-center"><div className="w-8 h-8 border-4 border-orange-500 border-t-transparent rounded-full animate-spin"></div></div>
            ) : (
                <>
                    <div className="grid grid-cols-1 md:grid-cols-3 gap-6">

                        {/* Total Cost Highlight */}
                        <div className="card p-6 border-l-4 border-l-emerald-500 flex flex-col justify-center bg-emerald-50/30">
                            <div className="flex items-center gap-2 text-emerald-600 mb-2">
                                <Wallet size={20} /> <span className="font-bold text-sm tracking-wider uppercase">Current Month Cost</span>
                            </div>
                            <div className="text-4xl font-black text-gray-900">${parseFloat(stats.totalCostUsd).toLocaleString('en-US', { minimumFractionDigits: 2 })}</div>
                            <div className="text-xs text-gray-500 mt-2 font-medium">Billing Period: {stats.month}</div>
                        </div>

                        {/* AI Usage */}
                        <div className="card p-6 bg-white/50">
                            <div className="flex items-center justify-between mb-4">
                                <div className="flex items-center gap-2 text-purple-600">
                                    <Cpu size={18} /> <span className="font-bold text-sm">AI Inference</span>
                                </div>
                                <div className="text-sm font-bold text-gray-700">${llmCost}</div>
                            </div>
                            <div className="space-y-4">
                                <div>
                                    <div className="flex justify-between text-xs mb-1">
                                        <span className="text-gray-500">LLM Tokens (Callex-1.3)</span>
                                        <span className="font-mono text-gray-900">{stats.llmTokens.toLocaleString()}</span>
                                    </div>
                                    <div className="w-full h-1.5 bg-gray-100 rounded-full overflow-hidden">
                                        <div className="h-full bg-purple-500 w-[60%]"></div>
                                    </div>
                                </div>
                                <div>
                                    <div className="flex justify-between text-xs mb-1">
                                        <span className="text-gray-500">Voice Tokens (Callex-1.1)</span>
                                        <span className="font-mono text-gray-900">{stats.sttMinutes.toLocaleString()}</span>
                                    </div>
                                    <div className="w-full h-1.5 bg-gray-100 rounded-full overflow-hidden">
                                        <div className="h-full bg-purple-400 w-[80%]"></div>
                                    </div>
                                </div>
                            </div>
                        </div>

                        {/* Telecom Usage */}
                        <div className="card p-6 bg-white/50">
                            <div className="flex items-center justify-between mb-4">
                                <div className="flex items-center gap-2 text-blue-600">
                                    <PhoneCall size={18} /> <span className="font-bold text-sm">Telecom Trunking</span>
                                </div>
                                <div className="text-sm font-bold text-gray-700">${telecomCost}</div>
                            </div>
                            <div>
                                <div className="flex justify-between text-xs mb-1">
                                    <span className="text-gray-500">Twilio SIP Minutes</span>
                                    <span className="font-mono text-gray-900">{stats.telecomMins.toLocaleString()}</span>
                                </div>
                                <div className="w-full h-1.5 bg-gray-100 rounded-full overflow-hidden">
                                    <div className="h-full bg-blue-500 w-[75%]"></div>
                                </div>
                            </div>
                        </div>

                    </div>

                    <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">

                        {/* Chart */}
                        <div className="col-span-2 card p-6">
                            <h3 className="text-sm font-bold text-gray-800 mb-6 flex items-center gap-2">
                                <Activity size={16} className="text-orange-500" /> Cost Trajectory
                            </h3>
                            <div className="h-[300px] w-full">
                                <ResponsiveContainer width="100%" height="100%">
                                    <AreaChart data={historyData}>
                                        <defs>
                                            <linearGradient id="colorCost" x1="0" y1="0" x2="0" y2="1">
                                                <stop offset="5%" stopColor="#f97316" stopOpacity={0.3} />
                                                <stop offset="95%" stopColor="#f97316" stopOpacity={0} />
                                            </linearGradient>
                                        </defs>
                                        <CartesianGrid strokeDasharray="3 3" vertical={false} stroke="#f1f5f9" />
                                        <XAxis dataKey="date" axisLine={false} tickLine={false} tick={{ fontSize: 12, fill: '#94a3b8' }} dy={10} />
                                        <YAxis axisLine={false} tickLine={false} tick={{ fontSize: 12, fill: '#94a3b8' }} tickFormatter={(v) => `$${v}`} />
                                        <Tooltip
                                            contentStyle={{ borderRadius: '8px', border: 'none', boxShadow: '0 4px 6px -1px rgb(0 0 0 / 0.1)' }}
                                            formatter={(v) => [`$${v}`, 'Cumulative Cost']}
                                        />
                                        <Area type="monotone" dataKey="cost" stroke="#f97316" strokeWidth={3} fillOpacity={1} fill="url(#colorCost)" />
                                    </AreaChart>
                                </ResponsiveContainer>
                            </div>
                        </div>

                        {/* Invoices List */}
                        <div className="card flex flex-col">
                            <div className="p-4 border-b border-gray-100">
                                <h3 className="text-sm font-bold text-gray-800 flex items-center gap-2">
                                    <FileText size={16} className="text-gray-500" /> Past Invoices
                                </h3>
                            </div>
                            <div className="divide-y divide-gray-50 flex-1 overflow-y-auto">
                                {['Jan 2026', 'Dec 2025', 'Nov 2025', 'Oct 2025'].map((month, i) => (
                                    <div key={month} className="px-4 py-3 flex items-center justify-between hover:bg-gray-50 transition-colors cursor-pointer group">
                                        <div>
                                            <div className="text-sm font-semibold text-gray-700">{month}</div>
                                            <div className="text-xs text-gray-400">Paid • Visa ending in 4242</div>
                                        </div>
                                        <div className="flex items-center gap-3">
                                            <div className="text-sm font-bold text-gray-900">${(2000 - i * 150).toLocaleString()}</div>
                                            <button className="text-gray-300 group-hover:text-orange-500 transition-colors"><DownloadCloud size={16} /></button>
                                        </div>
                                    </div>
                                ))}
                            </div>
                        </div>

                    </div>
                </>
            )}
        </div>
    );
}
