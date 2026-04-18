import React, { useEffect, useState } from 'react';
import { api } from '../lib/api.js';
import { useStore } from '../store/index.js';
import AudioPlayer from '../components/AudioPlayer.jsx';
import {
    FileAudio, Search, Filter, Calendar, ChevronRight, ChevronDown,
    User, Clock, MessageSquare, ShieldCheck, Download, AlertCircle, Phone, CheckCircle2
} from 'lucide-react';

export default function Analytics() {
    const [calls, setCalls] = useState([]);
    const [loading, setLoading] = useState(true);
    const [total, setTotal] = useState(0);
    const [expandedRow, setExpandedRow] = useState(null);

    // Filters
    const [filters, setFilters] = useState({ sentiment: '', minDuration: '', disposition: '' });
    const [dispositions, setDispositions] = useState([]);

    useEffect(() => {
        // Load dispositions for the filter dropdown
        api.get('/qa/dispositions').then(setDispositions).catch(() => { });
        loadCalls();
    }, []);

    useEffect(() => { loadCalls(); }, [filters]); // Reload when filters change

    async function loadCalls() {
        setLoading(true);
        try {
            const qs = new URLSearchParams();
            if (filters.sentiment) qs.append('sentiment', filters.sentiment);
            if (filters.minDuration) qs.append('minDuration', filters.minDuration);
            if (filters.disposition) qs.append('disposition', filters.disposition);

            const res = await api.get(`/analytics?${qs.toString()}`);
            setCalls(res.calls || []);
            setTotal(res.total || 0);
        } catch (err) {
            console.error(err);
        } finally {
            setLoading(false);
        }
    }

    const formatLen = (secs) => {
        if (!secs) return '0s';
        const m = Math.floor(secs / 60);
        const s = secs % 60;
        return m > 0 ? `${m}m ${s}s` : `${s}s`;
    };

    const sentimentColor = (s) => {
        switch (s?.toLowerCase()) {
            case 'positive': return 'text-emerald-600 bg-emerald-50 border-emerald-200';
            case 'negative': return 'text-red-600 bg-red-50 border-red-200';
            case 'angry': return 'text-rose-700 bg-rose-50 border-rose-200';
            default: return 'text-gray-600 bg-gray-50 border-gray-200';
        }
    };

    return (
        <div className="space-y-6">
            {/* Header */}
            <div className="flex items-end justify-between">
                <div>
                    <h1 className="text-2xl font-bold text-gray-900">Call Logs & Recordings</h1>
                    <p className="text-sm text-gray-400 mt-0.5">Filter, play, and audit past conversations ({total} total)</p>
                </div>
                <button className="btn-secondary text-sm py-2">
                    <Download size={14} className="mr-1.5" /> Export CSV
                </button>
            </div>

            {/* Advanced Filters */}
            <div className="card p-4 flex items-center gap-4 flex-wrap bg-white/50">
                <div className="flex items-center gap-2 text-sm font-medium text-gray-500 mr-2">
                    <Filter size={16} /> Filters:
                </div>

                <select
                    className="input-field max-w-[160px] py-1.5 text-sm"
                    value={filters.sentiment}
                    onChange={e => setFilters(f => ({ ...f, sentiment: e.target.value }))}
                >
                    <option value="">All Sentiments</option>
                    <option value="positive">Positive</option>
                    <option value="neutral">Neutral</option>
                    <option value="negative">Negative</option>
                    <option value="angry">Angry</option>
                </select>

                <select
                    className="input-field max-w-[160px] py-1.5 text-sm"
                    value={filters.minDuration}
                    onChange={e => setFilters(f => ({ ...f, minDuration: e.target.value }))}
                >
                    <option value="">Any Duration</option>
                    <option value="60">&gt; 1 minute</option>
                    <option value="300">&gt; 5 minutes</option>
                    <option value="600">&gt; 10 minutes</option>
                </select>

                <select
                    className="input-field max-w-[180px] py-1.5 text-sm"
                    value={filters.disposition}
                    onChange={e => setFilters(f => ({ ...f, disposition: e.target.value }))}
                >
                    <option value="">All Dispositions</option>
                    {dispositions.map(d => <option key={d.id} value={d.id}>{d.name}</option>)}
                </select>

                {Object.values(filters).some(Boolean) && (
                    <button
                        onClick={() => setFilters({ sentiment: '', minDuration: '', disposition: '' })}
                        className="text-xs text-orange-500 hover:underline font-medium ml-auto"
                    >
                        Clear Filters
                    </button>
                )}
            </div>

            {/* Table */}
            <div className="card overflow-hidden">
                <div className="overflow-x-auto">
                    <table className="w-full text-left border-collapse">
                        <thead>
                            <tr className="bg-gray-50/80 border-b border-gray-100/50">
                                <th className="p-4 text-xs font-semibold text-gray-500 w-10"></th>
                                <th className="p-4 text-xs font-semibold text-gray-500">Contact</th>
                                <th className="p-4 text-xs font-semibold text-gray-500">Agent / Campaign</th>
                                <th className="p-4 text-xs font-semibold text-gray-500">Duration</th>
                                <th className="p-4 text-xs font-semibold text-gray-500">Sentiment</th>
                                <th className="p-4 text-xs font-semibold text-gray-500">Disposition</th>
                                <th className="p-4 text-xs font-semibold text-gray-500">QA Score</th>
                                <th className="p-4 text-xs font-semibold text-gray-500">Date</th>
                            </tr>
                        </thead>
                        <tbody className="divide-y divide-gray-50">
                            {loading ? (
                                <tr><td colSpan="8" className="p-8 text-center text-sm text-gray-400">Loading calls...</td></tr>
                            ) : calls.length === 0 ? (
                                <tr><td colSpan="8" className="p-8 text-center text-sm text-gray-400">No calls found.</td></tr>
                            ) : calls.map(call => (
                                <React.Fragment key={call.id}>
                                    {/* Row */}
                                    <tr
                                        onClick={() => setExpandedRow(expandedRow === call.id ? null : call.id)}
                                        className={`hover:bg-orange-50/30 cursor-pointer transition-colors ${expandedRow === call.id ? 'bg-orange-50/50' : ''}`}
                                    >
                                        <td className="p-4 text-gray-400">
                                            {expandedRow === call.id ? <ChevronDown size={16} /> : <ChevronRight size={16} />}
                                        </td>
                                        <td className="p-4">
                                            <div className="font-semibold text-sm text-gray-900 font-mono">{call.phoneNumber}</div>
                                            <div className="text-xs text-gray-400 flex items-center gap-1 mt-0.5">
                                                {call.direction === 'inbound' ? '↓ Inbound' : '↑ Outbound'}
                                            </div>
                                        </td>
                                        <td className="p-4">
                                            <div className="text-sm font-medium text-gray-800">{call.agent?.name || 'Human Queue'}</div>
                                            <div className="text-xs text-gray-400">{call.campaign?.name || 'Direct Line'}</div>
                                        </td>
                                        <td className="p-4 text-sm text-gray-600">{formatLen(call.duration)}</td>
                                        <td className="p-4">
                                            <span className={`px-2 py-0.5 rounded border text-[10px] font-bold uppercase tracking-wider ${sentimentColor(call.sentiment)}`}>
                                                {call.sentiment || 'N/A'}
                                            </span>
                                        </td>
                                        <td className="p-4">
                                            {call.disposition ? (
                                                <span className="text-xs font-medium text-slate-700 bg-slate-100 px-2 py-1 rounded-md">
                                                    {call.disposition.name}
                                                </span>
                                            ) : <span className="text-xs text-gray-300">—</span>}
                                        </td>
                                        <td className="p-4">
                                            {call.QAScore ? (
                                                <div className="flex items-center gap-1.5">
                                                    <CheckCircle2 size={14} className={call.QAScore.score >= 80 ? 'text-emerald-500' : 'text-orange-500'} />
                                                    <span className="text-sm font-bold text-gray-700">{call.QAScore.score}%</span>
                                                </div>
                                            ) : <span className="text-xs text-gray-300">Unscored</span>}
                                        </td>
                                        <td className="p-4 text-sm text-gray-500">
                                            {new Date(call.startedAt).toLocaleDateString()} {new Date(call.startedAt).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })}
                                        </td>
                                    </tr>

                                    {/* Expanded Audit Panel */}
                                    {expandedRow === call.id && (
                                        <tr>
                                            <td colSpan="8" className="p-0 border-b border-orange-100">
                                                <div className="bg-orange-50/50 p-6 shadow-inner border-y border-orange-100/50 flex flex-col gap-6">

                                                    {/* Audio Player */}
                                                    <div className="w-full max-w-3xl">
                                                        {call.Recording ? (
                                                            <AudioPlayer src={call.Recording.url} waveformData={JSON.parse(call.Recording.waveform || '[]')} />
                                                        ) : (
                                                            <div className="bg-white border rounded-xl p-6 flex flex-col items-center justify-center text-gray-400 gap-2">
                                                                <AlertCircle size={24} className="text-gray-300" />
                                                                <span className="text-sm font-medium">Recording not available for this call</span>
                                                            </div>
                                                        )}
                                                    </div>

                                                    {/* Data columns */}
                                                    <div className="grid grid-cols-2 lg:grid-cols-3 gap-6">

                                                        {/* Transcript */}
                                                        <div className="bg-white rounded-xl shadow-sm border border-gray-100 p-4 col-span-2">
                                                            <h4 className="text-xs font-bold text-gray-400 uppercase tracking-wider mb-3 flex items-center gap-1.5">
                                                                <MessageSquare size={14} /> Transcript
                                                            </h4>
                                                            <div className="h-48 overflow-y-auto text-sm text-gray-700 leading-relaxed pr-2 space-y-3">
                                                                {call.transcriptMessages && call.transcriptMessages.length > 0 ? (
                                                                    call.transcriptMessages.map((msg, i) => {
                                                                        const isBot = msg.role === 'ai' || msg.role === 'model' || msg.role === 'bot' || msg.role === 'agent';
                                                                        return (
                                                                            <div key={i} className={`p-2 rounded-lg ${isBot ? 'bg-orange-50/50 border border-orange-100/30' : 'bg-gray-50'}`}>
                                                                                <span className={`font-semibold text-xs mr-2 ${isBot ? 'text-orange-600' : 'text-blue-600'}`}>
                                                                                    {isBot ? 'AI' : 'Customer'}
                                                                                </span>
                                                                                {msg.text}
                                                                            </div>
                                                                        );
                                                                    })
                                                                ) : call.transcript ? (
                                                                    call.transcript.split('\n').filter(l => l.trim()).map((line, i) => {
                                                                        const isBot = line.startsWith('AI:') || line.startsWith('Agent:') || line.startsWith('Bot:');
                                                                        return (
                                                                            <div key={i} className={`p-2 rounded-lg ${isBot ? 'bg-orange-50/50 border border-orange-100/30' : 'bg-gray-50'}`}>
                                                                                <span className={`font-semibold text-xs mr-2 ${isBot ? 'text-orange-600' : 'text-blue-600'}`}>
                                                                                    {isBot ? 'AI' : 'Customer'}
                                                                                </span>
                                                                                {line.replace(/^(AI|Agent|Bot|Customer|User):\s*/, '')}
                                                                            </div>
                                                                        );
                                                                    })
                                                                ) : <span className="text-gray-400 italic">No transcript available.</span>}
                                                            </div>
                                                        </div>

                                                        {/* Summary & QA */}
                                                        <div className="space-y-4">
                                                            <div className="bg-white rounded-xl shadow-sm border border-gray-100 p-4">
                                                                <h4 className="text-xs font-bold text-gray-400 uppercase tracking-wider mb-2">AI Summary</h4>
                                                                <p className="text-sm text-gray-700 leading-relaxed">
                                                                    {call.summary || <span className="text-gray-400 italic">No summary generated.</span>}
                                                                </p>

                                                                {/* Custom Extracted Data Section */}
                                                                {call.structuredData && (
                                                                    <div className="mt-4 pt-4 border-t border-gray-50">
                                                                        <h4 className="text-xs font-bold text-orange-500 uppercase tracking-wider mb-3 flex items-center gap-1.5">
                                                                            <ShieldCheck size={14} /> Extracted Data
                                                                        </h4>
                                                                        <div className="grid grid-cols-2 gap-3">
                                                                            {Object.entries(
                                                                                (typeof call.structuredData === 'string' && call.structuredData.startsWith('{')) 
                                                                                    ? JSON.parse(call.structuredData) 
                                                                                    : (typeof call.structuredData === 'object' ? call.structuredData : {})
                                                                            ).map(([key, val]) => (
                                                                                <div key={key} className="bg-orange-50/50 rounded-lg p-2.5 border border-orange-100/50">
                                                                                    <div className="text-[10px] text-orange-600/80 font-bold uppercase mb-0.5 tracking-wider">{key.replace(/_/g, ' ')}</div>
                                                                                    <div className="text-sm font-bold text-gray-900 truncate" title={String(val)}>{String(val)}</div>
                                                                                </div>
                                                                            ))}
                                                                        </div>
                                                                    </div>
                                                                )}
                                                            </div>

                                                            <div className="bg-white rounded-xl shadow-sm border border-emerald-100 bg-emerald-50/30 p-4">
                                                                <h4 className="text-xs font-bold text-emerald-600 uppercase tracking-wider mb-2 flex items-center gap-1.5">
                                                                    <ShieldCheck size={14} /> QA Audit Score
                                                                </h4>
                                                                {call.QAScore ? (
                                                                    <div>
                                                                        <div className="text-3xl font-black text-emerald-700 mb-1">{call.QAScore.score}<span className="text-lg text-emerald-500">%</span></div>
                                                                        <p className="text-xs text-emerald-800 leading-relaxed">{call.QAScore.feedback}</p>
                                                                    </div>
                                                                ) : (
                                                                    <button className="w-full py-2 bg-white border border-emerald-200 text-emerald-600 rounded-lg text-sm font-semibold hover:bg-emerald-50 transition-colors">
                                                                        Grade Call (Manual QA)
                                                                    </button>
                                                                )}
                                                            </div>
                                                        </div>

                                                    </div>
                                                </div>
                                            </td>
                                        </tr>
                                    )}
                                </React.Fragment>
                            ))}
                        </tbody>
                    </table>

                    <div className="p-4 border-t border-gray-100 flex items-center justify-between text-sm text-gray-500 bg-gray-50/50">
                        <span>Showing {calls.length} of {total} calls</span>
                        <div className="flex gap-2">
                            <button disabled className="px-3 py-1 border rounded bg-white disabled:opacity-50">Prev</button>
                            <button disabled className="px-3 py-1 border rounded bg-white disabled:opacity-50">Next</button>
                        </div>
                    </div>
                </div>
            </div>
        </div>
    );
}
