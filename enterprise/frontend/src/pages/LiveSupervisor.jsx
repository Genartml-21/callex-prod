import { useEffect, useState, useRef } from 'react';
import { api } from '../lib/api.js';
import { useStore } from '../store/index.js';
import { Phone, Headphones, Mic, MessageSquare, AlertTriangle, CheckCircle, X, Send, Eye, Trash2 } from 'lucide-react';

const WS_URL = `ws://${window.location.host.replace('3000', '4000')}`;
const sentimentBadge = (s) => ({
    positive: 'badge-green', neutral: 'badge-gray', negative: 'badge-blue', angry: 'badge-red'
}[s] || 'badge-gray');

export default function LiveSupervisor() {
    const [calls, setCalls] = useState([]);
    const [selectedCall, setSelectedCall] = useState(null);
    const [transcript, setTranscript] = useState([]);
    const [whisper, setWhisper] = useState('');
    const [loading, setLoading] = useState(false);
    const wsRef = useRef(null);
    const { showToast } = useStore();

    const fetchCalls = () => api.activeCalls().then(setCalls).catch(() => { });

    useEffect(() => {
        fetchCalls();
        const interval = setInterval(fetchCalls, 5000);
        return () => clearInterval(interval);
    }, []);

    function openCall(call) {
        setSelectedCall(call);
        setTranscript([]);
        if (wsRef.current) wsRef.current.close();
        const ws = new WebSocket(`${WS_URL}?type=supervisor&callId=${call.id}`);
        wsRef.current = ws;
        ws.onmessage = (e) => {
            const msg = JSON.parse(e.data);
            if (msg.type === 'transcript_line') setTranscript(t => [...t, msg.line]);
            if (msg.type === 'whisper_ack') showToast('Whisper injected into AI context', 'success');
        };
    }

    async function sendWhisper() {
        if (!whisper.trim() || !selectedCall) return;
        await api.whisper(selectedCall.id, whisper);
        showToast('Whisper sent!', 'success');
        setWhisper('');
    }

    async function handleBarge(callId) {
        await api.barge(callId);
        showToast('Barged into call — You are now in control', 'success');
        fetchCalls();
    }

    async function startTestCall() {
        setLoading(true);
        const agents = await api.agents();
        await api.simulateCall({ agentId: agents[0]?.id });
        showToast('Test call started', 'info');
        setLoading(false);
        fetchCalls();
    }

    async function endCall(id) {
        await api.endCall(id);
        showToast('Call ended', 'info');
        if (selectedCall?.id === id) { setSelectedCall(null); setTranscript([]); wsRef.current?.close(); }
        fetchCalls();
    }

    return (
        <div className="space-y-6">
            <div className="page-header">
                <div>
                    <h1 className="text-2xl font-bold text-gray-900">Live Supervisor</h1>
                    <p className="text-sm text-gray-400">Monitor, barge, and whisper to live AI agents</p>
                </div>
                <button className="btn-primary" onClick={startTestCall} disabled={loading}>
                    <Phone size={15} /> {loading ? 'Starting…' : 'Test Call'}
                </button>
            </div>

            <div className="flex gap-6">
                {/* Call table */}
                <div className="flex-1 card overflow-hidden p-0">
                    <div className="px-6 py-4 border-b border-gray-100">
                        <h2 className="section-title">Active Calls ({calls.length})</h2>
                    </div>
                    <table className="w-full">
                        <thead className="bg-gray-50 border-b border-gray-100">
                            <tr>
                                <th className="table-header">Phone</th>
                                <th className="table-header">Agent</th>
                                <th className="table-header">Sentiment</th>
                                <th className="table-header">Duration</th>
                                <th className="table-header">Actions</th>
                            </tr>
                        </thead>
                        <tbody>
                            {calls.length === 0 && (
                                <tr><td colSpan={5} className="text-center py-12 text-gray-400 text-sm">No active calls. Click "Test Call" to start one.</td></tr>
                            )}
                            {calls.map(call => (
                                <tr key={call.id} className={`hover:bg-gray-50 transition-colors cursor-pointer ${selectedCall?.id === call.id ? 'bg-orange-50' : ''} ${call.sentiment === 'angry' ? 'border-l-4 border-red-500' : ''}`}>
                                    <td className="table-cell font-mono font-medium" onClick={() => openCall(call)}>{call.phoneNumber}</td>
                                    <td className="table-cell text-gray-500" onClick={() => openCall(call)}>{call.agent?.name || 'Default Agent'}</td>
                                    <td className="table-cell" onClick={() => openCall(call)}>
                                        <span className={sentimentBadge(call.sentiment)}>
                                            {call.sentiment === 'angry' && <AlertTriangle size={11} className="mr-1" />}
                                            {call.sentiment || 'neutral'}
                                        </span>
                                    </td>
                                    <td className="table-cell text-gray-400">
                                        {Math.floor((Date.now() - new Date(call.startedAt)) / 1000)}s
                                    </td>
                                    <td className="table-cell">
                                        <div className="flex gap-1">
                                            <button className="p-1.5 hover:bg-blue-50 rounded-lg transition-colors text-blue-500" onClick={() => openCall(call)} title="Listen In"><Eye size={14} /></button>
                                            <button className="p-1.5 hover:bg-orange-50 rounded-lg transition-colors text-orange-500" onClick={() => handleBarge(call.id)} title="Barge"><Headphones size={14} /></button>
                                            <button className="p-1.5 hover:bg-red-50 rounded-lg transition-colors text-red-500" onClick={() => endCall(call.id)} title="End"><X size={14} /></button>
                                        </div>
                                    </td>
                                </tr>
                            ))}
                        </tbody>
                    </table>
                </div>

                {/* Transcript panel */}
                {selectedCall && (
                    <div className="w-96 card flex flex-col gap-4">
                        <div className="flex items-center justify-between">
                            <div>
                                <div className="font-semibold text-gray-800 text-sm">{selectedCall.phoneNumber}</div>
                                <span className={`${sentimentBadge(selectedCall.sentiment)} mt-1`}>{selectedCall.sentiment}</span>
                            </div>
                            <button onClick={() => { setSelectedCall(null); wsRef.current?.close(); }} className="p-1.5 hover:bg-gray-100 rounded-lg"><X size={14} /></button>
                        </div>

                        <div className="flex-1 bg-gray-50 rounded-xl p-3 text-sm space-y-2 max-h-80 overflow-y-auto">
                            {transcript.length === 0 && <p className="text-gray-400 text-xs text-center pt-4">Connecting to live stream...</p>}
                            {transcript.map((line, i) => {
                                const isBot = line.startsWith('Bot:') || line.startsWith('[SYSTEM');
                                return (
                                    <div key={i} className={`p-2 rounded-lg text-xs ${isBot ? 'bg-orange-100 text-orange-800 ml-4' : 'bg-white text-gray-700 mr-4 border border-gray-100'}`}>
                                        {line}
                                    </div>
                                );
                            })}
                        </div>

                        {/* Whisper box */}
                        <div>
                            <label className="label flex items-center gap-1.5"><Mic size={11} /> Whisper to AI (hidden from customer)</label>
                            <div className="flex gap-2">
                                <input className="input-field flex-1" placeholder="e.g. Offer a 10% discount now..." value={whisper} onChange={e => setWhisper(e.target.value)} onKeyDown={e => e.key === 'Enter' && sendWhisper()} />
                                <button className="btn-primary" onClick={sendWhisper}><Send size={14} /></button>
                            </div>
                        </div>
                    </div>
                )}
            </div>
        </div>
    );
}
