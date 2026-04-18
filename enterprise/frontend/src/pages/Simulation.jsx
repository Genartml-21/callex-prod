import { useState } from 'react';
import { api } from '../lib/api.js';
import { useStore } from '../store/index.js';
import { Play, Zap, Bot, CheckCircle, X, AlertTriangle } from 'lucide-react';

export default function Simulation() {
    const [advResult, setAdvResult] = useState(null);
    const [batchResult, setBatchResult] = useState(null);
    const [loading, setLoading] = useState({ adv: false, batch: false, softphone: false });
    const [botCount, setBotCount] = useState(50);
    const [sfStatus, setSfStatus] = useState('idle');
    const [sfLog, setSfLog] = useState([]);
    const { showToast } = useStore();

    async function runAdversarial() {
        setLoading(l => ({ ...l, adv: true })); setAdvResult(null);
        const res = await api.runAdversarial({ botCount });
        setAdvResult(res); setLoading(l => ({ ...l, adv: false }));
        showToast(`Adversarial test done: ${res.passRate}% pass`, res.passRate >= 80 ? 'success' : 'error');
    }

    async function runBatch() {
        setLoading(l => ({ ...l, batch: true })); setBatchResult(null);
        const scenarios = [{ input: 'I want to cancel my plan', expected: 'retention' }, { input: 'My recharge failed', expected: 'support' }];
        const res = await api.runBatch({ scenarios });
        setBatchResult(res); setLoading(l => ({ ...l, batch: false }));
        showToast('Batch evaluation queued', 'info');
    }

    function startSoftphone() {
        setSfStatus('connecting'); setSfLog([]);
        const ws = new WebSocket(`ws://${window.location.host.replace('3000', '4000')}?type=softphone`);
        ws.onopen = () => { setSfStatus('connected'); setSfLog(l => [...l, '[System] Connected to AI agent...']); };
        ws.onmessage = e => { try { const m = JSON.parse(e.data); setSfLog(l => [...l, `[AI] ${m.text}`]); } catch { } };
        ws.onclose = () => setSfStatus('idle');
        ws.onerror = () => { setSfStatus('error'); setSfLog(l => [...l, '[Error] Connection failed']); };
        setTimeout(() => {
            if (ws.readyState === 1) ws.send(JSON.stringify({ type: 'audio', text: 'Test speech input' }));
        }, 1000);
    }

    return (
        <div className="space-y-6">
            <div className="page-header">
                <div><h1 className="text-2xl font-bold text-gray-900">Simulation & QA</h1><p className="text-sm text-gray-400">Test AI agents with softphone, batch scenarios, and adversarial attacks</p></div>
            </div>

            <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
                {/* Browser Softphone */}
                <div className="card">
                    <h2 className="section-title mb-4 flex items-center gap-2"><Bot size={16} className="text-orange-500" /> Browser Softphone</h2>
                    <div className="bg-gray-900 rounded-xl p-4 font-mono text-xs text-green-400 h-48 overflow-y-auto mb-4">
                        {sfLog.length === 0 ? <span className="text-gray-500">Click "Start" to connect to the AI agent...</span> : sfLog.map((l, i) => <div key={i}>{l}</div>)}
                    </div>
                    <div className="flex items-center gap-3">
                        <div className={`w-3 h-3 rounded-full ${sfStatus === 'connected' ? 'bg-green-500 animate-pulse' : sfStatus === 'connecting' ? 'bg-yellow-500' : 'bg-red-400'}`}></div>
                        <span className="text-sm text-gray-500 capitalize">{sfStatus}</span>
                        <button className="btn-primary ml-auto" onClick={startSoftphone} disabled={sfStatus === 'connected'}><Play size={14} /> Start</button>
                    </div>
                </div>

                {/* Adversarial Test */}
                <div className="card">
                    <h2 className="section-title mb-4 flex items-center gap-2"><Zap size={16} className="text-orange-500" /> Adversarial AI Attack</h2>
                    <p className="text-sm text-gray-500 mb-4">Spawn multiple bots that simultaneously attack the agent with edge cases, jailbreaks, and off-topic queries.</p>
                    <div className="mb-4">
                        <label className="label">Bot Count: {botCount}</label>
                        <input type="range" min="5" max="100" step="5" value={botCount} onChange={e => setBotCount(+e.target.value)} className="w-full accent-orange-500" />
                        <div className="flex justify-between text-xs text-gray-300 mt-1"><span>5</span><span>100</span></div>
                    </div>
                    <button className="btn-danger w-full mb-4" onClick={runAdversarial} disabled={loading.adv}>
                        <Zap size={14} /> {loading.adv ? `Attacking with ${botCount} bots...` : `Launch ${botCount} Bot Attack`}
                    </button>
                    {advResult && (
                        <div className="space-y-2">
                            <div className={`p-3 rounded-xl font-semibold text-sm text-center ${advResult.passRate >= 80 ? 'bg-green-50 text-green-700' : 'bg-red-50 text-red-700'}`}>
                                {advResult.passRate >= 80 ? '✓' : '⚠'} {advResult.passRate}% Pass Rate ({advResult.results.filter(r => r.passed).length}/{advResult.botCount} bots)
                            </div>
                            <div className="grid grid-cols-3 gap-2">
                                {advResult.results.slice(0, 6).map(r => (
                                    <div key={r.botId} className={`p-2 rounded-lg text-xs text-center ${r.passed ? 'bg-green-50 text-green-700' : 'bg-red-50 text-red-700'}`}>
                                        Bot #{r.botId} · {r.latencyMs}ms
                                        {r.issue && <div className="text-red-500">{r.issue}</div>}
                                    </div>
                                ))}
                            </div>
                        </div>
                    )}
                </div>

                {/* Batch Eval */}
                <div className="card lg:col-span-2">
                    <h2 className="section-title mb-4">Batch Scenario Evaluation</h2>
                    <p className="text-sm text-gray-500 mb-4">Run predefined test scenarios and score with an LLM-as-judge. Upload a CSV or use default scenarios.</p>
                    <div className="flex gap-3">
                        <button className="btn-primary" onClick={runBatch} disabled={loading.batch}>
                            <Play size={14} /> {loading.batch ? 'Running...' : 'Run Batch Evaluation'}
                        </button>
                        <button className="btn-secondary"><Play size={14} /> Upload CSV</button>
                    </div>
                    {batchResult && (
                        <div className="mt-4 p-4 bg-blue-50 rounded-xl border border-blue-100">
                            <div className="font-semibold text-blue-700 text-sm">Job {batchResult.jobId} queued</div>
                            <div className="text-xs text-blue-500 mt-1">{batchResult.total} scenarios · Status: {batchResult.status}</div>
                        </div>
                    )}
                </div>
            </div>
        </div>
    );
}
