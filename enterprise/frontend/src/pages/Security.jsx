import { useEffect, useState } from 'react';
import { api } from '../lib/api.js';
import { useStore } from '../store/index.js';
import { Plus, Trash2, ShieldCheck, Lock, Fingerprint } from 'lucide-react';

export default function Security() {
    const [sigs, setSigs] = useState([]);
    const [phrase, setPhrase] = useState('');
    const [desc, setDesc] = useState('');
    const [pciEnabled, setPciEnabled] = useState(false);
    const [hashResult, setHashResult] = useState(null);
    const [hashInput, setHashInput] = useState('');
    const { showToast } = useStore();

    const fetchSigs = () => api.voiceSignatures().then(setSigs).catch(() => { });
    useEffect(() => { fetchSigs(); }, []);

    async function addSig(e) {
        e.preventDefault();
        const sig = await api.createVoiceSig({ phrase, description: desc });
        setSigs(s => [sig, ...s]); setPhrase(''); setDesc('');
        showToast(`Voice signature "${phrase}" registered`, 'success');
    }

    async function delSig(id) {
        await api.deleteVoiceSig(id);
        setSigs(s => s.filter(x => x.id !== id));
        showToast('Signature removed', 'info');
    }

    async function hashAudio() {
        const res = await fetch('/api/security/hash-audio', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ audioChunk: hashInput, callId: 'demo' }) });
        const data = await res.json();
        setHashResult(data); showToast('SHA-256 hash generated', 'success');
    }

    const Toggle = ({ label, hint, val, onChange }) => (
        <div className="flex items-center justify-between p-4 bg-gray-50 rounded-xl">
            <div><div className="text-sm font-semibold text-gray-800">{label}</div><div className="text-xs text-gray-400 mt-0.5">{hint}</div></div>
            <button onClick={() => onChange(!val)} className={`relative inline-flex h-6 w-11 rounded-full transition-colors ${val ? 'bg-orange-500' : 'bg-gray-200'}`}>
                <span className={`inline-block h-4 w-4 mt-1 ml-1 rounded-full bg-white shadow transition-transform ${val ? 'translate-x-5' : ''}`}></span>
            </button>
        </div>
    );

    return (
        <div className="space-y-6">
            <div className="page-header">
                <div><h1 className="text-2xl font-bold text-gray-900">Security, Compliance & Legal</h1><p className="text-sm text-gray-400">Voice signatures, PCI-DSS, and SHA-256 audit trail</p></div>
            </div>

            <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
                {/* Voice Signatures */}
                <div className="card space-y-4">
                    <h2 className="section-title flex items-center gap-2"><Fingerprint size={16} className="text-orange-500" />Acoustic Voice Signatures</h2>
                    <p className="text-sm text-gray-400">Register trigger phrases for legal consent verification (e.g., "I agree"). Each is SHA-256 hashed for audit proof.</p>
                    <form onSubmit={addSig} className="space-y-3">
                        <div><label className="label">Trigger Phrase</label><input required className="input-field" placeholder='e.g. "I agree to the terms"' value={phrase} onChange={e => setPhrase(e.target.value)} /></div>
                        <div><label className="label">Description</label><input className="input-field" placeholder="Purpose of this signature..." value={desc} onChange={e => setDesc(e.target.value)} /></div>
                        <button type="submit" className="btn-primary"><Plus size={14} />Register Signature</button>
                    </form>
                    <div className="space-y-2 max-h-56 overflow-y-auto">
                        {sigs.length === 0 && <p className="text-sm text-gray-400 text-center py-4">No signatures registered yet.</p>}
                        {sigs.map(s => (
                            <div key={s.id} className="flex items-start gap-3 p-3 bg-gray-50 rounded-xl">
                                <ShieldCheck size={15} className="text-orange-500 mt-0.5 shrink-0" />
                                <div className="flex-1 min-w-0">
                                    <div className="font-medium text-gray-800 text-sm">"{s.phrase}"</div>
                                    {s.description && <div className="text-xs text-gray-400">{s.description}</div>}
                                    <div className="text-xs font-mono text-gray-300 mt-1 truncate">{s.hashExample}</div>
                                </div>
                                <button className="p-1 hover:bg-red-50 rounded-lg text-red-400 shrink-0" onClick={() => delSig(s.id)}><Trash2 size={12} /></button>
                            </div>
                        ))}
                    </div>
                </div>

                {/* PCI & Hashing */}
                <div className="card space-y-4">
                    <h2 className="section-title flex items-center gap-2"><Lock size={16} className="text-orange-500" />PCI-DSS & Compliance</h2>
                    <div className="space-y-3">
                        <Toggle
                            label="Dynamic Recording Pause"
                            hint="Automatically pause audio + text recording when credit card numbers are detected"
                            val={pciEnabled} onChange={v => setPciEnabled(v)}
                        />
                        <Toggle label="Redact PII in Transcripts" hint="Mask phone numbers, emails, and Aadhaar/PAN in stored transcripts" val={true} onChange={() => showToast('Setting saved', 'success')} />
                        <Toggle label="GDPR Consent Enforcement" hint="Require verbal consent before recording begins" val={false} onChange={() => showToast('Setting saved', 'success')} />
                    </div>

                    <div className="pt-4 border-t border-gray-100">
                        <h3 className="font-semibold text-gray-800 text-sm mb-3">SHA-256 Audio Chunk Hash Tool</h3>
                        <p className="text-xs text-gray-400 mb-2">Hash an audio segment for legal contract verification</p>
                        <textarea className="input-field h-16 text-xs font-mono" placeholder="Paste audio chunk data or text transcript segment..." value={hashInput} onChange={e => setHashInput(e.target.value)} />
                        <button className="btn-primary mt-2" onClick={hashAudio}><ShieldCheck size={14} />Generate SHA-256 Hash</button>
                        {hashResult && (
                            <div className="mt-3 p-3 bg-gray-900 rounded-xl">
                                <div className="text-xs text-gray-400 mb-1">SHA-256 Hash:</div>
                                <div className="text-xs font-mono text-green-400 break-all">{hashResult.hash}</div>
                                <div className="text-xs text-gray-500 mt-1">{hashResult.ts}</div>
                            </div>
                        )}
                    </div>
                </div>
            </div>
        </div>
    );
}
