import { useEffect, useState, useRef } from 'react';
import { api } from '../lib/api.js';
import { useStore } from '../store/index.js';
import { Upload, Trash2, RefreshCw, FileText, Link, Globe, CheckCircle, Clock } from 'lucide-react';

export default function KnowledgeBase() {
    const [docs, setDocs] = useState([]);
    const [topK, setTopK] = useState(5);
    const [simThresh, setSimThresh] = useState(0.75);
    const [urlInput, setUrlInput] = useState('');
    const fileRef = useRef(null);
    const { showToast } = useStore();

    const fetchDocs = () => api.docs().then(setDocs).catch(() => { });
    useEffect(() => { fetchDocs(); const t = setInterval(fetchDocs, 4000); return () => clearInterval(t); }, []);

    async function uploadFile(e) {
        const file = e.target.files[0]; if (!file) return;
        const fd = new FormData(); fd.append('file', file); fd.append('name', file.name); fd.append('type', 'pdf');
        await api.uploadDoc(fd);
        showToast('Document uploaded — processing...', 'info'); fetchDocs();
    }

    async function addUrl() {
        if (!urlInput.trim()) return;
        await api.uploadDoc(Object.assign(new FormData(), { append() { } }) || (() => { const fd = new FormData(); fd.append('type', 'web'); fd.append('name', urlInput); fd.append('sourceUrl', urlInput); return fd; })());
        // Use direct fetch for URL-only
        const fd = new FormData(); fd.append('type', 'web'); fd.append('name', urlInput); fd.append('sourceUrl', urlInput);
        await fetch('/api/knowledge', { method: 'POST', body: fd });
        showToast('Web source added', 'info'); setUrlInput(''); fetchDocs();
    }

    async function resync(id) { await api.resyncDoc(id); showToast('Resyncing...', 'info'); fetchDocs(); }
    async function del(id) { await api.deleteDoc(id); showToast('Document deleted', 'info'); fetchDocs(); }

    const typeIcon = { pdf: <FileText size={15} />, api: <Link size={15} />, web: <Globe size={15} /> };

    return (
        <div className="space-y-6">
            <div className="page-header">
                <div><h1 className="text-2xl font-bold text-gray-900">Vector Knowledge Base</h1><p className="text-sm text-gray-400">RAG-powered document repository for AI agents</p></div>
                <div className="flex gap-2">
                    <input ref={fileRef} type="file" accept=".pdf,.txt,.docx" className="hidden" onChange={uploadFile} />
                    <button className="btn-primary" onClick={() => fileRef.current?.click()}><Upload size={15} /> Upload PDF</button>
                </div>
            </div>

            <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
                {/* Document list */}
                <div className="lg:col-span-2 card p-0 overflow-hidden">
                    <div className="px-6 py-4 border-b border-gray-100 flex items-center justify-between">
                        <h2 className="section-title">Documents ({docs.length})</h2>
                    </div>
                    {/* URL add */}
                    <div className="px-6 py-3 border-b border-gray-50 flex gap-2">
                        <input className="input-field flex-1" placeholder="Add web URL or API endpoint..." value={urlInput} onChange={e => setUrlInput(e.target.value)} onKeyDown={e => e.key === 'Enter' && addUrl()} />
                        <button className="btn-secondary" onClick={addUrl}><Globe size={14} /> Add URL</button>
                    </div>
                    <div className="divide-y divide-gray-50">
                        {docs.length === 0 && <div className="text-center py-12 text-gray-400 text-sm">No documents yet. Upload a PDF or add a URL.</div>}
                        {docs.map(doc => (
                            <div key={doc.id} className="flex items-center gap-4 px-6 py-4 hover:bg-gray-50 transition-colors">
                                <div className={`p-2 rounded-lg ${doc.type === 'pdf' ? 'bg-red-50 text-red-500' : doc.type === 'web' ? 'bg-blue-50 text-blue-500' : 'bg-purple-50 text-purple-500'}`}>
                                    {typeIcon[doc.type] || <FileText size={15} />}
                                </div>
                                <div className="flex-1 min-w-0">
                                    <div className="font-medium text-gray-800 text-sm truncate">{doc.name}</div>
                                    <div className="text-xs text-gray-400 mt-0.5">{doc.chunkCount} chunks · {doc.type.toUpperCase()}</div>
                                </div>
                                <div>
                                    {doc.status === 'synced' ? <span className="badge-green"><CheckCircle size={11} className="mr-1" /> Synced</span>
                                        : <span className="badge-orange"><Clock size={11} className="mr-1 animate-spin" /> Processing</span>}
                                </div>
                                <div className="flex gap-1 shrink-0">
                                    <button className="p-1.5 hover:bg-blue-50 rounded-lg text-blue-400" onClick={() => resync(doc.id)} title="Resync"><RefreshCw size={13} /></button>
                                    <button className="p-1.5 hover:bg-red-50 rounded-lg text-red-400" onClick={() => del(doc.id)} title="Delete"><Trash2 size={13} /></button>
                                </div>
                            </div>
                        ))}
                    </div>
                </div>

                {/* RAG Settings */}
                <div className="card space-y-6">
                    <h2 className="section-title">Global RAG Settings</h2>
                    <div>
                        <label className="label">Top-K Injection</label>
                        <p className="text-xs text-gray-400 mb-2">Number of chunks injected per query</p>
                        <div className="flex justify-between text-sm font-semibold text-gray-700 mb-1"><span>K = {topK}</span></div>
                        <input type="range" min="1" max="20" value={topK} onChange={e => setTopK(+e.target.value)} className="w-full accent-orange-500" />
                        <div className="flex justify-between text-xs text-gray-300 mt-1"><span>1</span><span>20</span></div>
                    </div>
                    <div>
                        <label className="label">Cosine Similarity Threshold</label>
                        <p className="text-xs text-gray-400 mb-2">Minimum relevance score to inject a chunk</p>
                        <div className="flex justify-between text-sm font-semibold text-gray-700 mb-1"><span>{simThresh.toFixed(2)}</span></div>
                        <input type="range" min="0.5" max="1" step="0.01" value={simThresh} onChange={e => setSimThresh(+e.target.value)} className="w-full accent-orange-500" />
                        <div className="flex justify-between text-xs text-gray-300 mt-1"><span>0.50</span><span>1.00</span></div>
                    </div>
                    <div className="p-3 bg-orange-50 rounded-xl border border-orange-100 text-xs text-orange-700">
                        <strong>Vector Model:</strong> Callex-1.3 Embeddings<br />
                        <strong>Vector DB:</strong> Pinecone (us-east-1)<br />
                        <strong>Index:</strong> callex-main-prod
                    </div>
                    <button className="btn-primary w-full" onClick={() => showToast('RAG settings saved', 'success')}>Save Settings</button>
                </div>
            </div>
        </div>
    );
}
