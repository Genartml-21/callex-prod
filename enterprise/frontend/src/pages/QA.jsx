import { useState, useEffect } from 'react';
import { api } from '../lib/api.js';
import { useStore } from '../store/index.js';
import { Plus, Trash2, ShieldCheck, Tag, Loader2, PlayCircle } from 'lucide-react';

export default function QA() {
    const [dispositions, setDispositions] = useState([]);
    const [newDispName, setNewDispName] = useState('');
    const [newDispCat, setNewDispCat] = useState('neutral');
    const [loading, setLoading] = useState(true);
    const { showToast } = useStore();

    useEffect(() => {
        loadDispositions();
    }, []);

    async function loadDispositions() {
        setLoading(true);
        try {
            const data = await api.get('/qa/dispositions');
            setDispositions(data);
        } catch (err) {
            showToast('error', 'Failed to load dispositions');
        } finally {
            setLoading(false);
        }
    }

    async function handleAddDisposition(e) {
        e.preventDefault();
        if (!newDispName.trim()) return;
        try {
            await api.post('/qa/dispositions', { name: newDispName, category: newDispCat, requiresNote: false });
            setNewDispName('');
            loadDispositions();
            showToast('success', 'Disposition added');
        } catch (err) {
            showToast('error', 'Failed to add disposition');
        }
    }

    const catColor = (cat) => {
        if (cat === 'positive') return 'bg-emerald-100 text-emerald-700';
        if (cat === 'negative') return 'bg-red-100 text-red-700';
        return 'bg-gray-100 text-gray-700';
    };

    return (
        <div className="space-y-8">
            {/* Header */}
            <div className="page-header">
                <div>
                    <h1 className="text-2xl font-bold text-gray-900">QA & Dispositions</h1>
                    <p className="text-sm text-gray-400 mt-0.5">Manage custom call outcomes and manual scorecards</p>
                </div>
            </div>

            <div className="grid grid-cols-1 lg:grid-cols-2 gap-8">

                {/* Left Col: Dispositions */}
                <div className="space-y-4">
                    <div className="flex items-center gap-2 mb-2">
                        <Tag size={18} className="text-orange-500" />
                        <h2 className="text-lg font-bold text-gray-800">Call Dispositions</h2>
                    </div>
                    <p className="text-sm text-gray-500 mb-4">
                        Custom tags that agents (or AI) can assign to summarize the final outcome of a call.
                    </p>

                    <form onSubmit={handleAddDisposition} className="card p-4 flex gap-3 bg-gray-50/50">
                        <input
                            type="text"
                            placeholder="e.g. Sale Closed, DNC, Not Interested"
                            className="input-field flex-1"
                            value={newDispName}
                            onChange={e => setNewDispName(e.target.value)}
                        />
                        <select
                            className="input-field w-32"
                            value={newDispCat}
                            onChange={e => setNewDispCat(e.target.value)}
                        >
                            <option value="neutral">Neutral</option>
                            <option value="positive">Positive</option>
                            <option value="negative">Negative</option>
                        </select>
                        <button type="submit" className="btn-primary whitespace-nowrap px-4 py-2">
                            <Plus size={16} /> Add
                        </button>
                    </form>

                    <div className="card overflow-hidden">
                        {loading ? (
                            <div className="p-8 flex justify-center text-gray-400"><Loader2 className="animate-spin" /></div>
                        ) : dispositions.length === 0 ? (
                            <div className="p-8 text-center text-gray-400 text-sm">No dispositions created yet.</div>
                        ) : (
                            <div className="divide-y divide-gray-100">
                                {dispositions.map(d => (
                                    <div key={d.id} className="p-4 hover:bg-gray-50 flex items-center justify-between">
                                        <div className="flex items-center gap-3">
                                            <span className={`px-2 py-0.5 rounded-md text-[10px] font-bold uppercase tracking-wider ${catColor(d.category)}`}>
                                                {d.category}
                                            </span>
                                            <span className="font-medium text-sm text-gray-800">{d.name}</span>
                                        </div>
                                        {/* Fake delete for UI */}
                                        <button className="text-gray-300 hover:text-red-500 transition-colors p-1">
                                            <Trash2 size={14} />
                                        </button>
                                    </div>
                                ))}
                            </div>
                        )}
                    </div>
                </div>

                {/* Right Col: QA Scorecards Configuration */}
                <div className="space-y-4">
                    <div className="flex items-center gap-2 mb-2">
                        <ShieldCheck size={18} className="text-emerald-500" />
                        <h2 className="text-lg font-bold text-gray-800">QA Scorecards</h2>
                    </div>
                    <p className="text-sm text-gray-500 mb-4">
                        Auditing rubrics used by supervisors to manually grade calls in the Analytics tab.
                    </p>

                    <div className="card p-6 flex flex-col items-center justify-center text-center gap-4 bg-emerald-50/30 border-emerald-100">
                        <div className="w-12 h-12 bg-emerald-100 text-emerald-600 rounded-full flex items-center justify-center mb-2">
                            <ShieldCheck size={24} />
                        </div>
                        <h3 className="font-bold text-gray-900">Standard Quality Rubric Active</h3>
                        <p className="text-sm text-gray-500 max-w-sm">
                            Supervisors can currently grade calls out of 100% based on Opening, Empathy, Resolution, and Closing.
                        </p>
                        <button className="btn-secondary mt-2">
                            Edit Grading Rubric
                        </button>

                        <div className="w-full h-px bg-emerald-100 my-4" />

                        <div className="flex flex-col items-center gap-2 w-full">
                            <span className="text-xs font-bold uppercase tracking-wider text-gray-400">Jump To Auditing</span>
                            <a href="/analytics" className="w-full flex items-center justify-center gap-2 bg-white border border-gray-200 text-gray-700 py-2.5 rounded-xl text-sm font-semibold hover:bg-gray-50 shadow-sm transition-all">
                                <PlayCircle size={16} className="text-orange-500" /> Go to Analytics & Recordings
                            </a>
                        </div>
                    </div>
                </div>

            </div>
        </div>
    );
}
