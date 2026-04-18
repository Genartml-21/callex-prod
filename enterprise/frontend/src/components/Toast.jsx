import { useStore } from '../store/index.js';
import { useEffect } from 'react';
import { CheckCircle, AlertCircle, Info, X } from 'lucide-react';

export default function Toast() {
    const { toast } = useStore();
    if (!toast) return null;
    const icons = { success: <CheckCircle size={16} className="text-emerald-500" />, error: <AlertCircle size={16} className="text-red-500" />, info: <Info size={16} className="text-blue-500" /> };
    const borders = { success: 'border-emerald-100', error: 'border-red-100', info: 'border-blue-100' };
    return (
        <div className={`fixed bottom-6 right-6 z-50 flex items-center gap-3 bg-white border ${borders[toast.type] || 'border-gray-100'} rounded-2xl shadow-lg px-5 py-4 min-w-[280px] animate-pulse-once`}>
            {icons[toast.type] || icons.info}
            <span className="text-sm text-gray-700 font-medium">{toast.msg}</span>
        </div>
    );
}
