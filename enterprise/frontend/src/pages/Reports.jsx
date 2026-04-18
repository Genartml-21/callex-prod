import { useState } from 'react';
import { Download, FileText, Calendar, Filter, FileSpreadsheet, Headset, CheckCircle2, ShieldCheck } from 'lucide-react';
import { useStore } from '../store/index.js';

import { jsPDF } from 'jspdf';
import 'jspdf-autotable';

const REPORT_TYPES = [
    { id: 'calls_log', title: 'Complete Call Log', icon: Headset, desc: 'Every inbound/outbound call with duration, sentiment, and AI summary.' },
    { id: 'qa_scores', title: 'QA Audit Scores', icon: CheckCircle2, desc: 'Manual grading rubrics, supervisor feedback, and final scores per call.' },
    { id: 'billing_usage', title: 'Usage & Costs', icon: FileSpreadsheet, desc: 'Telecom minutes, LLM tokens consumed, and STT usage by month.' },
    { id: 'transcript_pdf', title: 'Call Transcripts (PDF)', icon: FileText, desc: 'Export detailed bot vs human conversation transcripts.' }
];

export default function Reports() {
    const [selectedReport, setSelectedReport] = useState(REPORT_TYPES[0].id);
    const [dateRange, setDateRange] = useState('7d');
    const [generating, setGenerating] = useState(false);
    const { showToast } = useStore();

    const handleExport = async () => {
        setGenerating(true);
        try {
            if (selectedReport === 'transcript_pdf') {
                generatePDF();
                showToast('Report generated and downloaded successfully.', 'success');
                return;
            }

            // Generate Dummy CSV Data for testing locally
            let csvContent = "";
            let fileName = `${selectedReport}_${dateRange}_export.csv`;

            if (selectedReport === 'calls_log') {
                csvContent = "CallID,Date,Phone,Direction,Agent,Campaign,DurationSecs,Sentiment,Disposition,RecordingURL,Summary\n" +
                    "CALL-101,2026-02-23T10:00:00Z,+919876543210,Outbound,Rahul AI,Loan Promo,145,Positive,Interested,,Customer agreed to terms.\n" +
                    "CALL-102,2026-02-23T10:05:00Z,+919876543211,Outbound,Priya AI,Loan Promo,45,Negative,Not Interested,,Customer hung up quickly.\n" +
                    "CALL-103,2026-02-23T10:15:00Z,+919876543212,Inbound,Support AI,Support Queue,320,Neutral,Resolved,,Helped customer reset password.";
            } else if (selectedReport === 'qa_scores') {
                csvContent = "ScoreID,CallID,DateScored,AgentEvaluated,ScoredBy,FinalScore,Feedback\n" +
                    "QA-1,CALL-101,2026-02-23T11:00:00Z,Rahul AI,Supervisor Bot,95,Excellent opening and rebuttal handling.\n" +
                    "QA-2,CALL-102,2026-02-23T11:05:00Z,Priya AI,Supervisor Bot,70,Spoke too fast during the intro.";
            } else if (selectedReport === 'billing_usage') {
                csvContent = "Month,TelecomMins,VoiceSTT_Mins,LLM_Tokens,TotalCostUSD\n" +
                    "2026-02,4500,4500,12500000,145.50\n" +
                    "2026-01,12000,12000,35000000,412.00";
            }

            const blob = new Blob([csvContent], { type: 'text/csv;charset=utf-8;' });
            const downloadUrl = window.URL.createObjectURL(blob);
            const link = document.createElement("a");
            link.href = downloadUrl;
            link.download = fileName;
            document.body.appendChild(link);
            link.click();
            document.body.removeChild(link);

            showToast('Report generated and downloaded successfully.', 'success');
        } catch (err) {
            console.error(err);
            showToast('Error generating report.', 'error');
        } finally {
            setGenerating(false);
        }
    };

    const generatePDF = () => {
        const doc = new jsPDF();

        // Add Callex Logo (Text placeholder for simplicity in this example)
        doc.setFontSize(22);
        doc.setTextColor(249, 115, 22); // Orange-500
        doc.setFont("helvetica", "bold");
        doc.text("Callex", 14, 20);

        doc.setFontSize(10);
        doc.setTextColor(100);
        doc.setFont("helvetica", "normal");
        doc.text("Enterprise Voice AI Transcripts", 14, 26);

        // Title
        doc.setFontSize(16);
        doc.setTextColor(20);
        doc.setFont("helvetica", "bold");
        doc.text(`Call Transcript Report - ${dateRange}`, 14, 40);

        // Dummy Hindi Transcript Data
        const transcriptData = [
            ["Bot", "Namaste! Main Callex AI se baat kar raha hoon. Kya main Rahul se baat kar sakta hoon?"],
            ["Human", "Haan, Rahul bol raha hoon. Boliye."],
            ["Bot", "Sir, humari team ne note kiya hai ki aapne loan application adhi chhodi thi. Kya aapko process complete karne mein koi madad chahiye?"],
            ["Human", "Nahi, mujhe interest rate thoda zyada laga tha."],
            ["Bot", "Main samajh sakta hoon. Agar main aapko 10.5% ka special interest rate offer karu, toh kya aap interested honge?"],
            ["Human", "Hmm, ye better lag raha hai. Kya aap mujhe documents ki list WhatsApp kar sakte hain?"],
            ["Bot", "Bilkul sir! Main abhi aapko WhatsApp par saari details bhej deta hoon. Call karne ke liye dhanyawad. Aapka din shubh ho!"]
        ];

        doc.autoTable({
            startY: 48,
            head: [['Speaker', 'Message (Hindi)']],
            body: transcriptData,
            theme: 'striped',
            headStyles: { fillColor: [249, 115, 22] }, // Orange-500 header
            styles: {
                font: 'helvetica', // Hindi font required for production, using English transliteration for PDF compatibility
                fontSize: 11,
                cellPadding: 6
            },
            columnStyles: {
                0: { fontStyle: 'bold', cellWidth: 30 },
            }
        });

        doc.save(`Transcript_${dateRange}.pdf`);
    };

    return (
        <div className="space-y-8">
            {/* Header */}
            <div className="page-header">
                <div>
                    <h1 className="text-2xl font-bold text-gray-900">Reports & Exports</h1>
                    <p className="text-sm text-gray-400 mt-0.5">Download raw CSV data for external business intelligence tools</p>
                </div>
            </div>

            <div className="grid grid-cols-1 lg:grid-cols-3 gap-8">

                {/* Left Col: Report Selection */}
                <div className="lg:col-span-2 space-y-4">
                    <h2 className="text-md font-bold text-gray-800 flex items-center gap-2 mb-4">
                        <FileText size={18} className="text-orange-500" /> Select Report Type
                    </h2>

                    <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                        {REPORT_TYPES.map(report => {
                            const Icon = report.icon;
                            const isSelected = selectedReport === report.id;

                            return (
                                <div
                                    key={report.id}
                                    onClick={() => setSelectedReport(report.id)}
                                    className={`p-5 rounded-2xl border-2 cursor-pointer transition-all ${isSelected
                                        ? 'border-orange-500 bg-orange-50/30 shadow-md shadow-orange-500/10'
                                        : 'border-gray-100 bg-white hover:border-gray-200 hover:bg-gray-50'
                                        }`}
                                >
                                    <div className={`w-10 h-10 rounded-xl flex items-center justify-center mb-3 ${isSelected ? 'bg-orange-100 text-orange-600' : 'bg-gray-100 text-gray-500'
                                        }`}>
                                        <Icon size={20} />
                                    </div>
                                    <h3 className={`font-bold text-sm mb-1 ${isSelected ? 'text-gray-900' : 'text-gray-700'}`}>
                                        {report.title}
                                    </h3>
                                    <p className="text-xs text-gray-500 leading-relaxed">
                                        {report.desc}
                                    </p>
                                </div>
                            );
                        })}
                    </div>
                </div>

                {/* Right Col: Parameters & Export */}
                <div className="bg-white rounded-2xl shadow-sm border border-gray-100 p-6 flex flex-col h-fit sticky top-24">
                    <h2 className="text-md font-bold text-gray-800 flex items-center gap-2 mb-6 border-b border-gray-100 pb-4">
                        <Filter size={18} className="text-blue-500" /> Export Parameters
                    </h2>

                    <div className="space-y-5 flex-1">

                        {/* Date Range */}
                        <div>
                            <label className="block text-xs font-bold text-gray-500 uppercase tracking-wider mb-2">
                                Date Range
                            </label>
                            <div className="relative">
                                <Calendar size={16} className="absolute left-3 top-1/2 -translate-y-1/2 text-gray-400" />
                                <select
                                    className="input-field w-full pl-10"
                                    value={dateRange}
                                    onChange={e => setDateRange(e.target.value)}
                                >
                                    <option value="today">Today</option>
                                    <option value="7d">Last 7 Days</option>
                                    <option value="30d">Last 30 Days</option>
                                    <option value="this_month">This Month</option>
                                    <option value="last_month">Last Month</option>
                                    <option value="all">All Time</option>
                                </select>
                            </div>
                        </div>

                        {/* Optional Granular Filters depending on report */}
                        {selectedReport === 'calls_log' && (
                            <div>
                                <label className="block text-xs font-bold text-gray-500 uppercase tracking-wider mb-2">
                                    Include Audio URLs
                                </label>
                                <div className="flex items-center gap-3">
                                    <button className="flex-1 py-2 rounded-lg border-2 border-orange-500 bg-orange-50 text-orange-700 text-sm font-semibold">Yes</button>
                                    <button className="flex-1 py-2 rounded-lg border-2 border-gray-100 text-gray-500 text-sm font-medium hover:bg-gray-50">No</button>
                                </div>
                            </div>
                        )}

                    </div>

                    <div className="mt-8 pt-6 border-t border-gray-100">
                        <button
                            onClick={handleExport}
                            disabled={generating}
                            className="w-full btn-primary py-3 flex items-center justify-center gap-2 text-base shadow-orange-500/25 shadow-lg disabled:opacity-70"
                        >
                            {generating ? (
                                <>
                                    <div className="w-5 h-5 border-2 border-white/30 border-t-white rounded-full animate-spin" />
                                    Generating {selectedReport === 'transcript_pdf' ? 'PDF' : 'CSV'}...
                                </>
                            ) : (
                                <>
                                    <Download size={18} />
                                    Export {REPORT_TYPES.find(r => r.id === selectedReport)?.title}
                                </>
                            )}
                        </button>
                        <p className="text-center text-[11px] text-gray-400 mt-3 flex items-center justify-center gap-1">
                            <ShieldCheck size={12} /> Data is exported securely over TLS
                        </p>
                    </div>
                </div>

            </div>
        </div>
    );
}
