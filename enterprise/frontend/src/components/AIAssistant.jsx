import React, { useState, useEffect, useRef } from 'react';
import { Bot, X, Send, Sparkles, Loader2, MessageSquare } from 'lucide-react';
import { useAuth } from '../contexts/AuthContext.jsx';

export default function AIAssistant() {
    const [isOpen, setIsOpen] = useState(false);
    const [messages, setMessages] = useState([
        { role: 'assistant', text: "Hi there! I'm your Callex AI copilot. Need help setting up an agent or launching a campaign?" }
    ]);
    const [input, setInput] = useState('');
    const [loading, setLoading] = useState(false);
    const messagesEndRef = useRef(null);
    const { user } = useAuth();

    const scrollToBottom = () => {
        messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
    };

    useEffect(() => {
        if (isOpen) {
            scrollToBottom();
        }
    }, [messages, isOpen]);

    const handleSend = async (e) => {
        e.preventDefault();
        if (!input.trim()) return;

        const userMsg = { role: 'user', text: input };
        setMessages(prev => [...prev, userMsg]);
        setInput('');
        setLoading(true);

        // Process AI response
        setTimeout(() => {
            let botText = "I see! To set that up, navigate to the Agent Studio, select your agent, and adjust the telephony settings.";
            if (userMsg.text.toLowerCase().includes('campaign') || userMsg.text.toLowerCase().includes('dialer')) {
                botText = "To launch an outbound campaign, go to the Outbound Dialer page. Once there, click 'New Campaign', assign your agent, and upload your CSV list of contacts. Let me know if you need help with the advanced settings!";
            } else if (userMsg.text.toLowerCase().includes('agent')) {
                botText = "In the Agent Studio, you can configure your AI's voice, personality, and foundational rules. I recommend starting with the Callex-1.3 model for the best reasoning capabilities.";
            }

            setMessages(prev => [...prev, { role: 'assistant', text: botText }]);
            setLoading(false);
        }, 1500);
    };

    return (
        <div className="fixed bottom-6 right-6 z-50 flex flex-col items-end">

            {/* Chat Window */}
            {isOpen && (
                <div className="w-[380px] h-[550px] max-h-[80vh] bg-white/80 backdrop-blur-xl border border-white/40 shadow-2xl rounded-3xl mb-4 flex flex-col overflow-hidden animate-fade-in ring-1 ring-gray-900/5 transition-all duration-300">

                    {/* Header */}
                    <div className="px-5 py-4 bg-gradient-to-r from-orange-500 to-rose-500 flex items-center justify-between text-white shrink-0 shadow-sm relative overflow-hidden">
                        <div className="absolute top-0 right-0 -mr-8 -mt-8 w-32 h-32 bg-white/10 rounded-full blur-2xl"></div>
                        <div className="flex items-center gap-3 relative z-10">
                            <div className="w-10 h-10 rounded-full bg-white/20 backdrop-blur-sm flex items-center justify-center border border-white/20 shadow-inner">
                                <Sparkles size={20} className="text-white" />
                            </div>
                            <div>
                                <h3 className="font-bold text-sm tracking-wide">Callex AI Assistant</h3>
                                <div className="text-[10px] text-orange-100 flex items-center gap-1.5 font-medium lowercase tracking-wider">
                                    <span className="relative flex h-2 w-2">
                                        <span className="animate-ping absolute inline-flex h-full w-full rounded-full bg-emerald-300 opacity-75"></span>
                                        <span className="relative inline-flex rounded-full h-2 w-2 bg-emerald-400"></span>
                                    </span>
                                    Online & Ready
                                </div>
                            </div>
                        </div>
                        <button
                            onClick={() => setIsOpen(false)}
                            className="p-2 hover:bg-white/20 rounded-full transition-colors relative z-10"
                        >
                            <X size={18} />
                        </button>
                    </div>

                    {/* Messages Area */}
                    <div className="flex-1 overflow-y-auto p-5 space-y-4 bg-slate-50/50 custom-scroll">
                        {messages.map((msg, idx) => (
                            <div key={idx} className={`flex ${msg.role === 'user' ? 'justify-end' : 'justify-start'}`}>
                                <div className={`max-w-[85%] rounded-2xl p-4 text-sm shadow-sm ${msg.role === 'user'
                                    ? 'bg-gray-900 text-white rounded-br-sm'
                                    : 'bg-white border text-gray-800 rounded-bl-sm border-gray-100'
                                    }`}>
                                    {msg.text}
                                </div>
                            </div>
                        ))}
                        {loading && (
                            <div className="flex justify-start">
                                <div className="bg-white border border-gray-100 rounded-2xl rounded-bl-sm p-4 text-sm text-gray-500 shadow-sm flex items-center gap-2">
                                    <Loader2 size={16} className="animate-spin text-orange-500" />
                                    <span>Thinking...</span>
                                </div>
                            </div>
                        )}
                        <div ref={messagesEndRef} />
                    </div>

                    {/* Input Area */}
                    <div className="p-4 bg-white border-t border-gray-100 shrink-0">
                        <form onSubmit={handleSend} className="relative flex items-center">
                            <input
                                type="text"
                                value={input}
                                onChange={(e) => setInput(e.target.value)}
                                placeholder="Ask me anything..."
                                className="w-full bg-gray-50 border border-gray-200 text-sm rounded-full py-3.5 pl-5 pr-12 focus:outline-none focus:border-orange-300 focus:bg-white transition-all focus:ring-4 focus:ring-orange-500/10"
                            />
                            <button
                                type="submit"
                                disabled={!input.trim() || loading}
                                className="absolute right-2 p-2 bg-gradient-to-br from-orange-400 to-rose-500 text-white rounded-full hover:shadow-md disabled:opacity-50 disabled:hover:shadow-none transition-all"
                            >
                                <Send size={14} className="ml-0.5" />
                            </button>
                        </form>
                    </div>
                </div>
            )}

            {/* Bubble Toggle Button */}
            {!isOpen && (
                <button
                    onClick={() => setIsOpen(true)}
                    className="group relative flex items-center justify-center w-14 h-14 bg-gradient-to-r from-gray-900 to-gray-800 text-white rounded-full shadow-2xl hover:scale-105 transition-all duration-300 hover:shadow-orange-500/20"
                >
                    <div className="absolute inset-0 bg-gradient-to-r from-orange-500 to-rose-500 rounded-full opacity-0 group-hover:opacity-100 transition-opacity duration-300 blur-lg -z-10"></div>
                    <MessageSquare size={24} className="group-hover:scale-110 transition-transform duration-300" />

                    {/* Welcome tooltip */}
                    <div className="absolute right-full mr-4 top-1/2 -translate-y-1/2 bg-white text-gray-900 text-xs font-bold py-2 px-4 rounded-xl shadow-lg border border-gray-100 opacity-0 group-hover:opacity-100 translate-x-4 group-hover:translate-x-0 transition-all duration-300 whitespace-nowrap pointer-events-none">
                        Need help?
                        <div className="absolute right-[-6px] top-1/2 -translate-y-1/2 border-[6px] border-transparent border-l-white"></div>
                    </div>
                </button>
            )}
        </div>
    );
}
