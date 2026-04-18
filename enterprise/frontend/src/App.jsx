import { Routes, Route, Navigate } from 'react-router-dom';
import { Component } from 'react';
import { AuthProvider, useAuth } from './contexts/AuthContext.jsx';
import Layout from './components/Layout.jsx';
import Login from './pages/Login.jsx';
import Dashboard from './pages/Dashboard.jsx';
import LiveSupervisor from './pages/LiveSupervisor.jsx';
import AgentStudio from './pages/AgentStudio.jsx';
import KnowledgeBase from './pages/KnowledgeBase.jsx';
import Simulation from './pages/Simulation.jsx';
import Dialer from './pages/Dialer.jsx';
import Analytics from './pages/Analytics.jsx';
import Routing from './pages/Routing.jsx';
import Integrations from './pages/Integrations.jsx';
import Security from './pages/Security.jsx';
import Settings from './pages/Settings.jsx';

import QA from './pages/QA.jsx';
import Reports from './pages/Reports.jsx';
import Telecom from './pages/Telecom.jsx';
import Billing from './pages/Billing.jsx';
import FollowUps from './pages/FollowUps.jsx';
import AdminUsers from './pages/AdminUsers.jsx';
import Toast from './components/Toast.jsx';
import { Loader2, ServerCrash } from 'lucide-react';

// Global error boundary — catches uncaught render errors and shows a friendly page
class ErrorBoundary extends Component {
    constructor(props) { super(props); this.state = { hasError: false }; }
    static getDerivedStateFromError() { return { hasError: true }; }
    componentDidCatch(error, info) { console.error('ErrorBoundary caught:', error, info); }
    render() {
        if (this.state.hasError) {
            return (
                <div className="min-h-screen flex items-center justify-center bg-gray-50">
                    <div className="text-center p-8 max-w-md">
                        <div className="w-16 h-16 rounded-2xl bg-red-50 flex items-center justify-center mx-auto mb-4">
                            <ServerCrash size={28} className="text-red-500" />
                        </div>
                        <h1 className="text-xl font-bold text-gray-900 mb-2">Something went wrong</h1>
                        <p className="text-gray-500 text-sm mb-6">
                            A page failed to render. Please try navigating back.
                        </p>
                        <button
                            onClick={() => this.setState({ hasError: false })}
                            className="px-4 py-2 bg-orange-500 text-white rounded-xl text-sm font-semibold hover:bg-orange-600 transition-colors"
                        >
                            Try again
                        </button>
                    </div>
                </div>
            );
        }
        return this.props.children;
    }
}

function ProtectedRoute({ children }) {
    const { user, loading } = useAuth();
    if (loading) return (
        <div className="min-h-screen flex items-center justify-center bg-gray-50">
            <div className="flex flex-col items-center gap-3">
                <div className="w-10 h-10 rounded-xl bg-orange-500 flex items-center justify-center animate-pulse">
                    <Loader2 size={20} className="text-white animate-spin" />
                </div>
                <p className="text-sm text-gray-400 font-medium">Loading Callex...</p>
            </div>
        </div>
    );
    if (!user) return <Navigate to="/login" replace />;
    return children;
}

export default function App() {
    return (
        <AuthProvider>
            <Toast />
            <ErrorBoundary>
                <Routes>
                    <Route path="/login" element={<Login />} />
                    <Route path="/" element={<ProtectedRoute><Layout /></ProtectedRoute>}>
                        <Route index element={<Navigate to="/dashboard" replace />} />
                        <Route path="dashboard" element={<ErrorBoundary><Dashboard /></ErrorBoundary>} />
                        <Route path="supervisor" element={<ErrorBoundary><LiveSupervisor /></ErrorBoundary>} />
                        <Route path="agents" element={<ErrorBoundary><AgentStudio /></ErrorBoundary>} />
                        <Route path="knowledge" element={<ErrorBoundary><KnowledgeBase /></ErrorBoundary>} />
                        <Route path="simulation" element={<ErrorBoundary><Simulation /></ErrorBoundary>} />
                        <Route path="dialer" element={<ErrorBoundary><Dialer /></ErrorBoundary>} />
                        <Route path="analytics" element={<ErrorBoundary><Analytics /></ErrorBoundary>} />
                        <Route path="routing" element={<ErrorBoundary><Routing /></ErrorBoundary>} />
                        <Route path="integrations" element={<ErrorBoundary><Integrations /></ErrorBoundary>} />
                        <Route path="security" element={<ErrorBoundary><Security /></ErrorBoundary>} />
                        <Route path="settings" element={<ErrorBoundary><Settings /></ErrorBoundary>} />

                        <Route path="qa" element={<ErrorBoundary><QA /></ErrorBoundary>} />
                        <Route path="reports" element={<ErrorBoundary><Reports /></ErrorBoundary>} />
                        <Route path="telecom" element={<ErrorBoundary><Telecom /></ErrorBoundary>} />
                        <Route path="billing" element={<ErrorBoundary><Billing /></ErrorBoundary>} />
                        <Route path="followups" element={<ErrorBoundary><FollowUps /></ErrorBoundary>} />
                        <Route path="admin/users" element={<ErrorBoundary><AdminUsers /></ErrorBoundary>} />
                    </Route>
                    <Route path="*" element={<Navigate to="/dashboard" replace />} />
                </Routes>
            </ErrorBoundary>
        </AuthProvider>
    );
}
