import { useState } from 'react';
import { useNavigate } from 'react-router-dom';
import { useAuth } from '../contexts/AuthContext.jsx';
import { LogoImage } from '../lib/logo.jsx';
import { Eye, EyeOff, AlertCircle, Loader2, Phone, PhoneCall, Activity, Users, Zap, ShieldCheck } from 'lucide-react';

// Floating live-call stat badges for the left panel
const LIVE_STATS = [
    { icon: PhoneCall, label: 'Active Calls', value: '2,847', color: 'bg-white/15', pos: 'top-[18%] right-8' },
    { icon: Users, label: 'Agents Online', value: '340', color: 'bg-white/15', pos: 'top-[36%] left-6' },
    { icon: Activity, label: 'Calls / Min', value: '127', color: 'bg-white/15', pos: 'top-[56%] right-10' },
    { icon: Zap, label: 'Avg Latency', value: '182ms', color: 'bg-white/15', pos: 'top-[72%] left-8' },
];

export default function Login() {
    const [tab, setTab] = useState('login');
    const [form, setForm] = useState({ email: '', password: '', name: '', username: '' });
    const [role, setRole] = useState('user');
    const [showPw, setShowPw] = useState(false);
    const [error, setError] = useState('');
    const [info, setInfo] = useState('');
    const [loading, setLoading] = useState(false);
    const { login, signup, loginWithGoogle, resetPassword } = useAuth();
    const navigate = useNavigate();

    const F = (key) => ({ value: form[key], onChange: e => setForm(f => ({ ...f, [key]: e.target.value })) });

    async function handleSubmit(e) {
        e.preventDefault();
        setError(''); setInfo(''); setLoading(true);
        try {
            if (role === 'admin') {
                // Admin login — send username as email field to backend
                await login(form.username, form.password, 'admin');
                navigate('/dashboard');
            } else if (tab === 'login') {
                await login(form.email, form.password, role);
                navigate('/dashboard');
            } else if (tab === 'signup') {
                await signup(form.email, form.password, form.name, role);
                navigate('/dashboard');
            } else {
                await resetPassword(form.email);
                setInfo('Password reset email sent! Check your inbox.');
                setTab('login');
            }
        } catch (err) {
            const msgs = {
                'auth/invalid-credential': 'Invalid username or password.',
                'auth/user-not-found': 'No account found.',
                'auth/wrong-password': 'Incorrect password.',
                'auth/email-already-in-use': 'An account with this email already exists.',
                'auth/weak-password': 'Password must be at least 6 characters.',
                'auth/invalid-email': 'Please enter a valid email address.',
                'auth/too-many-requests': 'Too many attempts. Please try again later.',
                'auth/popup-closed-by-user': 'Google sign-in was cancelled.',
            };
            setError(msgs[err.code] || err.message);
        } finally { setLoading(false); }
    }

    async function handleGoogle() {
        setError(''); setLoading(true);
        try { await loginWithGoogle(role); navigate('/dashboard'); }
        catch (err) {
            console.error("Google Auth Error:", err);
            if (err.code === 'auth/unauthorized-domain') {
                setError('This domain is not authorized for OAuth. Please add it in Firebase Console.');
            } else if (err.code !== 'auth/popup-closed-by-user') {
                setError(err.message || 'Google sign-in failed. Please try again.');
            }
        }
        finally { setLoading(false); }
    }

    const isAdmin = role === 'admin';

    return (
        <div className="min-h-screen bg-gradient-to-br from-orange-50 via-white to-gray-50 flex">

            {/* ── Left branding panel ── */}
            <div className="hidden lg:flex lg:w-1/2 bg-gradient-to-br from-orange-500 to-orange-700 flex-col justify-between p-12 relative overflow-hidden">

                {/* Background bokeh blobs */}
                <div className="absolute top-0 right-0 w-[500px] h-[500px] bg-orange-400/20 rounded-full -translate-y-1/2 translate-x-1/4 blur-3xl" />
                <div className="absolute bottom-0 left-0 w-72 h-72 bg-orange-900/20 rounded-full translate-y-1/2 -translate-x-1/3 blur-2xl" />
                <div className="absolute top-1/2 left-1/2 w-40 h-40 bg-white/5 rounded-full -translate-x-1/2 -translate-y-1/2 blur-xl" />

                {/* Floating live stats */}
                {LIVE_STATS.map(({ icon: Icon, label, value, color, pos }) => (
                    <div key={label} className={`absolute ${pos} ${color} backdrop-blur-sm border border-white/20 rounded-2xl px-4 py-3 flex items-center gap-3 animate-pulse`} style={{ animationDuration: `${2 + Math.random() * 2}s` }}>
                        <div className="w-8 h-8 rounded-xl bg-white/20 flex items-center justify-center">
                            <Icon size={14} className="text-white" />
                        </div>
                        <div>
                            <div className="text-white font-bold text-sm leading-none">{value}</div>
                            <div className="text-orange-200 text-xs mt-0.5">{label}</div>
                        </div>
                    </div>
                ))}

                {/* Center logo with ringing rings */}
                <div className="relative z-10 flex flex-col items-center justify-center flex-1">
                    {/* Pulse rings */}
                    <div className="relative flex items-center justify-center mb-8">
                        <div className="absolute w-48 h-48 rounded-full border-2 border-white/10 animate-ping" style={{ animationDuration: '3s' }} />
                        <div className="absolute w-36 h-36 rounded-full border-2 border-white/15 animate-ping" style={{ animationDuration: '2.5s', animationDelay: '0.3s' }} />
                        <div className="absolute w-28 h-28 rounded-full border-2 border-white/20 animate-ping" style={{ animationDuration: '2s', animationDelay: '0.6s' }} />
                        <div className="relative z-10 p-3 bg-white/10 rounded-3xl backdrop-blur-sm border border-white/20 shadow-2xl">
                            <LogoImage size={96} />
                        </div>
                    </div>

                    <h1 className="text-4xl font-bold text-white mb-3 leading-tight text-center">
                        AI-Powered Voice Agents<br />at Enterprise Scale
                    </h1>
                    <p className="text-orange-100 text-base leading-relaxed text-center max-w-xs">
                        Deploy intelligent voice agents, monitor live calls, and automate your call center in real time.
                    </p>
                </div>

                {/* Stats row bottom */}
                <div className="relative z-10 grid grid-cols-3 gap-4">
                    {[['99.9%', 'Uptime SLA'], ['<200ms', 'Avg Latency'], ['50+', 'Calls/second']].map(([val, label]) => (
                        <div key={label} className="bg-white/10 backdrop-blur border border-white/10 rounded-2xl p-4 text-center">
                            <div className="text-2xl font-bold text-white">{val}</div>
                            <div className="text-orange-200 text-xs mt-0.5">{label}</div>
                        </div>
                    ))}
                </div>
            </div>

            {/* ── Right auth panel ── */}
            <div className="flex-1 flex items-center justify-center p-8">
                <div className="w-full max-w-md">

                    {/* Mobile logo */}
                    <div className="mb-8 lg:hidden flex justify-center">
                        <LogoImage size={64} />
                    </div>

                    <div className="mb-8">
                        <h2 className="text-2xl font-bold text-gray-900 mb-1">
                            {isAdmin ? 'Admin Login' : tab === 'login' ? 'Welcome back' : tab === 'signup' ? 'Create your account' : 'Reset password'}
                        </h2>
                        <p className="text-gray-400 text-sm">
                            {isAdmin ? 'Enter your admin credentials to access the control panel' : tab === 'login' ? 'Sign in to your Callex dashboard' : tab === 'signup' ? 'Start your enterprise journey' : "We'll send you a reset link"}
                        </p>
                    </div>

                    {/* Role toggle — always visible */}
                    <div className="flex bg-gray-100 p-1 rounded-xl mb-5">
                        <button type="button" onClick={() => { setRole('admin'); setTab('login'); setError(''); }} className={`flex-1 py-2 text-sm font-semibold rounded-lg transition-colors flex items-center justify-center gap-1.5 ${role === 'admin' ? 'bg-white shadow-sm text-gray-900' : 'text-gray-500 hover:text-gray-700'}`}>
                            <ShieldCheck size={14} />Admin
                        </button>
                        <button type="button" onClick={() => { setRole('user'); setError(''); }} className={`flex-1 py-2 text-sm font-semibold rounded-lg transition-colors ${role === 'user' ? 'bg-white shadow-sm text-gray-900' : 'text-gray-500 hover:text-gray-700'}`}>User</button>
                    </div>

                    {/* ═══ ADMIN LOGIN: username + password only ═══ */}
                    {isAdmin ? (
                        <form onSubmit={handleSubmit} className="space-y-4">
                            <div>
                                <label className="label">Username</label>
                                <input required type="text" className="input-field" placeholder="Enter admin username" {...F('username')} />
                            </div>
                            <div>
                                <div className="flex items-center justify-between mb-1">
                                    <label className="label m-0">Password</label>
                                </div>
                                <div className="relative">
                                    <input required type={showPw ? 'text' : 'password'} className="input-field pr-10" placeholder="••••••••" {...F('password')} />
                                    <button type="button" onClick={() => setShowPw(!showPw)} className="absolute right-3 top-1/2 -translate-y-1/2 text-gray-400 hover:text-gray-600">
                                        {showPw ? <EyeOff size={16} /> : <Eye size={16} />}
                                    </button>
                                </div>
                            </div>

                            {error && (
                                <div className="flex items-start gap-2 p-3 bg-red-50 border border-red-100 rounded-xl text-sm text-red-600">
                                    <AlertCircle size={15} className="shrink-0 mt-0.5" />{error}
                                </div>
                            )}

                            <button type="submit" disabled={loading}
                                className="w-full btn-primary justify-center py-3 text-base rounded-xl shadow-md shadow-orange-100">
                                {loading && <Loader2 size={17} className="animate-spin" />}
                                Sign In as Admin
                            </button>
                        </form>
                    ) : (
                        /* ═══ USER LOGIN: full flow with Google, signup, reset ═══ */
                        <>
                            {/* Google */}
                            {tab !== 'reset' && (
                                <button onClick={handleGoogle} disabled={loading}
                                    className="w-full flex items-center justify-center gap-3 bg-white border border-gray-200 rounded-xl py-3 px-4 text-sm font-semibold text-gray-700 hover:bg-gray-50 hover:border-gray-300 transition-all shadow-sm mb-5">
                                    <svg width="18" height="18" viewBox="0 0 24 24">
                                        <path fill="#4285F4" d="M22.56 12.25c0-.78-.07-1.53-.2-2.25H12v4.26h5.92c-.26 1.37-1.04 2.53-2.21 3.31v2.77h3.57c2.08-1.92 3.28-4.74 3.28-8.09z" />
                                        <path fill="#34A853" d="M12 23c2.97 0 5.46-.98 7.28-2.66l-3.57-2.77c-.98.66-2.23 1.06-3.71 1.06-2.86 0-5.29-1.93-6.16-4.53H2.18v2.84C3.99 20.53 7.7 23 12 23z" />
                                        <path fill="#FBBC05" d="M5.84 14.09c-.22-.66-.35-1.36-.35-2.09s.13-1.43.35-2.09V7.07H2.18C1.43 8.55 1 10.22 1 12s.43 3.45 1.18 4.93l2.85-2.22.81-.62z" />
                                        <path fill="#EA4335" d="M12 5.38c1.62 0 3.06.56 4.21 1.64l3.15-3.15C17.45 2.09 14.97 1 12 1 7.7 1 3.99 3.47 2.18 7.07l3.66 2.84c.87-2.6 3.3-4.53 6.16-4.53z" />
                                    </svg>
                                    Continue with Google
                                </button>
                            )}

                            {tab !== 'reset' && (
                                <div className="flex items-center gap-3 mb-5">
                                    <div className="flex-1 h-px bg-gray-100"></div>
                                    <span className="text-xs text-gray-400 font-medium">or with email</span>
                                    <div className="flex-1 h-px bg-gray-100"></div>
                                </div>
                            )}

                            <form onSubmit={handleSubmit} className="space-y-4">
                                {tab === 'signup' && (
                                    <div><label className="label">Full Name</label><input required className="input-field" placeholder="John Smith" {...F('name')} /></div>
                                )}
                                <div><label className="label">Email</label><input required type="email" className="input-field" placeholder="you@company.com" {...F('email')} /></div>
                                {tab !== 'reset' && (
                                    <div>
                                        <div className="flex items-center justify-between mb-1">
                                            <label className="label m-0">Password</label>
                                            {tab === 'login' && <button type="button" onClick={() => { setTab('reset'); setError(''); }} className="text-xs text-orange-500 hover:text-orange-600 font-medium">Forgot password?</button>}
                                        </div>
                                        <div className="relative">
                                            <input required type={showPw ? 'text' : 'password'} className="input-field pr-10" placeholder="••••••••" {...F('password')} />
                                            <button type="button" onClick={() => setShowPw(!showPw)} className="absolute right-3 top-1/2 -translate-y-1/2 text-gray-400 hover:text-gray-600">
                                                {showPw ? <EyeOff size={16} /> : <Eye size={16} />}
                                            </button>
                                        </div>
                                        {tab === 'signup' && <p className="text-xs text-gray-400 mt-1">Minimum 6 characters</p>}
                                    </div>
                                )}

                                {error && (
                                    <div className="flex items-start gap-2 p-3 bg-red-50 border border-red-100 rounded-xl text-sm text-red-600">
                                        <AlertCircle size={15} className="shrink-0 mt-0.5" />{error}
                                    </div>
                                )}
                                {info && (
                                    <div className="flex items-start gap-2 p-3 bg-emerald-50 border border-emerald-100 rounded-xl text-sm text-emerald-600">
                                        <AlertCircle size={15} className="shrink-0 mt-0.5" />{info}
                                    </div>
                                )}

                                <button type="submit" disabled={loading}
                                    className="w-full btn-primary justify-center py-3 text-base rounded-xl shadow-md shadow-orange-100">
                                    {loading && <Loader2 size={17} className="animate-spin" />}
                                    {tab === 'login' ? 'Sign In' : tab === 'signup' ? 'Create Account' : 'Send Reset Link'}
                                </button>
                            </form>

                            <div className="mt-6 text-center text-sm text-gray-500">
                                {tab === 'login' ? (<>Don't have an account?{' '}<button onClick={() => { setTab('signup'); setError(''); }} className="text-orange-500 font-semibold hover:text-orange-600">Sign up free</button></>)
                                    : tab === 'signup' ? (<>Already have an account?{' '}<button onClick={() => { setTab('login'); setError(''); }} className="text-orange-500 font-semibold hover:text-orange-600">Sign in</button></>)
                                        : (<button onClick={() => { setTab('login'); setError(''); }} className="text-orange-500 font-semibold hover:text-orange-600">← Back to sign in</button>)}
                            </div>
                        </>
                    )}
                </div>
            </div>
        </div>
    );
}
