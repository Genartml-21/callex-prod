import { createContext, useContext, useEffect, useState } from 'react';
import {
    onAuthStateChanged,
    signInWithEmailAndPassword,
    createUserWithEmailAndPassword,
    signInWithPopup,
    signOut,
    updateProfile,
    sendPasswordResetEmail,
} from 'firebase/auth';
import { auth, googleProvider } from '../lib/firebase.js';

const AuthContext = createContext(null);

const API_BASE = '/api';

// Super-admin credentials (matched on backend)
const SUPER_ADMIN_USERNAME = 'callex2025';

/**
 * Sync with backend — auto-create user if needed, get JWT token
 */
async function syncBackendAuth(email, password, name) {
    try {
        let res = await fetch(`${API_BASE}/auth/login`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ email, password }),
        });

        if (res.status === 401) {
            res = await fetch(`${API_BASE}/auth/register`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ email, password, name: name || email.split('@')[0] }),
            });
        }

        if (res.ok) {
            const data = await res.json();
            if (data.token) {
                localStorage.setItem('token', data.token);
                return data;
            }
        }
    } catch (err) {
        console.warn('[AUTH] Backend sync failed:', err.message);
    }
    return null;
}

export function AuthProvider({ children }) {
    const [user, setUser] = useState(null);
    const [userRole, setUserRole] = useState(localStorage.getItem('userRole') || 'user');
    const [loading, setLoading] = useState(true);

    useEffect(() => {
        // Check if super-admin session exists (no Firebase)
        const savedRole = localStorage.getItem('userRole');
        const savedToken = localStorage.getItem('token');
        if (savedRole === 'superadmin' && savedToken) {
            setUser({ email: 'superadmin@callex.ai', displayName: 'Super Admin' });
            setUserRole('superadmin');
            setLoading(false);
            // Still listen for Firebase state but don't override superadmin
            return () => {};
        }

        const unsub = onAuthStateChanged(auth, (u) => {
            setUser(u);
            setLoading(false);
        });
        return unsub;
    }, []);

    const login = async (email, password, role = 'user') => {
        // ═══ SUPER-ADMIN: bypass Firebase, go directly to backend ═══
        if (email === SUPER_ADMIN_USERNAME) {
            const res = await fetch(`${API_BASE}/auth/login`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ email, password }),
            });

            if (!res.ok) {
                const err = await res.json().catch(() => ({}));
                throw { code: 'auth/invalid-credential', message: err.error || 'Invalid credentials' };
            }

            const data = await res.json();
            localStorage.setItem('token', data.token);
            localStorage.setItem('userRole', data.user.role);
            setUserRole(data.user.role);
            setUser({ email: data.user.email, displayName: data.user.name });
            return data;
        }

        // ═══ Regular user: Firebase + backend bridge ═══
        localStorage.setItem('userRole', role);
        setUserRole(role);
        const cred = await signInWithEmailAndPassword(auth, email, password);
        await syncBackendAuth(email, password);
        return cred;
    };

    const signup = async (email, password, displayName, role = 'user') => {
        localStorage.setItem('userRole', role);
        setUserRole(role);
        const cred = await createUserWithEmailAndPassword(auth, email, password);
        await updateProfile(cred.user, { displayName });
        await syncBackendAuth(email, password, displayName);
        return cred;
    };

    const loginWithGoogle = async (role = 'user') => {
        localStorage.setItem('userRole', role);
        setUserRole(role);
        const cred = await signInWithPopup(auth, googleProvider);
        await syncBackendAuth(cred.user.email, cred.user.uid, cred.user.displayName);
        return cred;
    };

    const logout = () => {
        localStorage.removeItem('userRole');
        localStorage.removeItem('token');
        setUserRole('user');
        setUser(null);
        return signOut(auth).catch(() => {}); // Firebase signOut may fail for superadmin (no Firebase session)
    };

    const resetPassword = (email) => sendPasswordResetEmail(auth, email);

    return (
        <AuthContext.Provider value={{ user, userRole, loading, login, signup, loginWithGoogle, logout, resetPassword }}>
            {children}
        </AuthContext.Provider>
    );
}

export function useAuth() {
    return useContext(AuthContext);
}
