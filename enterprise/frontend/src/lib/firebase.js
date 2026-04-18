import { initializeApp } from 'firebase/app';
import { getAuth, GoogleAuthProvider } from 'firebase/auth';
import { getAnalytics } from 'firebase/analytics';

const firebaseConfig = {
    apiKey: "AIzaSyAlXi4CSvr07HTbu_bV4EGO59MXVjmHf54",
    authDomain: "lakhuteleservices-1f9e0.firebaseapp.com",
    projectId: "lakhuteleservices-1f9e0",
    storageBucket: "lakhuteleservices-1f9e0.firebasestorage.app",
    messagingSenderId: "855678452910",
    appId: "1:855678452910:web:b0347ec8dfd710104c593f",
    measurementId: "G-K12ZEMY8KK"
};

const app = initializeApp(firebaseConfig);
export const auth = getAuth(app);
export const googleProvider = new GoogleAuthProvider();

// Only init analytics in browser
try { getAnalytics(app); } catch { }

export default app;
