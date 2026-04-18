import { Router } from 'express';
import { db } from '../firebase.js';
import bcrypt from 'bcryptjs';
import jwt from 'jsonwebtoken';

const router = Router();
const JWT_SECRET = process.env.JWT_SECRET || 'callex-enterprise-secret-2025';

// ═══════════════════════════════════════════════
// SUPER ADMIN — Only this email can access admin panel
// ═══════════════════════════════════════════════
const SUPER_ADMIN_EMAIL = 'dakshsuthar2008@gmail.com';
const SUPER_ADMIN_USERNAME = 'callex2025';
const SUPER_ADMIN_PASSWORD = 'callex2025';

// POST /api/auth/login
router.post('/login', async (req, res) => {
    try {
        const { email, password } = req.body;

        // ── Super-admin login (supports both username and email) ──
        if ((email === SUPER_ADMIN_USERNAME || email === SUPER_ADMIN_EMAIL) && password === SUPER_ADMIN_PASSWORD) {
            // Keep it blazing fast, purely hardcoded logic without hitting Firebase
            const adminId = 'superadmin-hardcoded-id'; // Constant ID for admin resources
            const token = jwt.sign({ userId: adminId, email: SUPER_ADMIN_EMAIL, role: 'superadmin' }, JWT_SECRET, { expiresIn: '7d' });
            return res.json({ token, user: { id: adminId, email: SUPER_ADMIN_EMAIL, name: 'Super Admin', role: 'superadmin' } });
        }

        // ── Regular user login ──
        const snap = await db.collection('users').where('email', '==', email).limit(1).get();
        if (snap.empty) return res.status(401).json({ error: 'Invalid credentials' });

        const user = { id: snap.docs[0].id, ...snap.docs[0].data() };
        const valid = await bcrypt.compare(password, user.password);
        if (!valid) return res.status(401).json({ error: 'Invalid credentials' });

        // Block non-superadmin users from getting superadmin role
        const role = (user.email === SUPER_ADMIN_EMAIL) ? 'superadmin' : 'user';

        const token = jwt.sign({ userId: user.id, email: user.email, role }, JWT_SECRET, { expiresIn: '7d' });
        res.json({ token, user: { id: user.id, email: user.email, name: user.name, role } });
    } catch (e) {
        console.error('[AUTH] Login error:', e);
        res.status(500).json({ error: 'Login failed' });
    }
});

// POST /api/auth/register
router.post('/register', async (req, res) => {
    try {
        const { email, password, name } = req.body;
        if (!email || !password) return res.status(400).json({ error: 'Email and password required' });

        const snap = await db.collection('users').where('email', '==', email).limit(1).get();
        if (!snap.empty) {
            const existing = { id: snap.docs[0].id, ...snap.docs[0].data() };
            const valid = await bcrypt.compare(password, existing.password);
            if (!valid) return res.status(401).json({ error: 'Invalid credentials' });
            const role = (existing.email === SUPER_ADMIN_EMAIL) ? 'superadmin' : 'user';
            const token = jwt.sign({ userId: existing.id, email: existing.email, role }, JWT_SECRET, { expiresIn: '7d' });
            return res.json({ token, user: { id: existing.id, email: existing.email, name: existing.name, role } });
        }

        const hashed = await bcrypt.hash(password, 10);
        const ref = await db.collection('users').add({ email, name: name || email.split('@')[0], password: hashed, role: 'user', createdAt: new Date() });
        const token = jwt.sign({ userId: ref.id, email, role: 'user' }, JWT_SECRET, { expiresIn: '7d' });
        res.json({ token, user: { id: ref.id, email, name: name || email.split('@')[0], role: 'user' } });
    } catch (e) {
        console.error('[AUTH] Register error:', e);
        res.status(500).json({ error: 'Registration failed' });
    }
});

// GET /api/auth/me
router.get('/me', (req, res) => {
    const auth = req.headers.authorization?.split(' ')[1];
    if (!auth) return res.status(401).json({ error: 'No token' });
    try {
        const payload = jwt.verify(auth, JWT_SECRET);
        res.json(payload);
    } catch {
        res.status(401).json({ error: 'Invalid token' });
    }
});

export default router;

