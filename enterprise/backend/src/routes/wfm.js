import { Router } from 'express';
import { db, docToObj, queryToArray } from '../firebase.js';

const router = Router();

// GET /api/wfm/states
router.get('/states', async (req, res) => {
    try {
        const snap = await db.collection('wfmStates').orderBy('timestamp', 'desc').limit(100).get();
        const states = [];
        const seen = new Set();
        for (const doc of snap.docs) {
            const s = { id: doc.id, ...doc.data() };
            if (!seen.has(s.userId)) {
                seen.add(s.userId);
                if (s.userId) {
                    const userDoc = await db.collection('users').doc(s.userId).get();
                    s.user = userDoc.exists ? { id: userDoc.id, name: userDoc.data().name, role: userDoc.data().role } : null;
                }
                states.push(s);
            }
        }
        res.json(states);
    } catch (err) { res.status(500).json({ error: err.message }); }
});

// POST /api/wfm/states
router.post('/states', async (req, res) => {
    try {
        const { userId, state } = req.body;
        const lastSnap = await db.collection('wfmStates').where('userId', '==', userId).orderBy('timestamp', 'desc').limit(1).get();
        if (!lastSnap.empty) {
            const last = lastSnap.docs[0];
            const ts = last.data().timestamp?.toDate ? last.data().timestamp.toDate() : new Date(last.data().timestamp);
            const duration = Math.floor((new Date() - ts) / 1000);
            await db.collection('wfmStates').doc(last.id).update({ duration });
        }
        const data = { userId, state, timestamp: new Date(), duration: null };
        const ref = await db.collection('wfmStates').add(data);
        res.json({ id: ref.id, ...data });
    } catch (err) { res.status(500).json({ error: err.message }); }
});

export default router;
