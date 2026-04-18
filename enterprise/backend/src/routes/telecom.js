import { Router } from 'express';
import { db, docToObj, queryToArray } from '../firebase.js';

const router = Router();

// === Phone Numbers ===
router.get('/numbers', async (req, res) => {
    try {
        const snap = await db.collection('phoneNumbers').get();
        const numbers = [];
        for (const doc of snap.docs) {
            const num = { id: doc.id, ...doc.data() };
            if (num.routingRuleId) {
                const rr = await db.collection('routingRules').doc(num.routingRuleId).get();
                num.routingRule = rr.exists ? docToObj(rr) : null;
            }
            numbers.push(num);
        }
        res.json(numbers);
    } catch (err) { res.status(500).json({ error: err.message }); }
});

router.post('/numbers', async (req, res) => {
    try {
        const { number, friendlyName, provider } = req.body;
        const data = { number, friendlyName, provider: provider || 'twilio', status: 'active', createdAt: new Date() };
        const ref = await db.collection('phoneNumbers').add(data);
        res.json({ id: ref.id, ...data });
    } catch (err) { res.status(500).json({ error: err.message }); }
});

router.delete('/numbers/:id', async (req, res) => {
    try { await db.collection('phoneNumbers').doc(req.params.id).delete(); res.json({ success: true }); }
    catch (err) { res.status(500).json({ error: err.message }); }
});

// === DNC List ===
router.get('/dnc', async (req, res) => {
    try {
        const snap = await db.collection('dncList').orderBy('createdAt', 'desc').get();
        res.json(queryToArray(snap));
    } catch (err) { res.status(500).json({ error: err.message }); }
});

router.post('/dnc', async (req, res) => {
    try {
        const { number, reason, addedBy } = req.body;
        const data = { number, reason, addedBy, createdAt: new Date() };
        const ref = await db.collection('dncList').add(data);
        res.json({ id: ref.id, ...data });
    } catch (err) { res.status(500).json({ error: err.message }); }
});

router.delete('/dnc/:id', async (req, res) => {
    try { await db.collection('dncList').doc(req.params.id).delete(); res.json({ success: true }); }
    catch (err) { res.status(500).json({ error: err.message }); }
});

export default router;
