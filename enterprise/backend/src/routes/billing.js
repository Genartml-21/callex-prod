import { Router } from 'express';
import { db, docToObj, queryToArray } from '../firebase.js';

const router = Router();

// GET /api/billing/stats
router.get('/stats', async (req, res) => {
    try {
        const currentMonth = new Date().toISOString().substring(0, 7);
        const doc = await db.collection('billingStats').doc(currentMonth).get();
        if (!doc.exists) {
            const data = { month: currentMonth, telecomMins: 0, llmTokens: 0, sttMinutes: 0, totalCostUsd: 0.0, updatedAt: new Date() };
            await db.collection('billingStats').doc(currentMonth).set(data);
            return res.json({ id: currentMonth, ...data });
        }
        res.json(docToObj(doc));
    } catch (err) {
        res.status(500).json({ error: err.message });
    }
});

// POST /api/billing/increment
router.post('/increment', async (req, res) => {
    try {
        const { telecomMins, llmTokens, sttMinutes, costIncrement } = req.body;
        const currentMonth = new Date().toISOString().substring(0, 7);
        const doc = await db.collection('billingStats').doc(currentMonth).get();

        if (doc.exists) {
            const existing = doc.data();
            await db.collection('billingStats').doc(currentMonth).update({
                telecomMins: (existing.telecomMins || 0) + (telecomMins || 0),
                llmTokens: (existing.llmTokens || 0) + (llmTokens || 0),
                sttMinutes: (existing.sttMinutes || 0) + (sttMinutes || 0),
                totalCostUsd: (existing.totalCostUsd || 0) + (costIncrement || 0),
                updatedAt: new Date(),
            });
        } else {
            await db.collection('billingStats').doc(currentMonth).set({
                month: currentMonth, telecomMins: telecomMins || 0, llmTokens: llmTokens || 0,
                sttMinutes: sttMinutes || 0, totalCostUsd: costIncrement || 0, updatedAt: new Date(),
            });
        }
        const updated = await db.collection('billingStats').doc(currentMonth).get();
        res.json(docToObj(updated));
    } catch (err) {
        res.status(500).json({ error: err.message });
    }
});

export default router;
