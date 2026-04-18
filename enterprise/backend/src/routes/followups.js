import { Router } from 'express';
import { db, docToObj, queryToArray } from '../firebase.js';

const router = Router();

// GET /api/followups
router.get('/', async (req, res) => {
    try {
        const snap = await db.collection('followUps').where('userId', '==', req.userId).orderBy('scheduledFor', 'asc').get();
        const followups = [];
        for (const doc of snap.docs) {
            const fu = { id: doc.id, ...doc.data() };
            if (fu.agentId) {
                const agentDoc = await db.collection('agents').doc(fu.agentId).get();
                fu.agent = agentDoc.exists ? docToObj(agentDoc) : null;
            }
            followups.push(fu);
        }
        res.json(followups);
    } catch (error) { res.status(500).json({ error: error.message }); }
});

// POST /api/followups
router.post('/', async (req, res) => {
    try {
        const { phoneNumber, agentId, campaignId, scheduledFor, reason } = req.body;
        const data = { userId: req.userId, phoneNumber, agentId, campaignId, scheduledFor: new Date(scheduledFor), reason, status: 'pending', createdAt: new Date() };
        const ref = await db.collection('followUps').add(data);
        res.json({ id: ref.id, ...data });
    } catch (error) { res.status(500).json({ error: error.message }); }
});

// PATCH /api/followups/:id/status
router.patch('/:id/status', async (req, res) => {
    try {
        const { status } = req.body;
        await db.collection('followUps').doc(req.params.id).update({ status });
        const doc = await db.collection('followUps').doc(req.params.id).get();
        res.json(docToObj(doc));
    } catch (error) { res.status(500).json({ error: error.message }); }
});

export default router;
