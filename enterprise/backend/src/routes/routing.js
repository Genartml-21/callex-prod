import { Router } from 'express';
import { db, docToObj, queryToArray } from '../firebase.js';

const router = Router();

// GET /api/routing/rules
router.get('/rules', async (req, res) => {
    const snap = await db.collection('routingRules').where('userId', '==', req.userId).orderBy('priority', 'asc').get();
    res.json(queryToArray(snap));
});

// POST /api/routing/rules
router.post('/rules', async (req, res) => {
    const data = { ...req.body, userId: req.userId, createdAt: new Date() };
    const ref = await db.collection('routingRules').add(data);
    res.json({ id: ref.id, ...data });
});

// PATCH /api/routing/rules/:id
router.patch('/rules/:id', async (req, res) => {
    const data = { ...req.body };
    delete data.id;
    await db.collection('routingRules').doc(req.params.id).update(data);
    const doc = await db.collection('routingRules').doc(req.params.id).get();
    res.json(docToObj(doc));
});

// DELETE /api/routing/rules/:id
router.delete('/rules/:id', async (req, res) => {
    await db.collection('routingRules').doc(req.params.id).delete();
    res.json({ success: true });
});

// POST /api/routing/evaluate
router.post('/evaluate', async (req, res) => {
    const { intentTag } = req.body;
    const snap = await db.collection('routingRules').where('active', '==', true).orderBy('priority', 'asc').get();
    const rules = queryToArray(snap);
    const matched = rules.find(r => r.intentTag.toLowerCase() === intentTag?.toLowerCase());
    res.json({ matched: matched || null, action: matched?.destination || 'agent' });
});

export default router;
