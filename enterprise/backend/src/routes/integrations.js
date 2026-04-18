import { Router } from 'express';
import { db, docToObj, queryToArray } from '../firebase.js';

const router = Router();

// GET /api/integrations
router.get('/', async (req, res) => {
    const snap = await db.collection('integrations').orderBy('name', 'asc').get();
    res.json(queryToArray(snap));
});

// PATCH /api/integrations/:id/connect
router.patch('/:id/connect', async (req, res) => {
    const { config } = req.body;
    await db.collection('integrations').doc(req.params.id).update({ connected: true, config: JSON.stringify(config || {}) });
    const doc = await db.collection('integrations').doc(req.params.id).get();
    const integration = docToObj(doc);
    await db.collection('systemEvents').add({ type: 'integration.connected', message: `Integration "${integration.name}" connected`, severity: 'info', meta: '{}', createdAt: new Date() });
    res.json(integration);
});

// PATCH /api/integrations/:id/disconnect
router.patch('/:id/disconnect', async (req, res) => {
    await db.collection('integrations').doc(req.params.id).update({ connected: false, config: '{}' });
    const doc = await db.collection('integrations').doc(req.params.id).get();
    res.json(docToObj(doc));
});

export default router;
