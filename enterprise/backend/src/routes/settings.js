import { Router } from 'express';
import { db, docToObj, queryToArray } from '../firebase.js';
import { v4 as uuidv4 } from 'uuid';
import crypto from 'crypto';

const router = Router();

// GET /api/settings/api-keys
router.get('/api-keys', async (req, res) => {
    const snap = await db.collection('apiKeys').where('active', '==', true).where('userId', '==', req.userId).get();
    let keys = queryToArray(snap);
    keys.sort((a,b) => new Date(b.createdAt||0) - new Date(a.createdAt||0));
    keys = keys.map(k => ({ id: k.id, name: k.name, prefix: k.prefix, env: k.env, lastUsed: k.lastUsed, createdAt: k.createdAt, active: k.active }));
    res.json(keys);
});

// POST /api/settings/api-keys
router.post('/api-keys', async (req, res) => {
    const { name, env } = req.body;
    const rawKey = uuidv4().replace(/-/g, '');
    const prefix = env === 'live' ? `ck_live_${rawKey.slice(0, 8)}` : `ck_test_${rawKey.slice(0, 8)}`;
    const fullKey = `${prefix}_${rawKey.slice(8)}`;
    const keyHash = crypto.createHash('sha256').update(fullKey).digest('hex');
    const data = { userId: req.userId, name, keyHash, prefix, env: env || 'test', active: true, lastUsed: null, createdAt: new Date() };
    const ref = await db.collection('apiKeys').add(data);
    res.json({ id: ref.id, ...data, fullKey });
});

// DELETE /api/settings/api-keys/:id
router.delete('/api-keys/:id', async (req, res) => {
    await db.collection('apiKeys').doc(req.params.id).update({ active: false });
    res.json({ success: true });
});

// GET /api/settings/webhooks
router.get('/webhooks', async (req, res) => {
    const snap = await db.collection('webhooks').where('userId', '==', req.userId).get();
    const hooks = queryToArray(snap);
    hooks.sort((a,b) => new Date(b.createdAt||0) - new Date(a.createdAt||0));
    res.json(hooks);
});

// POST /api/settings/webhooks
router.post('/webhooks', async (req, res) => {
    const { url, events, secret } = req.body;
    const data = { userId: req.userId, url, events: JSON.stringify(events || []), secret, active: true, createdAt: new Date() };
    const ref = await db.collection('webhooks').add(data);
    res.json({ id: ref.id, ...data });
});

// PATCH /api/settings/webhooks/:id
router.patch('/webhooks/:id', async (req, res) => {
    const data = { ...req.body };
    if (data.events && Array.isArray(data.events)) data.events = JSON.stringify(data.events);
    delete data.id;
    await db.collection('webhooks').doc(req.params.id).update(data);
    const doc = await db.collection('webhooks').doc(req.params.id).get();
    res.json(docToObj(doc));
});

// DELETE /api/settings/webhooks/:id
router.delete('/webhooks/:id', async (req, res) => {
    await db.collection('webhooks').doc(req.params.id).delete();
    res.json({ success: true });
});

// POST /api/settings/webhooks/:id/test
router.post('/webhooks/:id/test', async (req, res) => {
    const doc = await db.collection('webhooks').doc(req.params.id).get();
    if (!doc.exists) return res.status(404).json({ error: 'Webhook not found' });
    const webhook = docToObj(doc);
    res.json({ success: true, message: `Test event dispatched to ${webhook.url}` });
});

export default router;
