import { Router } from 'express';
import { db, docToObj, queryToArray } from '../firebase.js';
import crypto from 'crypto';

const router = Router();

// GET /api/security/voice-signatures
router.get('/voice-signatures', async (req, res) => {
    const snap = await db.collection('voiceSignatures').orderBy('createdAt', 'desc').get();
    res.json(queryToArray(snap));
});

// POST /api/security/voice-signatures
router.post('/voice-signatures', async (req, res) => {
    const { phrase, description } = req.body;
    const hashExample = crypto.createHash('sha256').update(phrase + Date.now()).digest('hex');
    const data = { phrase, description, hashExample, active: true, createdAt: new Date() };
    const ref = await db.collection('voiceSignatures').add(data);
    res.json({ id: ref.id, ...data });
});

// DELETE /api/security/voice-signatures/:id
router.delete('/voice-signatures/:id', async (req, res) => {
    await db.collection('voiceSignatures').doc(req.params.id).delete();
    res.json({ success: true });
});

// POST /api/security/pci
router.post('/pci', async (req, res) => {
    const { callId, paused } = req.body;
    await db.collection('calls').doc(callId).update({ recordingPaused: paused });
    await db.collection('systemEvents').add({ type: 'pci.recording', message: `Recording ${paused ? 'paused' : 'resumed'} for call ${callId}`, severity: 'warning', meta: '{}', createdAt: new Date() });
    res.json({ success: true, paused });
});

// POST /api/security/hash-audio
router.post('/hash-audio', async (req, res) => {
    const { audioChunk, callId } = req.body;
    const hash = crypto.createHash('sha256').update(audioChunk || '').digest('hex');
    res.json({ hash, callId, ts: new Date().toISOString(), algorithm: 'SHA-256' });
});

export default router;
