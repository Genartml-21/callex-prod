import { Router } from 'express';
import { db, docToObj, queryToArray } from '../firebase.js';
import { broadcastToCall } from '../index.js';

const router = Router();

// GET /api/supervisor/calls
router.get('/calls', async (req, res) => {
    try {
        // Query BOTH 'active' and 'in-progress' for maximum compatibility
        const [activeSnap, inProgressSnap] = await Promise.all([
            db.collection('calls').where('status', '==', 'active').get(),
            db.collection('calls').where('status', '==', 'in-progress').get(),
        ]);

        const allDocs = [...activeSnap.docs, ...inProgressSnap.docs];
        const now = Date.now();
        const MAX_AGE_MS = 2 * 60 * 60 * 1000; // 2 hours — ghost call protection

        const calls = [];
        for (const doc of allDocs) {
            const callData = doc.data();

            // Skip ghost calls older than 2 hours
            const startedAt = callData.startedAt?.toDate ? callData.startedAt.toDate().getTime() : new Date(callData.startedAt || 0).getTime();
            if (now - startedAt > MAX_AGE_MS) continue;

            const call = { id: doc.id, ...callData };
            if (call.agentId && !call.agentName) {
                try {
                    const agentDoc = await db.collection('agents').doc(call.agentId).get();
                    if (agentDoc.exists) call.agentName = agentDoc.data().name || 'Unknown Agent';
                } catch (e) { /* ignore */ }
            }
            calls.push({
                id: call.id,
                phoneNumber: call.phoneNumber || 'Unknown',
                crmId: call.crmId || null,
                agentId: call.agentId || '',
                agentName: call.agentName || 'Unknown Agent',
                status: call.status || 'active',
                sentiment: call.sentiment || 'neutral',
                transcript: call.transcript || '',
                transcriptMessages: call.transcriptMessages || [],
                startedAt: call.startedAt,
                endedAt: call.endedAt || null,
            });
        }
        calls.sort((a, b) => {
            const da = a.startedAt?.toDate ? a.startedAt.toDate().getTime() : new Date(a.startedAt || 0).getTime();
            const db2 = b.startedAt?.toDate ? b.startedAt.toDate().getTime() : new Date(b.startedAt || 0).getTime();
            return db2 - da;
        });

        console.log(`[SUPERVISOR] Returning ${calls.length} active calls`);
        res.json({ success: true, data: calls, message: 'Active call fetched successfully' });
    } catch (e) {
        console.error('[SUPERVISOR ERROR] GET /calls failed:', e);
        res.status(500).json({ success: false, data: [], message: 'Failed to fetch active calls' });
    }
});

// POST /api/supervisor/calls
router.post('/calls', async (req, res) => {
    const { phoneNumber, agentId } = req.body;
    if (!phoneNumber) return res.status(400).json({ error: 'phoneNumber required' });
    const data = { 
        phoneNumber, 
        agentId: agentId || null, 
        userId: req.userId || null,
        status: 'active', 
        sentiment: 'neutral', 
        transcript: '', 
        mosScore: 4.5, 
        startedAt: new Date() 
    };
    const ref = await db.collection('calls').add(data);
    await db.collection('systemEvents').add({ type: 'call.started', message: `Call started from ${phoneNumber}`, severity: 'info', meta: '{}', createdAt: new Date() });
    import('../index.js').then(({ broadcastToDashboard }) => broadcastToDashboard({ type: 'refresh_kpi' }));
    res.json({ id: ref.id, ...data });
});

// PATCH /api/supervisor/calls/:id/end
router.patch('/calls/:id/end', async (req, res) => {
    const doc = await db.collection('calls').doc(req.params.id).get();
    const existing = docToObj(doc);
    const duration = existing?.startedAt ? Math.round((Date.now() - new Date(existing.startedAt.toDate ? existing.startedAt.toDate() : existing.startedAt).getTime()) / 1000) : 0;
    await db.collection('calls').doc(req.params.id).update({ status: 'completed', endedAt: new Date(), duration });
    await db.collection('systemEvents').add({ type: 'call.ended', message: `Call ended: ${existing?.phoneNumber}`, severity: 'info', meta: '{}', createdAt: new Date() });
    import('../index.js').then(({ broadcastToDashboard }) => broadcastToDashboard({ type: 'refresh_kpi' }));
    const updated = await db.collection('calls').doc(req.params.id).get();
    res.json(docToObj(updated));
});

// POST /api/supervisor/calls/:id/whisper
router.post('/calls/:id/whisper', async (req, res) => {
    const { message } = req.body;
    if (!message) return res.status(400).json({ error: 'message required' });
    broadcastToCall(req.params.id, { type: 'whisper', message, ts: Date.now() });
    const doc = await db.collection('calls').doc(req.params.id).get();
    const call = docToObj(doc);
    const newTranscript = (call?.transcript || '') + `\n[SYSTEM WHISPER]: ${message}`;
    await db.collection('calls').doc(req.params.id).update({ transcript: newTranscript });
    res.json({ success: true, message });
});

// POST /api/supervisor/calls/:id/barge
router.post('/calls/:id/barge', async (req, res) => {
    broadcastToCall(req.params.id, { type: 'barge', ts: Date.now() });
    await db.collection('calls').doc(req.params.id).update({ status: 'transferred' });
    await db.collection('systemEvents').add({ type: 'call.barged', message: `Supervisor barged into call ${req.params.id}`, severity: 'warning', meta: '{}', createdAt: new Date() });
    res.json({ success: true });
});

// GET /api/supervisor/calls/:id/transcript
router.get('/calls/:id/transcript', async (req, res) => {
    const doc = await db.collection('calls').doc(req.params.id).get();
    const call = docToObj(doc);
    if (!call) return res.json({ transcript: '', transcriptMessages: [], messageCount: 0, sentiment: 'neutral' });

    const messages = call.transcriptMessages || [];

    // Get agent name if not present
    let agentName = call.agentName || '';
    if (!agentName && call.agentId) {
        try {
            const agentDoc = await db.collection('agents').doc(call.agentId).get();
            if (agentDoc.exists) agentName = agentDoc.data().name || '';
        } catch (e) { /* ignore */ }
    }

    res.json({
        callId: call.id,
        phoneNumber: call.phoneNumber || '',
        agentId: call.agentId || '',
        agentName,
        duration: call.duration || 0,
        transcript: call.transcript || '',
        transcriptMessages: messages,
        messageCount: messages.length,
        sentiment: call.sentiment || 'neutral',
        startedAt: call.startedAt,
        endedAt: call.endedAt || null,
    });
});

// PATCH /api/supervisor/calls/:id/sentiment
router.patch('/calls/:id/sentiment', async (req, res) => {
    const { sentiment } = req.body;
    await db.collection('calls').doc(req.params.id).update({ sentiment });
    const doc = await db.collection('calls').doc(req.params.id).get();
    res.json(docToObj(doc));
});

export default router;
