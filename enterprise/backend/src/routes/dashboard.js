import { Router } from 'express';
import { db, docToObj, queryToArray } from '../firebase.js';

const router = Router();

// GET /api/dashboard/kpis
router.get('/kpis', async (req, res) => {
    const agentsSnap = await db.collection('agents').where('userId', '==', req.userId).get();
    const agentIds = agentsSnap.docs.map(d => d.id);

    let activeCalls = 0, completedToday = 0, allCalls = [], queueDepth = 0;
    const todayStart = new Date(); todayStart.setHours(0, 0, 0, 0);

    if (agentIds.length > 0) {
        // Firestore 'in' supports max 30 values — chunk if needed
        const chunks = [];
        for (let i = 0; i < agentIds.length; i += 30) chunks.push(agentIds.slice(i, i + 30));

        for (const chunk of chunks) {
            const callsSnap = await db.collection('calls').where('agentId', 'in', chunk).get();
            callsSnap.forEach(d => {
                const c = d.data();
                allCalls.push(c);
                if (c.status === 'active') activeCalls++;
                if (c.status === 'completed' && c.startedAt && new Date(c.startedAt.toDate ? c.startedAt.toDate() : c.startedAt) >= todayStart) completedToday++;
            });
        }
    }

    // Queue depth (calls without agent)
    const queueSnap = await db.collection('calls').where('status', '==', 'active').where('userId', '==', req.userId).get();
    queueSnap.forEach(d => { if (!d.data().agentId) queueDepth++; });

    const avgMOS = allCalls.filter(c => c.mosScore).reduce((a, b, _, arr) => a + b.mosScore / arr.length, 0) || 4.2;
    const angryCount = allCalls.filter(c => c.sentiment === 'angry').length;
    const slaRate = allCalls.length > 0 ? Math.round((1 - angryCount / allCalls.length) * 100) : 100;

    const errorSnap = await db.collection('systemEvents').where('severity', '==', 'error').get();
    const totalEventsSnap = await db.collection('systemEvents').get();
    const fallbackRate = totalEventsSnap.size > 0 ? (errorSnap.size / totalEventsSnap.size) * 100 : 0;

    res.json({
        activeCalls, completedToday,
        avgMOS: Math.round(avgMOS * 100) / 100,
        slaPercent: slaRate,
        apiFallbackRate: Math.round(fallbackRate * 10) / 10,
        aiAgentsAvailable: agentIds.length,
        humanAgentsAvailable: 2,
        queueDepth,
    });
});

// GET /api/dashboard/ab-test
router.get('/ab-test', async (req, res) => {
    const callsSnap = await db.collection('calls').where('status', '==', 'completed').get();
    const calls = queryToArray(callsSnap);

    const stats = {};
    for (const c of calls) {
        if (!c.agentId) continue;
        const agentDoc = await db.collection('agents').doc(c.agentId).get();
        if (!agentDoc.exists) continue;
        const model = agentDoc.data().llmModel;
        if (!stats[model]) stats[model] = { calls: 0, durationSum: 0, mosSum: 0, conversions: 0 };
        stats[model].calls++;
        stats[model].durationSum += c.duration || 0;
        stats[model].mosSum += c.mosScore || 0;
        if (c.sentiment === 'positive') stats[model].conversions++;
    }

    const models = Object.keys(stats).sort((a, b) => stats[b].calls - stats[a].calls);
    if (models.length < 2) {
        return res.json({
            champion: { model: 'Callex-1.3', csat: 0, avgDuration: 0, conversions: 0, calls: 0 },
            challenger: { model: 'Callex-1.2', csat: 0, avgDuration: 0, conversions: 0, calls: 0 },
            winner: 'none', confidence: 0
        });
    }

    const m1 = models[0], m2 = models[1];
    const calc = (m) => ({
        model: m,
        csat: stats[m].calls > 0 ? (stats[m].mosSum / stats[m].calls).toFixed(1) : 0,
        avgDuration: stats[m].calls > 0 ? Math.round(stats[m].durationSum / stats[m].calls) : 0,
        conversions: stats[m].conversions,
        calls: stats[m].calls
    });
    const champ = calc(m1), chal = calc(m2);
    res.json({ champion: champ, challenger: chal, winner: parseFloat(champ.csat) >= parseFloat(chal.csat) ? 'champion' : 'challenger', confidence: Math.min(99, 50 + (stats[m1].calls + stats[m2].calls)) });
});

// GET /api/dashboard/events
router.get('/events', async (req, res) => {
    const snap = await db.collection('systemEvents').orderBy('createdAt', 'desc').limit(50).get();
    res.json(queryToArray(snap));
});

// POST /api/dashboard/events
router.post('/events', async (req, res) => {
    const { type, message, severity, meta } = req.body;
    const data = { type, message, severity: severity || 'info', meta: JSON.stringify(meta || {}), createdAt: new Date() };
    const ref = await db.collection('systemEvents').add(data);
    res.json({ id: ref.id, ...data });
});

export default router;
