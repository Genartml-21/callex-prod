import { Router } from 'express';
import { db, docToObj, queryToArray } from '../firebase.js';

const router = Router();

// GET /api/qa/dispositions
router.get('/dispositions', async (req, res) => {
    try {
        const page = parseInt(req.query.page) || 1;
        const limit = parseInt(req.query.limit) || 50;

        const snap = await db.collection('dispositions').orderBy('name', 'asc').get();
        let dispositions = queryToArray(snap);

        if (req.query.linkedAgent) {
            dispositions = dispositions.filter(d => 
                d.linkedAgents && d.linkedAgents.includes(req.query.linkedAgent)
            );
        }

        if (req.query.pagination === 'false') {
            return res.json(dispositions);
        }

        const total = dispositions.length;
        const startIndex = (page - 1) * limit;
        const endIndex = page * limit;
        const paginatedDispositions = dispositions.slice(startIndex, endIndex);

        res.json({
            dispositions: paginatedDispositions,
            pagination: { page, limit, total, totalPages: Math.ceil(total / limit) }
        });
    } catch (err) {
        res.status(500).json({ error: err.message });
    }
});

// POST /api/qa/dispositions
router.post('/dispositions', async (req, res) => {
    try {
        const { name, category, requiresNote, tagline, requiredFields, linkedAgents, linkedCampaigns, callDirection } = req.body;
        
        if (!linkedAgents || !Array.isArray(linkedAgents) || linkedAgents.length === 0) {
            return res.status(400).json({ error: "You must select at least one Agent to link this disposition to." });
        }

        const data = { 
            name, 
            category, 
            requiresNote: requiresNote || false, 
            tagline: tagline || '',
            requiredFields: Array.isArray(requiredFields) ? requiredFields : [],
            linkedAgents,
            linkedCampaigns: Array.isArray(linkedCampaigns) ? linkedCampaigns : [],
            callDirection: callDirection || 'both', // 'inbound', 'outbound', 'both'
            active: true, 
            createdAt: new Date() 
        };
        const ref = await db.collection('dispositions').add(data);
        res.json({ id: ref.id, ...data });
    } catch (err) {
        res.status(500).json({ error: err.message });
    }
});

// PUT /api/qa/dispositions/:id
router.put('/dispositions/:id', async (req, res) => {
    try {
        const doc = await db.collection('dispositions').doc(req.params.id).get();
        if (!doc.exists) return res.status(404).json({ error: 'Disposition not found' });

        const { name, category, requiresNote, active, tagline, requiredFields, linkedAgents, linkedCampaigns, callDirection } = req.body;
        const updates = { updatedAt: new Date() };
        if (name !== undefined) updates.name = name;
        if (category !== undefined) updates.category = category;
        if (requiresNote !== undefined) updates.requiresNote = requiresNote;
        if (active !== undefined) updates.active = active;
        if (tagline !== undefined) updates.tagline = tagline;
        if (requiredFields !== undefined) updates.requiredFields = Array.isArray(requiredFields) ? requiredFields : [];
        if (linkedAgents !== undefined) updates.linkedAgents = Array.isArray(linkedAgents) ? linkedAgents : [];
        if (linkedCampaigns !== undefined) updates.linkedCampaigns = Array.isArray(linkedCampaigns) ? linkedCampaigns : [];
        if (callDirection !== undefined) updates.callDirection = callDirection;

        await db.collection('dispositions').doc(req.params.id).update(updates);
        res.json({ id: req.params.id, ...doc.data(), ...updates });
    } catch (err) {
        res.status(500).json({ error: err.message });
    }
});

// DELETE /api/qa/dispositions/:id
router.delete('/dispositions/:id', async (req, res) => {
    try {
        const doc = await db.collection('dispositions').doc(req.params.id).get();
        if (!doc.exists) return res.status(404).json({ error: 'Disposition not found' });
        await db.collection('dispositions').doc(req.params.id).delete();
        res.json({ message: 'Disposition deleted successfully', id: req.params.id });
    } catch (err) {
        res.status(500).json({ error: err.message });
    }
});

// GET /api/qa/scores/:callId
router.get('/scores/:callId', async (req, res) => {
    try {
        const snap = await db.collection('qaScores').where('callId', '==', req.params.callId).limit(1).get();
        if (snap.empty) return res.json(null);
        const score = { id: snap.docs[0].id, ...snap.docs[0].data() };
        if (score.scoredByUserId) {
            const userDoc = await db.collection('users').doc(score.scoredByUserId).get();
            score.user = userDoc.exists ? { name: userDoc.data().name, email: userDoc.data().email } : null;
        }
        res.json(score);
    } catch (err) {
        res.status(500).json({ error: err.message });
    }
});

// POST /api/qa/scores
router.post('/scores', async (req, res) => {
    try {
        const { callId, scoredByUserId, score, feedback, rubric } = req.body;
        const snap = await db.collection('qaScores').where('callId', '==', callId).limit(1).get();
        const data = { callId, score, feedback, rubric: JSON.stringify(rubric), scoredByUserId, createdAt: new Date() };

        if (!snap.empty) {
            await db.collection('qaScores').doc(snap.docs[0].id).update(data);
            res.json({ id: snap.docs[0].id, ...data });
        } else {
            const ref = await db.collection('qaScores').add(data);
            res.json({ id: ref.id, ...data });
        }
    } catch (err) {
        res.status(500).json({ error: err.message });
    }
});

export default router;
