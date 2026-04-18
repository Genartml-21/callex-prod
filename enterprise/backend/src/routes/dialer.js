import { Router } from 'express';
import { db, docToObj, queryToArray } from '../firebase.js';
import { wss } from '../index.js';

const router = Router();

// GET /api/dialer/campaigns
router.get('/campaigns', async (req, res) => {
    const snap = await db.collection('campaigns').where('userId', '==', req.userId).orderBy('createdAt', 'desc').get();
    res.json(queryToArray(snap));
});

// POST /api/dialer/campaigns
router.post('/campaigns', async (req, res) => {
    const {
        name, agentId, dialingMode, callsPerSecond, tcpaLock, dncScrubbing, amdEnabled, voicemailDrop,
        startDate, endDate, audience,
        scriptOverride, maxDuration, maxRetries, retryDelayMin, voicemailDropAudio,
        localCallerId, strictLitigatorScrub, sentimentTransfer, timezoneRespect, costCapTokens, postCallSmsTemplate,
        recordCalls, concurrentCallLimit, autoPauseFailureRate, webhookUrl, dynamicVariables,
        transferNumber, transferWhisper, maxBudgetUsd, smsOnNoAnswer, amdAction, analysisSchema
    } = req.body;

    const leads = Array.isArray(audience) ? audience : [];

    const data = {
        userId: req.userId, name, agentId, dialingMode: dialingMode || 'predictive',
        callsPerSecond: callsPerSecond || 5, tcpaLock: tcpaLock ?? true, dncScrubbing: dncScrubbing ?? true,
        amdEnabled: amdEnabled ?? true, voicemailDrop: voicemailDrop ?? false,
        scriptOverride: scriptOverride || null,
        maxDuration: isNaN(maxDuration) ? 10 : parseInt(maxDuration, 10),
        maxRetries: isNaN(maxRetries) ? 0 : parseInt(maxRetries, 10),
        retryDelayMin: isNaN(retryDelayMin) ? 60 : parseInt(retryDelayMin, 10),
        voicemailDropAudio: voicemailDropAudio || null,
        localCallerId: localCallerId ?? false,
        strictLitigatorScrub: strictLitigatorScrub ?? false,
        sentimentTransfer: sentimentTransfer ?? false,
        timezoneRespect: timezoneRespect ?? true,
        costCapTokens: isNaN(costCapTokens) ? 5000 : parseInt(costCapTokens, 10),
        postCallSmsTemplate: postCallSmsTemplate || null,
        recordCalls: recordCalls ?? true,
        concurrentCallLimit: isNaN(concurrentCallLimit) ? 0 : parseInt(concurrentCallLimit, 10),
        autoPauseFailureRate: isNaN(autoPauseFailureRate) ? 0 : parseInt(autoPauseFailureRate, 10),
        webhookUrl: webhookUrl || null, dynamicVariables: dynamicVariables ?? false,
        transferNumber: transferNumber || null, transferWhisper: transferWhisper || null,
        maxBudgetUsd: maxBudgetUsd ? parseFloat(maxBudgetUsd) : null,
        smsOnNoAnswer: smsOnNoAnswer || null, amdAction: amdAction || 'hangup',
        analysisSchema: analysisSchema || '[]',
        totalLeads: leads.length, dialedLeads: 0, connectedLeads: 0,
        startDate: startDate ? new Date(startDate) : null, endDate: endDate ? new Date(endDate) : null,
        audience: JSON.stringify(leads), status: 'draft', createdAt: new Date(),
    };

    const ref = await db.collection('campaigns').add(data);
    await db.collection('systemEvents').add({ type: 'campaign.created', message: `Campaign "${name}" created with ${leads.length} leads`, severity: 'info', meta: '{}', createdAt: new Date() });
    res.json({ id: ref.id, ...data });
});

// PATCH /api/dialer/campaigns/:id
router.patch('/campaigns/:id', async (req, res) => {
    const data = { ...req.body };
    if (data.audience && Array.isArray(data.audience)) data.audience = JSON.stringify(data.audience);
    delete data.id;
    await db.collection('campaigns').doc(req.params.id).update(data);
    const doc = await db.collection('campaigns').doc(req.params.id).get();
    res.json(docToObj(doc));
});

// PATCH /api/dialer/campaigns/:id/status
router.patch('/campaigns/:id/status', async (req, res) => {
    const { status } = req.body;
    await db.collection('campaigns').doc(req.params.id).update({ status });
    const doc = await db.collection('campaigns').doc(req.params.id).get();
    const campaign = docToObj(doc);
    if (status === 'running') {
        await db.collection('systemEvents').add({ type: 'campaign.started', message: `Campaign "${campaign.name}" started`, severity: 'info', meta: '{}', createdAt: new Date() });
    }
    res.json(campaign);
});

// DELETE /api/dialer/campaigns/:id
router.delete('/campaigns/:id', async (req, res) => {
    await db.collection('campaigns').doc(req.params.id).delete();
    res.json({ success: true });
});

export default router;
