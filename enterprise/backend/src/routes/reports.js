import { Router } from 'express';
import { db, docToObj, queryToArray } from '../firebase.js';
import { stringify } from 'csv-stringify/sync';

const router = Router();

function getDateFilter(range) {
    const now = new Date();
    if (range === 'today') { now.setHours(0, 0, 0, 0); return now; }
    if (range === '7d') { const d = new Date(); d.setDate(d.getDate() - 7); return d; }
    if (range === '30d') { const d = new Date(); d.setDate(d.getDate() - 30); return d; }
    if (range === 'this_month') return new Date(now.getFullYear(), now.getMonth(), 1);
    if (range === 'last_month') return new Date(now.getFullYear(), now.getMonth() - 1, 1);
    return null;
}

// GET /api/reports/export
router.get('/export', async (req, res) => {
    try {
        const { type, range } = req.query;
        const startDate = getDateFilter(range);
        let csvString = "";

        if (type === 'calls_log') {
            let query = db.collection('calls').orderBy('startedAt', 'desc');
            const snap = await query.get();
            let calls = queryToArray(snap);
            if (startDate) calls = calls.filter(c => {
                const d = c.startedAt?.toDate ? c.startedAt.toDate() : new Date(c.startedAt);
                return d >= startDate;
            });

            const rows = [];
            for (const c of calls) {
                let agentName = 'Unknown', campaignName = 'None', dispName = '';
                if (c.agentId) { const a = await db.collection('agents').doc(c.agentId).get(); if (a.exists) agentName = a.data().name; }
                if (c.campaignId) { const camp = await db.collection('campaigns').doc(c.campaignId).get(); if (camp.exists) campaignName = camp.data().name; }
                if (c.dispositionId) { const disp = await db.collection('dispositions').doc(c.dispositionId).get(); if (disp.exists) dispName = disp.data().name; }
                rows.push({
                    CallID: c.id, Date: new Date(c.startedAt?.toDate ? c.startedAt.toDate() : c.startedAt).toISOString(),
                    Phone: c.phoneNumber, Direction: c.direction, Agent: agentName, Campaign: campaignName,
                    DurationSecs: c.duration, Sentiment: c.sentiment, Disposition: dispName, RecordingURL: c.recordingUrl || '', Summary: c.summary || ''
                });
            }
            csvString = stringify(rows, { header: true });

        } else if (type === 'qa_scores') {
            const snap = await db.collection('qaScores').orderBy('createdAt', 'desc').get();
            const scores = queryToArray(snap);
            const rows = scores.map(s => ({ ScoreID: s.id, CallID: s.callId, FinalScore: s.score, Feedback: s.feedback }));
            csvString = stringify(rows, { header: true });

        } else if (type === 'billing_usage') {
            const snap = await db.collection('billingStats').orderBy('month', 'desc').get();
            const stats = queryToArray(snap);
            const rows = stats.map(s => ({ Month: s.month, TelecomMins: s.telecomMins, VoiceSTT_Mins: s.sttMinutes, LLM_Tokens: s.llmTokens, TotalCostUSD: s.totalCostUsd }));
            csvString = stringify(rows, { header: true });

        } else {
            return res.status(400).json({ error: 'Invalid report type' });
        }

        const filename = `${type}_export_${range || 'all'}.csv`;
        res.setHeader('Content-Type', 'text/csv');
        res.setHeader('Content-Disposition', `attachment; filename="${filename}"`);
        res.status(200).send(csvString);
    } catch (err) {
        console.error("Export Error:", err);
        res.status(500).send("Error generating report");
    }
});

export default router;
