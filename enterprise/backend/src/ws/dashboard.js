import { broadcastToDashboard } from '../index.js';
import { db, queryToArray } from '../firebase.js';

let dashboardInterval = null;

export function setupDashboardWS(ws) {
    sendDashboardUpdate(ws);

    if (!dashboardInterval) {
        dashboardInterval = setInterval(async () => {
            const data = await buildKPIPayload();
            broadcastToDashboard({ type: 'kpi', data });
        }, 5000);
    }

    ws.on('close', () => { });
    ws.on('message', (msg) => {
        try {
            const parsed = JSON.parse(msg);
            if (parsed.type === 'ping') ws.send(JSON.stringify({ type: 'pong' }));
        } catch { }
    });
}

async function buildKPIPayload() {
    const activeCallsSnap = await db.collection('calls').where('status', '==', 'active').get();
    const eventsSnap = await db.collection('systemEvents').orderBy('createdAt', 'desc').limit(5).get();
    const allCallsSnap = await db.collection('calls').get();

    const allCalls = queryToArray(allCallsSnap);
    const avgMOS = allCalls.filter(c => c.mosScore).reduce((a, b, _, arr) => a + b.mosScore / arr.length, 0) || 4.2;
    const angryCount = allCalls.filter(c => c.sentiment === 'angry').length;
    const slaPercent = allCalls.length > 0 ? Math.round((1 - angryCount / allCalls.length) * 100) : 100;

    const agentsSnap = await db.collection('agents').where('status', '==', 'active').get();

    let queueDepth = 0;
    activeCallsSnap.forEach(d => { if (!d.data().agentId) queueDepth++; });

    const errorSnap = await db.collection('systemEvents').where('severity', '==', 'error').get();
    const totalEventsSnap = await db.collection('systemEvents').get();
    const fallbackRate = totalEventsSnap.size > 0 ? (errorSnap.size / totalEventsSnap.size) * 100 : 0;

    return {
        activeCalls: activeCallsSnap.size,
        avgMOS: avgMOS.toFixed(2),
        slaPercent,
        apiFallbackRate: fallbackRate.toFixed(1),
        aiAgentsAvailable: agentsSnap.size || 0,
        humanAgentsAvailable: 2,
        queueDepth,
        events: queryToArray(eventsSnap),
    };
}

async function sendDashboardUpdate(ws) {
    try {
        const data = await buildKPIPayload();
        if (ws.readyState === 1) ws.send(JSON.stringify({ type: 'kpi', data }));
    } catch { }
}
