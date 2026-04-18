import { Router } from 'express';
import { db, docToObj, queryToArray } from '../firebase.js';
import bcrypt from 'bcryptjs';

const router = Router();

// Middleware: Super-admin only
function requireSuperAdmin(req, res, next) {
    if (req.userRole !== 'superadmin') return res.status(403).json({ error: 'Super-admin access required' });
    next();
}
router.use(requireSuperAdmin);

// GET /api/admin/stats
router.get('/stats', async (req, res) => {
    try {
        const [usersSnap, agentsSnap, callsSnap, campaignsSnap, apiKeysSnap, activeCallsSnap, docsSnap] = await Promise.all([
            db.collection('users').get(),
            db.collection('agents').get(),
            db.collection('calls').get(),
            db.collection('campaigns').get(),
            db.collection('apiKeys').where('active', '==', true).get(),
            db.collection('calls').where('status', '==', 'active').get(),
            db.collection('knowledgeDocs').get(),
        ]);
        res.json({
            totalUsers: usersSnap.size,
            totalAgents: agentsSnap.size,
            totalCalls: callsSnap.size,
            totalCampaigns: campaignsSnap.size,
            totalApiKeys: apiKeysSnap.size,
            activeCalls: activeCallsSnap.size,
            totalDocs: docsSnap.size,
        });
    } catch (e) {
        console.error('[ADMIN] Stats error:', e);
        res.status(500).json({ error: 'Failed to fetch stats' });
    }
});

// GET /api/admin/users
router.get('/users', async (req, res) => {
    try {
        const usersSnap = await db.collection('users').orderBy('createdAt', 'desc').get();
        const users = queryToArray(usersSnap);

        const enriched = await Promise.all(users.map(async u => {
            const [agents, campaigns, calls, apiKeys, docs, webhooks, rules, followups] = await Promise.all([
                db.collection('agents').where('userId', '==', u.id).get(),
                db.collection('campaigns').where('userId', '==', u.id).get(),
                db.collection('calls').where('userId', '==', u.id).get(),
                db.collection('apiKeys').where('userId', '==', u.id).get(),
                db.collection('knowledgeDocs').where('userId', '==', u.id).get(),
                db.collection('webhooks').where('userId', '==', u.id).get(),
                db.collection('routingRules').where('userId', '==', u.id).get(),
                db.collection('followUps').where('userId', '==', u.id).get(),
            ]);
            return {
                id: u.id, email: u.email, name: u.name, role: u.role, createdAt: u.createdAt,
                agents: agents.size, campaigns: campaigns.size, calls: calls.size,
                apiKeys: apiKeys.size, knowledgeDocs: docs.size, webhooks: webhooks.size,
                routingRules: rules.size, followUps: followups.size,
            };
        }));
        res.json(enriched);
    } catch (e) {
        console.error('[ADMIN] Users error:', e);
        res.status(500).json({ error: 'Failed to fetch users' });
    }
});

// GET /api/admin/users/:id
router.get('/users/:id', async (req, res) => {
    try {
        const doc = await db.collection('users').doc(req.params.id).get();
        const user = docToObj(doc);
        if (!user) return res.status(404).json({ error: 'User not found' });

        const [agents, campaigns, apiKeys, calls, docs, webhooks, followups] = await Promise.all([
            db.collection('agents').where('userId', '==', user.id).get(),
            db.collection('campaigns').where('userId', '==', user.id).get(),
            db.collection('apiKeys').where('userId', '==', user.id).get(),
            db.collection('calls').where('userId', '==', user.id).get(),
            db.collection('knowledgeDocs').where('userId', '==', user.id).get(),
            db.collection('webhooks').where('userId', '==', user.id).get(),
            db.collection('followUps').where('userId', '==', user.id).get(),
        ]);

        res.json({
            user: { id: user.id, email: user.email, name: user.name, role: user.role, createdAt: user.createdAt },
            agents: queryToArray(agents), campaigns: queryToArray(campaigns),
            apiKeys: queryToArray(apiKeys),
            recentCalls: queryToArray(calls).sort((a, b) => {
                const da = a.startedAt?.toDate ? a.startedAt.toDate().getTime() : new Date(a.startedAt || 0).getTime();
                const db2 = b.startedAt?.toDate ? b.startedAt.toDate().getTime() : new Date(b.startedAt || 0).getTime();
                return db2 - da;
            }).slice(0, 50),
            knowledgeDocs: queryToArray(docs), webhooks: queryToArray(webhooks), followups: queryToArray(followups),
        });
    } catch (e) {
        console.error('[ADMIN] User detail error:', e);
        res.status(500).json({ error: 'Failed to fetch user details' });
    }
});

// POST /api/admin/users
router.post('/users', async (req, res) => {
    try {
        const { email, name, password, role } = req.body;
        if (!email || !password) return res.status(400).json({ error: 'Email and password required' });

        const existing = await db.collection('users').where('email', '==', email).limit(1).get();
        if (!existing.empty) return res.status(409).json({ error: 'User with this email already exists' });

        const hashed = await bcrypt.hash(password, 10);
        const ref = await db.collection('users').add({ email, name: name || email.split('@')[0], password: hashed, role: role || 'user', createdAt: new Date() });
        res.json({ id: ref.id, email, name: name || email.split('@')[0], role: role || 'user', createdAt: new Date() });
    } catch (e) {
        console.error('[ADMIN] Create user error:', e);
        res.status(500).json({ error: 'Failed to create user' });
    }
});

// PATCH /api/admin/users/:id
router.patch('/users/:id', async (req, res) => {
    try {
        const { name, role } = req.body;
        const data = {};
        if (name) data.name = name;
        if (role) data.role = role;
        await db.collection('users').doc(req.params.id).update(data);
        const doc = await db.collection('users').doc(req.params.id).get();
        const user = docToObj(doc);
        res.json({ id: user.id, email: user.email, name: user.name, role: user.role });
    } catch (e) {
        console.error('[ADMIN] Update user error:', e);
        res.status(500).json({ error: 'Failed to update user' });
    }
});

// DELETE /api/admin/users/:id
router.delete('/users/:id', async (req, res) => {
    try {
        const doc = await db.collection('users').doc(req.params.id).get();
        const user = docToObj(doc);
        if (!user) return res.status(404).json({ error: 'User not found' });
        if (user.role === 'superadmin') return res.status(403).json({ error: 'Cannot delete super-admin' });

        // Delete user's data
        const collections = ['apiKeys', 'webhooks', 'followUps', 'knowledgeDocs', 'routingRules', 'campaigns'];
        for (const col of collections) {
            const snap = await db.collection(col).where('userId', '==', user.id).get();
            const batch = db.batch();
            snap.forEach(d => batch.delete(d.ref));
            if (!snap.empty) await batch.commit();
        }

        // Delete agents and their prompt versions
        const agentsSnap = await db.collection('agents').where('userId', '==', user.id).get();
        if (!agentsSnap.empty) {
            for (const agentDoc of agentsSnap.docs) {
                const pvSnap = await db.collection('promptVersions').where('agentId', '==', agentDoc.id).get();
                const pvBatch = db.batch();
                pvSnap.forEach(d => pvBatch.delete(d.ref));
                if (!pvSnap.empty) await pvBatch.commit();
                await agentDoc.ref.delete();
            }
        }

        await db.collection('users').doc(req.params.id).delete();
        res.json({ success: true, message: `User ${user.email} and all their data deleted` });
    } catch (e) {
        console.error('[ADMIN] Delete user error:', e);
        res.status(500).json({ error: 'Failed to delete user' });
    }
});

// GET /api/admin/agents
router.get('/agents', async (req, res) => {
    try {
        const snap = await db.collection('agents').orderBy('createdAt', 'desc').get();
        const agents = [];
        for (const doc of snap.docs) {
            const agent = { id: doc.id, ...doc.data() };
            if (agent.userId) {
                const userDoc = await db.collection('users').doc(agent.userId).get();
                agent.user = userDoc.exists ? { id: userDoc.id, email: userDoc.data().email, name: userDoc.data().name } : null;
            }
            agents.push(agent);
        }
        res.json(agents);
    } catch (e) {
        console.error('[ADMIN] Agents error:', e);
        res.status(500).json({ error: 'Failed to fetch agents' });
    }
});

// PATCH /api/admin/agents/:id
router.patch('/agents/:id', async (req, res) => {
    try {
        const { prosodyRate, llmModel, patienceMs, bargeInMode } = req.body;
        const data = {};
        if (prosodyRate !== undefined) data.prosodyRate = prosodyRate;
        if (llmModel !== undefined) data.llmModel = llmModel;
        if (patienceMs !== undefined) data.patienceMs = patienceMs;
        if (bargeInMode !== undefined) data.bargeInMode = bargeInMode;
        data.updatedAt = new Date();

        await db.collection('agents').doc(req.params.id).update(data);
        const doc = await db.collection('agents').doc(req.params.id).get();
        res.json(docToObj(doc));
    } catch (e) {
        console.error('[ADMIN] Update agent error:', e);
        res.status(500).json({ error: 'Failed to update agent' });
    }
});

// POST /api/admin/maintenance
router.post('/maintenance', async (req, res) => {
    try {
        const { durationMinutes = 60 } = req.body;
        const until = new Date(Date.now() + durationMinutes * 60 * 1000);

        await db.collection('systemEvents').add({
            type: 'maintenance',
            severity: 'warning',
            message: `Maintenance mode activated until ${until.toLocaleTimeString()}. All agents paused for ${durationMinutes} minutes.`,
            meta: '{}',
            createdAt: new Date(),
        });

        // Pause all active agents
        const activeSnap = await db.collection('agents').where('status', '==', 'active').get();
        const batch = db.batch();
        activeSnap.forEach(d => batch.update(d.ref, { status: 'paused', updatedAt: new Date() }));
        if (!activeSnap.empty) await batch.commit();

        res.json({ success: true, maintenanceUntil: until, agentsPaused: true });
    } catch (e) {
        console.error('[ADMIN] Maintenance error:', e);
        res.status(500).json({ error: 'Failed to activate maintenance mode' });
    }
});

// GET /api/admin/agents-by-user — Agents grouped by user
router.get('/agents-by-user', async (req, res) => {
    try {
        const usersSnap = await db.collection('users').orderBy('createdAt', 'desc').get();
        const result = [];

        for (const userDoc of usersSnap.docs) {
            const u = { id: userDoc.id, ...userDoc.data() };
            const agentsSnap = await db.collection('agents').where('userId', '==', u.id).get();
            const agents = queryToArray(agentsSnap);
            agents.sort((a, b) => new Date(b.createdAt || 0) - new Date(a.createdAt || 0));
            result.push({
                user: { id: u.id, email: u.email, name: u.name, role: u.role },
                agents,
                totalAgents: agents.length,
            });
        }
        res.json(result);
    } catch (e) {
        console.error('[ADMIN] Agents by user error:', e);
        res.status(500).json({ error: 'Failed to fetch agents by user' });
    }
});

// GET /api/admin/users/:userId/export — Full data export for a specific user
router.get('/users/:userId/export', async (req, res) => {
    try {
        const userDoc = await db.collection('users').doc(req.params.userId).get();
        if (!userDoc.exists) return res.status(404).json({ error: 'User not found' });
        const user = { id: userDoc.id, ...userDoc.data() };
        delete user.password;

        const [agentsSnap, callsSnap, campaignsSnap, apiKeysSnap, docsSnap, webhooksSnap, rulesSnap, followupsSnap] = await Promise.all([
            db.collection('agents').where('userId', '==', user.id).get(),
            db.collection('calls').where('userId', '==', user.id).get(),
            db.collection('campaigns').where('userId', '==', user.id).get(),
            db.collection('apiKeys').where('userId', '==', user.id).get(),
            db.collection('knowledgeDocs').where('userId', '==', user.id).get(),
            db.collection('webhooks').where('userId', '==', user.id).get(),
            db.collection('routingRules').where('userId', '==', user.id).get(),
            db.collection('followUps').where('userId', '==', user.id).get(),
        ]);

        const agents = queryToArray(agentsSnap);
        const calls = queryToArray(callsSnap);
        const campaigns = queryToArray(campaignsSnap);

        // Get calls linked to this user's agents
        const agentIds = agents.map(a => a.id);
        let agentCalls = [];
        if (agentIds.length > 0) {
            for (let i = 0; i < agentIds.length; i += 30) {
                const chunk = agentIds.slice(i, i + 30);
                const snap = await db.collection('calls').where('agentId', 'in', chunk).get();
                agentCalls = agentCalls.concat(queryToArray(snap));
            }
        }
        // Combine user's direct calls + agent-related calls (deduplicate)
        const allCallIds = new Set(calls.map(c => c.id));
        agentCalls.forEach(c => { if (!allCallIds.has(c.id)) calls.push(c); });

        const exportData = {
            exportDate: new Date().toISOString(),
            user,
            summary: {
                totalAgents: agents.length,
                totalCalls: calls.length,
                totalCampaigns: campaigns.length,
                totalApiKeys: queryToArray(apiKeysSnap).length,
                totalDocs: queryToArray(docsSnap).length,
                totalWebhooks: queryToArray(webhooksSnap).length,
                totalRules: queryToArray(rulesSnap).length,
                totalFollowUps: queryToArray(followupsSnap).length,
            },
            agents: agents.map(a => {
                const { userId, ...rest } = a;
                return rest;
            }),
            calls: calls.map(c => ({
                id: c.id,
                phoneNumber: c.phoneNumber,
                direction: c.direction || 'inbound',
                status: c.status,
                duration: c.duration,
                sentiment: c.sentiment,
                startedAt: c.startedAt,
                endedAt: c.endedAt,
                summary: c.summary,
                transcript: c.transcript,
                recordingUrl: c.recordingUrl,
                agentId: c.agentId,
                campaignId: c.campaignId,
            })),
            campaigns: queryToArray(campaignsSnap).map(c => {
                const { userId, ...rest } = c;
                return rest;
            }),
            apiKeys: queryToArray(apiKeysSnap).map(k => ({
                id: k.id, name: k.name, prefix: k.prefix, env: k.env,
                active: k.active, createdAt: k.createdAt, lastUsed: k.lastUsed,
            })),
            knowledgeDocs: queryToArray(docsSnap),
            webhooks: queryToArray(webhooksSnap),
            routingRules: queryToArray(rulesSnap),
            followUps: queryToArray(followupsSnap),
        };

        res.setHeader('Content-Type', 'application/json');
        res.setHeader('Content-Disposition', `attachment; filename="user_${user.email}_export.json"`);
        res.json(exportData);
    } catch (e) {
        console.error('[ADMIN] User export error:', e);
        res.status(500).json({ error: 'Failed to export user data' });
    }
});

// GET /api/admin/users/:userId/calls — Get call logs for a specific user
router.get('/users/:userId/calls', async (req, res) => {
    try {
        const { startDate, endDate } = req.query;
        const agentsSnap = await db.collection('agents').where('userId', '==', req.params.userId).get();
        const agentIds = agentsSnap.docs.map(d => d.id);

        let calls = [];
        // Get calls by userId
        const directSnap = await db.collection('calls').where('userId', '==', req.params.userId).get();
        calls = queryToArray(directSnap);

        // Also get calls by agentId
        const callIds = new Set(calls.map(c => c.id));
        if (agentIds.length > 0) {
            for (let i = 0; i < agentIds.length; i += 30) {
                const chunk = agentIds.slice(i, i + 30);
                const snap = await db.collection('calls').where('agentId', 'in', chunk).get();
                snap.docs.forEach(d => {
                    if (!callIds.has(d.id)) {
                        calls.push({ id: d.id, ...d.data() });
                        callIds.add(d.id);
                    }
                });
            }
        }

        // Apply Date Filters
        if (startDate) {
            const startMs = new Date(startDate).getTime();
            if (!isNaN(startMs)) {
                calls = calls.filter(c => {
                    const ts = c.startedAt?.toDate ? c.startedAt.toDate().getTime() : new Date(c.startedAt || 0).getTime();
                    return ts >= startMs;
                });
            }
        }
        if (endDate) {
            const endMs = new Date(endDate).getTime();
            if (!isNaN(endMs)) {
                calls = calls.filter(c => {
                    const ts = c.startedAt?.toDate ? c.startedAt.toDate().getTime() : new Date(c.startedAt || 0).getTime();
                    return ts <= endMs;
                });
            }
        }

        // Enrich with agent names
        for (const call of calls) {
            if (call.agentId) {
                const agentDoc = await db.collection('agents').doc(call.agentId).get();
                call.agentName = agentDoc.exists ? agentDoc.data().name : 'Unknown';
            }
        }

        // Sort by date descending
        calls.sort((a, b) => {
            const da = a.startedAt?.toDate ? a.startedAt.toDate() : new Date(a.startedAt || 0);
            const db2 = b.startedAt?.toDate ? b.startedAt.toDate() : new Date(b.startedAt || 0);
            return db2 - da;
        });

        res.json(calls);
    } catch (e) {
        console.error('[ADMIN] User calls error:', e);
        res.status(500).json({ error: 'Failed to fetch user calls' });
    }
});

export default router;
