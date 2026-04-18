import { db } from '../firebase.js';
import crypto from 'crypto';
import jwt from 'jsonwebtoken';

const JWT_SECRET = process.env.JWT_SECRET || 'callex-enterprise-secret-2025';

/**
 * JWT Auth middleware
 */
export function requireAuth(req, res, next) {
    const authHeader = req.headers.authorization;
    if (!authHeader || !authHeader.startsWith('Bearer ')) {
        return res.status(401).json({ error: 'Authentication required' });
    }
    const token = authHeader.split(' ')[1];
    try {
        const payload = jwt.verify(token, JWT_SECRET);
        req.userId = payload.userId;
        req.userEmail = payload.email;
        req.userRole = payload.role;
        next();
    } catch (err) {
        return res.status(401).json({ error: 'Invalid or expired token' });
    }
}

export async function requireApiKey(req, res, next) {
    let token = '';
    const authHeader = req.headers.authorization;

    if (authHeader) {
        if (/^bearer /i.test(authHeader)) {
            token = authHeader.split(' ')[1];
        } else {
            // Support raw API key in Authorization header without 'Bearer ' prefix
            token = authHeader.trim();
        }
    } else if (req.headers['x-api-key']) {
        token = req.headers['x-api-key'];
    } else if (req.headers['api-key']) {
        token = req.headers['api-key'];
    }

    if (!token) {
        return res.status(401).json({ error: 'Missing or invalid Authorization header.' });
    }

    try {
        // Try API key lookup first
        const keyHash = crypto.createHash('sha256').update(token).digest('hex');
        const snap = await db.collection('apiKeys').where('keyHash', '==', keyHash).limit(1).get();

        if (!snap.empty) {
            const apiKey = { id: snap.docs[0].id, ...snap.docs[0].data() };
            if (!apiKey.active) return res.status(403).json({ error: 'Invalid or revoked API Key.' });

            // Update lastUsed asynchronously
            db.collection('apiKeys').doc(apiKey.id).update({ lastUsed: new Date() }).catch(e => console.error('[AUTH ERROR]', e));

            req.apiUser = { userId: apiKey.userId, env: apiKey.env, keyId: apiKey.id };
            req.userId = apiKey.userId;
            return next();
        }

        // Fallback: try JWT token (allows dashboard users to use /v1/ endpoints too)
        try {
            const payload = jwt.verify(token, JWT_SECRET);
            req.apiUser = { userId: payload.userId, env: 'production', keyId: null };
            req.userId = payload.userId;
            req.userEmail = payload.email;
            req.userRole = payload.role;
            return next();
        } catch (_jwtErr) {
            // Neither API key nor JWT matched
            return res.status(403).json({ error: 'Invalid or revoked API Key.' });
        }
    } catch (e) {
        console.error('[AUTH ERROR]', e);
        return res.status(500).json({ error: 'Internal server error during authentication' });
    }
}
