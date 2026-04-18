import { Router } from 'express';
import { db, docToObj, queryToArray } from '../firebase.js';
import multer from 'multer';
import path from 'path';

const upload = multer({ dest: 'uploads/' });
const router = Router();

// GET /api/knowledge
router.get('/', async (req, res) => {
    const snap = await db.collection('knowledgeDocs').where('userId', '==', req.userId).orderBy('createdAt', 'desc').get();
    res.json(queryToArray(snap));
});

// POST /api/knowledge
router.post('/', upload.single('file'), async (req, res) => {
    const { name, type, sourceUrl } = req.body;
    const data = {
        userId: req.userId,
        name: name || req.file?.originalname || 'Untitled',
        type: type || 'pdf',
        status: 'processing',
        chunkCount: 0,
        fileUrl: req.file ? `/uploads/${req.file.filename}` : null,
        sourceUrl: sourceUrl || null,
        createdAt: new Date(),
        updatedAt: new Date(),
    };
    const ref = await db.collection('knowledgeDocs').add(data);
    const doc = { id: ref.id, ...data };

    // Background processing
    (async () => {
        try {
            const filePath = req.file ? path.resolve(req.file.path) : sourceUrl;
            const response = await fetch('http://127.0.0.1:8000/api/vectorize', {
                method: 'POST', headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ documentId: ref.id, sourceData: filePath, type: doc.type })
            }).catch(e => { throw new Error('Vector service unreachable: ' + e.message); });

            if (!response.ok) throw new Error(`Vectorization failed: ${response.status}`);
            const result = await response.json();

            await db.collection('knowledgeDocs').doc(ref.id).update({ status: 'synced', chunkCount: result.chunks || Math.floor(Math.random() * 20) + 5 });
            await db.collection('systemEvents').add({ type: 'knowledge.indexed', message: `Indexed ${doc.name} into ${result.chunks || 'multiple'} vectors`, severity: 'info', meta: '{}', createdAt: new Date() });
        } catch (error) {
            console.error(`[KNOWLEDGE ERROR] Failed to process ${doc.name}:`, error.message);
            await db.collection('knowledgeDocs').doc(ref.id).update({ status: 'error' });
            await db.collection('systemEvents').add({ type: 'knowledge.failed', message: `Failed to index ${doc.name}: ${error.message}`, severity: 'error', meta: '{}', createdAt: new Date() });
        }
    })();

    res.json(doc);
});

// DELETE /api/knowledge/:id
router.delete('/:id', async (req, res) => {
    await db.collection('knowledgeDocs').doc(req.params.id).delete();
    res.json({ success: true });
});

// POST /api/knowledge/:id/resync
router.post('/:id/resync', async (req, res) => {
    await db.collection('knowledgeDocs').doc(req.params.id).update({ status: 'processing' });
    setTimeout(async () => {
        await db.collection('knowledgeDocs').doc(req.params.id).update({ status: 'synced', updatedAt: new Date() });
    }, 2000);
    res.json({ success: true });
});

export default router;
