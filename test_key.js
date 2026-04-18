import { db } from './enterprise/backend/src/firebase.js';
import crypto from 'crypto';

async function run() {
    const token = 'ck_live_d3188b50_94a44765acfc97d19ced4416';
    const keyHash = crypto.createHash('sha256').update(token).digest('hex');
    const snap = await db.collection('apiKeys').where('keyHash', '==', keyHash).get();
    
    if (snap.empty) {
        console.log("API Key not found!");
    } else {
        const keyData = snap.docs[0].data();
        console.log(`API Key User ID: ${keyData.userId}`);
    }
    process.exit(0);
}
run();
