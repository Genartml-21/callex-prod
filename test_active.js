import { db } from './enterprise/backend/src/firebase.js';

async function run() {
    const snap = await db.collection('calls').where('status', '==', 'active').get();
    console.log(`Found ${snap.size} active calls`);
    snap.forEach(doc => console.log(doc.id, doc.data()));
    process.exit(0);
}
run();
