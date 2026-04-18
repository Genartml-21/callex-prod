import { db } from './enterprise/backend/src/firebase.js';

async function cleanupOldActiveCalls() {
    try {
        console.log("Fetching active calls for cleanup...");
        const activeSnap = await db.collection('calls').where('status', '==', 'active').get();
        console.log(`Found ${activeSnap.size} currently active calls.`);

        const now = Date.now();
        // 2 hours in ms
        const MAX_AGE_MS = 2 * 60 * 60 * 1000;
        
        const batch = db.batch();
        let updatedCount = 0;

        activeSnap.forEach(doc => {
            const call = doc.data();
            const startedAt = call.startedAt?.toDate ? call.startedAt.toDate().getTime() : new Date(call.startedAt || 0).getTime();
            
            // If the call is older than 2 hours, it's definitely a ghost/stuck call
            if (now - startedAt > MAX_AGE_MS) {
                batch.update(doc.ref, { status: 'completed', endedAt: new Date(), outcome: 'system_cleanup' });
                updatedCount++;
            }
        });

        if (updatedCount > 0) {
            await batch.commit();
            console.log(`Successfully cleaned up ${updatedCount} stuck ghost calls, marking them as completed.`);
        } else {
            console.log("No old ghost calls to clean up.");
        }
    } catch (e) {
        console.error("Cleanup error:", e);
    }
    process.exit(0);
}

cleanupOldActiveCalls();
