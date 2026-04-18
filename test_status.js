import { db } from './enterprise/backend/src/firebase.js';

async function run() {
    console.log("Fetching all calls...");
    const snap = await db.collection('calls').get();
    let statusCounts = {};
    snap.forEach(doc => {
        let status = doc.data().status || 'undefined';
        statusCounts[status] = (statusCounts[status] || 0) + 1;
    });
    console.log("Total calls:", snap.size);
    console.log("Status counts:", statusCounts);
    process.exit(0);
}
run();
