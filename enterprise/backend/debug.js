const fetch = (...args) => import('node-fetch').then(({default: fetch}) => fetch(...args));
async function run() {
    try {
        console.log("Testing POST to agent-chat again to trigger new logs...");
        const res = await fetch('http://localhost:4000/api/simulation/agent-chat', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ agentId: "1", history: [], userText: "" }) // Will trigger Agent Not Found but write into logs
        });
        console.log("Status:", res.status);
    } catch(e) { }
}
run();
