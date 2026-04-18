(async () => {
    const agentsRes = await fetch('http://localhost:4000/api/agents');
    const agents = await agentsRes.json();
    if (agents.length === 0) { console.log("No agents"); return; }
    const agentId = agents[0].id;
    console.log("Using agent ID:", agentId);
    
    const chatRes = await fetch('http://localhost:4000/api/simulation/agent-chat', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ agentId, history: [], userText: "hello debugging" })
    });
    console.log("Chat Status:", chatRes.status);
    // don't read the whole body of chunked output properly, just check if it fails
    // actually, let's just see if the backend crashes.
})();
