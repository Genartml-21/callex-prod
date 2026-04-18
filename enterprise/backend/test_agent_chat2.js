(async () => {
    try {
        const agentsRes = await fetch('http://localhost:4000/api/agents');
        const agents = await agentsRes.json();
        const agentId = agents[0].id;
        console.log("Using agent ID:", agentId);
        
        const chatRes = await fetch('http://localhost:4000/api/simulation/agent-chat', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ agentId, history: [], userText: "hello debugging" })
        });
        console.log("Chat Status:", chatRes.status);
    } catch(e) {
        console.error(e);
    }
})();
