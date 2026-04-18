(async () => {
    try {
        const agentsRes = await fetch('http://localhost:4000/api/agents');
        const agents = await agentsRes.json();
        const agentId = agents[0].id;
        
        console.log("Testing OUTBOUND CONNECT (Empty UserText):");
        const chatRes1 = await fetch('http://localhost:4000/api/simulation/agent-chat', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ 
                agentId, 
                history: [], 
                userText: "", 
                agentOverride: { language: 'hi-IN', systemPrompt: 'You are calling from DishTV. Remind the user their ₹350 recharge is due today. Convince them to pay online.' }
            })
        });
        console.log("First Turn Status:", chatRes1.status);
        console.log("X-Agent-Text:", chatRes1.headers.get('x-agent-text'));
        
        console.log("\Testing BARGE-IN RESPONSE:");
        const chatRes2 = await fetch('http://localhost:4000/api/simulation/agent-chat', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ 
                agentId, 
                history: [
                    { role: 'model', text: 'Namaste, main DishTV se baat kar rahi hoon. Aapka Rs 350 ka recharge aaj due hai.' }
                ], 
                userText: "mere paas paise nahi hai abhi thoda time do", 
                agentOverride: { language: 'hi-IN', systemPrompt: 'You are calling from DishTV. Remind the user their ₹350 recharge is due today. Convince them to pay online.' }
            })
        });
        console.log("Second Turn Status:", chatRes2.status);
        console.log("X-Agent-Text:", chatRes2.headers.get('x-agent-text'));
        
    } catch(e) {
        console.error(e);
    }
})();
