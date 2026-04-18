import { db, docToObj } from '../firebase.js';

export function setupSupervisorWS(ws, callId) {
    let lastTranscriptLength = 0;

    // Listen to real-time updates from Python engine
    const unsubscribe = db.collection('calls').doc(callId).onSnapshot((doc) => {
        if (!doc.exists) return;
        const call = doc.data();
        
        const messages = call.transcriptMessages || [];
        if (messages.length > lastTranscriptLength) {
            // New messages arrived!
            for (let i = lastTranscriptLength; i < messages.length; i++) {
                const msg = messages[i];
                // Frontend LiveSupervisor expects "Bot: " or "User: " prefix
                let prefix = msg.role === 'model' ? 'Bot: ' : 'User: ';
                if (msg.role === 'system' || msg.text.startsWith('[System')) prefix = '';
                
                if (ws.readyState === 1) { // 1 = OPEN
                    ws.send(JSON.stringify({ 
                        type: 'transcript_line', 
                        callId, 
                        line: prefix + msg.text, 
                        ts: msg.timestamp || Date.now() 
                    }));
                }
            }
            lastTranscriptLength = messages.length;
        }
    }, (err) => {
        console.error('[WS] Supervisor snapshot error:', err);
    });

    ws.on('message', (msg) => {
        try {
            const parsed = JSON.parse(msg);
            if (parsed.type === 'whisper') {
                ws.send(JSON.stringify({ type: 'whisper_ack', message: parsed.message, ts: Date.now() }));
                // the HTTP endpoint /calls/:id/whisper already handles adding whisper to DB
            } else if (parsed.type === 'listen_in') {
                ws.send(JSON.stringify({ type: 'listen_ack', ts: Date.now() }));
            }
        } catch { }
    });

    ws.on('close', () => { 
        unsubscribe(); 
    });
}
