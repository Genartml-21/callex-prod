const fetch = (...args) => import('node-fetch').then(({default: fetch}) => fetch(...args));

async function run() {
  try {
    const res = await fetch('http://localhost:4000/api/agents/tts-preview', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        voiceId: 'MF4J4IDTRo0AxOO4dpFR',
        prosodyRate: 1.5,
        prosodyPitch: 1.5
      })
    });
    
    console.log("Status:", res.status);
    if (!res.ok) {
        console.log("Error:", await res.text());
        return;
    }
    
    // Read the first few bytes to verify it's an audio stream
    const buffer = await res.arrayBuffer();
    console.log("Received bytes:", buffer.byteLength);
    console.log("Success! Audio stream generated.");
  } catch (e) {
    console.log("Error:", e);
  }
}
run();
