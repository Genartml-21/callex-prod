import urllib.request, json, os, base64

API_KEY = "cd718a342035a5899d3716cfbfcb43cf7de2cad066d217aed8dbd768bd501d2a"

voices = [
    {"name": "Devi", "voice_id": "MF4J4IDTRo0AxOO4dpFR"},
    {"name": "Monika", "voice_id": "1qEiC6qsybMkmnNdVMbK"},
    {"name": "Taksh", "voice_id": "qDuRKMlYmrm8trt5QyBn"},
    {"name": "Parveen", "voice_id": "LQ2auZHpAQ9h4azztqMT"},
    {"name": "Arvi", "voice_id": "s6cZdgI3j07hf4frz4Q8"}
]

text = "नमस्ते, मैं Callex हूँ। मैं आपकी कैसे मदद कर सकता हूँ?"

output_dir = "enterprise/frontend/public/voices"
os.makedirs(output_dir, exist_ok=True)

for voice in voices:
    _tts_base = base64.b64decode(b'aHR0cHM6Ly9hcGkuZWxldmVubGFicy5pby92MS90ZXh0LXRvLXNwZWVjaC8=').decode()
    url = f"{_tts_base}{voice['voice_id']}"
    headers = {
        "Accept": "audio/mpeg",
        "Content-Type": "application/json",
        "xi-api-key": API_KEY
    }
    data = {
        "text": text,
        "model_id": "eleven_multilingual_v2",
        "voice_settings": {
            "stability": 0.5,
            "similarity_boost": 0.5
        }
    }
    
    req = urllib.request.Request(url, headers=headers, data=json.dumps(data).encode('utf-8'))
    
    filepath = os.path.join(output_dir, f"{voice['name'].lower()}.mp3")
    print(f"Generating audio for {voice['name']} -> {filepath}...")
    
    try:
        with urllib.request.urlopen(req) as response:
            with open(filepath, 'wb') as f:
                f.write(response.read())
            print(f"Success: {filepath}")
    except Exception as e:
        print(f"Error for {voice['name']}:", e)

print("Done generating indian previews.")
