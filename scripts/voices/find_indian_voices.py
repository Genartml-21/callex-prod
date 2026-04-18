import urllib.request, json

API_KEY = "cd718a342035a5899d3716cfbfcb43cf7de2cad066d217aed8dbd768bd501d2a"

import base64
_voice_api_base = base64.b64decode(b'aHR0cHM6Ly9hcGkuZWxldmVubGFicy5pby92MS92b2ljZXM=').decode()
url = _voice_api_base
headers = {"xi-api-key": API_KEY}
req = urllib.request.Request(url, headers=headers)

try:
    with urllib.request.urlopen(req) as response:
        voices = json.loads(response.read())['voices']
        indian_voices = []
        for v in voices:
            labels = v.get('labels', {})
            accent = labels.get('accent', '').lower()
            if 'indian' in accent or 'india' in accent:
                indian_voices.append(v)
        
        print(f"Found {len(indian_voices)} Indian voices")
        for i, v in enumerate(indian_voices[:6]): # get first 6
            print(f"{v['name']} | ID: {v['voice_id']} | Labels: {v.get('labels')}")
except Exception as e:
    print('Error:', e)
