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
            # check name
            if "india" in v['name'].lower() or "hindi" in v['name'].lower():
                indian_voices.append(v)
                continue
            
            # check labels
            labels = v.get('labels', {})
            if labels:
                for key, val in labels.items():
                    if "india" in str(val).lower() or "hindi" in str(val).lower():
                        indian_voices.append(v)
                        break
        
        print(f"Found {len(indian_voices)} Indian voices")
        for i, v in enumerate(indian_voices[:6]): # get first 6
            print(f"{v['name']} | ID: {v['voice_id']} | Labels: {v.get('labels')}")
except Exception as e:
    print('Error:', e)
