import os
import re

SAFE_LOG_FUNC = """
def __safe_log(msg) -> str:
    import builtins
    if msg is None: return "None"
    s = builtins.str(msg)
    s = s.replace("sarvam", "cx-stt").replace("Sarvam", "CX-STT").replace("SARVAM", "CX-STT")
    s = s.replace("saaras", "cx-asr-core").replace("Saaras", "CX-ASR-Core")
    s = s.replace("gemini", "cx-model").replace("Gemini", "CX-Model").replace("GEMINI", "CX-MODEL")
    s = s.replace("elevenlabs", "cx-voice").replace("ElevenLabs", "CX-Voice").replace("ELEVENLABS", "CX-VOICE")
    s = s.replace("generativelanguage.googleapis.com", "llm-api.callex.ai")
    s = s.replace("api.sarvam.ai", "stt-api.callex.ai")
    s = s.replace("api.elevenlabs.io", "voice-api.callex.ai")
    s = s.replace("Google GenAI", "CX-AI-Engine")
    return s
"""

def process_file(filepath):
    with open(filepath, 'r') as f:
        content = f.read()

    original = content

    if "__safe_log" not in content:
        # inject after imports
        parts = content.split("import json")
        if len(parts) > 1:
            content = parts[0] + "import json\n" + SAFE_LOG_FUNC + parts[1]
        else:
            content = SAFE_LOG_FUNC + "\n" + content

    # replace {e} with {__safe_log(e)}
    content = re.sub(r'\{e\}', '{__safe_log(e)}', content)
    # replace {last_error} with {__safe_log(last_error)}
    content = re.sub(r'\{last_error\}', '{__safe_log(last_error)}', content)
    # replace r.text[:200]
    content = content.replace("r.text[:200]", "__safe_log(r.text)[:200]")
    # replace {error_msg}
    content = re.sub(r'\{error_msg\}', '{__safe_log(error_msg)}', content)
    # replace str(msg)
    content = content.replace("str(msg)", "__safe_log(msg)")
    content = content.replace("str(raw_msg)", "__safe_log(raw_msg)")

    if content != original:
        with open(filepath, 'w') as f:
            f.write(content)
        print(f"Updated {filepath}")

process_file('app/main.py')
process_file('app/audio/sst_model_2_streaming.py')
