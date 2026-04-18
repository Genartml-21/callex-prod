import asyncio
import time
import numpy as np
import webrtcvad
from faster_whisper import WhisperModel
from concurrent.futures import ThreadPoolExecutor

class LocalWhisperSTT:
    """
    1-to-1 Drop-in Replacement for SarvamStreamingSTT.
    Uses Faster-Whisper locally for secure, ultra-fast real-time Hindi open-source STT.
    """
    
    # Cache model at class-level so it doesn't reload into GPU for every call
    _global_model = None

    def __init__(
        self,
        api_key: str = None, # Left for compatibility, unused
        on_transcript=None,
        on_speech_started=None,
        on_speech_ended=None,
        model: str = "large-v3", # Use fast open-source model
        language: str = "hi", # 'hi' for Hindi in Whisper
        mode: str = "transcribe",
        sample_rate: int = 16000,
        vad_signals: bool = True,
        high_vad_sensitivity: bool = True,
        key_manager=None, # unused
    ):
        self._on_transcript = on_transcript
        self._on_speech_started = on_speech_started
        self._on_speech_ended = on_speech_ended
        self._language = language[:2] if language else "hi"  # Whisper expects 'hi', not 'hi-IN'
        self._sample_rate = sample_rate
        
        # Determine aggressiveness (3 is most aggressive for filtering non-speech)
        self._vad = webrtcvad.Vad(3 if high_vad_sensitivity else 2)
        
        self._audio_buffer = bytearray()
        self._is_connected = False
        self._is_speaking = False
        self._silence_frames = 0
        
        self._executor = ThreadPoolExecutor(max_workers=3)
        
        # Load global model if not loaded
        if LocalWhisperSTT._global_model is None:
            print(f"[Callex AI STT] Loading Faster-Whisper '{model}' on 'auto' device...")
            # Use FP16 if GPU, else int8
            try:
                LocalWhisperSTT._global_model = WhisperModel(model, device="auto", compute_type="default")
                print(f"[Callex AI STT] ✅ Model loaded successfully.")
            except Exception as e:
                print(f"[Callex AI STT] ⚠️ Fallback to small model due to error: {e}")
                LocalWhisperSTT._global_model = WhisperModel("small", device="cpu", compute_type="int8")

    @property
    def is_connected(self) -> bool:
        return self._is_connected
        
    async def connect(self):
        self._is_connected = True
        print(f"[Callex AI STT] ✅ Connected (Local instance initialized for language={self._language})")
        
    def send_audio(self, pcm16_bytes: bytes):
        """Append incoming audio bytes to buffer and process chunks for VAD."""
        if not self._is_connected or not pcm16_bytes:
            return
            
        self._audio_buffer.extend(pcm16_bytes)
        
        # webrtcvad requires 10, 20 or 30ms frames. 
        # 20ms at 16000Hz = 320 samples = 640 bytes (16-bit PCM)
        CHUNK_SIZE = 640
        
        # Read from tail of buffer to check current VAD strictly on latest chunk
        if len(pcm16_bytes) >= CHUNK_SIZE:
            chunk = pcm16_bytes[:CHUNK_SIZE]
            try:
                is_speech = self._vad.is_speech(chunk, self._sample_rate)
            except Exception:
                is_speech = False
                
            if is_speech:
                self._silence_frames = 0
                if not self._is_speaking:
                    self._is_speaking = True
                    if self._on_speech_started:
                        asyncio.create_task(self._on_speech_started())
            else:
                self._silence_frames += 1
                # If we have ~ 40 frames of silence (0.8 seconds), count as ended
                if self._is_speaking and self._silence_frames > 40:
                    self._is_speaking = False
                    if self._on_speech_ended:
                        asyncio.create_task(self._on_speech_ended())
                    
                    # Trigger transcript asynchronously so we don't block audio loop
                    buffer_to_process = bytearray(self._audio_buffer)
                    self._audio_buffer.clear()
                    asyncio.create_task(self._process_transcript(buffer_to_process))

    def send_flush(self):
        """Force flush buffer and transcribe immediately."""
        if len(self._audio_buffer) > 6400: # ~0.2 seconds min
            buffer_to_process = bytearray(self._audio_buffer)
            self._audio_buffer.clear()
            self._is_speaking = False
            asyncio.create_task(self._process_transcript(buffer_to_process))

    async def _process_transcript(self, audio_data: bytearray):
        if not audio_data or len(audio_data) < 16000: # Ignore audio fragments < 0.5 sec
            return
            
        # Convert PCM16 bytes to numpy array of floats for Faster-Whisper Model (-1.0 to 1.0)
        audio_np = np.frombuffer(audio_data, dtype=np.int16).astype(np.float32) / 32768.0
        
        def run_model():
            start = time.time()
            segments, _ = LocalWhisperSTT._global_model.transcribe(audio_np, beam_size=1, language=self._language, condition_on_previous_text=False)
            text = " ".join([seg.text for seg in segments]).strip()
            latency = time.time() - start
            return text, latency
            
        # Run inference in a threadpool so the main asyncio VoIP loop isn't blocked!
        loop = asyncio.get_running_loop()
        try:
            text, latency = await loop.run_in_executor(self._executor, run_model)
        except Exception as e:
            print(f"[Callex AI STT] ❌ Inference failed: {e}")
            return
            
        if text and self._on_transcript:
             audio_dur = len(audio_data) / 2 / self._sample_rate
             print(f"[Whisper STT] 📝 Transcript: '{text[:80]}' (latency={latency:.2f}s, audio={audio_dur:.2f}s)")
             await self._on_transcript(text)
             
    async def disconnect(self):
        self._is_connected = False
        self._executor.shutdown(wait=False)
        print("[Whisper STT] 🔌 Disconnected")
