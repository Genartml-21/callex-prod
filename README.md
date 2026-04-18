# Callex AI Voice Assistant — Complete Developer Documentation

> **Production-grade, real-time AI voice calling system** built on FreeSWITCH + Python (FastAPI) + Node.js (Express).
> Handles 100+ concurrent outbound phone calls with sub-second latency using CX-Model LLM, CX STT Engine STT, and Callex-Voice-Engine TTS.

---

## Table of Contents

1.  [High-Level Architecture](#1-high-level-architecture)
2.  [Technology Stack](#2-technology-stack)
3.  [Project Structure (Directory Tree)](#3-project-structure-directory-tree)
4.  [Environment Variables (`.env`)](#4-environment-variables-env)
5.  [Configuration System](#5-configuration-system)
6.  [Core Application (`app/main.py`)](#6-core-application-appmainpy)
7.  [Audio Processing Pipeline (`app/audio/`)](#7-audio-processing-pipeline-appaudio)
8.  [Core Logic Modules (`app/core/`)](#8-core-logic-modules-appcore)
9.  [API Layer (`app/api/`)](#9-api-layer-appapi)
10. [Managers (`app/managers/`)](#10-managers-appmanagers)
11. [Services (`app/services/`)](#11-services-appservices)
12. [Utilities (`app/utils/`)](#12-utilities-apputils)
13. [Enterprise Platform (`enterprise/`)](#13-enterprise-platform-enterprise)
14. [Telemetry Dashboard (`dashboard.html`)](#14-telemetry-dashboard-dashboardhtml)
15. [Database Schema](#15-database-schema)
16. [Firebase / Firestore Integration](#16-firebase--firestore-integration)
17. [API Key Management & Load Balancing](#17-api-key-management--load-balancing)
18. [Call Lifecycle (End-to-End Flow)](#18-call-lifecycle-end-to-end-flow)
19. [Production Scaling & Concurrency](#19-production-scaling--concurrency)
20. [Deployment & Operations](#20-deployment--operations)
21. [Files NOT In Active Use (Dead Code / Legacy)](#21-files-not-in-active-use-dead-code--legacy)
22. [Root-Level Misc Files](#22-root-level-misc-files)
23. [Troubleshooting](#23-troubleshooting)

---

## 1. High-Level Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                        PHONE NETWORK (PSTN)                        │
└──────────────────────────────┬──────────────────────────────────────┘
                               │ SIP / RTP
                    ┌──────────▼──────────┐
                    │     FreeSWITCH      │   ← PBX / Telephony engine
                    │   (SIP + Media)     │
                    └──────────┬──────────┘
                               │ WebSocket (PCM16 audio frames)
                    ┌──────────▼──────────┐
                    │   app/main.py       │   ← Python FastAPI server
                    │  (Voice AI Core)    │
                    └──┬───┬───┬───┬──────┘
                       │   │   │   │
          ┌────────────┘   │   │   └────────────┐
          ▼                ▼   ▼                 ▼
   ┌─────────────┐  ┌──────────┐  ┌──────────┐  ┌─────────────────┐
   │  CX STT Engine  │  │ CX-Model   │  │Callex-Voice-Engine│  │   Firestore     │
   │ STT (ASR)   │  │  LLM     │  │   TTS    │  │  (Agent Config) │
   │ WebSocket   │  │  API     │  │ Streaming│  │                 │
   └─────────────┘  └──────────┘  └──────────┘  └────────┬────────┘
                                                          │
                                               ┌──────────▼──────────┐
                                               │  Enterprise Platform│
                                               │  (Node.js + React)  │
                                               │  Agent Studio / API │
                                               └─────────────────────┘
```

### Data Flow per Call

1. **FreeSWITCH** receives an inbound/outbound SIP call and opens a **WebSocket** to `app/main.py`.
2. The WebSocket handler receives the `agent_id` from FreeSWITCH channel variables.
3. **Agent config** is loaded from **Firestore** (via `agent_loader.py`).
4. The bot **speaks the opening line** using cached or freshly generated **Callex-Voice-Engine TTS** audio.
5. Customer audio frames arrive as base64-encoded PCM16 over WebSocket.
6. Audio passes through the **6-layer audio gating pipeline**: DeepFilterNet3 → Silero VAD → Speaker Verification → YAMNet → Semantic Filter → Duration Gates.
7. Validated speech is streamed to **CX STT Engine Streaming STT** via a persistent WebSocket.
8. Transcripts are checked by `ConversationBrain` for echoes / hallucinations.
9. Valid transcripts are sent to **CX-Model LLM** with conversation history + system prompt + tone instructions.
10. LLM response is sanitized by `ConversationBrain`, then streamed to **Callex-Voice-Engine TTS**.
11. TTS audio chunks are sent back to FreeSWITCH over the same WebSocket → customer hears the response.
12. On call end, **CX-Model** analyzes the full transcript for disposition, sentiment, and structured data.
13. Results are saved to both **SQLite** (local analytics DB) and **Firestore** (enterprise dashboard).

---

## 2. Technology Stack

### Python Backend (Voice AI Core)

| Component | Technology | Purpose |
|---|---|---|
| Web Framework | **FastAPI** + Uvicorn | WebSocket server + REST API |
| LLM | **Google CX-Model 2.5 Flash** | Conversational AI responses |
| STT (Speech-to-Text) | **CX STT Engine** (WebSocket streaming) | Real-time Hindi/English/Gujarati ASR |
| TTS (Text-to-Speech) | **Callex-Voice-Engine** (HTTP streaming) | Voice synthesis with configurable speed/stability |
| VAD (Voice Activity) | **Silero VAD** (PyTorch) | ML-based speech detection |
| Noise Suppression | **DeepFilterNet3** (PyTorch) | Neural noise suppression for traffic/crowd/wind |
| Sound Classification | **YAMNet** (TensorFlow Hub) | Rejects coughs, sneezes, dog barks, mic bumps |
| Speaker Verification | **Resemblyzer** | Identifies caller vs. background voices |
| Telephony | **FreeSWITCH** (via ESL) | SIP gateway, call control, hangup |
| Database (Local) | **SQLite** via SQLAlchemy | Call records, recordings, outcomes |
| Database (Cloud) | **Google Firestore** | Agent configs, call transcripts, enterprise data |
| Storage | **Firebase Storage** | Call recording uploads (signed URLs) |
| Process Manager | **PM2** | Production process management & auto-restart |

### Enterprise Platform (Node.js + React)

| Component | Technology | Purpose |
|---|---|---|
| API Server | **Express.js** (ESM) | Agent CRUD, campaign management, auth |
| Frontend | **React 18** + Vite | Agent Studio dashboard |
| State Management | **Zustand** | Client-side state |
| Charts | **Recharts** | Analytics visualizations |
| UI Framework | **TailwindCSS** | Styling |
| Auth | **JWT** + bcryptjs | User authentication |
| Telephony Control | **modesl** (FreeSWITCH ESL) | Originate calls from dashboard |
| File Upload | **express-fileupload** / Multer | Knowledge base uploads |

---

## 3. Project Structure (Directory Tree)

```
.
├── .env                          # Environment variables (GITIGNORED)
├── .gitignore                    # Git ignore rules
├── requirements.txt              # Python dependencies
├── firebase_credentials.json     # Firebase service account key (GITIGNORED)
├── firestore.rules               # Firestore security rules (open — auth at API layer)
├── bot_config.json               # Runtime bot config (GITIGNORED, auto-generated)
├── background_noise.mp3          # Call center ambiance audio (~14MB)
├── dashboard.html                # Glassmorphic real-time telemetry dashboard
│
├── app/                          # ═══ PYTHON VOICE AI CORE ═══
│   ├── __init__.py
│   ├── main.py                   # ★ MAIN ENTRY POINT (3074 lines) — WebSocket handler,
│   │                             #   TTS streaming, LLM calls, barge-in, recording, everything
│   ├── api/
│   │   ├── __init__.py
│   │   └── routes.py             # REST API: calls, analytics, settings, scripts, CSV export
│   ├── audio/
│   │   ├── __init__.py
│   │   ├── README.md             # (Empty placeholder)
│   │   ├── call_context.py       # Per-call audio pipeline isolation (bundles all per-call state)
│   │   ├── classifier.py         # YAMNet sound event classifier (Layer 5.5)
│   │   ├── deepfilter_denoiser.py# DeepFilterNet3 neural noise suppression
│   │   ├── gating.py             # 6-layer audio gating pipeline (AudioGatingPipeline)
│   │   ├── semantic.py           # Semantic speech filter (filler word rejection)
│   │   ├── speaker_verifier.py   # Resemblyzer-based caller voice identification
│   │   ├── sst_model_2_streaming.py # CX STT Engine WebSocket streaming STT client
│   │   ├── vad_silero.py         # Silero VAD model wrapper with hysteresis
│   │   └── verification.py       # ⚠️ LEGACY — pyannote.audio speaker verifier (replaced by speaker_verifier.py)
│   ├── core/
│   │   ├── __init__.py
│   │   ├── agent_loader.py       # Firestore agent config loader (bridge between DB and calls)
│   │   ├── config.py             # Central config constants (DB URL, recording paths, Firebase)
│   │   ├── config_manager.py     # Pydantic-based bot_config.json manager (VAD, voice, API keys)
│   │   ├── conversation_brain.py # Per-call conversation state + anti-hallucination engine
│   │   ├── database.py           # SQLAlchemy ORM models (Call, Recording, CallOutcome)
│   │   ├── db.py                 # Async Firestore wrappers (thread-pooled)
│   │   ├── fast_reply_cache.py   # Zero-latency FAQ cache (~5ms replies bypassing LLM)
│   │   └── tone_analyzer.py      # Real-time NLP emotion detection + adaptive LLM instructions
│   ├── managers/
│   │   ├── __init__.py
│   │   ├── recordings.py         # WAV file recording manager with retention cleanup
│   │   └── scripts.py            # S3-backed script CRUD manager (legacy)
│   ├── services/
│   │   ├── __init__.py
│   │   └── analytics.py          # Post-call AI analysis (CX-Model), transcript export, auto-training
│   └── utils/
│       ├── __init__.py
│       └── logger.py             # Call tracker: DB records, outcome detection, conversation logging
│
├── enterprise/                   # ═══ ENTERPRISE DASHBOARD PLATFORM ═══
│   ├── start.sh                  # Starts both backend (port 4000) and frontend (port 3000)
│   ├── backend/
│   │   ├── package.json
│   │   ├── .env                  # Backend env (Firebase creds, JWT secret, ports)
│   │   └── src/
│   │       ├── index.js          # Express server entry point
│   │       ├── firebase.js       # Firebase Admin SDK init
│   │       ├── middleware/       # Auth middleware (JWT verification)
│   │       ├── routes/           # REST routes (agents, calls, campaigns, auth, dispositions)
│   │       ├── services/         # Business logic services
│   │       └── ws/               # WebSocket proxy for live call monitoring
│   └── frontend/
│       ├── package.json
│       ├── vite.config.js
│       ├── tailwind.config.js
│       ├── vercel.json           # Vercel deployment config
│       └── src/                  # React app (Agent Studio, Call Logs, Analytics)
│
├── data/
│   └── call_recordings.db        # SQLite database (auto-created, GITIGNORED)
│
├── cache/
│   └── *.pcm                     # Cached opener audio files (content-hash named)
│
├── recordings/                   # Local call recording WAV files (GITIGNORED)
│
├── scripts/                      # ═══ DEVELOPMENT SCRIPTS ═══
│   └── deployment/               # Deployment helper scripts
│   └── voices/                   # Voice configuration scripts
│
├── _archive/                     # ═══ ARCHIVED/UNUSED FILES ═══ (GITIGNORED)
│   ├── main.py                   # Old monolithic main.py backup
│   ├── test copy 4.py            # Old test file backup
│   └── ...                       # Various old/superseded files
│
└── [Root-level misc files]       # See Section 22
```

---

## 4. Environment Variables (`.env`)

The `.env` file is loaded by `python-dotenv` at startup. **Never commit this file.** It is in `.gitignore`.

| Variable | Purpose | Example |
|---|---|---|
| `GENARTML_SECRET_KEY` | Callex-Voice-Engine TTS API secret key (hexadecimal) | `ebc0cf6c...` |
| `CALLEX_VOICE_KEY_1` through `_5` | Callex-Voice-Engine API key pool for round-robin load balancing | `030a62b1...` |
| `SST_MODEL_2_API_KEY_1` through `_5` | CX STT Engine STT API key pool for round-robin | `sk_kgi72r...` |
| `AWS_ACCESS_KEY_ID` | AWS credentials for S3 script storage | `AKIA6QKR...` |
| `AWS_SECRET_ACCESS_KEY` | AWS secret for S3 | `aC4jt4ij...` |
| `FIREBASE_CREDENTIALS_PATH` | Path to Firebase service account JSON | `firebase_credentials.json` |
| `FIREBASE_STORAGE_BUCKET` | Firebase Storage bucket name | `lakhuteleservices-1f9e0.firebasestorage.app` |
| `FS_PASSWORD` | FreeSWITCH ESL password | `ClueCon` |
| `DEEPGRAM_API_KEY` | Deepgram backup ASR key (currently unused) | `22db3ee2...` |
| `CALLEX_WS_AUTH_TOKEN` | WebSocket authentication token (commented out) | Not set |
| `CORS_ORIGINS` | Allowed CORS origins (commented out, defaults to `*`) | Not set |

### Variables NOT in `.env` but used in code

| Variable | Where Used | Default |
|---|---|---|
| `CX-MODEL_API_KEY_1` through `_4` | `app/main.py` — CX-Model LLM key pool | Empty (falls back to `bot_config.json`) |
| `FS_HOST` | `app/main.py` — FreeSWITCH ESL host | `127.0.0.1` |
| `FS_PORT` | `app/main.py` — FreeSWITCH ESL port | `8021` |
| `TTS_MAX_CONCURRENT` | `app/main.py` — TTS concurrency limiter | `15` |
| `DATABASE_URL` | `app/core/config.py` — SQLAlchemy DB URL | `sqlite:///data/call_recordings.db` |
| `RECORDINGS_DIR` | `app/core/config.py` — Recording save path | `<project_root>/recordings` |
| `RETENTION_DAYS` | `app/core/config.py` — Recording retention | `90` |
| `AWS_REGION` | `app/core/config.py` | `ap-south-1` |
| `AWS_BUCKET_NAME` | `app/core/config.py` | `callex-recordings` |

---

## 5. Configuration System

The system has **three layers** of configuration:

### Layer 1: `bot_config.json` (Runtime Config)

Managed by `app/core/config_manager.py`. This file is auto-generated and **gitignored**.

```json
{
  "vad": {
    "min_speech_duration": 0.1,
    "silence_timeout": 1.5,
    "interruption_threshold_db": -35.0,
    "noise_gate_db": -55.0,
    "spectral_flatness_threshold": 0.55
  },
  "api_credentials": {
    "server_key": "<cx-model-api-key>",
    "secret_key": "<cx-voice-secret-key>",
    "voice_id": "<cx-voice-voice-id>"
  },
  "voice": {
    "speed": 1.25,
    "stability": 0.85,
    "similarity_boost": 0.80,
    "style": 0.0,
    "use_speaker_boost": true
  }
}
```

**Important**: The `ConfigManager` class uses **Pydantic** for validation with min/max constraints. Changes via the `/api/settings` endpoint automatically save to `bot_config.json` and trigger a PM2 restart.

### Layer 2: `.env` (Secrets)

API keys and credentials. See Section 4.

### Layer 3: Firestore (Per-Agent Dynamic Config)

Each agent document in the `agents` collection contains its own:
- `systemPrompt` — The full LLM system prompt
- `openingLine` — First thing the bot says
- `voice` — Callex-Voice-Engine voice ID override
- `language` — STT language code (`hi-IN`, `en-IN`, `gu-IN`)
- `temperature`, `maxTokens` — LLM parameters
- `voiceSpeed`, `prosodyRate`, `prosodyPitch` — Voice tuning
- `backgroundNoiseVolume` — Per-agent noise level (0.0–1.0)
- `bargeInMode` — `balanced` / `aggressive` / `passive`
- `patienceMs` — How long to wait before responding
- `fillerPhrases` — JSON array of filler phrases to use while thinking
- `ipaLexicon` — Custom pronunciation lexicon
- `tools` — External tool definitions
- `enableNLP` — Toggle tone analysis
- `knowledgeBase` — FAQ text for FastReplyCache
- `analysisSchema` — Custom fields for post-call analysis
- `customDispositions` — Custom disposition categories

---

## 6. Core Application (`app/main.py`)

**This is the heart of the system** — a single 3074-line file that handles everything from WebSocket connections to TTS streaming.

### Key Sections (by line range)

| Lines | Section | Description |
|---|---|---|
| 1–9 | `.env` Loading | Loads `.env` before any other imports |
| 17–23 | `__safe_log()` | Log sanitizer — replaces internal vendor names ("cx-stt" → "SST_MODEL_2", "cx-asr-core" → "genartml-callex") for public-facing logs |
| 44–66 | Imports | All modular imports from `app/` subpackages |
| 72–121 | CX-Model Key Pool | Round-robin API key rotation with per-key semaphores (max 5 concurrent per key) |
| 124–230 | `CallexVoiceKeyManager` | Production-grade API key load balancer with automatic failover, rate-limit cooldown (60s), and exhaustion detection |
| 232–264 | Key Pool Initialization | Loads Callex-Voice-Engine keys (`CALLEX_VOICE_KEY_1-5`) and CX-STT keys (`SST_MODEL_2_API_KEY_1-5`) from environment |
| 266–334 | Configuration Constants | Audio config (16kHz, VAD thresholds, silence timeouts), voice settings, Firestore prompt cache |
| 336–408 | FreeSWITCH ESL | `freeswitch_hangup()` and `freeswitch_command()` — TCP socket communication with FreeSWITCH |
| 411–433 | Firebase Upload | `upload_to_firebase()` — Uploads call recordings with 24h signed URLs |
| 436–498 | `LocalRecorder` | Stereo WAV recorder (L=Customer, R=Bot) — interleaves customer and bot audio in real-time |
| 500–542 | Opener Cache | Content-hash based caching of TTS opener audio — auto-invalidates when text changes |
| 545–611 | Startup (`lifespan`) | Loads global AI models at startup: DeepFilterNet3 → YAMNet → Silero VAD. Scales thread pool to 500 workers. |
| 613–748 | Telemetry API | `/telemetry` serves dashboard HTML, `/api/telemetry/live` returns real-time metrics (latency, CPU, memory, active calls, analytics) |
| 750–768 | Legacy SCRIPTS Dict | **NOT IN ACTIVE USE** — Hardcoded fallback script. Agents now load from Firestore. |
| 770–806 | Background Noise | Loads `background_noise.mp3` into memory as PCM array, mixed into outgoing TTS audio at configurable volume |
| 808–894 | `NoiseFilter` | Per-call noise gate with highpass (80Hz), bandpass (80–4000Hz), spectral flatness, and energy dB checks |
| 896+ | TTS / WebSocket / LLM | The main `tts_stream_generate()`, `generate_response()`, and WebSocket handler — handles all real-time call logic |

### WebSocket Protocol

FreeSWITCH sends JSON messages over WebSocket:

```json
// Incoming audio frame
{
  "event": "media",
  "media": {
    "payload": "<base64-encoded-PCM16>"
  }
}

// Call metadata (sent once at connection)
{
  "event": "start",
  "start": {
    "callSid": "<UUID>",
    "from": "+91XXXXXXXXXX",
    "customParameters": {
      "agent_id": "<firestore-agent-id>"
    }
  }
}

// Call ended
{
  "event": "stop"
}
```

The server responds with:

```json
// Send audio to caller
{
  "event": "media",
  "media": {
    "payload": "<base64-encoded-PCM16>"
  }
}

// Clear audio queue (barge-in)
{
  "event": "clear"
}
```

### Important Internal Functions (in `main.py`)

| Function | Purpose |
|---|---|
| `get_cx-model_key()` | Async round-robin CX-Model API key selector |
| `get_sst_model_2_key()` | Gets a healthy CX STT Engine STT key |
| `ensure_opener_cache()` | Pre-generates and caches opening line TTS audio |
| `freeswitch_hangup(uuid)` | Terminates a call via ESL |
| `upload_to_firebase(path)` | Uploads recording to Firebase Storage |
| `load_bg_noise()` | Loads background noise MP3 into memory |
| `tts_stream_generate()` | Streams text → Callex-Voice-Engine → PCM16 chunks |
| `generate_response()` | Sends conversation to CX-Model and streams response |
| WebSocket handler | The main `async def websocket_endpoint()` — orchestrates the entire call |

---

## 7. Audio Processing Pipeline (`app/audio/`)

The audio pipeline has **6 layers** of validation before customer speech reaches the LLM.

### Layer Architecture

```
Raw PCM from FreeSWITCH
    │
    ▼
┌───────────────────────────────────────────┐
│ Layer 1: DeepFilterNet3 (deepfilter_denoiser.py)  │
│   Neural noise suppression (traffic, wind, crowd) │
│   16kHz → 48kHz → Process → 16kHz                │
│   Per-call DF state (isolated FFT buffers)        │
└───────────────────────┬───────────────────┘
                        ▼
┌───────────────────────────────────────────┐
│ Layer 2: NoiseFilter (in main.py)         │
│   Highpass 80Hz → Bandpass 80-4000Hz      │
│   Energy dB gate (-55dB) + Spectral       │
│   Flatness check (rejects fan/hum noise)  │
└───────────────────────┬───────────────────┘
                        ▼
┌───────────────────────────────────────────┐
│ Layer 3: Silero VAD (vad_silero.py)       │
│   ML speech detection with hysteresis     │
│   Per-call deep-copied model (isolated RNN│
│   state prevents cross-call interference) │
└───────────────────────┬───────────────────┘
                        ▼
┌───────────────────────────────────────────┐
│ Layer 4: Speaker Verification             │
│   (speaker_verifier.py — Resemblyzer)     │
│   Enrolls caller voice in first ~3s       │
│   Rejects background speakers by cosine   │
│   similarity (threshold: 0.48)            │
└───────────────────────┬───────────────────┘
                        ▼
┌───────────────────────────────────────────┐
│ Layer 5: Duration + Semantic Gates        │
│   Min speech duration: 150ms              │
│   Semantic filter (semantic.py) rejects   │
│   fillers: "um", "हाँ", "ok ok ok"        │
└───────────────────────┬───────────────────┘
                        ▼
┌───────────────────────────────────────────┐
│ Layer 5.5: YAMNet Sound Classifier        │
│   (classifier.py — TensorFlow Hub)        │
│   Rejects: coughs, sneezes, dog barks,    │
│   breathing, mic bumps, wind, clicking    │
└───────────────────────┬───────────────────┘
                        ▼
              Validated Speech PCM
                        │
                        ▼
┌───────────────────────────────────────────┐
│ CX STT Engine Streaming STT                   │
│   (sst_model_2_streaming.py)              │
│   Persistent WebSocket to CX-STT API      │
│   Returns transcript + VAD signals        │
└───────────────────────┬───────────────────┘
                        ▼
              Text Transcript → LLM
```

### File-by-File Breakdown

#### `app/audio/call_context.py` — `CallAudioContext`

**Purpose**: Bundles ALL per-call audio processing state into a single object. Every WebSocket connection creates its own `CallAudioContext` to guarantee **zero state leakage** between concurrent calls.

**What it creates per call**:
1. **Silero VAD** — Deep-copied PyTorch model (`copy.deepcopy`) to isolate RNN hidden state
2. **DeepFilterNet3** — New `DF` instance for isolated FFT/ISTFT buffers
3. **SpeakerVerifier** — Fresh enrollment and embedding state
4. **SemanticFilter** — Per-call filler word filter
5. **YAMNet reference** — Shared globally (stateless inference, safe to share)

**Critical detail**: Without `copy.deepcopy` on the Silero model, concurrent calls corrupt each other's RNN state, causing barge-in to fail completely.

```python
# Usage:
ctx = CallAudioContext(call_uuid="abc123")
is_speech, confidence = ctx.silero_vad.is_speech(audio_chunk)
clean_audio = ctx.deepfilter.process(raw_pcm)
is_caller, similarity = ctx.speaker_verifier.verify(audio)
ctx.cleanup()  # Call when WebSocket disconnects
```

#### `app/audio/deepfilter_denoiser.py` — DeepFilterNet3

**Purpose**: State-of-the-art neural noise suppression trained on DNS Challenge dataset.

**Architecture**: `Raw 16kHz PCM → Upsample 48kHz → DeepFilterNet3 → Downsample 16kHz → Clean PCM`

**Key details**:
- **Global model**: Loaded once at startup (`load_deepfilter_model()`)
- **Per-call state**: Each call gets its own `DF` instance (from `libdf`) for isolated FFT buffers
- Processes in 480-sample windows (30ms @ 16kHz)
- Falls back to raw audio passthrough if model fails to load
- Logs diagnostics every 200 frames (~1 minute of audio)

#### `app/audio/vad_silero.py` — `SileroVADFilter`

**Purpose**: ML-based voice activity detection using pretrained Silero VAD model.

**Key details**:
- **Accuracy**: 99%+ across diverse noise conditions
- **Model size**: ~2MB, <1ms inference per chunk
- Requires exactly 512 samples for 16kHz input
- Handles larger chunks by processing in non-overlapping windows and averaging
- **Hysteresis**: Requires 3 consecutive "speech" frames to turn ON, 5 consecutive "noise" frames to turn OFF — prevents rapid switching
- **Adaptive noise floor**: Learns from first 10 frames of low-confidence audio
- Global model is cached (`_GLOBAL_SILERO_MODEL`) so it's not re-downloaded per call

#### `app/audio/speaker_verifier.py` — `SpeakerVerifier` (Resemblyzer)

**Purpose**: Verifies that incoming speech belongs to the actual caller, not background speakers.

**Pipeline**:
1. **Enrollment** (first ~3 seconds): Accumulates clear speech (energy-gated, rejects quiet audio)
2. **Multi-segment embedding**: Splits enrollment into overlapping 1.6s segments, averages embeddings
3. **Verification**: Compares new speech against reference via cosine similarity
4. **Rolling update**: Reference embedding adapts over time (80% old + 20% new, max 10 updates)
5. **Verification buffer**: Accumulates 0.6s+ of speech before comparing for reliability

**Thresholds**:
- `similarity_threshold`: 0.48 (production — low enough for degraded phone audio)
- `enrollment_energy_db`: -35.0 dB minimum
- Fail-open policy: If model fails to load, everything passes through

#### `app/audio/classifier.py` — `SoundEventClassifier` (YAMNet)

**Purpose**: Layer 5.5 — Non-Linguistic Vocal Sound Filter. Rejects non-speech sounds.

**Blocklist** (38 sound types):
- `Cough`, `Sneeze`, `Throat clearing`, `Breathing`, `Wheeze`, `Sniff`, `Gasp`
- `Finger snapping`, `Knock`, `Tap`, `Clicking` (mic handling)
- `Wind`, `Rustle`, `Static`, `White noise`
- `Dog`, `Bark`, `Meow`, `Cat`, `Growling`

**Notes**: 
- Loads class names CSV from GitHub at startup (falls back gracefully if offline)
- YAMNet expects 16kHz float32 audio
- Averages scores across 0.48s frames for the whole segment

#### `app/audio/semantic.py` — `SemanticFilter`

**Purpose**: Filters out non-meaningful utterances that shouldn't trigger barge-in.

**Rules**:
1. Length check (min 3 characters by default)
2. Exact filler word match (Hindi: हाँ, हम्म, ओह | English: um, uh, hmm, ok)
3. All-filler check (space-separated words all in filler list)
4. Repetition detection (same word repeated 3+ times consecutively)
5. Excessive punctuation (>30% punctuation → gibberish)
6. Single character repetition pattern (e.g., "क क क")

**Combined filler list**: ~40+ Hindi + English + Gujarati filler words.

#### `app/audio/sst_model_2_streaming.py` — `SSTModel2StreamingSTT`

**Purpose**: Production WebSocket streaming STT client for CX STT Engine.

**Key details**:
- **Connection**: Persistent WebSocket to `wss://api.cx-stt.ai/speech-to-text/ws`
- **Model**: `cx-asr-core:v3` (encoded as base64 in code for public-facing log safety)
- **Protocol**: Audio sent as binary WAV frames (PCM16 with WAV header)
- **VAD signals**: Server returns `speech_start`, `speech_end`, and `data` (transcript) events
- **Key rotation**: Tries all available keys on connection failure, reports failures to key manager
- **Auto-reconnect**: Up to 3 attempts with 1s delay
- **Callbacks**: async `on_transcript`, `on_speech_started`, `on_speech_ended`

#### `app/audio/gating.py` — `AudioGatingPipeline`

**Purpose**: Multi-stage audio gating pipeline designed to prevent false TTS interruptions in noisy environments.

> **⚠️ PARTIALLY IN USE**: The `AudioGatingPipeline` class defines the full 6-layer architecture but in production, `main.py` uses individual components directly rather than this unified pipeline class. The utility functions (`bandpass_filter`, `ai_denoise`, `create_wav_header`, `quick_asr_cx-model`) are used independently.

**Layers defined**:
1. AI Noise Suppression (RNNoise/Spectral Gating via `noisereduce`)
2. SNR & Clarity Filter (signal-to-noise + spectral flatness consensus)
3. WebRTC VAD (Mode 3 — maximum aggressiveness)
4. Speaker Verification (pyannote/Resemblyzer)
5. Temporal Stability (min speech duration, bot ignore window)
5.5. NSVF — YAMNet classifier
6. Semantic Intent Confirmation (ASR word-level)

#### `app/audio/verification.py` — Legacy Speaker/Semantic Verifiers

> **⚠️ NOT IN ACTIVE USE**: This file contains `SpeakerVerifier` (using **pyannote.audio** — requires HuggingFace auth token) and `SemanticIntentVerifier` (using **faster-whisper**). Both have been **replaced** by the production equivalents:
> - Speaker verification → `speaker_verifier.py` (Resemblyzer — no auth needed)
> - Semantic intent → `semantic.py` (pure regex — 0ms latency, no model)
>
> Kept for potential future use. The pyannote model requires a HuggingFace token and is heavier.

---

## 8. Core Logic Modules (`app/core/`)

### `app/core/agent_loader.py` — Firestore Agent Bridge

**Purpose**: The single source of truth bridge between the Enterprise Dashboard/API and the live calling system.

```
Enterprise Dashboard (React) → Express API → Firestore
External API → Firestore
Calling System (main.py) → agent_loader.py → Firestore
```

**Key functions**:
- `load_agent(agent_id)` — Loads a single agent by UUID from Firestore, including linked custom dispositions
- `get_default_agent()` — Returns the first active agent as fallback
- `get_active_prompt(agent_id)` — Gets the active prompt version (if using versioning system)
- `get_linked_dispositions(agent_id)` — Fetches custom dispositions linked to an agent
- `_doc_to_dict(doc)` — Converts Firestore document to clean dict with defaults for missing fields

**Fallback agent**: `FALLBACK_AGENT` is a hardcoded Hindi DishTV recharge script used when Firestore is completely unavailable.

**Default field values** (applied when missing from Firestore):
```python
'language': 'en-US'
'temperature': 0.7
'maxTokens': 250
'bargeInMode': 'balanced'
'patienceMs': 800
'backgroundNoiseVolume': 0.20
'voiceSpeed': 1.0
```

### `app/core/conversation_brain.py` — `ConversationBrain`

**Purpose**: Per-call conversation state manager with built-in anti-hallucination engine.

**Problems it solves**:
1. **Opening line repetition** — Detects and blocks bot from repeating its opener
2. **Echo-loop hallucination** — Filters out bot's own TTS output transcribed by STT
3. **Cross-call history leakage** — All state is per-instance, protected by `asyncio.Lock`
4. **Repeat detection** — Blocks LLM from saying the same thing twice, even if rephrased
5. **Intra-reply deduplication** — Removes duplicate sentences within a single LLM response
6. **Loop truncation** — Detects and truncates repeating phrase loops

**Echo detection algorithm**:
- Compares transcript against last 8 bot messages
- Uses both exact substring matching and fuzzy similarity (>65% threshold)
- Word overlap check for partial echoes (>75% overlap)
- Tracks currently-speaking text with 2-second linger window after TTS ends

**Response sanitization**:
- Opening line repetition check (>70% similarity)
- Exact repeat of last reply (>80% similarity)
- Check against ALL recent bot messages (>80% similarity)
- Duplicate sentence removal within the reply
- Looping phrase truncation
- **Response fingerprinting**: SHA256 of normalized, word-sorted text catches rephrased duplicates (sliding window of 20)

**History management**:
- `history` — Trimmed to last 20 messages (context window for LLM)
- `full_history` — Complete untruncated transcript (for analytics)
- All mutations protected by `asyncio.Lock`
- Separate `_llm_lock` gates LLM calls (only 1 at a time per call)

### `app/core/fast_reply_cache.py` — `FastReplyCache`

**Purpose**: Zero-latency (~5ms) reply system for questions whose answers exist in the agent's prompt or knowledge base, completely bypassing the LLM.

**How it works**:
1. At call startup, parses agent's `systemPrompt` + `knowledgeBase` into FAQ pairs
2. Three extraction strategies:
   - **Explicit Q&A patterns**: `Q: ... A: ...` format
   - **If-asks patterns**: `If customer asks about X, say Y`
   - **Key-value facts**: `Address: 123 Main St` → triggers for "address", "pata", "kahan"
3. Customer utterance is fuzzy-matched (SequenceMatcher + word overlap blend) against triggers
4. Returns cached reply if score > 0.72, else falls through to LLM

**Cache TTL**: 60 seconds — agent prompt changes propagate within 1 minute.

**Acknowledgment patterns**: Language-aware regex patterns for Hindi, Gujarati, and English (हाँ, ji, ok, nahi, etc.) — these return `None` which means "let LLM handle" since they are context-dependent.

### `app/core/tone_analyzer.py` — `ToneAnalyzer`

**Purpose**: Real-time NLP emotion detection from customer transcripts with adaptive LLM prompt injection and TTS voice parameter hints.

**Supported emotions**: `angry`, `frustrated`, `confused`, `sad`, `happy`, `polite`, `rushed`, `skeptical`, `neutral`

**Detection method**: Multi-language regex pattern matching (~0.1ms per call):
- Hindi/Hinglish keywords (e.g., `bakwas`, `pagal`, `gussa`)
- English keywords (e.g., `terrible`, `horrible`, `fed up`)
- Gujarati keywords (e.g., `bakvaas`, `shu chhe aa`)
- Each pattern has a weight (0.4–1.0), combined into a confidence score

**Rolling state**: Maintains a deque of last 5 emotion readings, weighted by recency. Only switches emotion if new confidence > 0.4.

**Output**:
- `get_tone_instruction()` — Dynamic LLM system prompt injection (e.g., "🔴 CUSTOMER IS ANGRY. Be extremely empathetic...")
- `get_tts_hints()` — Voice parameter adjustments (stability, style) per emotion

### `app/core/config_manager.py` — `ConfigManager`

**Purpose**: Manages `bot_config.json` with Pydantic validation.

**Pydantic models**:
- `VADSettings` — min_speech_duration, silence_timeout, interruption_threshold_db, noise_gate_db, spectral_flatness_threshold
- `APICredentials` — server_key (CX-Model), secret_key (Callex-Voice-Engine), voice_id
- `VoiceSettings` — speed, stability, similarity_boost, style, use_speaker_boost
- `BotConfig` — Combines all three

**Key methods**:
- `load_config()` — Loads from file or environment defaults
- `save_config()` — Saves to `bot_config.json`
- `update_settings(updates)` — Deep-merges updates, validates, saves
- `get_env_dict()` — Flat dictionary for backward compatibility

### `app/core/config.py` — Central Constants

**Purpose**: Single source for path constants and environment-derived settings.

| Constant | Value | Description |
|---|---|---|
| `BASE_DIR` | Project root | Resolved from `__file__` |
| `DATABASE_URL` | `sqlite:///data/call_recordings.db` | SQLAlchemy connection string |
| `RECORDINGS_DIR` | `<root>/recordings` | Auto-created at import |
| `RECORDING_SAMPLE_RATE` | `16000` | Hz |
| `RECORDING_CHANNELS` | `1` (Mono) | |
| `RETENTION_DAYS` | `90` | Delete recordings older than this |
| `API_CORS_ORIGINS` | `*` | CORS allowed origins |
| `API_PREFIX` | `/api` | REST API url prefix |
| `DEFAULT_PAGE_SIZE` | `50` | Pagination |
| `MAX_PAGE_SIZE` | `200` | Max pagination |
| `FIREBASE_CREDENTIALS_PATH` | `firebase_credentials.json` | |
| `FIREBASE_STORAGE_BUCKET` | `lakhuteleservices-1f9e0.appspot.com` | |
| `AWS_ACCESS_KEY` / `AWS_SECRET_KEY` / `AWS_REGION` / `AWS_BUCKET_NAME` | From env | S3 config |

### `app/core/database.py` — SQLAlchemy ORM

**Purpose**: Local SQLite database for call recording metadata and analytics.

See [Section 15: Database Schema](#15-database-schema) for full model details.

**Key functions**:
- `init_db()` — Creates tables + runs column migrations
- `get_db()` — Dependency injection generator
- `get_db_session()` — Direct session creation
- `update_call_outcome()` — Heuristic disposition parser (keywords → agreed/declined/unclear)

### `app/core/db.py` — Async Firestore Wrappers

**Purpose**: Thread-pooled async wrappers for synchronous Firestore SDK operations.

**Functions**: `db_get_doc()`, `db_set_doc()`, `db_update_doc()`, `db_add_doc()`, `db_query_where()`

All wrap synchronous Firestore calls in `asyncio.to_thread()` to prevent blocking the WebSocket audio event loop.

---

## 9. API Layer (`app/api/`)

### `app/api/routes.py` — REST API

Mounted at `/api` prefix. FastAPI `APIRouter` with SQLAlchemy session dependency injection.

#### Call Management

| Endpoint | Method | Description |
|---|---|---|
| `/api/calls` | GET | List calls with filtering (date range, phone, status, outcome) + pagination |
| `/api/calls/{id}` | GET | Get single call details |
| `/api/calls/{id}` | DELETE | Delete call + recording from disk + DB |
| `/api/calls/{id}/recording` | GET | Stream/download call recording WAV (supports S3 URLs) |
| `/api/calls/{id}/notes` | PATCH | Update call notes |
| `/api/calls/{id}/disposition` | PATCH | Update disposition (auto-sets agreed/declined/unclear flags) |
| `/api/calls/bulk-delete` | POST | Delete multiple calls at once |
| `/api/calls/export/csv` | GET | Export filtered calls as CSV download |

#### Analytics

| Endpoint | Method | Description |
|---|---|---|
| `/api/analytics/summary` | GET | Total calls, agreed/declined/unclear counts, avg duration, recording size |
| `/api/analytics/daily` | GET | Daily call stats for trend charts (up to 365 days) |
| `/api/analytics/today` | GET | Today-only summary stats |
| `/api/analytics/dispositions` | GET | Disposition breakdown for pie chart |
| `/api/analytics/hourly` | GET | Calls grouped by hour for heatmap |

#### Settings & Scripts

| Endpoint | Method | Description |
|---|---|---|
| `/api/settings` | GET | Current bot_config.json settings |
| `/api/settings` | POST | Update settings + trigger PM2 restart |
| `/api/restart` | POST | Manually restart PM2 process |
| `/api/scripts` | GET | List all saved scripts |
| `/api/scripts/active` | GET | Get currently active script |
| `/api/scripts/{id}` | GET | Get single script |
| `/api/scripts` | POST | Create/update script |
| `/api/scripts/{id}` | DELETE | Delete script |
| `/api/scripts/{id}/activate` | POST | Activate script + trigger restart |
| `/api/health` | GET | Server health (uptime, memory, CPU, active calls) |

#### Telemetry (Defined in `main.py`, not `routes.py`)

| Endpoint | Method | Description |
|---|---|---|
| `/telemetry` | GET | Serves the dashboard HTML page |
| `/api/telemetry/live` | GET | Real-time JSON metrics (active calls, latency, CPU, analytics, agents) |

---

## 10. Managers (`app/managers/`)

### `app/managers/scripts.py` — `ScriptManager` (S3-Backed)

**Purpose**: CRUD operations for bot scripts stored as JSON files in AWS S3.

> **⚠️ PARTIALLY LEGACY**: This was the original script management system before Firestore agents were implemented. It is still functional and used by the `/api/scripts` endpoints, but the **primary** agent configuration now lives in Firestore (managed via `agent_loader.py` and the Enterprise Dashboard).

**S3 Structure**:
```
s3://callex-recordings/scripts/
  ├── script1.json
  ├── script_abc123.json
  └── _active.json         # Pointer to active script ID
```

**Features**: Auto-seeds default script if S3 is empty, active script tracking, timestamp management.

### `app/managers/recordings.py` — `RecordingManager`

**Purpose**: Manages real-time call recording to disk as WAV files.

**Key details**:
- Creates WAV with placeholder header, updates on close with actual data size
- Supports async chunk writing via `write_chunk_async()`
- `cleanup_old_recordings()` — Deletes files older than `RETENTION_DAYS` (90 by default)
- `get_storage_stats()` — Returns total files, bytes, MB, GB

> **Note**: In production, `main.py` uses its own `LocalRecorder` class (stereo recording) rather than this `RecordingManager` (mono recording). Both exist and could be used.

---

## 11. Services (`app/services/`)

### `app/services/analytics.py` — Post-Call AI Analysis

**Purpose**: AI-powered call outcome analysis and transcript export.

#### `analyze_call_outcome()`

Uses **CX-Model** to evaluate a complete conversation transcript and extract:
- `agreed` (boolean) — Did the customer agree?
- `commitment` — "today" / "tomorrow" / "later" / "refused"
- `disposition` — Category label (uses custom dispositions if configured)
- `sentiment` — "positive" / "negative" / "neutral"
- `summary` — 2-3 sentence call summary
- `notes` — Why the disposition was assigned
- `highlighted_points` — Key Q&A pairs extracted from conversation
- Custom fields defined in agent's `analysisSchema`

**Custom dispositions**: If the agent has custom dispositions linked, the prompt includes their names and matching rules.

#### `auto_train_sandbox_agent()`

**Purpose**: Background meta-reflection task for Training Sandbox agents.

How it works:
1. Takes the call transcript and current system prompt
2. Sends to **CX-Model Pro** (not Flash) as a "Master AI Training Architect"
3. CX-Model rewrites the system prompt incorporating corrections from the trainer's conversation
4. Updates Firestore: Creates new prompt version, sets it as active, updates agent document
5. Changes take effect on the next call (prompt cache TTL is 30s)

#### `export_transcript_threaded()`

**Purpose**: Saves complete transcript and outcome metrics to Firestore (call document).

Runs entirely in `asyncio.to_thread()` to prevent blocking. Updates or creates the call document in the `calls` Firestore collection with transcript, recording URL, outcome, sentiment, disposition, and structured data.

---

## 12. Utilities (`app/utils/`)

### `app/utils/logger.py` — `CallTracker`

**Purpose**: Thread-safe call lifecycle tracking with SQLite database integration.

**Features**:
- `start_call()` — Creates `Call` record in SQLite, tracks in `active_calls` dict
- `log_message()` — Appends messages to in-memory conversation history
- `end_call()` — Calculates duration, detects outcome from conversation, saves `CallOutcome` + `Recording` to DB
- `_detect_outcome()` — Rule-based agreement detection using Hindi/English keyword matching on user's last 3 messages

**Outcome detection keywords**:
- **Agreement**: हाँ, han, yes, ठीक है, ok, कर दूंगा, आज, कल, अभी
- **Decline**: नहीं, nahi, no, बंद कर दो, cancel, मत करो
- **Unclear**: Neither agreement nor decline detected

**IST timezone**: All times are in IST (UTC+5:30).

**Global instance**: `tracker = CallTracker()` — imported throughout the codebase.

---

## 13. Enterprise Platform (`enterprise/`)

A full-stack web application for managing AI voice agents, campaigns, and call analytics.

### Backend (`enterprise/backend/`)

| Technology | Purpose |
|---|---|
| Express.js (ESM) | HTTP API server on port 4000 |
| Firebase Admin SDK | Firestore + Storage access |
| JWT + bcryptjs | Authentication |
| modesl | FreeSWITCH ESL client (originate calls) |
| express-fileupload / Multer | File uploads |
| csv-stringify | CSV export |
| Prisma (optional) | DB schema management |
| ws | WebSocket support |

**Source structure** (`enterprise/backend/src/`):
- `index.js` — Express server entry point
- `firebase.js` — Firebase Admin SDK initialization
- `middleware/` — JWT auth middleware
- `routes/` — Agent CRUD, calls, campaigns, auth, dispositions
- `services/` — Business logic
- `ws/` — WebSocket proxy for live call monitoring

### Frontend (`enterprise/frontend/`)

| Technology | Purpose |
|---|---|
| React 18 | UI framework |
| Vite | Build tool (port 3000) |
| TailwindCSS | Styling |
| Zustand | State management |
| Recharts | Chart visualizations |
| react-router-dom | Routing |
| lucide-react | Icons |
| jspdf / jspdf-autotable | PDF export |
| Firebase SDK | Client-side Firestore/Auth |
| @sapphi-red/web-noise-suppressor | In-browser noise suppression for live monitoring |

**Deployment**: Configured for **Vercel** (`vercel.json` present).

### Starting the Enterprise Platform

```bash
cd enterprise
bash start.sh
# Backend: http://localhost:4000
# Frontend: http://localhost:3000
```

---

## 14. Telemetry Dashboard (`dashboard.html`)

A **glassmorphic, dark-mode** real-time monitoring dashboard served at `/telemetry`.

**Features**:
- Live active call count with phone numbers and durations
- LLM latency (TTFB) with color-coded progress bars (<800ms green, <1500ms yellow, >1500ms red)
- TTS latency tracking
- Conversion rate percentage
- CPU/Memory/Error monitoring
- Today vs Yesterday call comparison with delta badges
- Sentiment breakdown bars (positive/neutral/negative)
- Top disposition categories
- Average call duration
- Deployed agents table (name, NLP status, voice speed, agent ID)

**Refresh interval**: 1.5 seconds via `setInterval(refresh, 1500)`.

**Data source**: Polls `/api/telemetry/live` which aggregates data from:
- In-memory deques (`GLOBAL_LATENCY_TRACKER`, `GLOBAL_TTS_LATENCY_TRACKER`)
- SQLite database (analytics totals)
- `psutil` (system resources)
- Firestore (active agents list)
- `tracker.active_calls` (live calls)

---

## 15. Database Schema

### SQLite (`data/call_recordings.db`)

Three tables, managed by SQLAlchemy ORM in `app/core/database.py`:

#### `calls` Table

| Column | Type | Description |
|---|---|---|
| `id` | INTEGER PK | Auto-increment |
| `call_uuid` | VARCHAR(100) UNIQUE | FreeSWITCH call UUID |
| `phone_number` | VARCHAR(20) | Caller phone number |
| `start_time` | DATETIME | Call start (IST) |
| `end_time` | DATETIME | Call end (IST) |
| `duration_seconds` | FLOAT | Call duration |
| `status` | VARCHAR(20) | `in_progress`, `completed`, `disconnected`, `error` |
| `created_at` | DATETIME | Record creation time |

#### `recordings` Table

| Column | Type | Description |
|---|---|---|
| `id` | INTEGER PK | Auto-increment |
| `call_id` | INTEGER FK → calls.id | CASCADE delete |
| `file_path` | VARCHAR(500) | Path or URL (Firebase/S3) |
| `file_size_bytes` | INTEGER | File size |
| `format` | VARCHAR(10) | `wav` |
| `created_at` | DATETIME | |

#### `call_outcomes` Table

| Column | Type | Description |
|---|---|---|
| `id` | INTEGER PK | Auto-increment |
| `call_id` | INTEGER FK → calls.id | CASCADE delete |
| `customer_agreed` | BOOLEAN | True=agreed, False=declined, None=unclear |
| `commitment_date` | DATETIME | When customer promised to act |
| `unclear_response` | BOOLEAN | True if response was ambiguous |
| `disposition` | VARCHAR(100) | Category label (e.g., "Interested", "Busy") |
| `notes` | TEXT | Free-form notes |
| `transcript` | TEXT | Full conversation transcript (`User: ... / Bot: ...`) |
| `summary` | TEXT | AI-generated call summary |
| `sentiment` | VARCHAR(20) | `positive` / `negative` / `neutral` |
| `structured_data` | TEXT | JSON string of highlighted_points + custom fields |
| `created_at` | DATETIME | |

**Auto-migration**: `_migrate_add_columns()` safely adds new columns (`transcript`, `summary`, `sentiment`, `structured_data`) to existing databases without data loss.

### Firestore Collections

| Collection | Document ID | Purpose |
|---|---|---|
| `agents` | Auto-generated UUID | Agent configurations |
| `calls` | Call UUID | Enterprise call records with transcripts |
| `dispositions` | Auto-generated | Custom disposition categories |
| `promptVersions` | Auto-generated | Agent prompt version history |

---

## 16. Firebase / Firestore Integration

### Firebase Admin SDK Initialization

Initialized in `app/main.py` using `firebase_credentials.json`:
```python
cred = credentials.Certificate(FIREBASE_CREDENTIALS_PATH)
firebase_admin.initialize_app(cred, {'storageBucket': FIREBASE_STORAGE_BUCKET})
```

### Firebase Storage

Used for call recording uploads:
- Recordings uploaded to `recordings/` prefix
- Signed URLs generated with 24-hour expiry (not public)
- Function: `upload_to_firebase(file_path)` in `main.py`

### Firestore Security Rules

```
allow read, write: if true;
```

> **⚠️ IMPORTANT**: Firestore rules are completely open. Security is enforced at the **API layer** (JWT auth, userId scoping, superadmin checks in the Enterprise backend). The Firebase Admin SDK bypasses rules entirely. Client-side direct access is not used.

---

## 17. API Key Management & Load Balancing

### `CallexVoiceKeyManager` (in `main.py`)

Production-grade API key load balancer used for both Callex-Voice-Engine TTS and CX STT Engine STT.

**Features**:
- **True round-robin**: Concurrent requests are instantly distributed across all keys
- **Exhaustion detection**: HTTP 401/402/403 → key permanently marked dead
- **Rate-limit cooldown**: HTTP 429 → key put on 60-second cooldown, auto-recovers
- **Thread-safe**: Uses `threading.Lock` for 100+ concurrent access
- **Last resort**: If ALL keys are exhausted, tries the first key anyway
- **Pool status**: `healthy=3, exhausted=0, cooldown=1, total=4`

### CX-Model Key Pool

- Loads up to 4 keys from `CX-MODEL_API_KEY_1` through `_4` + fallback from `bot_config.json`
- Round-robin via `asyncio.Lock`
- Per-key semaphore: max 5 concurrent requests per key (prevents 429 storms)
- 4 keys × 5 concurrent = **20 max inflight LLM requests**

### TTS Concurrency Limiter

- `_TTS_MAX_CONCURRENT = 15` (configurable via env)
- `asyncio.Semaphore` prevents HTTP connection pool exhaustion at 100+ simultaneous calls
- Each Callex-Voice-Engine stream holds an HTTP connection open for 1-3 seconds

---

## 18. Call Lifecycle (End-to-End Flow)

```
1. FreeSWITCH → WebSocket "start" event
   │
   ├─ Extract agent_id from customParameters
   ├─ Load agent from Firestore (agent_loader.load_agent)
   ├─ Fall back to FALLBACK_AGENT if not found
   ├─ Create CallAudioContext (isolated per-call audio pipeline)
   ├─ Create ConversationBrain (isolated per-call state)
   ├─ Create SSTModel2StreamingSTT (CX STT Engine WebSocket)
   ├─ Create ToneAnalyzer
   ├─ Build FastReplyCache from agent prompt + KB
   ├─ Start SQLite call record (tracker.start_call)
   │
2. Speak Opening Line
   │
   ├─ Check opener cache (content-hash based)
   ├─ If cached: load PCM from disk
   ├─ If not: stream TTS → cache to disk + send to FreeSWITCH
   ├─ brain.mark_opening_spoken()
   ├─ Mix background noise (if background_noise.mp3 exists)
   │
3. Audio Loop (while WebSocket connected)
   │
   ├─ Receive PCM16 frame from FreeSWITCH
   ├─ DeepFilter denoise → NoiseFilter gate → Silero VAD
   ├─ If speech detected:
   │   ├─ Speaker verification (enroll or verify)
   │   ├─ Stream to CX STT Engine STT
   │   ├─ Accumulate speech duration
   │   ├─ If barge-in: send "clear" to stop bot audio
   │   └─ Record customer audio (LocalRecorder)
   │
   ├─ On STT transcript callback:
   │   ├─ brain.is_echo(transcript) → skip if echo
   │   ├─ semantic_filter.is_meaningful(transcript) → skip if filler
   │   ├─ fast_reply_cache.match(transcript) → instant reply if match
   │   ├─ tone_analyzer.analyze(transcript) → detect emotion
   │   ├─ brain.add_user_message(transcript)
   │   ├─ Build LLM prompt = system_prompt + tone_instruction + history
   │   ├─ Call CX-Model API (with key rotation + semaphore)
   │   ├─ brain.sanitize_response(reply) → anti-hallucination
   │   ├─ Check for [HANGUP] tag → schedule call termination
   │   ├─ Stream reply to Callex-Voice-Engine TTS → PCM chunks
   │   ├─ Mix background noise into TTS audio
   │   ├─ Send PCM chunks to FreeSWITCH
   │   └─ Record bot audio (LocalRecorder)
   │
4. Call End (WebSocket disconnect / [HANGUP] / FreeSWITCH hangup)
   │
   ├─ Disconnect CX-STT STT WebSocket
   ├─ Finalize stereo recording → /tmp/call_<uuid>.wav
   ├─ Upload to Firebase Storage → signed URL
   ├─ AI analysis (CX-Model) → disposition, sentiment, summary
   ├─ Save to SQLite (tracker.end_call with outcome)
   ├─ Save to Firestore (export_transcript_threaded)
   ├─ If sandbox agent: auto_train_sandbox_agent()
   ├─ brain.cleanup() + ctx.cleanup()
   └─ GC collect
```

---

## 19. Production Scaling & Concurrency

### Thread Pool Scaling

```python
# main.py lifespan()
loop.set_default_executor(ThreadPoolExecutor(max_workers=500))
```

The default asyncio thread pool (15 workers) causes severe queueing at 50+ concurrent calls. Scaled to 500 for all `asyncio.to_thread()` operations (ML models, Firestore, file I/O).

### PyTorch Thread Restriction

```python
torch.set_num_threads(1)
```

Without this, 50 concurrent calls with Silero VAD / Resemblyzer create thousands of thread contentions → 100% CPU → 30s+ latency.

### Key Concurrency Controls

| Component | Mechanism | Limit | Purpose |
|---|---|---|---|
| CX-Model LLM | Per-key semaphore | 5 per key | Prevent 429 rate limits |
| Callex-Voice-Engine TTS | Global semaphore | 15 total | Prevent connection pool exhaustion |
| CX-STT STT | Key rotation | Pool of 4 keys | Distribute load |
| Silero VAD | Per-call model clone | 1 per call | Prevent RNN state corruption |
| DeepFilterNet3 | Per-call DF state | 1 per call | Prevent FFT buffer corruption |
| ConversationBrain | Per-call `_llm_lock` | 1 LLM call per call | Prevent race conditions |
| Firestore reads | Prompt cache | 30s TTL | Reduce network calls |

### Memory Management

- `brain.cleanup()` and `ctx.cleanup()` explicitly release per-call memory
- `gc.collect()` called after each call ends
- DeepFilter model clone released via `del` in `CallAudioContext.cleanup()`
- Speaker verifier buffers cleared on call end

---

## 20. Deployment & Operations

### Prerequisites

- **Python 3.11+** (for `asyncio.to_thread`, f-string improvements)
- **Node.js 18+** (for Enterprise platform)
- **FreeSWITCH** installed and configured for WebSocket connections
- **ffmpeg** (for background noise conversion)
- **PM2** (process manager)

### Installation

```bash
# Python dependencies
pip install -r requirements.txt

# Enterprise platform
cd enterprise/backend && npm install
cd enterprise/frontend && npm install
```

### Running Locally

```bash
# Voice AI Core (development)
uvicorn app.main:app --host 0.0.0.0 --port 8765 --ws-max-size 16777216

# Production (via PM2)
pm2 start "uvicorn app.main:app --host 0.0.0.0 --port 8765" --name callex-AI-AMD

# Enterprise Platform
cd enterprise && bash start.sh
```

### PM2 Process Names (tried during restart)

The `/api/restart` endpoint tries these names in order:
1. `callex-AI-AMD`
2. `test`
3. `voice-bot`

### Key Production Settings

| Setting | Value | Rationale |
|---|---|---|
| `torch.set_num_threads(1)` | 1 thread per PyTorch op | Prevents CPU thread explosion |
| `ThreadPoolExecutor(500)` | 500 workers | Handles 100+ concurrent calls |
| `TTS_MAX_CONCURRENT` | 15 | Prevents httpx pool exhaustion |
| `CX-MODEL_MAX_CONCURRENT_PER_KEY` | 5 | Prevents 429 storms |
| `SILENCE_TIMEOUT` | 0.45s | Safety net (primary VAD is server-side) |
| `PROMPT_CACHE_TTL` | 30s | Balance freshness vs. network calls |
| `FAQ_CACHE_TTL` | 60s | Agent changes reflect within 1 min |

---

## 21. Files NOT In Active Use (Dead Code / Legacy)

### ⚠️ `app/audio/verification.py` — REPLACED

Contains two classes that have been **superseded**:

1. **`SpeakerVerifier` (pyannote.audio)** → Replaced by `speaker_verifier.py` (Resemblyzer)
   - Reason: pyannote requires HuggingFace auth token, heavier model
   - The Resemblyzer version is lighter and requires no auth

2. **`SemanticIntentVerifier` (faster-whisper)** → Replaced by `semantic.py` (regex)
   - Reason: faster-whisper adds ~200ms latency per check, regex is ~0ms
   - The regex version is sufficient for filler word detection

### ⚠️ `app/managers/scripts.py` — PARTIALLY LEGACY

The S3-backed script manager is still functional but **secondary** to Firestore-based agent management. The primary workflow is now:
1. Enterprise Dashboard → Express API → Firestore → `agent_loader.py`
2. The `/api/scripts` endpoints still work but are rarely used

### ⚠️ `app/audio/gating.py` — `AudioGatingPipeline` class PARTIALLY USED

The **class itself** (`AudioGatingPipeline`) is not instantiated in production. Individual utility functions from this file (`bandpass_filter`, `ai_denoise`, `create_wav_header`, `quick_asr_cx-model`) may be used independently. The production audio pipeline in `main.py` uses the individual components directly.

### ⚠️ Legacy `SCRIPTS` dict in `main.py` (line 754)

```python
SCRIPTS = {
    "script1": { ... }
}
```

This hardcoded dict is kept only as an "absolute last-resort fallback." Agent configs are loaded dynamically from Firestore.

### ⚠️ `DEEPGRAM_API_KEY` in `.env`

Deepgram was explored as a backup ASR option but is **not used** in the current codebase. No code references it.

### ⚠️ `CALLEX_WS_AUTH_TOKEN` in `.env`

Commented out. WebSocket authentication is not currently enforced.

### ⚠️ `app/audio/README.md`

Empty placeholder file (27 bytes, contains only `# Audio Processing Module`).

### ⚠️ Root-level test files

The following files are development/testing artifacts and **not part of production**:
- `test.py`, `test_cx-stt.py` — Python test scripts
- `test_active.js`, `test_key.js`, `test_multer.js`, `test_status.js` — Node.js test scripts
- `set_agent_speed.py` — One-off utility to update agent voice speed
- `cleanup_ghost_calls.js` — One-off Firestore cleanup script

### ⚠️ `_archive/` directory

Contains old/superseded files. Gitignored. Includes old `main.py` backups, test files, and deprecated modules.

### ⚠️ `pipecat-main/` directory

Referenced in the directory listing but appears to be a vendored/downloaded library. Not actively imported by the main codebase.

### ⚠️ `scratch/` directory

Temporary scratch files directory.

### ⚠️ Multiple API documentation files

Historical versioned documentation files (not code, just docs):
- `API_DOCUMENTATION.md` / `.pdf`
- `API_TESTING_GUIDE.md`
- `CALLEX_API_DOCUMENTATION_APRIL_8.md` / `_APRIL_10.md` / `_MARCH_23.md` / `_MARCH_25.md`
- `CALLEX_API_URLS.md` / `.pdf`
- `CALLEX_CUSTOM_DISPOSITION_API.md`
- `CALLEX_MASTER_API_REFERENCE.md`
- `CALLEX_SERVER_RESTART_AND_TEST_GUIDE.md` / `.pdf`
- `Callex-10th.md`
- `callex_api_documentation.md`
- `message_output.md`
- `TUNING.md`

These are kept for historical reference but are not part of the codebase.

---

## 22. Root-Level Misc Files

| File | Purpose | In Use? |
|---|---|---|
| `background_noise.mp3` | 14MB call center ambiance audio, loaded into memory at startup | ✅ Yes |
| `dashboard.html` | Real-time telemetry dashboard (served at `/telemetry`) | ✅ Yes |
| `firebase_credentials.json` | Firebase service account key (gitignored) | ✅ Yes |
| `firestore.rules` | Firestore security rules (open read/write) | ✅ Yes |
| `requirements.txt` | Python pip dependencies | ✅ Yes |
| `main.zip` | Compressed backup of the entire codebase (~21MB) | ❌ No |
| `set_agent_speed.py` | One-off Firestore update script | ❌ No |
| `test.py` | Simple Python test | ❌ No |
| `test_cx-stt.py` | CX-STT API connectivity test | ❌ No |
| `test_active.js` | Tests if agent is active in Firestore | ❌ No |
| `test_key.js` | Tests API key validity | ❌ No |
| `test_multer.js` | Tests file upload | ❌ No |
| `test_status.js` | Tests server status | ❌ No |
| `cleanup_ghost_calls.js` | Cleans up orphaned call records in Firestore | ❌ No (one-off) |

---

## 23. Troubleshooting

### Common Issues

| Issue | Cause | Fix |
|---|---|---|
| `NameError: tracker` | Circular import with telemetry module | Ensure `from app.utils.logger import tracker` is at module level |
| `400 Bad Request` from STT | Wrong language code for English | Use `en-IN` (not `en-US`) for CX STT Engine |
| Silent audio on calls | FreeSWITCH expects binary WebSocket frames | Ensure `send_audio()` wraps PCM in WAV header |
| High CPU (>90%) | PyTorch thread explosion | Verify `torch.set_num_threads(1)` is set before model loading |
| Latency spikes (>3s) | Blocking Firestore calls in async loop | All Firestore calls must use `asyncio.to_thread()` |
| Barge-in not working | Shared Silero VAD model state | Ensure `copy.deepcopy` is used in `CallAudioContext` |
| Echo loops | Bot hearing its own TTS output | `ConversationBrain.is_echo()` should detect; check `set_bot_speaking()` calls |
| Firebase init error | Missing/invalid credentials file | Ensure `firebase_credentials.json` exists at project root |
| All API keys exhausted | Rate limiting across all keys | Check `voice_key_manager.pool_status`, add more keys to `.env` |
| PM2 restart fails | Wrong process name | Check `pm2 list` and update names in `restart_server_delayed()` |

### Useful Commands

```bash
# Check PM2 status
pm2 status

# View live logs
pm2 logs callex-AI-AMD --lines 200

# Restart server
pm2 restart callex-AI-AMD

# Check active calls
curl http://localhost:8765/api/telemetry/live | jq '.active_calls'

# Health check
curl http://localhost:8765/api/health

# Export calls CSV
curl "http://localhost:8765/api/calls/export/csv?outcome=agreed" -o calls.csv
```

---

## License & Ownership

**Callex AI** © Lakhu Teleservices. All rights reserved.  
Internal use only. Do not distribute without authorization.
