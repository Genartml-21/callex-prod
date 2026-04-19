# Callex TTS — Architecture Documentation

> **Callex Proprietary GPU-Accelerated Text-to-Speech Synthesis Engine**
> Copyright © 2024-2026 Callex AI Research / Lakhu Teleservices Pvt. Ltd.

---

## System Architecture

```
┌────────────────────────────────────────────────────────────────────┐
│                     CALLEX TTS v2.0 PIPELINE                       │
├────────────────────────────────────────────────────────────────────┤
│                                                                    │
│  INPUT                                                             │
│  ──────                                                            │
│  Raw Text / SSML ──→ SSML Parser                                  │
│                          │                                         │
│                          ▼                                         │
│  TEXT PROCESSING PIPELINE                                          │
│  ────────────────────────                                          │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐            │
│  │  Hindi Text   │→│  Grapheme-   │→│  Symbol      │            │
│  │  Normalizer   │  │  to-Phoneme  │  │  Tokenizer   │            │
│  │              │  │  (G2P)       │  │              │            │
│  │ • NFC norm   │  │ • Epitran    │  │ • IPA→int   │            │
│  │ • ₹ expand   │  │ • eSpeak     │  │ • BOS/EOS   │            │
│  │ • Schwa del  │  │ • IPA output │  │ • Greedy     │            │
│  │ • Sandhi     │  │              │  │   longest    │            │
│  │ • Code-sw    │  │              │  │   match      │            │
│  └──────────────┘  └──────────────┘  └──────────────┘            │
│                                         │                          │
│                                         ▼                          │
│  NEURAL SYNTHESIS (VITS2)                                          │
│  ────────────────────────                                          │
│  ┌─────────────────────────────────────────────────────┐          │
│  │                                                     │          │
│  │  ┌────────────┐    ┌────────────┐                  │          │
│  │  │   Text     │───→│ Stochastic │──→ durations     │          │
│  │  │  Encoder   │    │ Duration   │                  │          │
│  │  │            │    │ Predictor  │                  │          │
│  │  │ • RoPE     │    │            │                  │          │
│  │  │ • Pre-LN   │    │ • Flow     │                  │          │
│  │  │ • SwiGLU   │    │ • MAS      │                  │          │
│  │  │ • 6 layers │    └────────────┘                  │          │
│  │  └─────┬──────┘                                    │          │
│  │        │ prior (μ, σ)                              │          │
│  │        ▼                                            │          │
│  │  ┌────────────┐    ┌────────────┐                  │          │
│  │  │ Normalizing│◄──→│ Posterior  │◄── mel (train)   │          │
│  │  │   Flow     │    │ Encoder    │                  │          │
│  │  │            │    │            │                  │          │
│  │  │ • Affine   │    │ • WaveNet  │                  │          │
│  │  │   coupling │    │ • 16 layer │                  │          │
│  │  │ • 4 flows  │    │ • Reparam  │                  │          │
│  │  └─────┬──────┘    └────────────┘                  │          │
│  │        │ z (latent)                                 │          │
│  │        ▼                                            │          │
│  │  ┌────────────┐                                    │          │
│  │  │ HiFi-GAN   │──→ waveform                       │          │
│  │  │ v2 Vocoder │                                    │          │
│  │  │            │                                    │          │
│  │  │ • Snake    │                                    │          │
│  │  │ • MRF      │                                    │          │
│  │  │ • 4 upsamp │                                    │          │
│  │  └────────────┘                                    │          │
│  │                                                     │          │
│  │  ADVERSARIAL TRAINING                               │          │
│  │  ┌──────┐ ┌──────┐ ┌──────┐                       │          │
│  │  │ MPD  │ │ MSD  │ │ MRD  │                       │          │
│  │  │ 5per │ │ 3scl │ │ 3res │                       │          │
│  │  └──────┘ └──────┘ └──────┘                       │          │
│  └─────────────────────────────────────────────────────┘          │
│                          │                                         │
│                          ▼                                         │
│  AUDIO POST-PROCESSING                                            │
│  ──────────────────────                                            │
│  ┌──────────────────────────────────────────────────────┐         │
│  │ HPF → Gate → Compressor → De-Esser → Warmth →       │         │
│  │ Loudness Norm (LUFS) → Peak Limiter                  │         │
│  └──────────────────────────────────────────────────────┘         │
│                          │                                         │
│                          ▼                                         │
│  ┌──────────────────────────────────────────────────────┐         │
│  │ Prosody: Pitch Shift │ Time Stretch │ Energy Scale    │         │
│  └──────────────────────────────────────────────────────┘         │
│                          │                                         │
│                          ▼                                         │
│  OUTPUT: 16kHz PCM16 mono (telephony)                             │
│                                                                    │
└────────────────────────────────────────────────────────────────────┘
```

## Component Deep-Dive

### Text Encoder (VITS2)
- **6-layer Transformer** with Pre-Layer Normalization
- **Rotary Position Embeddings (RoPE)** — encodes relative position directly in Q/K vectors
- **SwiGLU Feed-Forward** — gated activation (Shazeer, 2020) outperforms standard ReLU FFN
- Outputs prior distribution parameters (μ, log σ) for the flow model

### Stochastic Duration Predictor
- **Flow-based** — produces natural variation in speaking rhythm
- **Monotonic Alignment Search (MAS)** during training extracts ground-truth durations
- Inference: samples from learned distribution, controlled by `noise_scale_w`

### Normalizing Flow
- **4 Affine Coupling Layers** with WaveNet-based parameter networks
- Channel flips between layers for full dimensional mixing
- Transforms between prior (text-conditioned) and posterior (mel-conditioned) latent spaces
- Analytically invertible with tractable log-determinant Jacobian

### HiFi-GAN v2 Vocoder
- **Multi-Receptive Field Fusion (MRF)** — parallel residual blocks capture different temporal scales
- **Snake activation** — `x + (1/α)sin²(αx)` — learns periodic structure naturally
- **4-stage upsampling**: 8× → 8× → 2× → 2× (total 256×, matching hop_length)

### Triple Discriminator
| Discriminator | Domain | Catches |
|--------------|--------|---------|
| **MPD** (Multi-Period) | 2D periodic | Pitch artifacts, harmonic distortion |
| **MSD** (Multi-Scale) | 1D temporal | Temporal discontinuities |
| **MRD** (Multi-Resolution) | 2D STFT | Spectral artifacts, metallic tones |

### Audio Effects Chain (Signal Flow)
```
Input → HPF (80Hz) → Noise Gate → Compressor (soft knee) →
De-Esser (6kHz band) → Warmth (200Hz shelf) →
LUFS Normalization (-16 LUFS) → Peak Limiter (-1 dBFS) → Output
```

## Training

```bash
# Single GPU
make train

# Config-driven
python -m callex_tts.training.trainer --config configs/training/distributed.yaml

# Dry run (validate config)
make train-dry
```

## Inference

```bash
# Start server
make serve

# Health check
curl http://localhost:8124/health

# Synthesize (plain text)
curl -X POST http://localhost:8124/v2/synthesize \
     -H "Content-Type: application/json" \
     -d '{"text": "नमस्ते, आप कैसे हैं?"}' \
     --output audio.pcm

# Synthesize (SSML with prosody)
curl -X POST http://localhost:8124/v2/synthesize \
     -H "Content-Type: application/json" \
     -d '{"text": "<speak><prosody rate=\"fast\" pitch=\"+2st\">नमस्ते!</prosody></speak>"}' \
     --output audio.pcm
```

## Deployment

```bash
# Docker build
make docker

# Full stack (TTS + Prometheus + Grafana)
make docker-dev

# Grafana dashboard at http://localhost:3001
```
