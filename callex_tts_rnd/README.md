<h1 align="center">
  🔊 Callex TTS
</h1>

<p align="center">
  <strong>Proprietary GPU-Accelerated Hindi Text-to-Speech Synthesis Engine</strong>
</p>

<p align="center">
  <em>VITS2 · HiFi-GAN v2 · Snake Activation · RoPE · SSML · Production Audio Effects</em>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.10+-blue.svg" alt="Python">
  <img src="https://img.shields.io/badge/PyTorch-2.1+-red.svg" alt="PyTorch">
  <img src="https://img.shields.io/badge/CUDA-11.8-green.svg" alt="CUDA">
  <img src="https://img.shields.io/badge/License-Proprietary-yellow.svg" alt="License">
  <img src="https://img.shields.io/badge/Version-2.0.0-purple.svg" alt="Version">
</p>

---

## Overview

Callex TTS is a production-grade, end-to-end neural text-to-speech system built for Hindi telephony. It synthesizes natural-sounding speech from text input with full prosody control, audio post-processing, and SSML support.

### Key Capabilities

| Feature | Description |
|---------|-------------|
| **VITS2 Architecture** | End-to-end: text → latent → waveform in a single model |
| **Hindi-First NLP** | Schwa deletion, sandhi rules, code-switching, currency/number expansion |
| **SSML Support** | `<prosody>`, `<break>`, `<emphasis>`, `<say-as>` tags |
| **Audio Effects Chain** | Compressor, limiter, de-esser, warmth filter, LUFS normalization |
| **Prosody Control** | Pitch shifting (±12st), speed control (0.5x–2x), energy scaling |
| **Production Server** | FastAPI with Prometheus metrics, request tracing, model versioning |
| **Docker + k8s** | Multi-stage GPU Docker build, Compose stack with monitoring |

## Quick Start

```bash
# Install
pip install -e ".[dev,train]"

# Run tests
make test

# Start inference server
make serve

# Build Docker image
make docker
```

## Project Structure

```
callex_tts_rnd/
├── configs/              # YAML configurations
│   ├── model/           # Model hyperparameters
│   ├── training/        # Training configs (DDP, augmentation)
│   └── inference/       # Server configs (rate limits, SSML)
├── src/callex_tts/      # Source code
│   ├── models/          # Neural network architecture
│   │   ├── encoder.py   # Transformer text encoder (RoPE + SwiGLU)
│   │   ├── flow.py      # Normalizing flows (affine coupling)
│   │   ├── vocoder.py   # HiFi-GAN v2 (Snake + MRF)
│   │   ├── discriminator.py  # MPD + MSD + MRD
│   │   ├── duration_predictor.py  # Stochastic duration predictor
│   │   └── generator.py # Full VITS2 synthesis network
│   ├── audio/           # Audio processing
│   │   ├── features.py  # Mel spectrogram extraction
│   │   ├── prosody.py   # Pitch/speed/energy control
│   │   ├── effects.py   # Compressor, limiter, de-esser, warmth
│   │   └── vocoder_postnet.py  # Neural post-net
│   ├── text/            # Text processing
│   │   ├── normalizer.py # 10-stage Hindi normalization
│   │   ├── phonemizer.py # G2P (Epitran/eSpeak)
│   │   ├── tokenizer.py  # IPA → integer tokenization
│   │   └── ssml.py       # SSML parser
│   ├── data/            # Dataset & augmentation
│   ├── training/        # Training loop & losses
│   └── serving/         # Production FastAPI server
├── tests/               # Test suite
├── scripts/             # Benchmarks & export tools
├── docs/                # Architecture documentation
├── Dockerfile           # Multi-stage GPU build
├── docker-compose.yml   # TTS + Prometheus + Grafana
├── Makefile             # Developer workflow
└── pyproject.toml       # PEP 621 packaging
```

## Architecture

See [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) for the complete system design.

## License

Proprietary — Callex AI Research / Lakhu Teleservices Pvt. Ltd.
All rights reserved.
