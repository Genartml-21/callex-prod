# ╔══════════════════════════════════════════════════════════════════════╗
# ║  CALLEX TTS — Multi-Stage GPU Docker Build                        ║
# ║                                                                    ║
# ║  Stage 1: Builder — compile Python dependencies with CUDA          ║
# ║  Stage 2: Runtime — slim image with only runtime deps              ║
# ╚══════════════════════════════════════════════════════════════════════╝

# ── Stage 1: Builder ──────────────────────────────────────────────────
FROM nvidia/cuda:11.8.0-devel-ubuntu22.04 AS builder

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.10 python3.10-dev python3.10-venv python3-pip \
    build-essential git curl \
    libsndfile1-dev libffi-dev \
    && rm -rf /var/lib/apt/lists/*

RUN python3.10 -m pip install --upgrade pip setuptools wheel

WORKDIR /build

# Install PyTorch with CUDA 11.8
RUN pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu118

# Install project dependencies
COPY pyproject.toml .
COPY src/ src/
RUN pip install --no-cache-dir ".[train]"

# ── Stage 2: Runtime ──────────────────────────────────────────────────
FROM nvidia/cuda:11.8.0-runtime-ubuntu22.04 AS runtime

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV NVIDIA_VISIBLE_DEVICES=all
ENV NVIDIA_DRIVER_CAPABILITIES=compute,utility

# Runtime system deps only
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.10 python3.10-venv \
    libsndfile1 \
    curl \
    tini \
    && rm -rf /var/lib/apt/lists/* \
    && ln -sf /usr/bin/python3.10 /usr/bin/python3

# Copy installed Python packages from builder
COPY --from=builder /usr/local/lib/python3.10/dist-packages /usr/local/lib/python3.10/dist-packages
COPY --from=builder /usr/local/bin /usr/local/bin

# Non-root user
RUN groupadd -r callex && useradd -r -g callex -m callex
WORKDIR /app

# Copy application
COPY --chown=callex:callex . /app/

USER callex

# Expose ports
EXPOSE 8124

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=120s --retries=3 \
    CMD curl -f http://localhost:8124/health || exit 1

# Use tini for proper signal handling
ENTRYPOINT ["tini", "--"]

# Default: run inference server
CMD ["python3", "-m", "callex_tts.serving.server"]
