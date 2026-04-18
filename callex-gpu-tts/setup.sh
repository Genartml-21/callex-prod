#!/bin/bash
# ╔══════════════════════════════════════════════════════════════╗
# ║  CALLEX GPU TTS — One-Command Setup Script                  ║
# ║  Run this on your GPU server and everything installs itself  ║
# ╚══════════════════════════════════════════════════════════════╝

set -e

echo ""
echo "═══════════════════════════════════════════════════════════"
echo "  🚀 CALLEX GPU TTS SERVER — Automated Setup"
echo "═══════════════════════════════════════════════════════════"
echo ""

# 1. Create virtual environment
echo "[1/4] Creating Python virtual environment..."
python3 -m venv venv
source venv/bin/activate

# 2. Install GPU PyTorch
echo "[2/4] Installing GPU-accelerated PyTorch (CUDA 11.8)..."
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# 3. Install TTS dependencies
echo "[3/4] Installing TTS engine dependencies..."
pip install -r requirements.txt

# 4. Create reference voice placeholder
echo "[4/4] Checking reference voice file..."
if [ ! -f "reference.wav" ]; then
    echo ""
    echo "  ⚠️  IMPORTANT: You need to place your reference voice file!"
    echo "  Copy a 5-10 second WAV recording of your desired voice to:"
    echo "  $(pwd)/reference.wav"
    echo ""
fi

echo ""
echo "═══════════════════════════════════════════════════════════"
echo "  ✅ SETUP COMPLETE!"
echo ""
echo "  To start the server:"
echo "    source venv/bin/activate"
echo "    python tts_server.py"
echo ""
echo "  Or with PM2 (recommended for production):"
echo "    pm2 start tts_server.py --name cx-gpu-tts --interpreter ./venv/bin/python3"
echo ""
echo "  Health check:"
echo "    curl http://localhost:8124/health"
echo "═══════════════════════════════════════════════════════════"
