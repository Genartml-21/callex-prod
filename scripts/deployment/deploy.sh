#!/bin/bash
#
# Complete Deployment Script for Enterprise Speaker Verification
# Run this script on your server after uploading files
#

set -e  # Exit on any error

echo "════════════════════════════════════════════════════════════"
echo "  Enterprise Speaker Verification - Server Setup"
echo "════════════════════════════════════════════════════════════"
echo ""

# Configuration
APP_DIR="/path/to/application"  # UPDATE THIS PATH
HF_TOKEN="${HF_TOKEN:-}"  # Set via: export HF_TOKEN="your-token"

echo "📍 Working directory: $APP_DIR"
cd "$APP_DIR"

echo ""
echo "🔐 Setting up HuggingFace authentication..."
export HF_TOKEN="$HF_TOKEN"
export HUGGINGFACE_HUB_TOKEN="$HF_TOKEN"

# Login to HuggingFace
pip3 install -q huggingface_hub
huggingface-cli login --token "$HF_TOKEN" --add-to-git-credential

echo "✅ HuggingFace authentication configured"
echo ""

echo "📦 Installing dependencies..."
echo "   (This may take 5-10 minutes for first-time installation)"
pip3 install -r requirements.txt

echo ""
echo "✅ Dependencies installed successfully"
echo ""

echo "🧪 Testing imports..."
python3 -c "
from speaker_verification import SpeakerVerifier, SemanticIntentVerifier
print('  ✓ speaker_verification module loaded')
from audio_gating import AudioGatingPipeline, AudioGatingConfig
print('  ✓ audio_gating module loaded')
import main
print('  ✓ main module loaded')
print('')
print('✅ All modules imported successfully!')
"

echo ""
echo "🚀 Restarting application..."
pm2 stop Callex-AI 2>/dev/null || true
pm2 delete Callex-AI 2>/dev/null || true
pm2 start main.py --name "Callex-AI" --interpreter python3
pm2 save

echo ""
echo "════════════════════════════════════════════════════════════"
echo "  ✅ DEPLOYMENT COMPLETE!"
echo "════════════════════════════════════════════════════════════"
echo ""
echo "📊 Monitor logs with:"
echo "   pm2 logs Callex-AI"
echo ""
echo "Expected startup messages:"
echo "  [Speaker Verification] Loading pyannote embedding model..."
echo "  [Speaker Verification] Model loaded successfully"
echo "  [Semantic Intent] Loading faster-whisper (tiny) model..."
echo "  [Semantic Intent] Model loaded successfully"
echo "  [Audio Gating] Initialized 5-layer validation pipeline"
echo ""
echo "🎉 Your system now has enterprise speaker verification!"
