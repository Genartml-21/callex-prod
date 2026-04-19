"""
Callex TTS — Latency/Throughput Benchmark Script

Usage:
    python scripts/benchmark.py              # Full GPU benchmark
    python scripts/benchmark.py --dry-run    # CPU-only validation
"""

import argparse
import time
import logging
import torch

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(message)s")
logger = logging.getLogger("benchmark")


def benchmark_text_pipeline():
    """Benchmark the text processing pipeline."""
    from callex_tts.text.normalizer import HindiTextNormalizer
    from callex_tts.text.tokenizer import CallexTokenizer

    normalizer = HindiTextNormalizer()
    tokenizer = CallexTokenizer()

    test_texts = [
        "नमस्ते, मैं कैलेक्स एआई हूँ।",
        "₹500 का EMI भरना है Dr. शर्मा",
        "आपका OTP 4 5 6 7 है। कृपया किसी को न बताएं।",
        "आज मौसम बहुत अच्छा है, बाहर चलते हैं।",
        "कृपया 1 से 9 तक कोई नंबर दबाएं।",
    ]

    # Warmup
    for text in test_texts:
        normalizer.normalize(text)

    # Benchmark
    n_iterations = 1000
    start = time.time()
    for _ in range(n_iterations):
        for text in test_texts:
            normalized = normalizer.normalize(text)
            tokens = tokenizer.encode(normalized)
    elapsed = time.time() - start

    total_calls = n_iterations * len(test_texts)
    logger.info(
        "Text Pipeline: %d calls in %.2fs → %.1f calls/sec (%.2f ms/call)",
        total_calls, elapsed,
        total_calls / elapsed,
        elapsed / total_calls * 1000,
    )


def benchmark_audio_effects():
    """Benchmark the audio effects chain."""
    import numpy as np
    from callex_tts.audio.effects import AudioEffectsChain, EffectsChainConfig

    chain = AudioEffectsChain(EffectsChainConfig(sample_rate=24000))

    # Simulate 5 seconds of audio
    audio = np.random.randn(24000 * 5).astype(np.float32) * 0.5

    # Warmup
    chain.process(audio[:24000])

    # Benchmark
    n_iterations = 50
    start = time.time()
    for _ in range(n_iterations):
        chain.process(audio)
    elapsed = time.time() - start

    logger.info(
        "Audio Effects: %d iterations of 5s audio in %.2fs → %.1f× realtime",
        n_iterations, elapsed,
        (n_iterations * 5.0) / elapsed,
    )


def benchmark_model_construction():
    """Benchmark model instantiation."""
    from callex_tts.models.generator import CallexSynthesisNetwork

    start = time.time()
    model = CallexSynthesisNetwork(
        vocab_size=196,
        hidden_channels=192,
        use_snake=False,  # CPU doesn't benefit from Snake
    )
    elapsed = time.time() - start

    total_params = sum(p.numel() for p in model.parameters())
    logger.info(
        "Model Construction: %.2fs, %.2fM parameters",
        elapsed, total_params / 1e6,
    )


def main():
    parser = argparse.ArgumentParser(description="Callex TTS Benchmark")
    parser.add_argument("--dry-run", action="store_true", help="CPU-only, no GPU required")
    args = parser.parse_args()

    logger.info("═" * 60)
    logger.info("  CALLEX TTS — Performance Benchmark")
    logger.info("═" * 60)
    logger.info("Device: %s", "CUDA" if torch.cuda.is_available() and not args.dry_run else "CPU")
    logger.info("")

    benchmark_text_pipeline()
    benchmark_audio_effects()
    benchmark_model_construction()

    logger.info("")
    logger.info("═" * 60)
    logger.info("  Benchmark complete ✅")
    logger.info("═" * 60)


if __name__ == "__main__":
    main()
