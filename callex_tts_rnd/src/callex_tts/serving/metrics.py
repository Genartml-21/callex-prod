"""
╔══════════════════════════════════════════════════════════════════════╗
║  CALLEX TTS — Prometheus Metrics Exporter                           ║
╚══════════════════════════════════════════════════════════════════════╝
"""

from __future__ import annotations

import logging
from fastapi.responses import Response

logger = logging.getLogger("callex.tts.serving.metrics")

try:
    from prometheus_client import (
        Counter, Histogram, Gauge, Info,
        generate_latest, CONTENT_TYPE_LATEST,
    )
    PROMETHEUS_AVAILABLE = True
except ImportError:
    PROMETHEUS_AVAILABLE = False
    logger.warning("prometheus_client not installed — metrics disabled")


class _DummyMetric:
    """No-op metric when prometheus_client is not installed."""
    def inc(self, *args, **kwargs): pass
    def dec(self, *args, **kwargs): pass
    def set(self, *args, **kwargs): pass
    def observe(self, *args, **kwargs): pass
    def info(self, *args, **kwargs): pass
    def labels(self, *args, **kwargs): return self


class PrometheusMetrics:
    """
    Prometheus metrics for the TTS inference server.
    
    Exports:
      • callex_tts_requests_total — Total HTTP requests
      • callex_tts_request_latency_seconds — Request latency histogram
      • callex_tts_synthesis_requests_total — Synthesis requests
      • callex_tts_synthesis_latency_seconds — Synthesis latency
      • callex_tts_synthesis_errors_total — Failed syntheses
      • callex_tts_gpu_memory_bytes — GPU memory usage gauge
      • callex_tts_model_info — Model version metadata
    """

    def __init__(self):
        if PROMETHEUS_AVAILABLE:
            self.request_count = Counter(
                "callex_tts_requests_total",
                "Total HTTP requests to TTS server",
            )
            self.request_latency = Histogram(
                "callex_tts_request_latency_seconds",
                "HTTP request latency",
                buckets=(0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0),
            )
            self.synthesis_requests = Counter(
                "callex_tts_synthesis_requests_total",
                "Total synthesis requests",
            )
            self.synthesis_latency = Histogram(
                "callex_tts_synthesis_latency_seconds",
                "Synthesis latency",
                buckets=(0.05, 0.1, 0.25, 0.5, 1.0, 2.0, 5.0, 10.0),
            )
            self.synthesis_errors = Counter(
                "callex_tts_synthesis_errors_total",
                "Total failed synthesis requests",
            )
            self.gpu_memory = Gauge(
                "callex_tts_gpu_memory_bytes",
                "GPU memory usage in bytes",
            )
            self.active_requests = Gauge(
                "callex_tts_active_requests",
                "Currently in-flight requests",
            )
            self.model_info = Info(
                "callex_tts_model",
                "Loaded model version info",
            )
        else:
            self.request_count = _DummyMetric()
            self.request_latency = _DummyMetric()
            self.synthesis_requests = _DummyMetric()
            self.synthesis_latency = _DummyMetric()
            self.synthesis_errors = _DummyMetric()
            self.gpu_memory = _DummyMetric()
            self.active_requests = _DummyMetric()
            self.model_info = _DummyMetric()

    def generate_latest(self) -> Response:
        """Generate Prometheus-formatted metrics response."""
        if PROMETHEUS_AVAILABLE:
            # Update GPU memory gauge
            try:
                import torch
                if torch.cuda.is_available():
                    self.gpu_memory.set(torch.cuda.memory_allocated(0))
            except Exception:
                pass

            return Response(
                content=generate_latest(),
                media_type=CONTENT_TYPE_LATEST,
            )
        return Response(content="# prometheus_client not installed\n", media_type="text/plain")
