# ╔══════════════════════════════════════════════════════════════════╗
# ║  CALLEX TTS — Development Workflow Makefile                    ║
# ╚══════════════════════════════════════════════════════════════════╝

.PHONY: help install install-dev install-train test lint serve train benchmark clean docker

PYTHON ?= python3
VENV   ?= .venv

help: ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-20s\033[0m %s\n", $$1, $$2}'

# ── Environment ──────────────────────────────────────────────────

install: ## Install core dependencies
	$(PYTHON) -m pip install -e .

install-dev: ## Install with development tools
	$(PYTHON) -m pip install -e ".[dev]"

install-train: ## Install with training dependencies (WandB, DeepSpeed)
	$(PYTHON) -m pip install -e ".[train]"

install-all: ## Install everything
	$(PYTHON) -m pip install -e ".[dev,train,export]"

# ── Quality ──────────────────────────────────────────────────────

test: ## Run test suite
	$(PYTHON) -m pytest tests/ -v --tb=short

lint: ## Run linter
	$(PYTHON) -m ruff check src/ tests/

lint-fix: ## Auto-fix lint issues
	$(PYTHON) -m ruff check --fix src/ tests/

typecheck: ## Run type checker
	$(PYTHON) -m mypy src/callex_tts/

# ── Serve ────────────────────────────────────────────────────────

serve: ## Start inference server (default: port 8124)
	$(PYTHON) -m callex_tts.serving.server

serve-dev: ## Start with hot-reload
	uvicorn callex_tts.serving.server:app --reload --host 0.0.0.0 --port 8124

# ── Train ────────────────────────────────────────────────────────

train: ## Start training with default config
	$(PYTHON) -m callex_tts.training.trainer --config configs/training/distributed.yaml

train-dry: ## Dry run — validate config without GPU
	$(PYTHON) -m callex_tts.training.trainer --config configs/training/distributed.yaml --dry-run

# ── Benchmark ────────────────────────────────────────────────────

benchmark: ## Run latency/throughput benchmarks
	$(PYTHON) scripts/benchmark.py

benchmark-dry: ## Benchmark dry run (no GPU required)
	$(PYTHON) scripts/benchmark.py --dry-run

# ── Export ────────────────────────────────────────────────────────

export-onnx: ## Export model to ONNX format
	$(PYTHON) scripts/export_onnx.py

# ── Docker ───────────────────────────────────────────────────────

docker: ## Build production Docker image
	docker build -t callex-tts:latest .

docker-dev: ## Start full development stack (TTS + Prometheus + Grafana)
	docker-compose up -d

docker-stop: ## Stop development stack
	docker-compose down

# ── Cleanup ──────────────────────────────────────────────────────

clean: ## Remove build artifacts and caches
	rm -rf build/ dist/ *.egg-info .pytest_cache .mypy_cache .ruff_cache
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete 2>/dev/null || true
