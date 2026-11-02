# Single entry point for every task. Run `make help` for the list.
# Requires uv (https://docs.astral.sh/uv/). On Windows, run from WSL2.

.DEFAULT_GOAL := help
.PHONY: help install install-gpu lint format typecheck test test-all check precommit \
        config gpu-check download bench bench-sweep report rag-up rag-down ingest ask eval-check eval-validate eval-smoke eval eval-report clean

LLAMA_IMAGE ?= ghcr.io/ggml-org/llama.cpp:server-cuda

help: ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## ' $(MAKEFILE_LIST) | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-14s\033[0m %s\n", $$1, $$2}'

install: ## Create .venv with core + dev deps and install git hooks
	uv sync
	uv run pre-commit install

install-gpu: ## Add the Transformers/bitsandbytes backend (NVIDIA GPU)
	uv sync --extra gpu

lint: ## Lint with ruff
	uv run ruff check .
	uv run ruff format --check .

format: ## Auto-fix lint issues and format code
	uv run ruff check --fix .
	uv run ruff format .

typecheck: ## Static type-check with mypy (strict)
	uv run mypy

test: ## Fast tests only (no model loading)
	uv run pytest -m "not slow and not gpu"

test-all: ## All tests, including slow ones that load real models
	uv run pytest

check: lint typecheck test ## Everything CI runs

precommit: ## Run all pre-commit hooks on every file
	uv run pre-commit run --all-files

config: ## Validate configs/models.yaml and list the variants
	uv run python -m llmbench.config

gpu-check: ## Check that Docker containers can see the NVIDIA GPU
	docker pull $(LLAMA_IMAGE)
	docker run --rm --gpus all --entrypoint nvidia-smi $(LLAMA_IMAGE)

download: ## Download every enabled GGUF model (~12 GB, cached outside the repo)
	uv run python -m llmbench.models.hub --all

bench: ## Performance benchmark of every enabled GGUF variant -> results/perf/
	uv run python -m llmbench.models.bench --all

bench-sweep: ## Mistral-7B GPU offload sweep (layers on GPU vs speed) -> results/perf/
	uv run python -m llmbench.models.bench --model mistral-7b --quant gguf_q4_k_m \
		--n-gpu-layers 0 8 16 20 24 28 32

report: ## Tables + figures from results/perf/summary.csv -> results/perf/report.md
	uv run python -m llmbench.models.report

rag-up: ## Start Qdrant + embedding server (first start downloads bge-m3, ~2.3 GB)
	docker compose up -d qdrant embedder

rag-down: ## Stop Qdrant + embedding server (index kept in a Docker volume)
	docker compose stop qdrant embedder

ingest: ## Parse data/raw/*.pdf, chunk, embed and index into Qdrant -> results/rag/ingest.json
	uv run python -m llmbench.rag.ingest

Q ?= What was the net income in 2025?
MODEL ?= qwen2.5-1.5b
QUANT ?= gguf_q4_k_m
ask: ## Ask the RAG pipeline: make ask Q="..." MODEL=qwen2.5-1.5b QUANT=gguf_q4_k_m
	uv run python -m llmbench.rag.ask "$(Q)" --model $(MODEL) --quant $(QUANT)

eval-check: ## Check every key fact of the eval set appears on its source page
	uv run python -m llmbench.eval.validate --check-only

eval-validate: ## Review draft evaluation questions by hand (validate / edit / reject)
	uv run python -m llmbench.eval.validate

eval-smoke: ## Quick end-to-end check: 3 questions, one model, drafts allowed (not reported)
	uv run python -m llmbench.eval.run --model qwen2.5-0.5b --include-drafts --limit 3 \
		--out-dir /tmp/llmbench-eval-smoke

eval: ## RAG quality, every variant of configs/eval.yaml (resumes: skips variants already done)
	uv run python -m llmbench.eval.run --all --resume

eval-report: ## Quality table + quality-vs-speed figure -> results/eval/report.md
	uv run python -m llmbench.eval.report

clean: ## Remove caches (keeps .venv, data and results)
	rm -rf .pytest_cache .mypy_cache .ruff_cache .coverage htmlcov
	find . -type d -name __pycache__ -prune -exec rm -rf {} +
