# Single entry point for every task. Run `make help` for the list.
# Requires uv (https://docs.astral.sh/uv/). On Windows, run from WSL2.

.DEFAULT_GOAL := help
.PHONY: help install install-gguf install-gpu lint format typecheck test test-all \
        check precommit config clean

help: ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## ' $(MAKEFILE_LIST) | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-14s\033[0m %s\n", $$1, $$2}'

install: ## Create .venv with core + dev deps and install git hooks
	uv sync
	uv run pre-commit install

install-gguf: ## Add the llama.cpp backend (CPU build)
	uv sync --extra gguf

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

clean: ## Remove caches (keeps .venv, data and results)
	rm -rf .pytest_cache .mypy_cache .ruff_cache .coverage htmlcov
	find . -type d -name __pycache__ -prune -exec rm -rf {} +
