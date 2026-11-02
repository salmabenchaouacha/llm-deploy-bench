# Open-Weight LLM Deployment & Benchmarking Platform

Deploy open-weight LLMs (Qwen, Llama, Mistral) locally in several quantization formats,
compare them on quality (RAG + benchmarks), latency and memory, serve them through an
OpenAI-compatible API, fine-tune a small model with QLoRA, and monitor everything.

> 🚧 Work in progress — sections are filled phase by phase. Every number in this README
> comes from a raw file in [`results/`](results/).

## Problem

_TODO (Phase 6)_

## Architecture

_TODO (Phase 6 — Mermaid diagram)_

## Key results

_TODO — generated from `results/`_

## Quickstart

Requirements: Python 3.11, [uv](https://docs.astral.sh/uv/), Git. On Windows, use WSL2.

```bash
make install        # core + dev dependencies
make check          # lint + type-check + tests
```

Optional backends are installed per need:

```bash
uv sync --extra gguf   # llama.cpp (CPU or GPU)
uv sync --extra gpu    # bitsandbytes / transformers on NVIDIA GPU
```

## Project layout

| Path | Content |
|---|---|
| `src/llmbench/` | Installable package: `models`, `rag`, `eval`, `serving`, `finetune`, `monitoring` |
| `configs/` | YAML configuration, validated by Pydantic (`src/llmbench/config.py`) |
| `data/` | Raw / processed corpus (git-ignored) and the hand-validated eval set (`data/eval/`) |
| `results/` | Raw measurements (CSV/JSON), versioned |
| `tests/` | Pytest suite |

## Technical choices

_TODO_

## Limitations & next steps

_TODO_
