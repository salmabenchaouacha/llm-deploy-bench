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

Requirements: Python 3.11, [uv](https://docs.astral.sh/uv/), Git, Docker with NVIDIA GPU
support. On Windows, use WSL2 + Docker Desktop (Settings → Resources → WSL integration).

```bash
make install        # core + dev dependencies
make check          # lint + type-check + tests
```

### Performance benchmark (Phase 1)

GGUF models run in the official `llama.cpp` server image (CUDA build), the same engine
used for serving. Weights are cached in `~/.cache/llmbench/models` (outside the repo).

```bash
make gpu-check      # the container must list the GPU
make download       # ~12 GB of GGUF files
make bench          # every enabled variant -> results/perf/summary.csv + runs.jsonl
make bench-sweep    # Mistral-7B: layers offloaded to GPU vs speed / VRAM
make report         # results/perf/report.md + figures, generated from the CSV
```

Protocol (`configs/bench.yaml`): 3 prompts (short, French, long RAG-like context),
1 warm-up + 3 repetitions each, greedy decoding, exactly 128 generated tokens
(`ignore_eos`), prompt cache disabled. Measured per variant: file size, load time,
time-to-first-token (client side), decode tokens/s, prefill tokens/s (server timings),
peak VRAM (NVML, device-level minus baseline) and peak container RAM. A variant that does
not fit is recorded with `error=cuda_oom`.

### RAG pipeline (Phase 2)

Corpus: three results press releases listed in `configs/rag.yaml` — Airbus FY2025,
TotalEnergies Q4/FY2025, Schneider Electric Q1 2025 (download each `url` into `data/raw/`;
PDFs are not versioned). Pipeline: pypdf (layout mode, keeps table columns) →
page-bounded chunks (~300 tokens, overlap) with a contextual header (`Company — Title`, so a
bare table row still says which company it belongs to) → `BAAI/bge-m3` embeddings served by
Text Embeddings Inference (CPU, keeps the GPU for the LLM) → Qdrant → top-k context → any
model/quantization served by llama.cpp. The retriever is fixed, so every model receives the
same context.

```bash
make rag-up                     # Qdrant + embedding server (Docker)
make ingest                     # PDFs -> chunks -> embeddings -> Qdrant -> results/rag/ingest.json
make ask Q="How many commercial aircraft did Airbus deliver in 2025?"
make ask Q="Quel est le résultat net ajusté de TotalEnergies en 2025 ?" MODEL=llama3.2-3b
uv run python -m llmbench.rag.ask "..." --retrieval-only   # inspect retrieved chunks only
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
