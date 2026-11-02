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

### Quality evaluation (Phase 3)

Evaluation set `data/eval/questions.jsonl`: questions drafted from the corpus, then each one
reviewed by hand (`make eval-validate`: validate / edit / reject, source page shown). Only
validated items are scored. Metrics are deterministic (no LLM judge, reproducible, free):

| Metric | Meaning |
|---|---|
| retrieval hit@k, MRR | a chunk from the source page is among the top-k retrieved |
| answer accuracy | every key fact (exact figure, normalised: `15,587` = `15587`, `15,6` = `15.6`) is in the answer |
| fact recall | share of key facts found |
| hallucination rate | unanswerable questions answered anyway |
| false refusal rate | answerable questions the model refused |

```bash
make eval-validate   # human review of the drafted questions
make eval-smoke      # 3 questions, 1 model: checks the whole chain
make eval            # every variant of configs/eval.yaml -> results/eval/summary.csv + answers/
make eval-report     # results/eval/report.md: accuracy table + quality-vs-speed figure
```

### Serving API and load test (Phase 4)

One model served permanently (default: Llama 3.2 3B Q4_K_M, the best quality/speed
trade-off of Phase 3) by llama-server with continuous batching (4 parallel slots), behind a
FastAPI gateway that speaks the OpenAI API:

| Endpoint | |
|---|---|
| `POST /v1/chat/completions` | OpenAI-compatible, `stream: true` relays Server-Sent Events token by token |
| `GET /v1/models` | served model |
| `POST /rag/query` | `{"question": "..."}` -> answer + sources (doc, page, score) + timings |
| `GET /health` | 200 only when llm, embedder and qdrant all answer |
| `GET /docs` | interactive OpenAPI documentation |

```bash
make serve-up        # builds the API image, starts llm + api + qdrant + embedder
curl -s localhost:8000/health
curl -s localhost:8000/rag/query -H 'Content-Type: application/json' \
     -d '{"question": "How many commercial aircraft did Airbus deliver in 2025?"}'
make load            # Locust, chat endpoint, 1 / 5 / 10 concurrent users -> results/load/
make load-rag        # same on the full RAG pipeline
make load-report     # results/load/report.md: throughput and latency vs users
make serve-down
```

Any OpenAI client works: `OpenAI(base_url="http://localhost:8000/v1", api_key="unused")`.

## Project layout

| Path | Content |
|---|---|
| `src/llmbench/` | Installable package: `models`, `rag`, `eval`, `serving`, `api`, `load`, `monitoring` |
| `load/` | Locust scenarios (`locustfile.py`) |
| `docker/` | API image (`api.Dockerfile`) |
| `configs/` | YAML configuration, validated by Pydantic (`src/llmbench/config.py`) |
| `data/` | Raw / processed corpus (git-ignored) and the hand-validated eval set (`data/eval/`) |
| `results/` | Raw measurements (CSV/JSON), versioned |
| `tests/` | Pytest suite |

## Technical choices

_TODO_

## Limitations & next steps

_TODO_
