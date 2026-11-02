"""Ask a question to the RAG pipeline with any model / quantization.

CLI::

    python -m llmbench.rag.ask "What was TotalEnergies' 2025 net income?" --retrieval-only
    python -m llmbench.rag.ask "Quel est le chiffre d'affaires 2025 d'Orange ?" \
        --model qwen2.5-1.5b --quant gguf_q4_k_m
"""

from __future__ import annotations

import argparse
import textwrap

from llmbench.config import (
    DEFAULT_BENCH_CONFIG,
    DEFAULT_MODELS_CONFIG,
    DEFAULT_RAG_CONFIG,
    ConfigError,
    load_bench_config,
    load_models_config,
    load_rag_config,
)
from llmbench.models.hub import ensure_gguf, gguf_relpath
from llmbench.rag.embed import TEIEmbedder
from llmbench.rag.pipeline import RagPipeline, Retriever
from llmbench.rag.store import Hit, QdrantStore
from llmbench.serving.llama_server import LlamaServer


def print_hits(hits: list[Hit]) -> None:
    for i, h in enumerate(hits, 1):
        preview = textwrap.shorten(h.text.replace("\n", " "), 160)
        print(f"  {i}. [{h.doc_id} p.{h.page}] score={h.score:.3f}  {preview}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Ask the RAG pipeline a question.")
    parser.add_argument("question")
    parser.add_argument("--model", default="qwen2.5-1.5b")
    parser.add_argument("--quant", default="gguf_q4_k_m")
    parser.add_argument("--top-k", type=int)
    parser.add_argument("--retrieval-only", action="store_true", help="no LLM, show chunks")
    parser.add_argument("--rag-config", default=DEFAULT_RAG_CONFIG)
    parser.add_argument("--models-config", default=DEFAULT_MODELS_CONFIG)
    parser.add_argument("--bench-config", default=DEFAULT_BENCH_CONFIG)
    args = parser.parse_args(argv)

    try:
        rag = load_rag_config(args.rag_config)
        variant = load_models_config(args.models_config).get(args.model).variant(args.quant)
        server_cfg = load_bench_config(args.bench_config).server
    except (ConfigError, KeyError) as exc:
        print(f"ERROR: {exc.args[0] if exc.args else exc}")
        return 2

    retriever = Retriever(TEIEmbedder(rag.embedder), QdrantStore(rag.qdrant),
                          args.top_k or rag.top_k)  # fmt: skip
    if args.retrieval_only:
        print(f"Top-{retriever.k} chunks for: {args.question}")
        print_hits(retriever.retrieve(args.question))
        return 0

    ensure_gguf(variant, server_cfg.models_path)
    pipeline = RagPipeline(retriever, rag.prompt_template, rag.max_new_tokens)
    with LlamaServer.from_config(server_cfg, gguf_relpath(variant), variant.n_gpu_layers) as srv:
        ans = pipeline.answer(args.question, srv.base_url)

    print(f"\nQ: {ans.question}\n")
    print(textwrap.fill(ans.answer, 100))
    print(f"\nSources retrieved ({args.model} {args.quant}):")
    print_hits(ans.hits)
    print(f"\nretrieval {ans.retrieval_ms:.0f} ms | TTFT {ans.ttft_ms:.0f} ms | "
          f"total {ans.total_ms:.0f} ms | prompt {ans.prompt_tokens} tok | "
          f"answer {ans.gen_tokens} tok")  # fmt: skip
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
