"""Build the vector index: PDFs -> pages -> chunks -> embeddings -> Qdrant.

Writes ``data/processed/chunks.jsonl`` (all chunks, used to draft evaluation questions)
and ``results/rag/ingest.json`` (corpus statistics + file hashes, versioned).

CLI::

    python -m llmbench.rag.ingest              # full rebuild
    python -m llmbench.rag.ingest --dry-run    # parse + chunk only, no services needed
"""

from __future__ import annotations

import argparse
import hashlib
import json
import statistics
import time
from dataclasses import replace
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from llmbench.config import DEFAULT_RAG_CONFIG, RESULTS_DIR, RagConfig, load_rag_config
from llmbench.log import get_logger
from llmbench.models.bench import git_commit
from llmbench.rag.chunk import Chunk, chunk_pages
from llmbench.rag.embed import Embedder, TEIEmbedder
from llmbench.rag.parse import extract_pages
from llmbench.rag.services import wait_for_rag_services
from llmbench.rag.store import QdrantStore

log = get_logger(__name__)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def missing_documents(cfg: RagConfig) -> list[str]:
    raw = cfg.resolve(cfg.raw_dir)
    return [
        f"  {raw / d.file}\n    download from {d.url}"
        for d in cfg.corpus
        if not (raw / d.file).is_file()
    ]


def parse_corpus(cfg: RagConfig) -> tuple[list[Chunk], list[dict[str, Any]]]:
    raw = cfg.resolve(cfg.raw_dir)
    chunks: list[Chunk] = []
    stats: list[dict[str, Any]] = []
    for doc in cfg.corpus:
        path = raw / doc.file
        pages = extract_pages(path, doc.doc_id)
        doc_chunks = chunk_pages(pages, cfg.chunking.max_chars, cfg.chunking.overlap_chars)
        header = cfg.chunk_header.format(company=doc.company, title=doc.title, doc_id=doc.doc_id)
        if header:
            doc_chunks = [replace(c, text=f"{header}\n{c.text}") for c in doc_chunks]
        chunks.extend(doc_chunks)
        stats.append({
            "doc_id": doc.doc_id,
            "file": doc.file,
            "sha256": sha256(path),
            "pages_with_text": len(pages),
            "chunks": len(doc_chunks),
            "chars": sum(len(p.text) for p in pages),
        })  # fmt: skip
        log.info("parsed", extra={"doc": doc.doc_id, "pages": len(pages),
                                  "chunks": len(doc_chunks)})  # fmt: skip
    return chunks, stats


def write_chunks(chunks: list[Chunk], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for c in chunks:
            fh.write(json.dumps(c.to_dict(), ensure_ascii=False) + "\n")


def index_chunks(chunks: list[Chunk], embedder: Embedder, store: QdrantStore) -> float:
    """Embed and upsert every chunk. Returns the embedding time in seconds."""
    t0 = time.perf_counter()
    vectors = embedder.embed_passages([c.text for c in chunks])
    embed_s = time.perf_counter() - t0
    store.recreate(embedder.dim)
    store.upsert(chunks, vectors)
    stored = store.count()
    if stored != len(chunks):
        raise RuntimeError(f"Qdrant holds {stored} points, expected {len(chunks)}")
    return embed_s


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Build the RAG vector index.")
    parser.add_argument("--config", default=DEFAULT_RAG_CONFIG)
    parser.add_argument("--dry-run", action="store_true", help="parse + chunk only")
    parser.add_argument("--results-dir", type=Path, default=RESULTS_DIR / "rag")
    args = parser.parse_args(argv)

    cfg = load_rag_config(args.config)
    missing = missing_documents(cfg)
    if missing:
        print("ERROR: missing corpus documents:\n" + "\n".join(missing))
        return 2

    chunks, doc_stats = parse_corpus(cfg)
    if not chunks:
        print("ERROR: no text extracted (scanned PDFs?)")
        return 2
    write_chunks(chunks, cfg.resolve(cfg.chunks_file))
    lengths = [len(c.text) for c in chunks]
    print(f"{len(chunks)} chunks from {len(doc_stats)} documents "
          f"(median {statistics.median(lengths):.0f} chars) -> {cfg.chunks_file}")  # fmt: skip
    if args.dry_run:
        return 0

    wait_for_rag_services(cfg)
    embedder = TEIEmbedder(cfg.embedder)
    embed_s = index_chunks(chunks, embedder, QdrantStore(cfg.qdrant))

    report = {
        "timestamp": datetime.now(UTC).isoformat(timespec="seconds"),
        "git_commit": git_commit(),
        "embedder": {
            "model_id": cfg.embedder.model_id,
            "image": cfg.embedder.image,
            "dim": cfg.embedder.dim,
        },
        "qdrant": {"image": cfg.qdrant.image, "collection": cfg.qdrant.collection},
        "chunking": cfg.chunking.model_dump(),
        "chunk_header": cfg.chunk_header,
        "documents": doc_stats,
        "total_chunks": len(chunks),
        "chunk_chars_median": statistics.median(lengths),
        "embed_seconds": round(embed_s, 2),
        "embed_chunks_per_s": round(len(chunks) / embed_s, 2) if embed_s else None,
    }
    args.results_dir.mkdir(parents=True, exist_ok=True)
    out = args.results_dir / "ingest.json"
    out.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"Indexed {len(chunks)} chunks in Qdrant in {embed_s:.1f}s of embedding -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
