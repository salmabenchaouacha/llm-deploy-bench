"""Tests for the RAG pipeline: parsing, chunking, embedder, Qdrant client, end-to-end."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pytest
from matplotlib.backends.backend_pdf import PdfPages

from llmbench.config import (
    EmbedderConfig,
    QdrantConfig,
    RagConfig,
    load_rag_config,
)
from llmbench.rag.chunk import chunk_id, chunk_page
from llmbench.rag.embed import TEIEmbedder
from llmbench.rag.ingest import index_chunks, missing_documents, parse_corpus
from llmbench.rag.parse import PageText, clean_text, extract_pages
from llmbench.rag.pipeline import RagPipeline, Retriever, build_prompt
from llmbench.rag.store import Hit, QdrantStore
from tests.fake_llama_server import FakeState, fake_llama_server
from tests.fake_services import DIM, fake_rag_services

# ---------------------------------------------------------------- fixtures


def make_pdf(path: Path, pages: list[list[str]]) -> Path:
    """Real PDF with extractable text (one list of lines per page)."""
    with PdfPages(path) as pdf:
        for lines in pages:
            fig = plt.figure(figsize=(8.27, 11.69))
            for i, line in enumerate(lines):
                fig.text(0.08, 0.92 - i * 0.04, line, fontsize=11, family="DejaVu Sans")
            pdf.savefig(fig)
            plt.close(fig)
    return path


def rag_cfg(tmp_path: Path, url: str = "http://x", **kw: object) -> RagConfig:
    base = load_rag_config()
    return base.model_copy(
        update={
            "raw_dir": tmp_path,
            "chunks_file": tmp_path / "chunks.jsonl",
            "corpus": [base.corpus[0].model_copy(update={"file": "acme.pdf", "doc_id": "acme"})],
            "embedder": EmbedderConfig(url=url, dim=DIM, batch_size=3),
            "qdrant": QdrantConfig(url=url, collection="test"),
            **kw,
        }
    )


# ---------------------------------------------------------------- config


def test_repository_rag_yaml_is_valid() -> None:
    cfg = load_rag_config()
    assert {d.doc_id for d in cfg.corpus} >= {
        "airbus_fy2025",
        "totalenergies_fy2025",
        "schneider_q1_2025",
    }
    assert "{context}" in cfg.prompt_template


def test_rag_config_rejects_template_without_question() -> None:
    data = load_rag_config().model_dump()
    data["prompt_template"] = "Context: {context} -- no question placeholder here"
    with pytest.raises(ValueError, match="question"):
        RagConfig.model_validate(data)


# ---------------------------------------------------------------- parsing


def test_clean_text_compacts_table_columns() -> None:
    assert clean_text("Revenue        4,812      4,532\n\n\n\nNet") == (
        "Revenue | 4,812 | 4,532\n\nNet"
    )


def test_extract_pages_from_real_pdf(tmp_path: Path) -> None:
    pdf = make_pdf(tmp_path / "r.pdf", [["Revenue 2025 was 4,812 M EUR"], [], ["Net income"]])
    pages = extract_pages(pdf, "acme")
    assert [p.page for p in pages] == [1, 3]  # empty page 2 skipped, numbering kept
    assert "4,812" in pages[0].text


# ---------------------------------------------------------------- chunking


def test_chunks_respect_size_and_overlap() -> None:
    lines = [f"line {i:03d} " + "x" * 50 for i in range(60)]
    page = PageText("acme", 4, "\n".join(lines))
    chunks = chunk_page(page, max_chars=400, overlap_chars=120)
    assert len(chunks) > 5
    assert all(len(c.text) <= 400 for c in chunks)
    assert all(c.page == 4 and c.doc_id == "acme" for c in chunks)
    first_last_line = chunks[0].text.splitlines()[-1]
    assert first_last_line in chunks[1].text  # overlap carries the tail over
    joined = "\n".join(c.text for c in chunks)
    assert all(line in joined for line in lines)  # nothing lost


def test_chunk_ids_are_deterministic_and_unique() -> None:
    page = PageText("acme", 1, "\n".join("y" * 90 for _ in range(30)))
    a = chunk_page(page, 300, 50)
    b = chunk_page(page, 300, 50)
    assert [c.chunk_id for c in a] == [c.chunk_id for c in b]
    assert len({c.chunk_id for c in a}) == len(a)
    assert a[0].chunk_id == chunk_id("acme", 1, 0)


def test_overlong_line_is_split() -> None:
    chunks = chunk_page(PageText("acme", 1, "z" * 1000), 300, 50)
    assert all(len(c.text) <= 300 for c in chunks)
    assert sum(len(c.text) for c in chunks) >= 1000


# ---------------------------------------------------------------- services


def test_embedder_batches_and_applies_prefixes() -> None:
    with fake_rag_services() as (url, _, batches):
        cfg = EmbedderConfig(url=url, dim=DIM, batch_size=2, query_prefix="query: ")
        emb = TEIEmbedder(cfg)
        vectors = emb.embed_passages(["a", "b", "c", "d", "e"])
        assert len(vectors) == 5 and batches == [2, 2, 1]
        assert len(emb.embed_query("revenue")) == DIM


def test_embedder_rejects_wrong_dimension() -> None:
    with fake_rag_services() as (url, _, _), pytest.raises(ValueError, match="dim"):
        TEIEmbedder(EmbedderConfig(url=url, dim=DIM + 1)).embed_query("x")


def test_qdrant_store_roundtrip() -> None:
    with fake_rag_services() as (url, qdrant, _):
        store = QdrantStore(QdrantConfig(url=url, collection="c"))
        assert not store.exists()
        page = PageText("acme", 2, "Revenue grew to 4,812 million euros\nCash flow was strong")
        chunks = chunk_page(page, 40, 0)
        vectors = TEIEmbedder(EmbedderConfig(url=url, dim=DIM)).embed_passages(
            [c.text for c in chunks]
        )
        store.recreate(DIM)
        store.upsert(chunks, vectors, batch=1)
        assert store.count() == len(chunks)
        store.recreate(DIM)  # rebuild drops old points
        assert store.count() == 0
        assert ("DELETE", "/collections/c") in qdrant.requests


# ---------------------------------------------------------------- end to end


def test_ingest_then_answer_end_to_end(tmp_path: Path) -> None:
    make_pdf(tmp_path / "acme.pdf", [
        ["ACME full year 2025 results", "Revenue reached 4,812 million euros in 2025."],
        ["Outlook", "The dividend proposed is 3.20 euros per share."],
    ])  # fmt: skip
    with fake_rag_services() as (url, _, _):
        cfg = rag_cfg(tmp_path, url)
        assert missing_documents(cfg) == []
        chunks, stats = parse_corpus(cfg)
        assert stats[0]["pages_with_text"] == 2 and len(stats[0]["sha256"]) == 64
        store = QdrantStore(cfg.qdrant)
        embedder = TEIEmbedder(cfg.embedder)
        index_chunks(chunks, embedder, store)

        retriever = Retriever(embedder, store, k=1)
        top = retriever.retrieve("What dividend per share is proposed?")
        assert top[0].page == 2  # the dividend page wins

        state = FakeState(["The", "dividend", "is", "3.20", "euros", "[acme", "p.2]"])
        with fake_llama_server(state) as llm_url:
            ans = RagPipeline(retriever, cfg.prompt_template, 64).answer(
                "What dividend per share is proposed?", llm_url
            )
    assert "3.20" in ans.answer and ans.sources == ["acme p.2"]
    sent = state.requests[0]["messages"]
    assert len(sent) == 1 and sent[0]["role"] == "user"  # single user turn
    assert "[acme p.2]" in sent[0]["content"] and "3.20 euros" in sent[0]["content"]
    assert state.requests[0]["cache_prompt"] is False


def test_missing_documents_lists_download_url(tmp_path: Path) -> None:
    cfg = rag_cfg(tmp_path)
    missing = missing_documents(cfg)
    assert len(missing) == 1 and "download from https://" in missing[0]


def test_build_prompt_contains_citations() -> None:
    hits = [Hit("id", 0.9, "orange_fy2025", 7, "Revenue 40.3 bn")]
    prompt = build_prompt("C:\n{context}\nQ: {question}", " Revenue? ", hits)
    assert prompt == "C:\n[orange_fy2025 p.7]\nRevenue 40.3 bn\nQ: Revenue?"


def test_chunks_file_is_jsonl(tmp_path: Path) -> None:
    from llmbench.rag.ingest import write_chunks

    chunks = chunk_page(PageText("acme", 1, "a\nb"), 100, 0)
    write_chunks(chunks, tmp_path / "out" / "chunks.jsonl")
    row = json.loads((tmp_path / "out" / "chunks.jsonl").read_text().splitlines()[0])
    assert row["doc_id"] == "acme" and row["page"] == 1


def test_contextual_header_lets_retrieval_tell_companies_apart(tmp_path: Path) -> None:
    """Two reports with identical tables: only the header says which company is which."""
    table = ["Revenue | 4,812 | 4,532", "Operating margin | 12.1% | 11.4%"]
    make_pdf(tmp_path / "acme.pdf", [table])
    make_pdf(tmp_path / "globex.pdf", [table])
    base = rag_cfg(tmp_path)
    doc = base.corpus[0]
    corpus = [
        doc.model_copy(update={"doc_id": "acme", "file": "acme.pdf", "company": "Acme"}),
        doc.model_copy(update={"doc_id": "globex", "file": "globex.pdf", "company": "Globex"}),
    ]
    with fake_rag_services() as (url, _, _):
        cfg = rag_cfg(tmp_path, url, corpus=corpus)
        chunks, _ = parse_corpus(cfg)
        assert chunks[0].text.startswith("Acme — ")
        store, embedder = QdrantStore(cfg.qdrant), TEIEmbedder(cfg.embedder)
        index_chunks(chunks, embedder, store)
        top = Retriever(embedder, store, k=1).retrieve("Globex revenue")
    assert top[0].doc_id == "globex"


def test_chunk_header_with_unknown_field_is_rejected() -> None:
    data = load_rag_config().model_dump()
    data["chunk_header"] = "{ceo}"
    with pytest.raises(ValueError, match="chunk_header"):
        RagConfig.model_validate(data)
