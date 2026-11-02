"""Retrieval-augmented generation: same retriever, any served LLM.

    question -> embed (TEI) -> top-k chunks (Qdrant) -> prompt -> LLM (llama-server)

The retriever is independent of the generator: for the quality benchmark every model
receives exactly the same chunks, so differences come from the model alone.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

from llmbench.rag.embed import Embedder
from llmbench.rag.store import Hit, QdrantStore
from llmbench.serving.client import StreamResult, stream_chat

NO_ANSWER = "The provided documents do not contain this information."


@dataclass
class RagAnswer:
    question: str
    answer: str
    hits: list[Hit]
    retrieval_ms: float
    ttft_ms: float
    total_ms: float
    gen_tokens: int | None = None
    prompt_tokens: int | None = None
    extra: dict[str, object] = field(default_factory=dict)

    @property
    def sources(self) -> list[str]:
        return [f"{h.doc_id} p.{h.page}" for h in self.hits]


class Retriever:
    def __init__(self, embedder: Embedder, store: QdrantStore, k: int) -> None:
        self.embedder = embedder
        self.store = store
        self.k = k

    def retrieve(self, question: str, k: int | None = None) -> list[Hit]:
        return self.store.search(self.embedder.embed_query(question), k or self.k)


def format_context(hits: list[Hit]) -> str:
    return "\n\n---\n\n".join(f"[{h.doc_id} p.{h.page}]\n{h.text}" for h in hits)


def build_prompt(template: str, question: str, hits: list[Hit]) -> str:
    return template.format(context=format_context(hits), question=question.strip())


def generate(base_url: str, prompt: str, max_tokens: int) -> StreamResult:
    return stream_chat(
        base_url,
        [{"role": "user", "content": prompt}],
        max_tokens=max_tokens,
        temperature=0.0,
        seed=42,
        cache_prompt=False,  # fair latency: every question pays its own prefill
    )


class RagPipeline:
    def __init__(self, retriever: Retriever, template: str, max_new_tokens: int) -> None:
        self.retriever = retriever
        self.template = template
        self.max_new_tokens = max_new_tokens

    def answer(self, question: str, llm_base_url: str) -> RagAnswer:
        t0 = time.perf_counter()
        hits = self.retriever.retrieve(question)
        retrieval_ms = (time.perf_counter() - t0) * 1000
        return self.answer_from_hits(question, hits, llm_base_url, retrieval_ms)

    def answer_from_hits(
        self, question: str, hits: list[Hit], llm_base_url: str, retrieval_ms: float = 0.0
    ) -> RagAnswer:
        """Generate from already-retrieved chunks (the evaluation retrieves once for all models)."""
        res = generate(llm_base_url, build_prompt(self.template, question, hits),
                       self.max_new_tokens)  # fmt: skip
        timings = res.server_timings or {}
        return RagAnswer(
            question=question,
            answer=res.text.strip(),
            hits=hits,
            retrieval_ms=retrieval_ms,
            ttft_ms=res.ttft_s * 1000,
            total_ms=res.total_s * 1000,
            gen_tokens=timings.get("predicted_n"),
            prompt_tokens=timings.get("prompt_n"),
        )
