"""Embeddings through a Text Embeddings Inference (TEI) server.

TEI runs as a container (CPU image) next to Qdrant: the embedding model is served the
same way as the LLM, keeps the 4 GB GPU free for generation, and the Python side stays a
thin HTTP client.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Protocol

from llmbench.config import EmbedderConfig
from llmbench.http import request_json


class Embedder(Protocol):
    dim: int

    def embed_passages(self, texts: Sequence[str]) -> list[list[float]]: ...

    def embed_query(self, text: str) -> list[float]: ...


class TEIEmbedder:
    def __init__(self, cfg: EmbedderConfig, timeout: float = 300.0) -> None:
        self.cfg = cfg
        self.dim = cfg.dim
        self.timeout = timeout

    def _embed(self, texts: Sequence[str]) -> list[list[float]]:
        vectors: list[list[float]] = []
        for i in range(0, len(texts), self.cfg.batch_size):
            batch = list(texts[i : i + self.cfg.batch_size])
            out = request_json(
                "POST",
                f"{self.cfg.url}/embed",
                {"inputs": batch, "normalize": True, "truncate": True},
                timeout=self.timeout,
            )
            if not isinstance(out, list) or len(out) != len(batch):
                raise ValueError(f"unexpected TEI answer for {len(batch)} inputs")
            vectors.extend(out)
        for v in vectors:
            if len(v) != self.dim:
                raise ValueError(f"embedding dim {len(v)} != configured {self.dim}")
        return vectors

    def embed_passages(self, texts: Sequence[str]) -> list[list[float]]:
        return self._embed([self.cfg.passage_prefix + t for t in texts])

    def embed_query(self, text: str) -> list[float]:
        return self._embed([self.cfg.query_prefix + text])[0]
