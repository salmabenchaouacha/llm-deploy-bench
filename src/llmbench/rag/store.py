"""Thin typed client for the Qdrant REST API (collections, upsert, query)."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

from llmbench.config import QdrantConfig
from llmbench.http import HttpError, request_json
from llmbench.rag.chunk import Chunk


@dataclass(frozen=True)
class Hit:
    chunk_id: str
    score: float
    doc_id: str
    page: int
    text: str


class QdrantStore:
    def __init__(self, cfg: QdrantConfig, timeout: float = 60.0) -> None:
        self.base = f"{cfg.url}/collections/{cfg.collection}"
        self.timeout = timeout

    def _call(self, method: str, path: str, payload: Any | None = None) -> Any:
        return request_json(method, self.base + path, payload, timeout=self.timeout)

    def exists(self) -> bool:
        try:
            self._call("GET", "")
            return True
        except HttpError as exc:
            if exc.status == 404:
                return False
            raise

    def count(self) -> int:
        out = self._call("POST", "/points/count", {"exact": True})
        return int(out["result"]["count"])

    def recreate(self, dim: int) -> None:
        """Drop and create the collection: ingestion is a full, reproducible rebuild."""
        if self.exists():
            self._call("DELETE", "")
        self._call("PUT", "", {"vectors": {"size": dim, "distance": "Cosine"}})

    def upsert(
        self, chunks: Sequence[Chunk], vectors: Sequence[Sequence[float]], batch: int = 64
    ) -> None:
        if len(chunks) != len(vectors):
            raise ValueError("chunks and vectors must have the same length")
        for i in range(0, len(chunks), batch):
            points = [
                {"id": c.chunk_id, "vector": list(v), "payload": c.to_dict()}
                for c, v in zip(chunks[i : i + batch], vectors[i : i + batch], strict=True)
            ]
            self._call("PUT", "/points?wait=true", {"points": points})

    def search(self, vector: Sequence[float], k: int) -> list[Hit]:
        out = self._call(
            "POST", "/points/query", {"query": list(vector), "limit": k, "with_payload": True}
        )
        return [
            Hit(
                chunk_id=str(p["id"]),
                score=float(p["score"]),
                doc_id=p["payload"]["doc_id"],
                page=int(p["payload"]["page"]),
                text=p["payload"]["text"],
            )
            for p in out["result"]["points"]
        ]
