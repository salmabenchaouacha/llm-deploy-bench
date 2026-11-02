"""Split page text into overlapping chunks.

Rules:
* a chunk never crosses a page boundary -> every chunk cites one exact page;
* chunks are packed from whole lines (a table row is never cut in half) unless a single
  line is longer than ``max_chars``;
* consecutive chunks share ~``overlap_chars`` of trailing lines, so a sentence cut at a
  boundary still exists whole in one chunk;
* chunk ids are deterministic (UUID5 of doc/page/index): re-ingesting the same corpus
  overwrites the same Qdrant points instead of duplicating them.
"""

from __future__ import annotations

import uuid
from collections.abc import Iterable
from dataclasses import asdict, dataclass
from typing import Any

from llmbench.rag.parse import PageText

_NAMESPACE = uuid.UUID("6f1c3c2e-9a7b-4d0e-8b51-2a7f6d1e4c90")


@dataclass(frozen=True)
class Chunk:
    chunk_id: str
    doc_id: str
    page: int
    index: int  # position within the page
    text: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def chunk_id(doc_id: str, page: int, index: int) -> str:
    return str(uuid.uuid5(_NAMESPACE, f"{doc_id}:{page}:{index}"))


def _split_long_line(line: str, max_chars: int) -> list[str]:
    return [line[i : i + max_chars] for i in range(0, len(line), max_chars)]


def chunk_page(page: PageText, max_chars: int, overlap_chars: int) -> list[Chunk]:
    lines: list[str] = []
    for line in page.text.splitlines():
        lines.extend(_split_long_line(line, max_chars) if len(line) > max_chars else [line])

    pieces: list[str] = []
    current: list[str] = []
    size = 0
    for line in lines:
        if current and size + len(line) + 1 > max_chars:
            pieces.append("\n".join(current).strip())
            # keep trailing lines as overlap
            tail: list[str] = []
            tail_size = 0
            for prev in reversed(current):
                if tail_size + len(prev) + 1 > overlap_chars:
                    break
                tail.insert(0, prev)
                tail_size += len(prev) + 1
            current, size = tail, tail_size
        current.append(line)
        size += len(line) + 1
    if current and "\n".join(current).strip():
        pieces.append("\n".join(current).strip())

    return [
        Chunk(chunk_id(page.doc_id, page.page, i), page.doc_id, page.page, i, text)
        for i, text in enumerate(p for p in pieces if p)
    ]


def chunk_pages(pages: Iterable[PageText], max_chars: int, overlap_chars: int) -> list[Chunk]:
    return [c for p in pages for c in chunk_page(p, max_chars, overlap_chars)]
