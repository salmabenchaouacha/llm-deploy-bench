"""PDF -> text, page by page.

pypdf's ``layout`` extraction keeps the horizontal alignment of table columns, which
matters for financial reports where most figures live in tables. Long runs of spaces
used for alignment are then compacted into `` | `` so they do not waste embedding tokens
while columns stay distinguishable ("Revenue | 4,812 | 4,532").
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

from pypdf import PdfReader

_COLUMN_GAP = re.compile(r"[ \t]{3,}")
_BLANK_LINES = re.compile(r"\n{3,}")


@dataclass(frozen=True)
class PageText:
    doc_id: str
    page: int  # 1-based, as printed in a PDF viewer
    text: str


def clean_text(text: str) -> str:
    lines = [_COLUMN_GAP.sub(" | ", line).strip(" |") for line in text.splitlines()]
    return _BLANK_LINES.sub("\n\n", "\n".join(lines)).strip()


def extract_pages(path: Path, doc_id: str) -> list[PageText]:
    """Text of every non-empty page. Scanned (image-only) pages yield no text."""
    reader = PdfReader(path)
    pages = []
    for number, page in enumerate(reader.pages, start=1):
        try:
            raw = page.extract_text(extraction_mode="layout")
        except Exception:  # some malformed pages break layout mode
            raw = page.extract_text()
        text = clean_text(raw or "")
        if text:
            pages.append(PageText(doc_id, number, text))
    return pages
