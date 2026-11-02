"""Deterministic RAG metrics: no LLM judge, reproducible, free.

Retrieval
* ``rank``   - 1-based position of the first retrieved chunk from a source page (None = miss)
* hit@k, MRR are aggregated from it.

Generation
* ``facts_found / facts_total`` - key facts present in the answer (figures normalised:
  "15,587" == "15587", French "15,6" == "15.6", "793" is not found inside "1793")
* ``correct``  - answerable item and all key facts present
* ``refused``  - the answer says the documents do not contain the information
* unanswerable items: refusing is correct, answering is a hallucination (an empty answer
  is neither: it asserts nothing).
"""

from __future__ import annotations

import re
import unicodedata
from collections.abc import Sequence
from dataclasses import dataclass

from llmbench.eval.dataset import EvalItem
from llmbench.rag.store import Hit

DEFAULT_REFUSAL_MARKERS = (
    "do not contain this information",
    "does not contain this information",
    "do not contain information",
    "not contain this information",
    "no information",
    "ne contiennent pas",
    "ne contient pas",
    "pas cette information",
    "aucune information",
)

_DIGIT_GROUP = re.compile(r"(?<=\d)[,\u00a0\u202f ](?=\d{3}\b)")
_DECIMAL_COMMA = re.compile(r"(?<=\d),(?=\d{1,2}\b)")  # French "15,6" -> "15.6"


def normalize(text: str) -> str:
    """Lowercase, unify unicode, drop thousands separators between digits."""
    text = unicodedata.normalize("NFKC", text).lower()
    text = text.replace("\u2019", "'").replace("\u2212", "-")
    previous = None
    while previous != text:  # "1,234,567" needs several passes
        previous = text
        text = _DIGIT_GROUP.sub("", text)
    text = _DECIMAL_COMMA.sub(".", text)
    return re.sub(r"\s+", " ", text)


def contains_fact(answer: str, alternative: str) -> bool:
    """True if ``alternative`` appears in ``answer``; numbers must not be part of a larger one."""
    needle = normalize(alternative).strip()
    if not needle:
        return False
    hay = normalize(answer)
    left = r"(?<![\d.])" if needle[0].isdigit() else ""
    right = r"(?![\d])" if needle[-1].isdigit() else ""
    return re.search(left + re.escape(needle) + right, hay) is not None


def is_refusal(answer: str, markers: Sequence[str] = DEFAULT_REFUSAL_MARKERS) -> bool:
    text = normalize(answer)
    return any(m in text for m in markers)


@dataclass(frozen=True)
class ItemScore:
    rank: int | None
    facts_found: int
    facts_total: int
    correct: bool
    refused: bool
    correct_refusal: bool  # unanswerable and refused
    hallucinated: bool  # unanswerable and answered anyway


def retrieval_rank(item: EvalItem, hits: Sequence[Hit]) -> int | None:
    if not item.answerable:
        return None
    pages = set(item.source_pages)
    for i, h in enumerate(hits, start=1):
        if h.doc_id == item.doc_id and h.page in pages:
            return i
    return None


def score_item(
    item: EvalItem,
    answer: str,
    hits: Sequence[Hit],
    markers: Sequence[str] = DEFAULT_REFUSAL_MARKERS,
) -> ItemScore:
    refused = is_refusal(answer, markers)
    if not item.answerable:
        # An empty answer (immediate EOS, server crash) asserts nothing: not a hallucination.
        hallucinated = not refused and bool(answer.strip())
        return ItemScore(None, 0, 0, False, refused, refused, hallucinated)
    found = sum(any(contains_fact(answer, alt) for alt in group) for group in item.key_facts)
    total = len(item.key_facts)
    return ItemScore(
        rank=retrieval_rank(item, hits),
        facts_found=found,
        facts_total=total,
        correct=found == total and not refused,
        refused=refused,
        correct_refusal=False,
        hallucinated=False,
    )
