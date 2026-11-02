"""Evaluation set: questions with reference answers and source pages (``data/eval/*.jsonl``).

Each item is drafted (by a script or an LLM), then validated by a human with
``python -m llmbench.eval.validate``. Only ``status: validated`` items are used for the
reported numbers.

``key_facts`` are what a correct answer must contain, as groups of alternatives:
``[["793"], ["2025"]]`` = must mention 793 AND 2025; ``[["15,587", "15.6"]]`` = either
spelling of the same figure. Facts are what makes the score objective and free (no LLM
judge needed): in financial QA, the exact figure is the answer.
"""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from enum import StrEnum
from pathlib import Path
from typing import Self

from pydantic import BaseModel, ConfigDict, Field, model_validator


class Status(StrEnum):
    DRAFT = "draft"
    VALIDATED = "validated"
    REJECTED = "rejected"


class EvalItem(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str = Field(..., pattern=r"^[a-z0-9_]+$")
    question: str = Field(..., min_length=5)
    lang: str = Field("en", pattern=r"^[a-z]{2}$")
    answerable: bool = True
    doc_id: str | None = None
    source_pages: list[int] = Field(default_factory=list)
    reference_answer: str = ""
    key_facts: list[list[str]] = Field(default_factory=list)
    category: str = "fact"  # fact | comparison | multi_hop | unanswerable
    status: Status = Status.DRAFT
    notes: str = ""

    @model_validator(mode="after")
    def _check(self) -> Self:
        if self.answerable:
            if not self.doc_id or not self.source_pages:
                raise ValueError(f"{self.id}: answerable item needs doc_id and source_pages")
            if not self.key_facts or any(not group for group in self.key_facts):
                raise ValueError(f"{self.id}: answerable item needs non-empty key_facts")
        elif self.key_facts:
            raise ValueError(f"{self.id}: unanswerable item must not have key_facts")
        return self


def load_items(path: Path) -> list[EvalItem]:
    items = [
        EvalItem.model_validate_json(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    ids = [i.id for i in items]
    dupes = sorted({i for i in ids if ids.count(i) > 1})
    if dupes:
        raise ValueError(f"duplicate ids in {path}: {dupes}")
    return items


def save_items(items: list[EvalItem], path: Path) -> None:
    """Atomic write: a crash mid-validation never corrupts the file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=path.parent, suffix=".tmp")
    with os.fdopen(fd, "w", encoding="utf-8") as fh:
        for item in items:
            fh.write(json.dumps(item.model_dump(mode="json"), ensure_ascii=False) + "\n")
    Path(tmp).replace(path)


def dataset_hash(items: list[EvalItem]) -> str:
    """Short fingerprint of the items actually evaluated (stored with every result)."""
    payload = json.dumps([i.model_dump(mode="json") for i in items], sort_keys=True)
    return hashlib.sha256(payload.encode()).hexdigest()[:12]
