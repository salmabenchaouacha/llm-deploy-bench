"""Interactive human validation of the evaluation set.

For each draft item, shows the question, the reference answer, the key facts and the text
of the source page(s), then asks:

    [v] validate   [e] edit   [r] reject   [s] skip   [q] quit

Every decision is saved immediately (atomic write), so the session can be stopped and
resumed at any time. Run with ``make eval-validate``.
"""

from __future__ import annotations

import argparse
import json
import textwrap
from collections.abc import Callable
from pathlib import Path

from llmbench.config import DEFAULT_EVAL_CONFIG, load_eval_config
from llmbench.eval.dataset import EvalItem, Status, load_items, save_items

Prompt = Callable[[str], str]


def load_pages(chunks_file: Path) -> dict[tuple[str, int], str]:
    """(doc_id, page) -> page text rebuilt from chunks (overlaps make it slightly redundant)."""
    pages: dict[tuple[str, int], list[str]] = {}
    if chunks_file.is_file():
        for line in chunks_file.read_text(encoding="utf-8").splitlines():
            c = json.loads(line)
            pages.setdefault((c["doc_id"], c["page"]), []).append(c["text"])
    return {k: "\n".join(v) for k, v in pages.items()}


def render(item: EvalItem, pages: dict[tuple[str, int], str], max_chars: int = 1800) -> str:
    facts = " AND ".join("(" + " | ".join(g) + ")" for g in item.key_facts) or "-"
    lines = [
        "=" * 100,
        f"[{item.id}] {item.category}  lang={item.lang}  answerable={item.answerable}",
        f"Q: {item.question}",
        f"Reference: {item.reference_answer or '-'}",
        f"Key facts: {facts}",
    ]
    if item.notes:
        lines.append(f"Notes: {item.notes}")
    for page in item.source_pages:
        text = pages.get((item.doc_id or "", page), "(page text not found — run ingest)")
        lines += [
            f"--- source {item.doc_id} p.{page} ---",
            textwrap.shorten(text.replace("\n", " ⏎ "), max_chars, placeholder=" [...]"),
        ]
    return "\n".join(lines)


def edit(item: EvalItem, ask: Prompt) -> EvalItem:
    """Field-by-field edit; empty input keeps the current value."""
    data = item.model_dump()
    q = ask(f"question [{item.question}]: ").strip()
    if q:
        data["question"] = q
    ref = ask(f"reference_answer [{item.reference_answer}]: ").strip()
    if ref:
        data["reference_answer"] = ref
    if item.answerable:
        facts = ask('key_facts as JSON, e.g. [["793"],["15,587","15.6"]] [keep]: ').strip()
        if facts:
            data["key_facts"] = json.loads(facts)
        pages = ask(f"source_pages, comma-separated [{item.source_pages}]: ").strip()
        if pages:
            data["source_pages"] = [int(p) for p in pages.split(",")]
    return EvalItem.model_validate(data)


def review(
    items: list[EvalItem],
    pages: dict[tuple[str, int], str],
    save: Callable[[list[EvalItem]], None],
    ask: Prompt = input,
    show: Callable[[str], None] = print,
) -> dict[str, int]:
    counts = {"validated": 0, "rejected": 0, "edited": 0, "skipped": 0}
    todo = [i for i, it in enumerate(items) if it.status is Status.DRAFT]
    show(f"{len(todo)} draft items to review ({len(items)} total)")
    for idx in todo:
        while True:
            show(render(items[idx], pages))
            choice = ask("[v]alidate [e]dit [r]eject [s]kip [q]uit > ").strip().lower()
            if choice == "v":
                items[idx] = items[idx].model_copy(update={"status": Status.VALIDATED})
                counts["validated"] += 1
            elif choice == "r":
                reason = ask("reason (optional): ").strip()
                items[idx] = items[idx].model_copy(
                    update={"status": Status.REJECTED, "notes": reason or items[idx].notes}
                )
                counts["rejected"] += 1
            elif choice == "e":
                try:
                    items[idx] = edit(items[idx], ask)
                    counts["edited"] += 1
                except (ValueError, json.JSONDecodeError) as exc:
                    show(f"! invalid edit, item unchanged: {exc}")
                continue  # show the edited item again before deciding
            elif choice == "s":
                counts["skipped"] += 1
            elif choice == "q":
                return counts
            else:
                continue
            save(items)
            break
    return counts


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Validate evaluation questions by hand.")
    parser.add_argument("--eval-config", default=DEFAULT_EVAL_CONFIG)
    args = parser.parse_args(argv)
    cfg = load_eval_config(args.eval_config)
    path = cfg.resolve(cfg.dataset)
    items = load_items(path)
    counts = review(
        items, load_pages(cfg.resolve(cfg.chunks_file)), lambda its: save_items(its, path)
    )
    status = {s.value: sum(i.status is s for i in items) for s in Status}
    print(f"\nSession: {counts}\nDataset: {status} -> {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
