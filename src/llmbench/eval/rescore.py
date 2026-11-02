"""Re-score saved answers with the current dataset and metrics, without re-running models.

Generation is the expensive part (about an hour for six variants on a 4 GB GPU); scoring
is pure Python. When a metric or a reference answer is corrected, this recomputes every
score from ``results/eval/answers/*.jsonl`` and appends one new summary row per variant
(``session_id`` starts with ``rescore-``). Latencies are the originally measured ones.

CLI::  python -m llmbench.eval.rescore
"""

from __future__ import annotations

import argparse
import csv
import json
import uuid
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from llmbench.config import DEFAULT_EVAL_CONFIG, load_eval_config
from llmbench.eval.dataset import EvalItem, dataset_hash, load_items
from llmbench.eval.metrics import score_item
from llmbench.eval.run import EVAL_DIR, AnswerRecord, aggregate, select_items, write_results
from llmbench.models.bench import git_commit
from llmbench.rag.store import Hit


def parse_source(source: str) -> Hit:
    """'airbus_fy2025 p.5' -> Hit (only doc and page matter for scoring)."""
    doc, page = source.rsplit(" p.", 1)
    return Hit("", 0.0, doc, int(page), "")


def rescore_file(path: Path, items: dict[str, EvalItem], markers: list[str]) -> list[AnswerRecord]:
    records = []
    for line in path.read_text(encoding="utf-8").splitlines():
        row = json.loads(line)
        item = items.get(row["item_id"])
        if item is None:  # question removed or rejected since the run
            continue
        hits = [parse_source(s) for s in row["retrieved"]]
        records.append(
            AnswerRecord(
                item_id=item.id,
                category=item.category,
                answerable=item.answerable,
                question=item.question,
                answer=row["answer"],
                reference_answer=item.reference_answer,
                retrieved=row["retrieved"],
                score=score_item(item, row["answer"], hits, markers),
                ttft_ms=row["ttft_ms"],
                total_ms=row["total_ms"],
                gen_tokens=row["gen_tokens"],
                error=row.get("error", ""),
            )
        )
    return records


def latest_rows(csv_path: Path) -> dict[tuple[str, str], dict[str, str]]:
    with csv_path.open(encoding="utf-8") as fh:
        rows = sorted(csv.DictReader(fh), key=lambda r: r["timestamp"])
    return {(r["model"], r["quant"]): r for r in rows if not r.get("error")}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Re-score saved answers.")
    parser.add_argument("--out-dir", type=Path, default=EVAL_DIR)
    parser.add_argument("--eval-config", default=DEFAULT_EVAL_CONFIG)
    args = parser.parse_args(argv)

    cfg = load_eval_config(args.eval_config)
    items_list = select_items(load_items(cfg.resolve(cfg.dataset)), cfg.only_validated)
    items = {i.id: i for i in items_list}
    previous = latest_rows(args.out_dir / "summary.csv")
    session: dict[str, Any] = {
        "session_id": f"rescore-{uuid.uuid4().hex[:6]}",
        "timestamp": datetime.now(UTC).isoformat(timespec="seconds"),
        "git_commit": git_commit(),
        "dataset_hash": dataset_hash(items_list),
    }
    for path in sorted((args.out_dir / "answers").glob("*__*.jsonl")):
        model, quant = path.stem.split("__", 1)
        records = rescore_file(path, items, cfg.refusal_markers)
        prev = previous.get((model, quant), {})
        summary = {
            **session,
            "model": model,
            "quant": quant,
            "n_gpu_layers": prev.get("n_gpu_layers", ""),
            "top_k": prev.get("top_k", ""),
            **aggregate(records),
        }
        write_results(records, summary, args.out_dir)
        print(
            f"{model} {quant}: accuracy {summary['answer_accuracy']:.0%}, "
            f"hallucination {summary['hallucination_rate']:.0%} ({len(records)} answers)"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
