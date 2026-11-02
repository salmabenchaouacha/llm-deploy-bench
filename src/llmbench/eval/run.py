"""Run the RAG quality evaluation for one or several model/quant variants.

For each variant: start llama-server, answer every evaluation question through the shared
RAG pipeline, score answers with deterministic metrics, then write

* ``results/eval/answers/<model>__<quant>.jsonl`` - every question, answer, chunks, scores
* ``results/eval/summary.csv``                    - one row per variant and run

CLI::

    python -m llmbench.eval.run --model qwen2.5-1.5b --quant gguf_q4_k_m
    python -m llmbench.eval.run --all            # every variant listed in configs/eval.yaml
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
import urllib.error
import uuid
from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from llmbench.config import (
    DEFAULT_BENCH_CONFIG,
    DEFAULT_EVAL_CONFIG,
    DEFAULT_MODELS_CONFIG,
    DEFAULT_RAG_CONFIG,
    RESULTS_DIR,
    ConfigError,
    load_bench_config,
    load_eval_config,
    load_models_config,
    load_rag_config,
)
from llmbench.eval.dataset import EvalItem, Status, dataset_hash, load_items
from llmbench.eval.metrics import ItemScore, score_item
from llmbench.http import HttpError
from llmbench.log import get_logger
from llmbench.models.bench import git_commit, percentile
from llmbench.models.hub import ensure_gguf, gguf_relpath
from llmbench.rag.embed import TEIEmbedder
from llmbench.rag.pipeline import RagAnswer, RagPipeline, Retriever
from llmbench.rag.services import wait_for_rag_services
from llmbench.rag.store import Hit, QdrantStore
from llmbench.serving.client import StreamError
from llmbench.serving.llama_server import DockerRunError, LlamaServer, ServerStartError

log = get_logger(__name__)

EVAL_DIR = RESULTS_DIR / "eval"

SUMMARY_FIELDS = [
    "session_id",
    "timestamp",
    "git_commit",
    "dataset_hash",
    "model",
    "quant",
    "n_gpu_layers",
    "top_k",
    "n_items",
    "n_answerable",
    "n_unanswerable",
    "retrieval_hit_at_k",
    "mrr",
    "fact_recall",
    "answer_accuracy",
    "false_refusal_rate",
    "refusal_accuracy",
    "hallucination_rate",
    "ttft_ms_p50",
    "total_ms_p50",
    "gen_tokens_mean",
    "error",
]


@dataclass
class AnswerRecord:
    item_id: str
    category: str
    answerable: bool
    question: str
    answer: str
    reference_answer: str
    retrieved: list[str]
    score: ItemScore
    ttft_ms: float
    total_ms: float
    gen_tokens: int | None
    error: str = ""

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["score"] = asdict(self.score)
        return d


def retrieve_all(items: Sequence[EvalItem], retriever: Retriever) -> dict[str, list[Hit]]:
    """Retrieve once for every question, before any LLM is started.

    Every variant then receives exactly the same chunks, and the embedding server is not
    needed (nor competing for RAM) while models generate.
    """
    return {item.id: retriever.retrieve(item.question) for item in items}


def write_retrieval(hits: dict[str, list[Hit]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for item_id, item_hits in hits.items():
            row = {
                "item_id": item_id,
                "hits": [
                    {
                        "doc_id": h.doc_id,
                        "page": h.page,
                        "score": round(h.score, 4),
                        "chunk_id": h.chunk_id,
                    }
                    for h in item_hits
                ],
            }
            fh.write(json.dumps(row) + "\n")


def done_variants(csv_path: Path, dataset: str) -> set[tuple[str, str]]:
    """(model, quant) already evaluated without error on this exact dataset version."""
    if not csv_path.is_file():
        return set()
    with csv_path.open(encoding="utf-8") as fh:
        return {
            (r["model"], r["quant"])
            for r in csv.DictReader(fh)
            if r.get("dataset_hash") == dataset and not r.get("error")
        }


def _is_connection_error(exc: Exception) -> bool:
    """The server is gone (crashed / restarting), as opposed to a bad answer."""
    if isinstance(exc, HttpError):
        return exc.status is None
    return isinstance(exc, (ConnectionError, urllib.error.URLError, TimeoutError))


def evaluate(
    items: Sequence[EvalItem],
    pipeline: RagPipeline,
    llm_base_url: str,
    refusal_markers: Sequence[str],
    hits_by_item: dict[str, list[Hit]] | None = None,
    recover: Callable[[], str] | None = None,
) -> list[AnswerRecord]:
    """Answer and score every item.

    A failed request never aborts the run: if the LLM server is unreachable, ``recover``
    (restart the server, return its URL) is called and the question retried once; any
    remaining failure is recorded as an empty, wrong answer with its error message.
    """
    records = []
    for n, item in enumerate(items, 1):
        hits = hits_by_item[item.id] if hits_by_item is not None else []
        ans: RagAnswer | None = None
        error = ""
        for attempt in (1, 2):
            try:
                if hits_by_item is not None:
                    ans = pipeline.answer_from_hits(item.question, hits, llm_base_url)
                else:
                    ans = pipeline.answer(item.question, llm_base_url)
                break
            except (StreamError, HttpError, OSError) as exc:
                error = f"{type(exc).__name__}: {exc}"[:300]
                if attempt == 1 and recover is not None and _is_connection_error(exc):
                    log.warning("LLM server unreachable, restarting", extra={"id": item.id})
                    llm_base_url = recover()
                    continue
                break
        if ans is None:
            log.warning("question failed", extra={"id": item.id, "error": error})
            records.append(
                AnswerRecord(
                    item_id=item.id,
                    category=item.category,
                    answerable=item.answerable,
                    question=item.question,
                    answer="",
                    reference_answer=item.reference_answer,
                    retrieved=[f"{h.doc_id} p.{h.page}" for h in hits],
                    score=score_item(item, "", hits, refusal_markers),
                    ttft_ms=math.nan,
                    total_ms=math.nan,
                    gen_tokens=None,
                    error=error,
                )
            )
            continue
        score = score_item(item, ans.answer, ans.hits, refusal_markers)
        records.append(
            AnswerRecord(
                item_id=item.id,
                category=item.category,
                answerable=item.answerable,
                question=item.question,
                answer=ans.answer,
                reference_answer=item.reference_answer,
                retrieved=ans.sources,
                score=score,
                ttft_ms=ans.ttft_ms,
                total_ms=ans.total_ms,
                gen_tokens=ans.gen_tokens,
            )
        )
        log.info(
            "answered",
            extra={
                "n": f"{n}/{len(items)}",
                "id": item.id,
                "correct": score.correct or score.correct_refusal,
                "empty": not ans.answer,
            },
        )
    return records


def _ratio(num: float, den: float) -> float:
    return num / den if den else math.nan


def aggregate(records: Sequence[AnswerRecord]) -> dict[str, Any]:
    ans = [r for r in records if r.answerable]
    una = [r for r in records if not r.answerable]
    ranks = [r.score.rank for r in ans]
    gen = [r.gen_tokens for r in records if r.gen_tokens is not None]
    return {
        "n_items": len(records),
        "n_answerable": len(ans),
        "n_unanswerable": len(una),
        "retrieval_hit_at_k": _ratio(sum(x is not None for x in ranks), len(ans)),
        "mrr": _ratio(sum(1 / x for x in ranks if x), len(ans)),
        "fact_recall": _ratio(
            sum(r.score.facts_found for r in ans), sum(r.score.facts_total for r in ans)
        ),
        "answer_accuracy": _ratio(sum(r.score.correct for r in ans), len(ans)),
        "false_refusal_rate": _ratio(sum(r.score.refused for r in ans), len(ans)),
        "refusal_accuracy": _ratio(sum(r.score.correct_refusal for r in una), len(una)),
        "hallucination_rate": _ratio(sum(r.score.hallucinated for r in una), len(una)),
        "ttft_ms_p50": percentile([r.ttft_ms for r in records], 50),
        "total_ms_p50": percentile([r.total_ms for r in records], 50),
        "gen_tokens_mean": statistics.fmean(gen) if gen else math.nan,
    }


def _fmt(v: Any) -> Any:
    if isinstance(v, float):
        return "" if math.isnan(v) else round(v, 4)
    return "" if v is None else v


def write_results(records: Sequence[AnswerRecord], summary: dict[str, Any], out_dir: Path) -> Path:
    answers_dir = out_dir / "answers"
    answers_dir.mkdir(parents=True, exist_ok=True)
    answers = answers_dir / f"{summary['model']}__{summary['quant']}.jsonl"
    with answers.open("w", encoding="utf-8") as fh:
        for r in records:
            fh.write(json.dumps(r.to_dict(), ensure_ascii=False) + "\n")
    csv_path = out_dir / "summary.csv"
    new = not csv_path.exists()
    if not new:
        with csv_path.open(encoding="utf-8") as fh:
            if next(csv.reader(fh), []) != SUMMARY_FIELDS:
                raise RuntimeError(f"{csv_path} has other columns; move it aside and rerun")
    with csv_path.open("a", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=SUMMARY_FIELDS)
        if new:
            writer.writeheader()
        writer.writerow({k: _fmt(summary.get(k)) for k in SUMMARY_FIELDS})
    return answers


def select_items(items: list[EvalItem], only_validated: bool) -> list[EvalItem]:
    kept = [i for i in items if i.status is not Status.REJECTED]
    return [i for i in kept if i.status is Status.VALIDATED] if only_validated else kept


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="RAG quality evaluation.")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--all", action="store_true", help="every variant in eval.yaml")
    group.add_argument("--model")
    parser.add_argument("--quant", default="gguf_q4_k_m")
    parser.add_argument(
        "--include-drafts",
        action="store_true",
        help="also use unvalidated items (debug only, not for reporting)",
    )
    parser.add_argument("--limit", type=int, help="first N items only (smoke test)")
    parser.add_argument(
        "--resume",
        action="store_true",
        help="skip variants already evaluated on this dataset version",
    )
    parser.add_argument("--out-dir", type=Path, default=EVAL_DIR)
    parser.add_argument("--eval-config", default=DEFAULT_EVAL_CONFIG)
    parser.add_argument("--rag-config", default=DEFAULT_RAG_CONFIG)
    parser.add_argument("--models-config", default=DEFAULT_MODELS_CONFIG)
    parser.add_argument("--bench-config", default=DEFAULT_BENCH_CONFIG)
    args = parser.parse_args(argv)

    try:
        ecfg = load_eval_config(args.eval_config)
        rag = load_rag_config(args.rag_config)
        models = load_models_config(args.models_config)
        server_cfg = load_bench_config(args.bench_config).server
        variants = ecfg.variants if args.all else [(args.model, args.quant)]
        resolved = [(m, models.get(m).variant(q)) for m, q in variants]
    except (ConfigError, KeyError) as exc:
        print(f"ERROR: {exc.args[0] if exc.args else exc}")
        return 2

    dataset_path = ecfg.resolve(ecfg.dataset)
    if not dataset_path.is_file():
        print(f"ERROR: {dataset_path} not found — draft and validate questions first")
        return 2
    items = select_items(load_items(dataset_path), ecfg.only_validated and not args.include_drafts)
    if args.limit:
        items = items[: args.limit]
    if not items:
        print("ERROR: no evaluation items (validate some with `make eval-validate`)")
        return 2
    print(f"{len(items)} items ({sum(i.answerable for i in items)} answerable)")

    data_version = dataset_hash(items) + ("-drafts" if args.include_drafts else "")
    session: dict[str, Any] = {
        "session_id": uuid.uuid4().hex[:8],
        "timestamp": datetime.now(UTC).isoformat(timespec="seconds"),
        "git_commit": git_commit(),
        "dataset_hash": data_version,
        "top_k": rag.top_k,
    }
    if args.resume:
        done = done_variants(args.out_dir / "summary.csv", data_version)
        skipped = [f"{m} {v.quant.value}" for m, v in resolved if (m, v.quant.value) in done]
        resolved = [(m, v) for m, v in resolved if (m, v.quant.value) not in done]
        if skipped:
            print(f"Already evaluated on this dataset, skipped: {', '.join(skipped)}")
        if not resolved:
            print("Nothing left to evaluate.")
            return 0

    wait_for_rag_services(rag)
    retriever = Retriever(TEIEmbedder(rag.embedder), QdrantStore(rag.qdrant), rag.top_k)
    pipeline = RagPipeline(retriever, rag.prompt_template, rag.max_new_tokens)
    hits_by_item = retrieve_all(items, retriever)
    write_retrieval(hits_by_item, args.out_dir / "retrieval.jsonl")
    print(f"Retrieved top-{rag.top_k} chunks for {len(items)} questions (shared by all models)")

    for model_id, variant in resolved:
        summary: dict[str, Any] = {
            **session,
            "model": model_id,
            "quant": variant.quant.value,
            "n_gpu_layers": variant.n_gpu_layers or "all",
        }
        ensure_gguf(variant, server_cfg.models_path)
        try:
            with LlamaServer.from_config(
                server_cfg, gguf_relpath(variant), variant.n_gpu_layers
            ) as srv:

                def recover(server: LlamaServer = srv) -> str:
                    tail = server.logs(25)
                    log.warning("llama-server logs before restart", extra={"tail": tail})
                    server.stop()
                    server.start()
                    return server.base_url

                records = evaluate(
                    items, pipeline, srv.base_url, ecfg.refusal_markers, hits_by_item, recover
                )
        except DockerRunError as exc:
            print(f"ERROR: {exc}")
            return 3
        except ServerStartError as exc:
            summary["error"] = "server_start_failed"
            log.error("server failed", extra={"model": model_id, "details": str(exc)[-500:]})
            records = []
        summary.update(aggregate(records) if records else {"n_items": 0})
        path = write_results(records, summary, args.out_dir)
        print(
            f"-> {model_id} {variant.quant.value}: "
            f"accuracy {summary.get('answer_accuracy', math.nan):.0%}, "
            f"fact recall {summary.get('fact_recall', math.nan):.0%}, "
            f"hallucination {summary.get('hallucination_rate', math.nan):.0%} -> {path}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
