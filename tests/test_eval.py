"""Tests for the evaluation set, metrics, runner, validation CLI and quality report."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest

from llmbench.config import DEFAULT_MODELS_CONFIG, load_eval_config
from llmbench.eval.dataset import EvalItem, Status, dataset_hash, load_items, save_items
from llmbench.eval.metrics import contains_fact, is_refusal, normalize, score_item
from llmbench.eval.report import build_report
from llmbench.eval.run import SUMMARY_FIELDS, aggregate, evaluate, select_items, write_results
from llmbench.eval.validate import review
from llmbench.models.bench import SUMMARY_FIELDS as PERF_FIELDS
from llmbench.rag.store import Hit


def item(**kw: object) -> EvalItem:
    base: dict[str, object] = {
        "id": "airbus_deliveries",
        "question": "How many aircraft did Airbus deliver?",
        "doc_id": "airbus_fy2025",
        "source_pages": [1],
        "reference_answer": "793 aircraft.",
        "key_facts": [["793"]],
    }
    base.update(kw)
    return EvalItem.model_validate(base)


def hit(doc: str, page: int) -> Hit:
    return Hit("x", 0.9, doc, page, "text")


# ---------------------------------------------------------------- dataset


def test_answerable_item_requires_facts_and_source() -> None:
    with pytest.raises(ValueError, match="key_facts"):
        item(key_facts=[])
    with pytest.raises(ValueError, match="source_pages"):
        item(source_pages=[])


def test_unanswerable_item_cannot_have_facts() -> None:
    ok = item(id="u1", answerable=False, doc_id=None, source_pages=[], key_facts=[])
    assert not ok.answerable
    with pytest.raises(ValueError, match="must not"):
        item(answerable=False, key_facts=[["1"]])


def test_save_load_roundtrip_and_duplicate_ids(tmp_path: Path) -> None:
    path = tmp_path / "q.jsonl"
    save_items([item(), item(id="b")], path)
    assert [i.id for i in load_items(path)] == ["airbus_deliveries", "b"]
    path.write_text(path.read_text() + path.read_text().splitlines()[0] + "\n")
    with pytest.raises(ValueError, match="duplicate"):
        load_items(path)


def test_dataset_hash_changes_with_content() -> None:
    assert dataset_hash([item()]) != dataset_hash([item(reference_answer="other")])


def test_select_items_keeps_only_validated() -> None:
    items = [item(id="a", status="validated"), item(id="b"), item(id="c", status="rejected")]
    assert [i.id for i in select_items(items, only_validated=True)] == ["a"]
    assert [i.id for i in select_items(items, only_validated=False)] == ["a", "b"]


# ---------------------------------------------------------------- metrics


@pytest.mark.parametrize(
    ("answer", "fact", "expected"),
    [
        ("Airbus delivered 793 aircraft.", "793", True),
        ("Airbus delivered 1793 aircraft.", "793", False),  # not inside a bigger number
        ("Adjusted net income was $15,587 million", "15587", True),
        ("Adjusted net income was $15 587 million", "15,587", True),
        ("Le résultat net ajusté est de 15,6 milliards $", "15.6", True),  # French decimal
        ("Revenue was €9.3 billion", "9.3", True),
        ("Revenue was €9.35 billion", "9.3", False),
        ("Organic growth of +7.4%", "7.4%", True),
        ("EBIT Adjusted € 7.1 billion", "7.1 billion", True),
    ],
)
def test_contains_fact(answer: str, fact: str, expected: bool) -> None:
    assert contains_fact(answer, fact) is expected


def test_normalize_thousands() -> None:
    assert normalize("1,234,567 and 4\u202f812") == "1234567 and 4812"


def test_refusal_detection_en_fr() -> None:
    assert is_refusal("The provided documents do not contain this information.")
    assert is_refusal("Les documents fournis ne contiennent pas cette information.")
    assert not is_refusal("Airbus delivered 793 aircraft.")


def test_score_answerable_item() -> None:
    it = item(key_facts=[["793"], ["2025"]], source_pages=[1, 2])
    s = score_item(
        it, "In 2025 Airbus delivered 793 aircraft.", [hit("x", 1), hit("airbus_fy2025", 2)]
    )
    assert (s.rank, s.facts_found, s.facts_total, s.correct) == (2, 2, 2, True)
    partial = score_item(it, "Airbus delivered 793 aircraft.", [])
    assert (partial.rank, partial.facts_found, partial.correct) == (None, 1, False)


def test_score_unanswerable_item() -> None:
    it = item(id="u", answerable=False, doc_id=None, source_pages=[], key_facts=[])
    refused = score_item(it, "The provided documents do not contain this information.", [])
    invented = score_item(it, "Orange revenue was 40 billion euros.", [])
    assert refused.correct_refusal and not refused.hallucinated
    assert invented.hallucinated and not invented.correct_refusal


# ---------------------------------------------------------------- runner


class FakePipeline:
    def __init__(self, answers: dict[str, str]) -> None:
        self.answers = answers

    def answer(self, question: str, url: str) -> object:
        from llmbench.rag.pipeline import RagAnswer

        return RagAnswer(
            question, self.answers[question], [hit("airbus_fy2025", 1)], 5.0, 100.0, 300.0, 20, 900
        )


def test_evaluate_aggregate_and_write(tmp_path: Path) -> None:
    items = [
        item(id="a", question="Q deliveries?"),
        item(id="b", question="Q revenue?", key_facts=[["73.4"]]),
        item(
            id="u",
            question="Q orange?",
            answerable=False,
            doc_id=None,
            source_pages=[],
            key_facts=[],
            category="unanswerable",
        ),
    ]
    pipe = FakePipeline(
        {
            "Q deliveries?": "793 aircraft were delivered.",
            "Q revenue?": "Revenue was 70 billion.",  # wrong figure
            "Q orange?": "The provided documents do not contain this information.",
        }
    )
    records = evaluate(items, pipe, "http://llm", load_eval_config().refusal_markers)  # type: ignore[arg-type]
    agg = aggregate(records)
    assert agg["answer_accuracy"] == 0.5
    assert agg["retrieval_hit_at_k"] == 1.0 and agg["mrr"] == 1.0
    assert agg["refusal_accuracy"] == 1.0 and agg["hallucination_rate"] == 0.0
    summary = {"model": "m", "quant": "gguf_q4_k_m", "timestamp": "t", **agg}
    answers = write_results(records, summary, tmp_path)
    assert len(answers.read_text().splitlines()) == 3
    row = next(csv.DictReader((tmp_path / "summary.csv").open()))
    assert list(row) == SUMMARY_FIELDS and row["answer_accuracy"] == "0.5"


# ---------------------------------------------------------------- validation CLI


def test_review_validates_rejects_and_edits(tmp_path: Path) -> None:
    items = [item(id="a"), item(id="b"), item(id="c"), item(id="d", status="validated")]
    answers = iter(
        [
            "v",  # a validated
            "e",
            "",
            "Airbus delivered 793.",
            "",
            "",  # b edited (reference only) ...
            "v",  # ... then validated
            "r",
            "ambiguous",  # c rejected
        ]
    )
    saved: list[list[EvalItem]] = []
    shown: list[str] = []
    counts = review(
        items,
        {("airbus_fy2025", 1): "793 commercial aircraft delivered"},
        lambda its: saved.append(list(its)),
        lambda _: next(answers),
        shown.append,
    )
    assert counts == {"validated": 2, "rejected": 1, "edited": 1, "skipped": 0}
    final = saved[-1]
    assert [i.status for i in final] == [
        Status.VALIDATED,
        Status.VALIDATED,
        Status.REJECTED,
        Status.VALIDATED,
    ]
    assert final[1].reference_answer == "Airbus delivered 793."
    assert final[2].notes == "ambiguous"
    assert any("793 commercial aircraft delivered" in s for s in shown)  # source page shown


def test_review_quit_saves_nothing_more() -> None:
    saved: list[object] = []
    counts = review([item()], {}, saved.append, lambda _: "q", lambda _: None)
    assert saved == [] and counts["validated"] == 0


# ---------------------------------------------------------------- report


def _write_csv(path: Path, fields: list[str], rows: list[dict[str, object]]) -> Path:
    with path.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in fields})
    return path


def test_quality_report_joins_speed(tmp_path: Path) -> None:
    common = {
        "timestamp": "t",
        "n_items": 30,
        "n_answerable": 26,
        "n_unanswerable": 4,
        "retrieval_hit_at_k": 0.92,
        "mrr": 0.8,
        "dataset_hash": "abc",
        "ttft_ms_p50": 900,
    }
    eval_csv = _write_csv(
        tmp_path / "summary.csv",
        SUMMARY_FIELDS,
        [
            {
                **common,
                "model": "qwen2.5-1.5b",
                "quant": "gguf_q4_k_m",
                "answer_accuracy": 0.6,
                "fact_recall": 0.7,
                "hallucination_rate": 0.25,
                "false_refusal_rate": 0.05,
            },
            {
                **common,
                "model": "qwen2.5-1.5b",
                "quant": "gguf_q8_0",
                "answer_accuracy": 0.65,
                "fact_recall": 0.72,
                "hallucination_rate": 0.25,
                "false_refusal_rate": 0.0,
            },
        ],
    )
    perf_csv = _write_csv(
        tmp_path / "perf.csv",
        PERF_FIELDS,
        [
            {
                "session_id": "s",
                "timestamp": "t",
                "model": "qwen2.5-1.5b",
                "quant": q,
                "n_gpu_layers": "all",
                "decode_tok_s_p50": v,
                "file_size_mb": 1,
                "vram_model_mib": 1,
                "vram_peak_mib": 100,
            }
            for q, v in (("gguf_q4_k_m", 93.1), ("gguf_q8_0", 70.7))
        ],
    )
    report = build_report(eval_csv, tmp_path / "out", perf_csv, DEFAULT_MODELS_CONFIG)
    text = report.read_text()
    assert "| qwen2.5-1.5b | Q8_0 | **65%** |" in text
    assert "| 93.1 |" in text and "hit@k **92%**" in text
    assert (tmp_path / "out" / "figures" / "quality_vs_speed.png").stat().st_size > 10_000


def test_repository_eval_yaml_is_valid() -> None:
    cfg = load_eval_config()
    assert ("qwen2.5-1.5b", "gguf_q4_k_m") in [(m, q.value) for m, q in cfg.variants]
    assert json.dumps(cfg.refusal_markers)
