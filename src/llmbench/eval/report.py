"""Quality report: RAG accuracy per variant, joined with the speed measured in Phase 1.

Reads ``results/eval/summary.csv`` (latest run per variant) and
``results/perf/summary.csv``, writes ``results/eval/report.md`` and two figures:

* ``figures/accuracy.png``         - answer accuracy per variant
* ``figures/quality_vs_speed.png`` - accuracy vs decode speed: the deployment trade-off

CLI::  python -m llmbench.eval.report
"""

from __future__ import annotations

import argparse
import csv
import math
from dataclasses import dataclass
from pathlib import Path

from llmbench.config import DEFAULT_MODELS_CONFIG, RESULTS_DIR, load_models_config
from llmbench.models.report import INK, INK_2, SERIES, SURFACE, _new, load_rows, split_rows

EVAL_DIR = RESULTS_DIR / "eval"
PERF_CSV = RESULTS_DIR / "perf" / "summary.csv"


def _f(v: str | None) -> float:
    try:
        return float(v) if v not in (None, "") else math.nan
    except ValueError:
        return math.nan


@dataclass(frozen=True)
class EvalRow:
    model: str
    quant: str
    n_items: int
    n_answerable: int
    n_unanswerable: int
    hit_at_k: float
    mrr: float
    fact_recall: float
    accuracy: float
    false_refusal: float
    refusal_accuracy: float
    hallucination: float
    ttft_ms: float
    dataset_hash: str
    decode_tok_s: float = math.nan

    @property
    def label(self) -> str:
        return f"{self.model} {self.quant.removeprefix('gguf_').upper()}"

    @property
    def family(self) -> str:
        return self.model.split("-")[0]


def load_eval_rows(csv_path: Path) -> list[EvalRow]:
    with csv_path.open(encoding="utf-8") as fh:
        raw = [r for r in csv.DictReader(fh) if not r.get("error")]
    latest: dict[tuple[str, str], dict[str, str]] = {}
    for r in sorted(raw, key=lambda r: r["timestamp"]):
        latest[(r["model"], r["quant"])] = r
    return [
        EvalRow(
            model=r["model"],
            quant=r["quant"],
            n_items=int(r["n_items"]),
            n_answerable=int(r["n_answerable"]),
            n_unanswerable=int(r["n_unanswerable"]),
            hit_at_k=_f(r["retrieval_hit_at_k"]),
            mrr=_f(r["mrr"]),
            fact_recall=_f(r["fact_recall"]),
            accuracy=_f(r["answer_accuracy"]),
            false_refusal=_f(r["false_refusal_rate"]),
            refusal_accuracy=_f(r["refusal_accuracy"]),
            hallucination=_f(r["hallucination_rate"]),
            ttft_ms=_f(r["ttft_ms_p50"]),
            dataset_hash=r["dataset_hash"],
        )
        for r in latest.values()
    ]


def attach_speed(rows: list[EvalRow], perf_csv: Path, models_config: Path) -> list[EvalRow]:
    """Add the decode speed of the same variant (configured GPU offload) from Phase 1."""
    if not perf_csv.is_file():
        return rows
    cfg = load_models_config(models_config)
    default_ngl = {
        m: next((v.n_gpu_layers for v in s.variants if v.enabled and v.is_gguf), None)
        for m, s in cfg.models.items()
    }
    comparison, _ = split_rows(load_rows(perf_csv), default_ngl)
    speed = {(p.model, p.quant): p.decode_p50 for p in comparison}
    return [
        EvalRow(**{**r.__dict__, "decode_tok_s": speed.get((r.model, r.quant), math.nan)})
        for r in rows
    ]


def _pct(v: float) -> str:
    return "—" if math.isnan(v) else f"{v:.0%}"


def table(rows: list[EvalRow]) -> str:
    head = (
        "| Model | Quant | Answer accuracy | Fact recall | Hallucination¹ | False refusal² "
        "| RAG TTFT p50 (ms) | Decode (tok/s) |\n|---|---|--:|--:|--:|--:|--:|--:|\n"
    )
    return head + "\n".join(
        f"| {r.model} | {r.quant.removeprefix('gguf_').upper()} | **{_pct(r.accuracy)}** "
        f"| {_pct(r.fact_recall)} | {_pct(r.hallucination)} | {_pct(r.false_refusal)} "
        f"| {'—' if math.isnan(r.ttft_ms) else f'{r.ttft_ms:,.0f}'} "
        f"| {'—' if math.isnan(r.decode_tok_s) else f'{r.decode_tok_s:.1f}'} |"
        for r in rows
    )


def plot_accuracy(rows: list[EvalRow], out: Path) -> None:
    ordered = sorted(rows, key=lambda r: r.accuracy)
    fig, (ax,) = _new(8, 0.5 * len(ordered) + 1.2)
    ax.grid(axis="y", visible=False)
    y = range(len(ordered))
    ax.barh(y, [r.accuracy * 100 for r in ordered], height=0.6, color=SERIES[0])
    ax.set_yticks(list(y), [r.label for r in ordered], color=INK, fontsize=9)
    for i, r in enumerate(ordered):
        ax.text(r.accuracy * 100, i, f"  {r.accuracy:.0%}", va="center", color=INK, fontsize=9)
    ax.set_xlim(0, 110)
    ax.set_xlabel("Answer accuracy on answerable questions (%, all key facts present)", color=INK_2)
    ax.set_title("RAG answer accuracy per model / quantization", loc="left", color=INK, fontsize=11)
    fig.tight_layout()
    fig.savefig(out, facecolor=SURFACE)


def plot_quality_vs_speed(rows: list[EvalRow], out: Path) -> None:
    pts = [r for r in rows if not math.isnan(r.decode_tok_s)]
    families = sorted({r.family for r in pts})
    colors = {f: SERIES[i % len(SERIES)] for i, f in enumerate(families)}
    fig, (ax,) = _new(8, 5)
    for fam in families:
        group = [r for r in pts if r.family == fam]
        ax.scatter(
            [r.decode_tok_s for r in group],
            [r.accuracy * 100 for r in group],
            s=70,
            color=colors[fam],
            edgecolor=SURFACE,
            linewidth=1.5,
            label=fam,
            zorder=3,
        )
        for r in group:
            ax.annotate(
                r.label,
                (r.decode_tok_s, r.accuracy * 100),
                textcoords="offset points",
                xytext=(7, 4),
                fontsize=8,
                color=INK,
            )
    ax.set_xlim(left=0)
    ax.set_ylim(0, 105)
    ax.set_xlabel("Decode speed on RTX 3050 4 GB (tokens/s, Phase 1)", color=INK_2)
    ax.set_ylabel("Answer accuracy (%)", color=INK_2)
    ax.set_title(
        "Quality / speed trade-off (top-right is best)", loc="left", color=INK, fontsize=11
    )
    ax.legend(frameon=False, labelcolor=INK, fontsize=9, title="family", title_fontsize=9)
    fig.tight_layout()
    fig.savefig(out, facecolor=SURFACE)


def build_report(eval_csv: Path, out_dir: Path, perf_csv: Path, models_config: Path) -> Path:
    import matplotlib.pyplot as plt

    rows = attach_speed(load_eval_rows(eval_csv), perf_csv, models_config)
    rows.sort(key=lambda r: (-r.accuracy, -_nan0(r.decode_tok_s)))
    fig_dir = out_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    parts = [
        "# RAG quality evaluation — results",
        "",
        "_Generated by `python -m llmbench.eval.report` from "
        "[`summary.csv`](summary.csv); per-question answers in [`answers/`](answers/)._",
        "",
    ]
    if rows:
        r0 = rows[0]
        hashes = sorted({r.dataset_hash for r in rows})
        parts += [
            f"Evaluation set: **{r0.n_items} hand-validated questions** "
            f"({r0.n_answerable} answerable, {r0.n_unanswerable} unanswerable), dataset "
            f"`{', '.join(hashes)}`. Same retriever for every model: retrieval hit@k "
            f"**{_pct(r0.hit_at_k)}**, MRR **{r0.mrr:.2f}**. Greedy decoding.",
            "",
            table(rows),
            "",
            "¹ Unanswerable questions answered anyway (invented). "
            "² Answerable questions the model refused.",
            "",
            "![Accuracy](figures/accuracy.png)",
            "",
            "![Quality vs speed](figures/quality_vs_speed.png)",
        ]
        if len(hashes) > 1:
            parts += ["", "> ⚠ Variants were evaluated on different dataset versions."]
        plot_accuracy(rows, fig_dir / "accuracy.png")
        plot_quality_vs_speed(rows, fig_dir / "quality_vs_speed.png")
        plt.close("all")
    report = out_dir / "report.md"
    report.write_text("\n".join(parts) + "\n", encoding="utf-8")
    return report


def _nan0(v: float) -> float:
    return 0.0 if math.isnan(v) else v


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Build the RAG quality report.")
    parser.add_argument("--csv", type=Path, default=EVAL_DIR / "summary.csv")
    parser.add_argument("--perf-csv", type=Path, default=PERF_CSV)
    parser.add_argument("--out-dir", type=Path, default=EVAL_DIR)
    parser.add_argument("--models-config", type=Path, default=DEFAULT_MODELS_CONFIG)
    args = parser.parse_args(argv)
    if not args.csv.is_file():
        print(f"ERROR: {args.csv} not found — run `make eval` first")
        return 2
    report = build_report(args.csv, args.out_dir, args.perf_csv, args.models_config)
    print(f"Report written to {report}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
