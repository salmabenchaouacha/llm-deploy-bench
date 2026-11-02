"""Comparative performance report generated from ``results/perf/summary.csv``.

Every number in the report is read from the CSV written by the benchmark: nothing is
typed by hand. Re-running the benchmark and this command refreshes the report.

Outputs (in ``results/perf/``):

* ``report.md``                - comparison table + offload-sweep table + figures
* ``figures/decode_speed.png``  - generation speed per variant
* ``figures/tradeoff.png``      - VRAM vs speed (the deployment trade-off)
* ``figures/offload_sweep.png`` - Mistral-7B: GPU layers vs speed / VRAM

CLI::

    python -m llmbench.models.report
"""

from __future__ import annotations

import argparse
import csv
import math
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless: CI, WSL, no display needed

import matplotlib.pyplot as plt

from llmbench.config import DEFAULT_MODELS_CONFIG, RESULTS_DIR, load_models_config

PERF_DIR = RESULTS_DIR / "perf"
GPU_VRAM_GIB = 4.0  # RTX 3050 Laptop (override with --vram-limit-gb)

# Reference palette (light mode), validated for colour-vision deficiencies.
INK, INK_2, GRID, SURFACE = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"
SERIES = ["#2a78d6", "#eb6834", "#1baf7a"]  # first 3 slots: safe for all-pairs (scatter)
OOM_COLOR = "#e34948"


# ---------------------------------------------------------------- data


@dataclass(frozen=True)
class Row:
    model: str
    quant: str
    ngl: str  # "all" or an integer as text
    family: str
    timestamp: str
    file_size_mb: float
    vram_model_mib: float
    ttft_ms_p50: float
    ttft_long_ms_p50: float
    decode_p50: float
    decode_std: float
    prefill_p50: float
    load_time_s: float
    error: str
    session_id: str

    @property
    def label(self) -> str:
        quant = self.quant.removeprefix("gguf_").upper()
        suffix = "" if self.ngl == "all" else f" ({self.ngl} GPU layers)"
        return f"{self.model} {quant}{suffix}"

    @property
    def ngl_int(self) -> int | None:
        return int(self.ngl) if self.ngl.isdigit() else None


def _f(value: str | None) -> float:
    try:
        return float(value) if value not in (None, "") else math.nan
    except ValueError:
        return math.nan


def load_rows(csv_path: Path) -> list[Row]:
    """Read the summary CSV and keep only the latest row per (model, quant, ngl)."""
    with csv_path.open(encoding="utf-8") as fh:
        raw = list(csv.DictReader(fh))
    latest: dict[tuple[str, str, str], dict[str, str]] = {}
    for r in sorted(raw, key=lambda r: r["timestamp"]):
        latest[(r["model"], r["quant"], r["n_gpu_layers"])] = r
    return [
        Row(
            model=r["model"],
            quant=r["quant"],
            ngl=r["n_gpu_layers"],
            family=r["model"].split("-")[0],
            timestamp=r["timestamp"],
            file_size_mb=_f(r["file_size_mb"]),
            vram_model_mib=_f(r["vram_model_mib"]),
            ttft_ms_p50=_f(r["ttft_ms_p50"]),
            ttft_long_ms_p50=_f(r["ttft_long_ms_p50"]),
            decode_p50=_f(r["decode_tok_s_p50"]),
            decode_std=_f(r["decode_tok_s_std"]),
            prefill_p50=_f(r["prefill_tok_s_p50"]),
            load_time_s=_f(r["load_time_s"]),
            error=r.get("error", ""),
            session_id=r["session_id"],
        )
        for r in latest.values()
    ]


def split_rows(rows: list[Row], default_ngl: dict[str, int | None]) -> tuple[list[Row], list[Row]]:
    """(comparison rows, offload-sweep rows).

    The comparison keeps, per (model, quant), the configured offload ("all" layers, or
    the value set in models.yaml for models that do not fit). The sweep holds every
    partial-offload measurement of models that have one.
    """
    comparison = [
        r
        for r in rows
        if not r.error
        and (r.ngl == "all" if default_ngl.get(r.model) is None
             else r.ngl_int == default_ngl[r.model])
    ]  # fmt: skip
    swept = {r.model for r in rows if r.ngl_int is not None}
    sweep = sorted((r for r in rows if r.model in swept and r.ngl_int is not None),
                   key=lambda r: (r.model, r.ngl_int or 0))  # fmt: skip
    comparison.sort(key=lambda r: -r.decode_p50)
    return comparison, sweep


# ---------------------------------------------------------------- figures


def _style(ax: plt.Axes) -> None:
    ax.set_facecolor(SURFACE)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
    ax.tick_params(colors=INK_2, labelsize=9)
    ax.grid(color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)


def _new(width: float, height: float, ncols: int = 1) -> tuple[plt.Figure, list[plt.Axes]]:
    fig, axes = plt.subplots(1, ncols, figsize=(width, height), dpi=150)
    fig.patch.set_facecolor(SURFACE)
    axes_list = list(axes) if ncols > 1 else [axes]
    for ax in axes_list:
        _style(ax)
    return fig, axes_list


def plot_decode_speed(rows: list[Row], out: Path) -> None:
    ordered = sorted(rows, key=lambda r: r.decode_p50)
    fig, (ax,) = _new(8, 0.5 * len(ordered) + 1.2)
    ax.grid(axis="y", visible=False)
    y = range(len(ordered))
    ax.barh(y, [r.decode_p50 for r in ordered], height=0.6, color=SERIES[0])
    ax.set_yticks(list(y), [r.label for r in ordered], color=INK, fontsize=9)
    for i, r in enumerate(ordered):
        ax.text(r.decode_p50, i, f"  {r.decode_p50:.1f}", va="center", color=INK, fontsize=9)
    ax.set_xlabel("Generation speed (tokens/s, median, higher is better)", color=INK_2)
    ax.set_xlim(0, max(r.decode_p50 for r in ordered) * 1.15)
    ax.set_title("Decode speed per model / quantization", loc="left", color=INK, fontsize=11)
    fig.tight_layout()
    fig.savefig(out, facecolor=SURFACE)
    plt.close(fig)


def plot_tradeoff(rows: list[Row], out: Path, vram_limit_gib: float) -> None:
    families = sorted({r.family for r in rows})
    colors = {f: SERIES[i % len(SERIES)] for i, f in enumerate(families)}
    fig, (ax,) = _new(8, 5)
    for fam in families:
        pts = [r for r in rows if r.family == fam]
        ax.scatter([r.vram_model_mib / 1024 for r in pts], [r.decode_p50 for r in pts],
                   s=70, color=colors[fam], edgecolor=SURFACE, linewidth=1.5,
                   label=fam, zorder=3)  # fmt: skip
        for r in pts:
            ax.annotate(r.label, (r.vram_model_mib / 1024, r.decode_p50),
                        textcoords="offset points", xytext=(7, 4), fontsize=8,
                        color=INK)  # fmt: skip
    ax.axvline(vram_limit_gib, color=INK_2, linestyle="--", linewidth=1)
    ax.text(vram_limit_gib, ax.get_ylim()[1], f" GPU limit {vram_limit_gib:.0f} GB",
            color=INK_2, fontsize=8, va="top")  # fmt: skip
    ax.set_xlim(0, vram_limit_gib * 1.25)
    ax.set_ylim(bottom=0)
    ax.set_xlabel("VRAM used by the model (GB)", color=INK_2)
    ax.set_ylabel("Decode speed (tokens/s)", color=INK_2)
    ax.set_title("Memory / speed trade-off (top-left is best)", loc="left", color=INK,
                 fontsize=11)  # fmt: skip
    ax.legend(frameon=False, labelcolor=INK, fontsize=9, title="family", title_fontsize=9)
    fig.tight_layout()
    fig.savefig(out, facecolor=SURFACE)
    plt.close(fig)


def plot_sweep(rows: list[Row], out: Path, vram_limit_gib: float) -> None:
    ok = [r for r in rows if not r.error]
    failed = [r for r in rows if r.error]
    fig, (ax_speed, ax_vram) = _new(10, 4, ncols=2)
    x = [r.ngl_int for r in ok]
    ax_speed.plot(x, [r.decode_p50 for r in ok], color=SERIES[0], linewidth=2, marker="o",
                  markersize=6, markeredgecolor=SURFACE)  # fmt: skip
    ax_vram.plot(x, [r.vram_model_mib / 1024 for r in ok], color=SERIES[0], linewidth=2,
                 marker="o", markersize=6, markeredgecolor=SURFACE)  # fmt: skip
    ax_vram.axhline(vram_limit_gib, color=INK_2, linestyle="--", linewidth=1)
    ax_vram.text(0, vram_limit_gib, f"GPU limit {vram_limit_gib:.0f} GB", color=INK_2,
                 fontsize=8, va="bottom")  # fmt: skip
    for r in ok:
        ax_speed.annotate(f"{r.decode_p50:.1f}", (r.ngl_int, r.decode_p50),
                          textcoords="offset points", xytext=(0, 8), ha="center",
                          fontsize=8, color=INK)  # fmt: skip
    for ax in (ax_speed, ax_vram):
        for r in failed:
            ax.axvline(r.ngl_int, color=OOM_COLOR, linewidth=1, linestyle=":")
            ax.text(r.ngl_int, 0.02, " OOM" if r.error == "cuda_oom" else " failed",
                    transform=ax.get_xaxis_transform(), color=INK_2, fontsize=8,
                    rotation=90, va="bottom")  # fmt: skip
        ax.set_xlabel("Layers offloaded to the GPU", color=INK_2)
        ax.set_ylim(bottom=0)
    ax_speed.set_ylabel("Decode speed (tokens/s)", color=INK_2)
    ax_vram.set_ylabel("VRAM used (GB)", color=INK_2)
    ax_vram.set_ylim(top=vram_limit_gib * 1.15)
    model = rows[0].model if rows else ""
    ax_speed.set_title(f"{model}: speed vs GPU offload", loc="left", color=INK, fontsize=11)
    ax_vram.set_title(f"{model}: VRAM vs GPU offload", loc="left", color=INK, fontsize=11)
    fig.tight_layout()
    fig.savefig(out, facecolor=SURFACE)
    plt.close(fig)


# ---------------------------------------------------------------- markdown


def _n(value: float, digits: int = 0) -> str:
    return "—" if math.isnan(value) else f"{value:,.{digits}f}"


def comparison_table(rows: list[Row]) -> str:
    head = (
        "| Model | Quant | GPU layers | File (GB) | VRAM (MiB) | TTFT short p50 (ms) "
        "| TTFT long p50 (ms) | Prefill (tok/s) | Decode (tok/s) |\n"
        "|---|---|---|--:|--:|--:|--:|--:|--:|\n"
    )
    lines = [
        f"| {r.model} | {r.quant.removeprefix('gguf_').upper()} | {r.ngl} "
        f"| {_n(r.file_size_mb / 1024, 2)} | {_n(r.vram_model_mib)} | {_n(r.ttft_ms_p50)} "
        f"| {_n(r.ttft_long_ms_p50)} | {_n(r.prefill_p50)} "
        f"| **{_n(r.decode_p50, 1)}** ± {_n(r.decode_std, 1)} |"
        for r in rows
    ]
    return head + "\n".join(lines)


def sweep_table(rows: list[Row]) -> str:
    head = (
        "| GPU layers | VRAM (MiB) | TTFT long p50 (ms) | Decode (tok/s) | Status |\n"
        "|--:|--:|--:|--:|---|\n"
    )
    lines = [
        f"| {r.ngl} | {_n(r.vram_model_mib)} | {_n(r.ttft_long_ms_p50)} "
        f"| {_n(r.decode_p50, 1)} | {'ok' if not r.error else r.error} |"
        for r in rows
    ]
    return head + "\n".join(lines)


def build_report(
    csv_path: Path, out_dir: Path, models_config: Path, vram_limit: float = GPU_VRAM_GIB
) -> Path:
    cfg = load_models_config(models_config)
    default_ngl = {
        m: next((v.n_gpu_layers for v in spec.variants if v.enabled and v.is_gguf), None)
        for m, spec in cfg.models.items()
    }
    rows = load_rows(csv_path)
    comparison, sweep = split_rows(rows, default_ngl)

    fig_dir = out_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    parts = [
        "# Performance benchmark — results",
        "",
        "_Generated by `python -m llmbench.models.report` from "
        f"[`{csv_path.name}`]({csv_path.name}). Do not edit by hand._",
        "",
        "Protocol: `configs/bench.yaml` — greedy decoding, 128 generated tokens per run, "
        "prompt cache disabled, 1 warm-up + 3 repetitions of 3 prompts. "
        "VRAM = device memory used minus the baseline before loading.",
    ]
    if comparison:
        plot_decode_speed(comparison, fig_dir / "decode_speed.png")
        plot_tradeoff(comparison, fig_dir / "tradeoff.png", vram_limit)
        parts += [
            "",
            "## Comparison",
            "",
            comparison_table(comparison),
            "",
            "![Decode speed](figures/decode_speed.png)",
            "",
            "![VRAM / speed trade-off](figures/tradeoff.png)",
        ]
    if sweep:
        plot_sweep(sweep, fig_dir / "offload_sweep.png", vram_limit)
        parts += [
            "",
            f"## GPU offload sweep — {sweep[0].model}",
            "",
            sweep_table(sweep),
            "",
            "![Offload sweep](figures/offload_sweep.png)",
        ]
    report = out_dir / "report.md"
    report.write_text("\n".join(parts) + "\n", encoding="utf-8")
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Build the performance report.")
    parser.add_argument("--csv", type=Path, default=PERF_DIR / "summary.csv")
    parser.add_argument("--out-dir", type=Path, default=PERF_DIR)
    parser.add_argument("--models-config", type=Path, default=DEFAULT_MODELS_CONFIG)
    parser.add_argument("--vram-limit-gb", type=float, default=GPU_VRAM_GIB)
    args = parser.parse_args(argv)
    if not args.csv.is_file():
        print(f"ERROR: {args.csv} not found — run `make bench` first")
        return 2
    report = build_report(args.csv, args.out_dir, args.models_config, args.vram_limit_gb)
    print(f"Report written to {report}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
