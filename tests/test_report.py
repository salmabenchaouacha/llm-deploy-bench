"""Tests for the performance report generator (synthetic CSV, never real results)."""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any

import pytest

from llmbench.config import DEFAULT_MODELS_CONFIG
from llmbench.models.bench import SUMMARY_FIELDS, append_results
from llmbench.models.report import build_report, load_rows, split_rows


def _row(model: str, quant: str, ngl: str, decode: float, ts: str, **kw: Any) -> dict[str, Any]:
    base = {k: "" for k in SUMMARY_FIELDS}
    base.update(
        session_id="s", timestamp=ts, model=model, quant=quant, n_gpu_layers=ngl,
        file_size_mb=1000, vram_model_mib=900, vram_peak_mib=1034,
        ttft_ms_p50=30, ttft_long_ms_p50=60,
        decode_tok_s_p50=decode, decode_tok_s_std=0.5, prefill_tok_s_p50=2000, load_time_s=2,
    )  # fmt: skip
    base.update(kw)
    return base


def _write(path: Path, rows: list[dict[str, Any]]) -> Path:
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=SUMMARY_FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    return path


@pytest.fixture
def csv_path(tmp_path: Path) -> Path:
    return _write(
        tmp_path / "summary.csv",
        [
            _row("qwen2.5-0.5b", "gguf_q4_k_m", "all", 100.0, "2026-01-01T00:00:00"),
            _row("qwen2.5-0.5b", "gguf_q4_k_m", "all", 150.0, "2026-01-02T00:00:00"),  # newer
            _row("qwen2.5-1.5b", "gguf_q8_0", "all", 70.0, "2026-01-02T00:00:00"),
            _row("mistral-7b", "gguf_q4_k_m", "0", 2.0, "2026-01-02T00:00:00"),
            _row("mistral-7b", "gguf_q4_k_m", "20", 5.0, "2026-01-02T00:00:00"),
            _row(
                "mistral-7b",
                "gguf_q4_k_m",
                "28",
                8.0,
                "2026-01-02T00:00:00",
                vram_model_mib=3909,
                vram_peak_mib=4043,
            ),
            _row(
                "mistral-7b",
                "gguf_q4_k_m",
                "32",
                0,
                "2026-01-02T00:00:00",
                error="cuda_oom",
                decode_tok_s_p50="",
            ),
        ],
    )


def test_latest_row_wins(csv_path: Path) -> None:
    rows = load_rows(csv_path)
    qwen = [r for r in rows if r.model == "qwen2.5-0.5b"]
    assert len(qwen) == 1 and qwen[0].decode_p50 == 150.0


def test_split_uses_configured_offload_and_keeps_failures_in_sweep(csv_path: Path) -> None:
    comparison, sweep = split_rows(load_rows(csv_path), {"mistral-7b": 20})
    assert [r.label for r in comparison] == [
        "qwen2.5-0.5b Q4_K_M", "qwen2.5-1.5b Q8_0", "mistral-7b Q4_K_M (20 GPU layers)",
    ]  # fmt: skip
    assert [(r.ngl, r.error) for r in sweep] == [
        ("0", ""), ("20", ""), ("28", ""), ("32", "cuda_oom"),
    ]  # fmt: skip


def test_build_report_writes_markdown_and_figures(csv_path: Path, tmp_path: Path) -> None:
    out = tmp_path / "out"
    report = build_report(csv_path, out, DEFAULT_MODELS_CONFIG)
    text = report.read_text()
    assert "| qwen2.5-0.5b | Q4_K_M | all |" in text
    assert "**150.0** ± 0.5" in text
    assert "| 32 | 900 | 60 | — | cuda_oom |" in text
    assert "| 28 | ≥ 3,909 ¹ | 60 | 8.0 | ok |" in text  # saturated card flagged
    assert "sysmem fallback" in text
    assert "| 20 | 900 |" in text  # not saturated: plain value
    for name in ("decode_speed", "tradeoff", "offload_sweep"):
        png = out / "figures" / f"{name}.png"
        assert png.stat().st_size > 10_000, name


def test_append_refuses_mismatched_header(tmp_path: Path) -> None:
    (tmp_path / "summary.csv").write_text("old,columns\n1,2\n")
    with pytest.raises(RuntimeError, match="different columns"):
        append_results([], {"session_id": "s"}, tmp_path)
