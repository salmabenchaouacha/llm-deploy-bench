"""Load-test aggregation and report (Locust itself is not needed)."""

from __future__ import annotations

import math
from pathlib import Path

import pytest

from llmbench.load.report import build_report
from llmbench.load.run import Record, append_summary, load_records, summarize


def test_summarize_throughput_latency_and_failures() -> None:
    records = [
        # two requests in parallel during 2 s, 100 tokens each, TTFT 0.5 s
        Record(start=100.0, ok=True, latency_ms=2000, ttft_ms=500, tokens=100),
        Record(start=100.0, ok=True, latency_ms=2000, ttft_ms=500, tokens=100),
        Record(start=101.0, ok=False, latency_ms=10, ttft_ms=None, tokens=None),
    ]
    s = summarize(records)
    assert s["requests"] == 2 and s["failures"] == 1
    assert s["throughput_tok_s"] == pytest.approx(100.0)  # 200 tokens / 2 s
    assert s["requests_per_s"] == pytest.approx(1.0)
    assert s["latency_ms_p95"] == 2000 and s["ttft_ms_p50"] == 500
    assert s["decode_tok_s_per_request_p50"] == pytest.approx(99 / 1.5)


def test_summarize_all_failed() -> None:
    assert summarize([Record(0.0, False, 5.0, None, None)]) == {"requests": 0, "failures": 1}


def test_records_roundtrip_and_report(tmp_path: Path) -> None:
    raw = tmp_path / "r.jsonl"
    raw.write_text(
        '{"scenario": "chat", "start": 1.0, "ok": true, "latency_ms": 1000.0, '
        '"ttft_ms": 100.0, "tokens": 50, "error": null}\n'
    )
    assert load_records(raw)[0].tokens == 50
    csv_path = tmp_path / "summary.csv"
    for users, tput in [(1, 40.0), (5, 120.0), (10, 130.0)]:
        append_summary({"session_id": "s", "timestamp": f"2026-01-01T00:00:0{users % 10}",
                        "scenario": "chat", "users": users, "model": "m", "llm_slots": 4,
                        "duration_s": 60, "requests": 10, "failures": 0,
                        "throughput_tok_s": tput, "latency_ms_p50": 1000.0,
                        "latency_ms_p95": 1500.0 * users, "ttft_ms_p50": 80.0,
                        "ttft_ms_p95": math.nan, "vram_peak_mib": 3000.0}, csv_path)  # fmt: skip
    report = build_report(csv_path, tmp_path).read_text()
    assert "| chat | 5 | 10 / 0 |" in report and "**120.0**" in report
    assert "4 parallel slots" in report
    assert (tmp_path / "figures" / "load.png").stat().st_size > 10_000
