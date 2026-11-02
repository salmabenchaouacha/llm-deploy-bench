"""Run the Locust load test at several concurrency levels and summarise each level.

For every number of concurrent users: one headless Locust run of ``--duration`` seconds,
the raw per-request records in ``results/load/raw/``, one summary row in
``results/load/summary.csv``. In-flight requests finish before a level ends
(``--stop-timeout``), so no request is cut and counted as a failure.

CLI::

    python -m llmbench.load.run --scenario chat --users 1 5 10 --duration 60
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import subprocess
import sys
import time
import urllib.error
import urllib.request
import uuid
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from llmbench.config import PROJECT_ROOT, RESULTS_DIR
from llmbench.log import get_logger
from llmbench.models.bench import git_commit, percentile
from llmbench.monitoring.memory import PeakSampler, gpu_info, nvml_used_mib_reader

log = get_logger(__name__)

LOAD_DIR = RESULTS_DIR / "load"
LOCUSTFILE = PROJECT_ROOT / "load" / "locustfile.py"
USER_CLASS = {"chat": "ChatUser", "rag": "RagUser"}

SUMMARY_FIELDS = [
    "session_id", "timestamp", "git_commit", "scenario", "users", "duration_s", "model",
    "llm_slots", "max_tokens", "requests", "failures", "requests_per_s",
    "throughput_tok_s", "latency_ms_p50", "latency_ms_p95", "ttft_ms_p50", "ttft_ms_p95",
    "decode_tok_s_per_request_p50", "vram_peak_mib", "gpu_name",
]  # fmt: skip


@dataclass(frozen=True)
class Record:
    start: float
    ok: bool
    latency_ms: float
    ttft_ms: float | None
    tokens: int | None


def load_records(path: Path) -> list[Record]:
    if not path.exists():
        return []
    out = []
    for line in path.read_text(encoding="utf-8").splitlines():
        r = json.loads(line)
        out.append(Record(r["start"], r["ok"], r["latency_ms"], r["ttft_ms"], r["tokens"]))
    return out


def summarize(records: Sequence[Record]) -> dict[str, Any]:
    """Aggregate one level. Throughput = tokens of successful requests / wall-clock span."""
    ok = [r for r in records if r.ok]
    if not ok:
        return {"requests": 0, "failures": len(records)}
    span_s = max(r.start + r.latency_ms / 1000 for r in ok) - min(r.start for r in ok)
    tokens = sum(r.tokens or 0 for r in ok)
    per_request = [
        (r.tokens - 1) / ((r.latency_ms - r.ttft_ms) / 1000)
        for r in ok
        if r.tokens and r.tokens > 1 and r.ttft_ms is not None and r.latency_ms > r.ttft_ms
    ]
    ttfts = [r.ttft_ms for r in ok if r.ttft_ms is not None]
    return {
        "requests": len(ok),
        "failures": len(records) - len(ok),
        "requests_per_s": len(ok) / span_s if span_s > 0 else math.nan,
        "throughput_tok_s": tokens / span_s if span_s > 0 else math.nan,
        "latency_ms_p50": percentile([r.latency_ms for r in ok], 50),
        "latency_ms_p95": percentile([r.latency_ms for r in ok], 95),
        "ttft_ms_p50": percentile(ttfts, 50),
        "ttft_ms_p95": percentile(ttfts, 95),
        "decode_tok_s_per_request_p50": percentile(per_request, 50),
    }


def append_summary(row: dict[str, Any], csv_path: Path) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    new_file = not csv_path.exists()
    with csv_path.open("a", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=SUMMARY_FIELDS)
        if new_file:
            writer.writeheader()
        writer.writerow({k: _fmt(row.get(k)) for k in SUMMARY_FIELDS})


def _fmt(value: Any) -> Any:
    if isinstance(value, float):
        return "" if math.isnan(value) else round(value, 2)
    return "" if value is None else value


def _get_json(url: str, timeout: float = 5.0) -> Any:
    with urllib.request.urlopen(url, timeout=timeout) as resp:
        return json.loads(resp.read())


def _post_json(url: str, payload: dict[str, Any], timeout: float = 300.0) -> Any:
    req = urllib.request.Request(url, data=json.dumps(payload).encode(), method="POST",
                                 headers={"Content-Type": "application/json"})  # fmt: skip
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read())


def wait_healthy(host: str, timeout_s: float) -> None:
    deadline = time.monotonic() + timeout_s
    last = "no answer"
    while time.monotonic() < deadline:
        try:
            _get_json(f"{host}/health")
            return
        except urllib.error.HTTPError as exc:  # 503: a service is still starting
            last = exc.read().decode(errors="replace")
        except (urllib.error.URLError, OSError) as exc:
            last = str(exc)
        time.sleep(2)
    raise SystemExit(
        f"API not healthy at {host}/health after {timeout_s:.0f}s: {last}\n"
        "Is it running? `make serve-up`, then `docker compose logs api llm`"
    )


def warm_up(host: str, scenario: str) -> None:
    """One unmeasured request: the first call pays CUDA graph / cache initialisation."""
    if scenario == "chat":
        hello = {"messages": [{"role": "user", "content": "Hello"}], "max_tokens": 8}
        _post_json(f"{host}/v1/chat/completions", hello)
    else:
        _post_json(f"{host}/rag/query", {"question": "What was the net income in 2025?"})


def served_model(host: str) -> str:
    try:
        return str(_get_json(f"{host}/v1/models")["data"][0]["id"])
    except (urllib.error.URLError, OSError, KeyError, IndexError, ValueError):
        return "unknown"


def llm_slots(llm_url: str) -> int | None:
    """Parallel decoding slots of llama-server (requests generated at the same time)."""
    try:
        return int(_get_json(f"{llm_url}/props")["total_slots"])
    except (urllib.error.URLError, OSError, KeyError, ValueError, TypeError):
        return None


def run_level(scenario: str, users: int, duration_s: int, host: str, max_tokens: int,
              records_path: Path) -> None:  # fmt: skip
    cmd = [
        sys.executable, "-m", "locust", "-f", str(LOCUSTFILE), USER_CLASS[scenario],
        "--headless", "--users", str(users), "--spawn-rate", str(users),
        "--run-time", f"{duration_s}s", "--stop-timeout", "300",
        "--host", host, "--only-summary", "--loglevel", "WARNING",
    ]  # fmt: skip
    env = {**os.environ, "LOAD_RECORDS": str(records_path), "LOAD_MAX_TOKENS": str(max_tokens)}
    # Locust exits with 1 when some requests failed: failures are in the records, not fatal.
    proc = subprocess.run(cmd, env=env, check=False)
    if proc.returncode not in (0, 1):
        raise SystemExit(f"locust exited with code {proc.returncode}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Load test the API with Locust.")
    parser.add_argument("--scenario", choices=sorted(USER_CLASS), default="chat")
    parser.add_argument("--users", type=int, nargs="+", default=[1, 5, 10])
    parser.add_argument("--duration", type=int, default=60, help="seconds per level")
    parser.add_argument("--max-tokens", type=int, default=128, help="chat: tokens per answer")
    parser.add_argument("--host", default="http://127.0.0.1:8000")
    parser.add_argument("--llm-url", default="http://127.0.0.1:8092", help="for /props")
    parser.add_argument("--out-dir", type=Path, default=LOAD_DIR)
    args = parser.parse_args(argv)

    wait_healthy(args.host, timeout_s=600)
    warm_up(args.host, args.scenario)
    session = uuid.uuid4().hex[:8]
    meta = {
        "session_id": session, "git_commit": git_commit(), "scenario": args.scenario,
        "duration_s": args.duration, "model": served_model(args.host),
        "llm_slots": llm_slots(args.llm_url), "gpu_name": gpu_info().get("gpu_name", ""),
        "max_tokens": args.max_tokens if args.scenario == "chat" else "",
    }  # fmt: skip
    raw_dir = args.out_dir / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)
    for users in args.users:
        records_path = raw_dir / f"{session}_{args.scenario}_u{users}.jsonl"
        # A service restarted by Docker (e.g. the embedder killed for lack of RAM) must be
        # back before the level starts, or the level only measures fast 503 errors.
        wait_healthy(args.host, timeout_s=600)
        log.info("load level", extra={"scenario": args.scenario, "users": users})
        try:
            sampler: PeakSampler | None = PeakSampler(nvml_used_mib_reader(), name="vram")
        except Exception:  # no NVML (CPU-only machine, CI)
            sampler = None
        if sampler is not None:
            sampler.start()
        try:
            run_level(args.scenario, users, args.duration, args.host, args.max_tokens,
                      records_path)  # fmt: skip
        finally:
            vram = sampler.stop() if sampler is not None else math.nan
        stats = summarize(load_records(records_path))
        row = {**meta, **stats, "users": users, "vram_peak_mib": vram,
               "timestamp": datetime.now(UTC).isoformat(timespec="seconds")}  # fmt: skip
        append_summary(row, args.out_dir / "summary.csv")
        if stats.get("failures", 0) > max(10, stats.get("requests", 0) // 10):
            print(f"WARNING: {stats['failures']} failed requests at {users} users: a service "
                  "probably crashed. Check `docker compose ps` (an 'Up X seconds' service "
                  "restarted) and `free -h`, then rerun this scenario.")  # fmt: skip
        print(f"-> {args.scenario} {users:>2} users: {stats.get('requests', 0)} ok / "
              f"{stats.get('failures', 0)} failed, "
              f"{row.get('throughput_tok_s', math.nan):.1f} tok/s total, "
              f"p95 latency {row.get('latency_ms_p95', math.nan) / 1000:.1f} s")  # fmt: skip
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
