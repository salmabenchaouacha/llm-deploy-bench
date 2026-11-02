"""Performance benchmark of GGUF variants served by llama.cpp.

For each (model, quant, n_gpu_layers) target:

1. download the GGUF file if needed, record its size on disk;
2. read the GPU memory baseline, start the VRAM / RAM peak samplers;
3. start llama-server in Docker, record the load time (until /health is OK);
4. warm up, then stream every prompt ``runs`` times (greedy, fixed max_new_tokens,
   prompt cache disabled) and time each generation;
5. append raw runs to ``results/perf/runs.jsonl`` and one summary row to
   ``results/perf/summary.csv``. A target that fails (e.g. CUDA OOM) still gets a row
   with its error: "does not fit" is a result too.

CLI::

    python -m llmbench.models.bench --all
    python -m llmbench.models.bench --model qwen2.5-1.5b --quant gguf_q4_k_m
    python -m llmbench.models.bench --model mistral-7b --quant gguf_q4_k_m \
        --n-gpu-layers 0 8 16 20 24        # offload sweep
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
import subprocess
import sys
import uuid
from collections.abc import Callable, Iterator, Sequence
from contextlib import AbstractContextManager
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Protocol

from llmbench.config import (
    DEFAULT_BENCH_CONFIG,
    DEFAULT_MODELS_CONFIG,
    PROJECT_ROOT,
    RESULTS_DIR,
    BenchConfig,
    ConfigError,
    ModelsConfig,
    Variant,
    load_bench_config,
    load_models_config,
)
from llmbench.log import get_logger
from llmbench.models.hub import ensure_gguf, gguf_relpath
from llmbench.monitoring.memory import (
    PeakSampler,
    Reader,
    docker_mem_mib_reader,
    gpu_info,
    nvml_used_mib_reader,
)
from llmbench.serving.client import StreamResult, count_tokens, stream_chat
from llmbench.serving.llama_server import DockerRunError, LlamaServer, ServerStartError

log = get_logger(__name__)

PERF_DIR = RESULTS_DIR / "perf"

SUMMARY_FIELDS = [
    "session_id", "timestamp", "git_commit", "server_image", "gpu_name", "gpu_driver",
    "model", "quant", "params_b", "n_gpu_layers", "n_ctx", "max_new_tokens",
    "file_size_mb", "load_time_s", "vram_baseline_mib", "vram_peak_mib", "vram_model_mib",
    "ram_peak_mib", "n_runs",
    "ttft_ms_p50", "ttft_ms_p95", "ttft_long_ms_p50",
    "decode_tok_s_p50", "decode_tok_s_mean", "decode_tok_s_std",
    "prefill_tok_s_p50", "total_ms_p50", "gen_tokens_mean",
    "error",
]  # fmt: skip


# ---------------------------------------------------------------- pure metrics


def decode_tokens_per_s(gen_tokens: int, ttft_s: float, total_s: float) -> float:
    """Generation speed after the first token: (n - 1) tokens over (total - ttft)."""
    dt = total_s - ttft_s
    if gen_tokens < 2 or dt <= 0:
        return math.nan
    return (gen_tokens - 1) / dt


def percentile(values: Sequence[float], q: float) -> float:
    """Linear-interpolated percentile (q in [0, 100]); NaN on empty input."""
    data = sorted(v for v in values if not math.isnan(v))
    if not data:
        return math.nan
    if len(data) == 1:
        return data[0]
    pos = (len(data) - 1) * q / 100
    lo, hi = math.floor(pos), math.ceil(pos)
    return data[lo] + (data[hi] - data[lo]) * (pos - lo)


def _mean_std(values: Sequence[float]) -> tuple[float, float]:
    data = [v for v in values if not math.isnan(v)]
    if not data:
        return math.nan, math.nan
    return statistics.fmean(data), statistics.stdev(data) if len(data) > 1 else 0.0


# ---------------------------------------------------------------- records


@dataclass
class RunRecord:
    session_id: str
    model: str
    quant: str
    n_gpu_layers: int | None
    prompt_idx: int
    repetition: int
    prompt_tokens: int | None
    gen_tokens: int
    gen_tokens_source: str
    ttft_ms: float
    total_ms: float
    decode_tok_s: float
    server_decode_tok_s: float | None
    server_prefill_tok_s: float | None
    finish_reason: str | None


def _gen_tokens(res: StreamResult, base_url: str) -> tuple[int, str]:
    """Most reliable generated-token count available, and where it came from."""
    if res.server_timings and res.server_timings.get("predicted_n"):
        return int(res.server_timings["predicted_n"]), "server_timings"
    if res.usage and res.usage.get("completion_tokens"):
        return int(res.usage["completion_tokens"]), "usage"
    try:
        return count_tokens(base_url, res.text), "tokenize"
    except Exception:
        return res.n_chunks, "chunks"


def _prompt_tokens(res: StreamResult) -> int | None:
    if res.server_timings and res.server_timings.get("prompt_n"):
        return int(res.server_timings["prompt_n"])
    if res.usage and res.usage.get("prompt_tokens"):
        return int(res.usage["prompt_tokens"])
    return None


def summarize(records: Sequence[RunRecord]) -> dict[str, Any]:
    ttft = [r.ttft_ms for r in records]
    decode = [r.decode_tok_s for r in records]
    prefill = [r.server_prefill_tok_s for r in records if r.server_prefill_tok_s is not None]
    long_idx = max((r.prompt_idx for r in records), default=0)
    ttft_long = [r.ttft_ms for r in records if r.prompt_idx == long_idx]
    mean, std = _mean_std(decode)
    return {
        "n_runs": len(records),
        "ttft_ms_p50": percentile(ttft, 50),
        "ttft_ms_p95": percentile(ttft, 95),
        "ttft_long_ms_p50": percentile(ttft_long, 50),
        "decode_tok_s_p50": percentile(decode, 50),
        "decode_tok_s_mean": mean,
        "decode_tok_s_std": std,
        "prefill_tok_s_p50": percentile(prefill, 50),
        "total_ms_p50": percentile([r.total_ms for r in records], 50),
        "gen_tokens_mean": statistics.fmean(r.gen_tokens for r in records) if records else math.nan,
    }  # fmt: skip


def _fmt(value: Any) -> Any:
    if isinstance(value, float):
        return "" if math.isnan(value) else round(value, 3)
    return "" if value is None else value


def append_results(
    records: Sequence[RunRecord], summary: dict[str, Any], results_dir: Path
) -> None:
    results_dir.mkdir(parents=True, exist_ok=True)
    with (results_dir / "runs.jsonl").open("a", encoding="utf-8") as fh:
        for r in records:
            fh.write(json.dumps(asdict(r), ensure_ascii=False) + "\n")
    csv_path = results_dir / "summary.csv"
    new_file = not csv_path.exists()
    if not new_file:
        with csv_path.open(encoding="utf-8") as fh:
            header = next(csv.reader(fh), [])
        if header != SUMMARY_FIELDS:  # appending would silently misalign columns
            raise RuntimeError(
                f"{csv_path} has different columns than this version writes; "
                "move it aside (e.g. summary_v1.csv) and rerun"
            )
    with csv_path.open("a", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=SUMMARY_FIELDS)
        if new_file:
            writer.writeheader()
        writer.writerow({k: _fmt(summary.get(k)) for k in SUMMARY_FIELDS})


# ---------------------------------------------------------------- environment


def git_commit() -> str:
    try:
        sha = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=PROJECT_ROOT, capture_output=True, text=True, check=True,
        ).stdout.strip()  # fmt: skip
        dirty = subprocess.run(
            ["git", "status", "--porcelain", "--", "src", "configs"],
            cwd=PROJECT_ROOT, capture_output=True, text=True, check=True,
        ).stdout.strip()  # fmt: skip
        return f"{sha}-dirty" if dirty else sha
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def image_digest(image: str) -> str:
    """Immutable digest of the server image, so a result can be reproduced exactly."""
    try:
        out = subprocess.run(
            ["docker", "image", "inspect", "--format", "{{index .RepoDigests 0}}", image],
            capture_output=True, text=True, check=True,
        ).stdout.strip()  # fmt: skip
        return out or image
    except (OSError, subprocess.CalledProcessError):
        return image


# ---------------------------------------------------------------- orchestration


@dataclass(frozen=True)
class Target:
    model_id: str
    variant: Variant
    n_gpu_layers: int | None
    params_b: float


class Server(Protocol):
    @property
    def base_url(self) -> str: ...

    @property
    def load_time_s(self) -> float | None: ...


ServerFactory = Callable[[Target, BenchConfig], AbstractContextManager[Server]]


def _default_server(target: Target, cfg: BenchConfig) -> AbstractContextManager[Server]:
    return LlamaServer.from_config(cfg.server, gguf_relpath(target.variant), target.n_gpu_layers)


def _measure(
    target: Target, cfg: BenchConfig, base_url: str, session_id: str
) -> Iterator[RunRecord]:
    def call(prompt: str) -> StreamResult:
        return stream_chat(
            base_url,
            [{"role": "user", "content": prompt}],
            max_tokens=cfg.max_new_tokens,
            temperature=0.0,
            seed=42,
            ignore_eos=cfg.ignore_eos,
            cache_prompt=False,
        )

    for _ in range(cfg.warmup_runs):
        call(cfg.prompts[0])

    for rep in range(cfg.runs):
        for idx, prompt in enumerate(cfg.prompts):
            res = call(prompt)
            n_gen, source = _gen_tokens(res, base_url)
            t = res.server_timings or {}
            yield RunRecord(
                session_id=session_id,
                model=target.model_id,
                quant=target.variant.quant.value,
                n_gpu_layers=target.n_gpu_layers,
                prompt_idx=idx,
                repetition=rep,
                prompt_tokens=_prompt_tokens(res),
                gen_tokens=n_gen,
                gen_tokens_source=source,
                ttft_ms=res.ttft_s * 1000,
                total_ms=res.total_s * 1000,
                decode_tok_s=decode_tokens_per_s(n_gen, res.ttft_s, res.total_s),
                server_decode_tok_s=t.get("predicted_per_second"),
                server_prefill_tok_s=t.get("prompt_per_second"),
                finish_reason=res.finish_reason,
            )


def run_target(
    target: Target,
    cfg: BenchConfig,
    *,
    session: dict[str, Any],
    ensure_model: Callable[[Variant, Path], Path] = ensure_gguf,
    server_factory: ServerFactory = _default_server,
    vram_reader: Reader | None = None,
    ram_reader: Reader | None = None,
) -> tuple[list[RunRecord], dict[str, Any]]:
    """Benchmark one target. Never raises for a server failure: it is reported in 'error'."""
    path = ensure_model(target.variant, cfg.server.models_path)
    summary: dict[str, Any] = {
        **session,
        "model": target.model_id,
        "quant": target.variant.quant.value,
        "params_b": target.params_b,
        "n_gpu_layers": "all" if target.n_gpu_layers is None else target.n_gpu_layers,
        "n_ctx": cfg.server.n_ctx,
        "max_new_tokens": cfg.max_new_tokens,
        "file_size_mb": path.stat().st_size / 1024**2,
    }
    baseline = vram_reader() if vram_reader else math.nan
    vram = PeakSampler(vram_reader, 0.2, "vram") if vram_reader else None
    ram = PeakSampler(ram_reader, 1.0, "ram") if ram_reader else None
    records: list[RunRecord] = []

    log.info("benchmarking", extra={"model": target.model_id, "quant": target.variant.quant.value,
                                    "ngl": summary["n_gpu_layers"]})  # fmt: skip
    if vram:
        vram.start()
    try:
        with server_factory(target, cfg) as server:
            if ram:
                ram.start()
            summary["load_time_s"] = server.load_time_s
            records = list(_measure(target, cfg, server.base_url, session["session_id"]))
            if ram:  # read while the container still exists: docker stats fails after
                ram.stop()
    except ServerStartError as exc:
        oom = "out of memory" in str(exc).lower()
        summary["error"] = "cuda_oom" if oom else "server_start_failed"
        log.error("server failed", extra={"error": summary["error"], "details": str(exc)[-800:]})
    finally:
        vram_peak = vram.stop() if vram else math.nan
        ram_peak = ram.stop() if ram else math.nan

    summary.update(
        vram_baseline_mib=baseline,
        vram_peak_mib=vram_peak,
        vram_model_mib=vram_peak - baseline,
        ram_peak_mib=ram_peak,
        **(summarize(records) if records else {"n_runs": 0}),
    )
    return records, summary


def select_targets(
    cfg: ModelsConfig, model: str | None, quant: str | None, ngl: list[int] | None
) -> list[Target]:
    if model is None:
        pairs = [(m, v) for m, v in cfg.iter_variants(gpu_available=True) if v.is_gguf]
    else:
        spec = cfg.get(model)
        pairs = (
            [(model, spec.variant(quant))]
            if quant
            else [(model, v) for v in spec.variants if v.enabled and v.is_gguf]
        )
    targets = []
    for model_id, variant in pairs:
        if not variant.is_gguf:
            raise SystemExit(f"{model_id}/{variant.quant}: Transformers backend is planned for V2")
        params = cfg.get(model_id).params_b
        layer_counts: list[int | None] = list(ngl) if ngl else [variant.n_gpu_layers]
        for n in layer_counts:
            targets.append(Target(model_id, variant, n, params))
    return targets


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Latency / throughput / memory benchmark.")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--all", action="store_true", help="every enabled GGUF variant")
    group.add_argument("--model", help="model id from configs/models.yaml")
    parser.add_argument("--quant", help="one quant of --model (default: all enabled GGUF ones)")
    parser.add_argument("--n-gpu-layers", type=int, nargs="+", help="override / sweep")
    parser.add_argument("--runs", type=int, help="override bench.yaml runs")
    parser.add_argument("--models-config", default=DEFAULT_MODELS_CONFIG)
    parser.add_argument("--bench-config", default=DEFAULT_BENCH_CONFIG)
    parser.add_argument("--results-dir", type=Path, default=PERF_DIR)
    parser.add_argument("--dry-run", action="store_true", help="list targets and exit")
    args = parser.parse_args(argv)

    try:
        models_cfg = load_models_config(args.models_config)
        bench_cfg = load_bench_config(args.bench_config)
        targets = select_targets(models_cfg, args.model, args.quant, args.n_gpu_layers)
    except (ConfigError, KeyError) as exc:
        print(f"ERROR: {exc.args[0] if exc.args else exc}", file=sys.stderr)
        return 2
    if args.runs:
        bench_cfg = bench_cfg.model_copy(update={"runs": args.runs})

    for t in targets:
        ngl = "all" if t.n_gpu_layers is None else t.n_gpu_layers
        print(f"{t.model_id:<16} {t.variant.quant.value:<12} ngl={ngl}")
    if args.dry_run:
        return 0

    info = gpu_info(bench_cfg.gpu_index)
    session = {
        "session_id": uuid.uuid4().hex[:8],
        "timestamp": datetime.now(UTC).isoformat(timespec="seconds"),
        "git_commit": git_commit(),
        "server_image": image_digest(bench_cfg.server.image),
        "gpu_name": info.get("gpu_name", "none"),
        "gpu_driver": info.get("gpu_driver", ""),
    }
    vram_reader = nvml_used_mib_reader(bench_cfg.gpu_index) if info else None
    ram_reader = docker_mem_mib_reader(bench_cfg.server.container_name)

    failures = 0
    for target in targets:
        try:
            records, summary = run_target(
                target, bench_cfg, session=session, vram_reader=vram_reader, ram_reader=ram_reader
            )
        except DockerRunError as exc:
            print(f"ERROR: {exc}\nNothing was written for this target. Fix Docker and rerun.",
                  file=sys.stderr)  # fmt: skip
            return 3
        append_results(records, summary, args.results_dir)
        failures += bool(summary.get("error"))
        print(
            f"-> {target.model_id} {target.variant.quant.value}: "
            + (summary["error"] if summary.get("error") else
               f"decode {summary['decode_tok_s_p50']:.1f} tok/s, "
               f"TTFT p50 {summary['ttft_ms_p50']:.0f} ms, "
               f"VRAM +{summary['vram_model_mib']:.0f} MiB")
        )  # fmt: skip
    print(f"Results appended to {args.results_dir}/ (session {session['session_id']})")
    return 1 if failures == len(targets) else 0


if __name__ == "__main__":
    raise SystemExit(main())
