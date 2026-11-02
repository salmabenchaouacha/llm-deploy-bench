"""Tests for the performance benchmark: metrics, target selection, end-to-end run."""

from __future__ import annotations

import csv
import itertools
import json
import math
import shutil
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

import pytest

from llmbench.config import BenchConfig, Variant, load_bench_config, load_models_config
from llmbench.models.bench import (
    SUMMARY_FIELDS,
    Target,
    append_results,
    decode_tokens_per_s,
    percentile,
    run_target,
    select_targets,
)
from llmbench.serving.llama_server import ServerStartError
from tests.fake_llama_server import FakeState, fake_llama_server

# ---------------------------------------------------------------- pure metrics


def test_decode_tokens_per_s_excludes_first_token() -> None:
    # 11 tokens: first at 0.5 s, last at 1.5 s -> 10 tokens in 1 s
    assert decode_tokens_per_s(11, 0.5, 1.5) == pytest.approx(10.0)


@pytest.mark.parametrize(("n", "ttft", "total"), [(1, 0.1, 1.0), (10, 1.0, 1.0)])
def test_decode_tokens_per_s_undefined_cases(n: int, ttft: float, total: float) -> None:
    assert math.isnan(decode_tokens_per_s(n, ttft, total))


def test_percentile() -> None:
    data = [1.0, 2.0, 3.0, 4.0, 5.0]
    assert percentile(data, 50) == 3.0
    assert percentile(data, 100) == 5.0
    assert percentile(data, 25) == 2.0
    assert percentile([7.0], 95) == 7.0
    assert math.isnan(percentile([], 50))
    assert percentile([math.nan, 4.0], 50) == 4.0


# ---------------------------------------------------------------- target selection


def test_select_all_targets_are_enabled_gguf() -> None:
    targets = select_targets(load_models_config(), None, None, None)
    assert targets and all(t.variant.is_gguf and t.variant.enabled for t in targets)
    mistral = [t for t in targets if t.model_id == "mistral-7b"]
    assert [t.n_gpu_layers for t in mistral] == [20]  # partial offload from models.yaml


def test_select_sweep_creates_one_target_per_layer_count() -> None:
    targets = select_targets(load_models_config(), "mistral-7b", "gguf_q4_k_m", [0, 8, 16])
    assert [t.n_gpu_layers for t in targets] == [0, 8, 16]


def test_select_transformers_variant_is_rejected_for_now() -> None:
    with pytest.raises(SystemExit, match="V2"):
        select_targets(load_models_config(), "qwen2.5-1.5b", "fp16", None)


# ---------------------------------------------------------------- end-to-end (fake server)


def _bench_cfg(tmp_path: Path) -> BenchConfig:
    real = load_bench_config()
    return real.model_copy(
        update={
            "runs": 2,
            "warmup_runs": 1,
            "max_new_tokens": 8,
            "server": real.server.model_copy(update={"models_dir": tmp_path}),
        }
    )


def _target() -> Target:
    variant = Variant(quant="gguf_q4_k_m", repo_id="org/repo", file="m.gguf")
    return Target("tiny", variant, None, 0.5)


def _fake_model(variant: Variant, models_dir: Path) -> Path:
    path = models_dir / "m.gguf"
    path.write_bytes(b"\0" * 1024 * 1024)  # 1 MiB
    return path


class _FakeServer:
    def __init__(self, base_url: str) -> None:
        self.base_url = base_url
        self.load_time_s = 1.25


def test_run_target_end_to_end(tmp_path: Path) -> None:
    cfg = _bench_cfg(tmp_path)
    state = FakeState([f"t{i}" for i in range(50)], delay_s=0.002)
    vram = itertools.chain([1000.0], itertools.repeat(2500.0))

    with fake_llama_server(state) as url:

        @contextmanager
        def server_factory(target: Target, c: BenchConfig) -> Iterator[_FakeServer]:
            yield _FakeServer(url)

        records, summary = run_target(
            _target(), cfg, session={"session_id": "s1"},
            ensure_model=_fake_model, server_factory=server_factory,
            vram_reader=lambda: next(vram), ram_reader=lambda: 800.0,
        )  # fmt: skip

    n_prompts = len(cfg.prompts)
    assert len(state.requests) == 1 + 2 * n_prompts  # 1 warmup + 2 runs x prompts
    assert all(r["cache_prompt"] is False for r in state.requests)
    assert len(records) == 2 * n_prompts
    assert all(r.gen_tokens == 8 and r.gen_tokens_source == "server_timings" for r in records)
    assert all(r.decode_tok_s > 0 for r in records)
    assert summary["n_runs"] == 2 * n_prompts
    assert summary["load_time_s"] == 1.25
    assert summary["file_size_mb"] == pytest.approx(1.0)
    assert summary["vram_baseline_mib"] == 1000.0
    assert summary["vram_model_mib"] == 1500.0
    assert summary["ram_peak_mib"] == 800.0
    assert summary["prefill_tok_s_p50"] == 500.0
    assert "error" not in summary


def test_run_target_reports_oom_instead_of_crashing(tmp_path: Path) -> None:
    @contextmanager
    def failing(target: Target, c: BenchConfig) -> Iterator[_FakeServer]:
        raise ServerStartError("exited\n--- logs ---\nCUDA error: out of memory")
        yield  # pragma: no cover

    records, summary = run_target(
        _target(), _bench_cfg(tmp_path), session={"session_id": "s2"},
        ensure_model=_fake_model, server_factory=failing,
    )  # fmt: skip
    assert records == []
    assert summary["error"] == "cuda_oom"
    assert summary["n_runs"] == 0


def test_append_results_writes_header_once(tmp_path: Path) -> None:
    summary = {"session_id": "s", "model": "m", "decode_tok_s_p50": math.nan, "n_runs": 0}
    append_results([], summary, tmp_path)
    append_results([], summary, tmp_path)
    rows = list(csv.DictReader((tmp_path / "summary.csv").open()))
    assert len(rows) == 2
    assert list(rows[0]) == SUMMARY_FIELDS
    assert rows[0]["decode_tok_s_p50"] == ""  # NaN written as empty, not "nan"


# ---------------------------------------------------------------- real model (local only)


@pytest.mark.slow
@pytest.mark.gpu
def test_real_tiny_model_benchmark(tmp_path: Path) -> None:
    """Downloads Qwen2.5-0.5B (~0.4 GB) and runs it in the real llama-server container."""
    if shutil.which("docker") is None:
        pytest.skip("docker not available")
    from llmbench.models.bench import main

    assert main(["--model", "qwen2.5-0.5b", "--runs", "1", "--results-dir", str(tmp_path)]) == 0
    row = next(csv.DictReader((tmp_path / "summary.csv").open()))
    assert row["error"] == "" and float(row["decode_tok_s_p50"]) > 0
    first = json.loads((tmp_path / "runs.jsonl").read_text().splitlines()[0])
    assert first["gen_tokens"] > 0


# ---------------------------------------------------------------- error handling


def test_cli_unknown_model_prints_clean_error(capsys: pytest.CaptureFixture[str]) -> None:
    from llmbench.models.bench import main

    assert main(["--model", "quen2.5-0.5b", "--dry-run"]) == 2
    err = capsys.readouterr().err
    assert err.startswith("ERROR: model 'quen2.5-0.5b' not configured")
    assert "Traceback" not in err


def test_docker_run_failure_is_not_recorded_as_a_result(tmp_path: Path) -> None:
    from llmbench.serving.llama_server import DockerRunError

    @contextmanager
    def docker_broken(target: Target, c: BenchConfig) -> Iterator[_FakeServer]:
        raise DockerRunError("docker run failed (exit 125): port is already allocated")
        yield  # pragma: no cover

    with pytest.raises(DockerRunError, match="port"):
        run_target(
            _target(), _bench_cfg(tmp_path), session={"session_id": "s3"},
            ensure_model=_fake_model, server_factory=docker_broken,
        )  # fmt: skip
