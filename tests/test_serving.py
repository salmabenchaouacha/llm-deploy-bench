"""Tests for the streaming client and the llama-server container wrapper."""

from __future__ import annotations

from pathlib import Path

import pytest

from llmbench.config import LlamaServerConfig
from llmbench.serving.client import StreamError, count_tokens, stream_chat
from llmbench.serving.llama_server import (
    ALL_LAYERS,
    LlamaServer,
    ServerStartError,
    wait_for_health,
)
from tests.fake_llama_server import FakeState, fake_llama_server

WORDS = [f"w{i}" for i in range(20)]


def test_stream_chat_collects_text_timings_and_ttft() -> None:
    state = FakeState(WORDS, delay_s=0.005)
    with fake_llama_server(state) as url:
        res = stream_chat(url, [{"role": "user", "content": "hi"}], max_tokens=5)
    assert res.text == "w0 w1 w2 w3 w4 "
    assert res.n_chunks == 5
    assert 0 < res.ttft_s < res.total_s
    assert res.finish_reason == "length"
    assert res.server_timings is not None and res.server_timings["predicted_n"] == 5


def test_stream_chat_sends_benchmark_options() -> None:
    state = FakeState(WORDS)
    with fake_llama_server(state) as url:
        stream_chat(url, [{"role": "user", "content": "x"}], max_tokens=3,
                    ignore_eos=True, cache_prompt=False, seed=7)  # fmt: skip
    sent = state.requests[0]
    assert sent["stream"] is True
    assert sent["ignore_eos"] is True
    assert sent["cache_prompt"] is False
    assert sent["seed"] == 7
    assert sent["temperature"] == 0.0


def test_stream_chat_omits_llama_extensions_by_default() -> None:
    state = FakeState(WORDS)
    with fake_llama_server(state) as url:
        stream_chat(url, [{"role": "user", "content": "x"}], max_tokens=3)
    assert "ignore_eos" not in state.requests[0]
    assert "cache_prompt" not in state.requests[0]


def test_stream_chat_without_content_raises() -> None:
    with fake_llama_server(FakeState([])) as url, pytest.raises(StreamError):
        stream_chat(url, [{"role": "user", "content": "x"}], max_tokens=3)


def test_count_tokens_uses_tokenize_endpoint() -> None:
    with fake_llama_server(FakeState(WORDS)) as url:
        assert count_tokens(url, "a b c d") == 4


def test_wait_for_health_returns_when_ready() -> None:
    with fake_llama_server(FakeState(WORDS)) as url:
        assert wait_for_health(f"{url}/health", timeout_s=5) >= 0


def test_wait_for_health_detects_crashed_container() -> None:
    state = FakeState(WORDS)
    state.healthy = False
    with fake_llama_server(state) as url, pytest.raises(ServerStartError):
        wait_for_health(f"{url}/health", timeout_s=5, is_alive=lambda: False, poll_s=0.01)


def test_wait_for_health_times_out() -> None:
    state = FakeState(WORDS)
    state.healthy = False
    with fake_llama_server(state) as url, pytest.raises(TimeoutError):
        wait_for_health(f"{url}/health", timeout_s=0.05, poll_s=0.01)


def _server(n_gpu_layers: int | None, use_gpu: bool = True) -> LlamaServer:
    cfg = LlamaServerConfig(models_dir=Path("/models-host"), use_gpu=use_gpu, port=9001)
    return LlamaServer.from_config(cfg, Path("org__repo/m.gguf"), n_gpu_layers)


def test_docker_command_mounts_model_and_offloads_all_layers() -> None:
    cmd = _server(None).spec.docker_run_command()
    assert cmd[:3] == ["docker", "run", "-d"]
    assert "--gpus" in cmd
    assert "/models-host:/models:ro" in cmd
    assert cmd[cmd.index("--model") + 1] == "/models/org__repo/m.gguf"
    assert cmd[cmd.index("--n-gpu-layers") + 1] == str(ALL_LAYERS)
    assert "127.0.0.1:9001:8080" in cmd


def test_docker_command_partial_offload_and_cpu() -> None:
    cmd = _server(20, use_gpu=False).spec.docker_run_command()
    assert cmd[cmd.index("--n-gpu-layers") + 1] == "20"
    assert "--gpus" not in cmd
