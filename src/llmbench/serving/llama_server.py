"""Start / stop the official llama.cpp server (``llama-server``) in a Docker container.

Why a container instead of the ``llama-cpp-python`` bindings:

* no CUDA toolkit nor C++ compilation in WSL: the image ships a CUDA build;
* the image tag pins the exact llama.cpp version -> reproducible measurements;
* it is the same engine the serving layer (Phase 4) talks to, so we benchmark the
  stack we actually deploy, through its OpenAI-compatible HTTP API.

Usage::

    with LlamaServer.from_config(server_cfg, model_rel_path, n_gpu_layers=None) as srv:
        ...  # srv.base_url -> "http://127.0.0.1:8080"
"""

from __future__ import annotations

import subprocess
import time
import urllib.error
import urllib.request
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from types import TracebackType
from typing import Self

from llmbench.config import LlamaServerConfig
from llmbench.log import get_logger

log = get_logger(__name__)

ALL_LAYERS = 999  # llama.cpp clamps -ngl to the real layer count


class ServerStartError(RuntimeError):
    """The container exited or never became healthy. Message includes its logs."""


class DockerRunError(RuntimeError):
    """``docker run`` itself failed (port taken, bad mount, GPU not exposed...).

    An environment problem, not a property of the model: callers should stop, not
    record it as a benchmark result.
    """


@dataclass(frozen=True)
class LlamaServerSpec:
    image: str
    models_dir: Path
    model_relpath: Path
    n_gpu_layers: int | None
    n_ctx: int
    host: str
    port: int
    container_name: str
    use_gpu: bool = True

    @property
    def base_url(self) -> str:
        return f"http://{self.host}:{self.port}"

    def docker_run_command(self) -> list[str]:
        ngl = ALL_LAYERS if self.n_gpu_layers is None else self.n_gpu_layers
        cmd = ["docker", "run", "-d", "--name", self.container_name]
        if self.use_gpu:
            cmd += ["--gpus", "all"]
        cmd += [
            "-p", f"{self.host}:{self.port}:8080",
            "-v", f"{self.models_dir}:/models:ro",
            self.image,
            "--model", f"/models/{self.model_relpath.as_posix()}",
            "--host", "0.0.0.0",
            "--port", "8080",
            "--ctx-size", str(self.n_ctx),
            "--n-gpu-layers", str(ngl),
            "--parallel", "1",
            "--metrics",
        ]  # fmt: skip
        return cmd


def http_ok(url: str, timeout: float = 2.0) -> bool:
    """True if GET ``url`` answers 200 (llama-server's /health is 503 while loading)."""
    try:
        with urllib.request.urlopen(url, timeout=timeout) as resp:
            return bool(resp.status == 200)
    except (urllib.error.URLError, ConnectionError, TimeoutError, OSError):
        return False


def wait_for_health(
    url: str,
    timeout_s: float,
    *,
    is_alive: Callable[[], bool] = lambda: True,
    poll_s: float = 0.5,
    clock: Callable[[], float] = time.monotonic,
) -> float:
    """Poll ``url`` until healthy. Returns the waiting time in seconds.

    Raises ``ServerStartError`` if ``is_alive()`` turns False (container crashed, e.g.
    CUDA out of memory) or ``TimeoutError`` after ``timeout_s``.
    """
    start = clock()
    while True:
        if http_ok(url):
            return clock() - start
        if not is_alive():
            raise ServerStartError("server process exited before becoming healthy")
        if clock() - start > timeout_s:
            raise TimeoutError(f"{url} not healthy after {timeout_s:.0f}s")
        time.sleep(poll_s)


def _docker(*args: str, check: bool = True) -> subprocess.CompletedProcess[str]:
    return subprocess.run(["docker", *args], capture_output=True, text=True, check=check)


class LlamaServer:
    """Context manager owning one llama-server container."""

    def __init__(self, spec: LlamaServerSpec, startup_timeout_s: float = 300.0) -> None:
        self.spec = spec
        self.startup_timeout_s = startup_timeout_s
        self.load_time_s: float | None = None

    @classmethod
    def from_config(
        cls, cfg: LlamaServerConfig, model_relpath: Path, n_gpu_layers: int | None
    ) -> Self:
        spec = LlamaServerSpec(
            image=cfg.image,
            models_dir=cfg.models_path,
            model_relpath=model_relpath,
            n_gpu_layers=n_gpu_layers,
            n_ctx=cfg.n_ctx,
            host=cfg.host,
            port=cfg.port,
            container_name=cfg.container_name,
            use_gpu=cfg.use_gpu,
        )
        return cls(spec, cfg.startup_timeout_s)

    @property
    def base_url(self) -> str:
        return self.spec.base_url

    def _is_running(self) -> bool:
        res = _docker("inspect", "-f", "{{.State.Running}}", self.spec.container_name, check=False)
        return res.stdout.strip() == "true"

    def logs(self, tail: int = 40) -> str:
        res = _docker("logs", "--tail", str(tail), self.spec.container_name, check=False)
        return (res.stdout + res.stderr).strip()

    def start(self) -> None:
        _docker("rm", "-f", self.spec.container_name, check=False)  # leftover from a crash
        cmd = self.spec.docker_run_command()
        log.info("starting llama-server", extra={"cmd": " ".join(cmd)})
        run = subprocess.run(cmd, capture_output=True, text=True, check=False)
        if run.returncode != 0:  # docker itself failed: port taken, bad mount, no GPU...
            self.stop()
            raise DockerRunError(
                f"docker run failed (exit {run.returncode}): {run.stderr.strip() or run.stdout}"
            )
        try:
            self.load_time_s = wait_for_health(
                f"{self.base_url}/health", self.startup_timeout_s, is_alive=self._is_running
            )
        except (ServerStartError, TimeoutError) as exc:
            logs = self.logs()
            self.stop()
            raise ServerStartError(f"{exc}\n--- container logs ---\n{logs}") from exc
        log.info("llama-server ready", extra={"load_time_s": round(self.load_time_s, 2)})

    def stop(self) -> None:
        _docker("rm", "-f", self.spec.container_name, check=False)

    def __enter__(self) -> Self:
        self.start()
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        self.stop()
