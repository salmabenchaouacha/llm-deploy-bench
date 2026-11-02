"""Peak memory sampling (GPU VRAM via NVML, container RAM via ``docker stats``).

A background thread polls a ``reader`` (any ``() -> MiB`` function) and keeps the max.
Readers are injected, so the sampler is unit-tested without a GPU.

Note on VRAM: NVML cannot list per-process usage under WSL2, so we read the *device*
memory used and subtract a baseline taken before the model is loaded. Close other GPU
apps during a benchmark.
"""

from __future__ import annotations

import re
import subprocess
import threading
from collections.abc import Callable
from types import TracebackType
from typing import Any, Self

from llmbench.log import get_logger

log = get_logger(__name__)

Reader = Callable[[], float]


class PeakSampler:
    """Context manager: samples ``reader`` every ``interval_s`` and records the peak."""

    def __init__(self, reader: Reader, interval_s: float = 0.25, name: str = "mem") -> None:
        self._reader = reader
        self._interval = interval_s
        self._name = name
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self.peak: float = float("nan")
        self.samples = 0
        self.errors = 0
        self._stopped = False

    def _sample_once(self) -> None:
        try:
            value = self._reader()
        except Exception:  # a failing reader must never kill the benchmark
            self.errors += 1
            return
        self.samples += 1
        if not self.peak >= value:  # also true when peak is NaN
            self.peak = value

    def _run(self) -> None:
        while not self._stop.is_set():
            self._sample_once()
            self._stop.wait(self._interval)

    def start(self) -> None:
        self._thread = threading.Thread(target=self._run, name=f"sampler-{self._name}")
        self._thread.daemon = True
        self._thread.start()

    def stop(self) -> float:
        """Stop sampling and return the peak. Idempotent: later calls return the same peak."""
        if self._stopped:
            return self.peak
        self._stopped = True
        self._stop.set()
        if self._thread is not None:
            self._thread.join()
        self._sample_once()  # final reading, so short runs still get a value
        if self.errors:
            log.warning("sampler errors", extra={"sampler": self._name, "errors": self.errors})
        return self.peak

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


# ---------------------------------------------------------------- GPU (NVML)


def _nvml() -> Any:
    import pynvml  # provided by the nvidia-ml-py package

    pynvml.nvmlInit()
    return pynvml


def nvml_used_mib_reader(index: int = 0) -> Reader:
    nv = _nvml()
    handle = nv.nvmlDeviceGetHandleByIndex(index)

    def read() -> float:
        return float(nv.nvmlDeviceGetMemoryInfo(handle).used) / 1024**2

    return read


def gpu_info(index: int = 0) -> dict[str, Any]:
    """Name, driver and total memory of the GPU, or ``{}`` when NVML is unavailable."""
    try:
        nv = _nvml()
        handle = nv.nvmlDeviceGetHandleByIndex(index)
        name = nv.nvmlDeviceGetName(handle)
        driver = nv.nvmlSystemGetDriverVersion()
        return {
            "gpu_name": name.decode() if isinstance(name, bytes) else name,
            "gpu_driver": driver.decode() if isinstance(driver, bytes) else driver,
            "gpu_total_mib": round(nv.nvmlDeviceGetMemoryInfo(handle).total / 1024**2),
        }
    except Exception as exc:
        log.warning("NVML unavailable", extra={"error": str(exc)})
        return {}


# ---------------------------------------------------------------- container RAM

_UNITS = {"b": 1 / 1024**2, "kib": 1 / 1024, "kb": 1 / 1024, "mib": 1.0, "mb": 1.0,
          "gib": 1024.0, "gb": 1024.0, "tib": 1024.0**2}  # fmt: skip
_MEM_RE = re.compile(r"^\s*([\d.]+)\s*([a-zA-Z]+)")


def parse_docker_mem(mem_usage: str) -> float:
    """``'1.234GiB / 15.5GiB'`` -> 1263.6 (MiB, the used part only)."""
    match = _MEM_RE.match(mem_usage)
    if not match or match.group(2).lower() not in _UNITS:
        raise ValueError(f"cannot parse docker memory usage: {mem_usage!r}")
    return float(match.group(1)) * _UNITS[match.group(2).lower()]


def docker_mem_mib_reader(container: str) -> Reader:
    def read() -> float:
        out = subprocess.run(
            ["docker", "stats", "--no-stream", "--format", "{{.MemUsage}}", container],
            capture_output=True,
            text=True,
            check=True,
            timeout=10,
        ).stdout
        return parse_docker_mem(out)

    return read
