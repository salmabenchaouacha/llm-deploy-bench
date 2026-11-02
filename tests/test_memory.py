"""Tests for the peak memory sampler and docker stats parsing."""

from __future__ import annotations

import itertools
import math

import pytest

from llmbench.monitoring.memory import PeakSampler, parse_docker_mem


def test_sampler_records_peak() -> None:
    values = itertools.chain([100.0, 350.0, 200.0], itertools.repeat(150.0))
    with PeakSampler(lambda: next(values), interval_s=0.001) as sampler:
        while sampler.samples < 4:
            pass
    assert sampler.peak == 350.0


def test_sampler_survives_failing_reader() -> None:
    def boom() -> float:
        raise RuntimeError("nvml gone")

    sampler = PeakSampler(boom, interval_s=0.001)
    sampler.start()
    assert math.isnan(sampler.stop())
    assert sampler.errors >= 1


def test_sampler_stop_takes_final_reading() -> None:
    sampler = PeakSampler(lambda: 42.0, interval_s=10)
    assert sampler.stop() == 42.0  # never started: still one reading


@pytest.mark.parametrize(
    ("raw", "mib"),
    [
        ("1.5GiB / 15.5GiB", 1536.0),
        ("512MiB / 8GiB", 512.0),
        ("2048KiB / 1GiB", 2.0),
        ("  3GB / 8GB\n", 3072.0),
    ],
)
def test_parse_docker_mem(raw: str, mib: float) -> None:
    assert parse_docker_mem(raw) == pytest.approx(mib)


def test_parse_docker_mem_rejects_garbage() -> None:
    with pytest.raises(ValueError):
        parse_docker_mem("--")
