"""Tests for the configuration schema and the real configs/models.yaml."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
import yaml

from llmbench.config import (
    DEFAULT_MODELS_CONFIG,
    Backend,
    ConfigError,
    ModelsConfig,
    Quant,
    load_models_config,
)


def _minimal(**overrides: Any) -> dict[str, Any]:
    cfg: dict[str, Any] = {
        "models": {
            "tiny": {
                "hf_repo": "org/tiny",
                "family": "test",
                "params_b": 0.5,
                "variants": [
                    {"quant": "fp16"},
                    {"quant": "gguf_q4_k_m", "repo_id": "org/tiny-GGUF", "file": "tiny.gguf"},
                ],
            }
        }
    }
    cfg.update(overrides)
    return cfg


def _write(tmp_path: Path, data: Any) -> Path:
    path = tmp_path / "models.yaml"
    path.write_text(yaml.safe_dump(data), encoding="utf-8")
    return path


# ------------------------------------------------------------------ real config


def test_repository_models_yaml_is_valid() -> None:
    cfg = load_models_config(DEFAULT_MODELS_CONFIG)
    assert len(cfg.models) >= 2


def test_every_real_model_has_a_cpu_runnable_variant() -> None:
    """The project must stay usable without an NVIDIA GPU."""
    cfg = load_models_config()
    cpu_models = {m for m, _ in cfg.iter_variants(gpu_available=False)}
    assert cpu_models == set(cfg.models)


# ------------------------------------------------------------------ schema rules


def test_minimal_config_loads(tmp_path: Path) -> None:
    cfg = load_models_config(_write(tmp_path, _minimal()))
    assert cfg.get("tiny").variant("gguf_q4_k_m").backend is Backend.LLAMA_CPP
    assert cfg.generation.temperature == 0.0  # default applied


def test_unknown_key_is_rejected(tmp_path: Path) -> None:
    data = _minimal()
    data["models"]["tiny"]["temprature"] = 0.7  # typo must not be silently ignored
    with pytest.raises(ConfigError, match="temprature"):
        load_models_config(_write(tmp_path, data))


def test_out_of_range_generation_param_is_rejected(tmp_path: Path) -> None:
    with pytest.raises(ConfigError, match="temperature"):
        load_models_config(_write(tmp_path, _minimal(generation={"temperature": 5})))


def test_gguf_without_file_is_rejected(tmp_path: Path) -> None:
    data = _minimal()
    data["models"]["tiny"]["variants"] = [{"quant": "gguf_q8_0"}]
    with pytest.raises(ConfigError, match=r"\.gguf"):
        load_models_config(_write(tmp_path, data))


def test_hf_variant_with_file_is_rejected(tmp_path: Path) -> None:
    data = _minimal()
    data["models"]["tiny"]["variants"] = [{"quant": "fp16", "file": "x.gguf"}]
    with pytest.raises(ConfigError, match="remove 'file'"):
        load_models_config(_write(tmp_path, data))


def test_duplicate_quant_is_rejected(tmp_path: Path) -> None:
    data = _minimal()
    data["models"]["tiny"]["variants"] = [{"quant": "fp16"}, {"quant": "fp16"}]
    with pytest.raises(ConfigError, match="duplicate"):
        load_models_config(_write(tmp_path, data))


def test_missing_file_raises_config_error(tmp_path: Path) -> None:
    with pytest.raises(ConfigError, match="not found"):
        load_models_config(tmp_path / "nope.yaml")


def test_invalid_yaml_raises_config_error(tmp_path: Path) -> None:
    path = tmp_path / "models.yaml"
    path.write_text("models: [unclosed", encoding="utf-8")
    with pytest.raises(ConfigError, match="invalid YAML"):
        load_models_config(path)


# ------------------------------------------------------------------ helpers


def test_unknown_model_error_lists_available() -> None:
    cfg = ModelsConfig.model_validate(_minimal())
    with pytest.raises(KeyError, match="tiny"):
        cfg.get("gpt-5")


def test_cpu_filter_skips_gpu_only_quants() -> None:
    cfg = ModelsConfig.model_validate(_minimal())
    cpu = [v.quant for _, v in cfg.iter_variants(gpu_available=False)]
    assert cpu == [Quant.GGUF_Q4_K_M]
    gpu = [v.quant for _, v in cfg.iter_variants(gpu_available=True)]
    assert gpu == [Quant.FP16, Quant.GGUF_Q4_K_M]


def test_model_generation_overrides_global() -> None:
    data = _minimal()
    data["models"]["tiny"]["generation"] = {"max_new_tokens": 64}
    cfg = ModelsConfig.model_validate(data)
    assert cfg.generation_for("tiny").max_new_tokens == 64


def test_config_is_immutable() -> None:
    cfg = ModelsConfig.model_validate(_minimal())
    with pytest.raises(Exception, match="frozen"):
        cfg.generation.temperature = 1.0  # type: ignore[misc]


@pytest.mark.parametrize(
    ("quant", "backend", "gpu"),
    [
        (Quant.FP16, Backend.TRANSFORMERS, True),
        (Quant.BNB_4BIT, Backend.TRANSFORMERS, True),
        (Quant.AWQ_INT4, Backend.TRANSFORMERS, True),
        (Quant.GGUF_Q8_0, Backend.LLAMA_CPP, False),
    ],
)
def test_quant_properties(quant: Quant, backend: Backend, gpu: bool) -> None:
    assert quant.backend is backend
    assert quant.requires_nvidia_gpu is gpu
