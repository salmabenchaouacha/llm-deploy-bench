"""Centralised, validated configuration.

Every YAML file in ``configs/`` is parsed into a Pydantic model before any code uses it.
Typos (unknown keys), impossible values (``temperature: 5``) and inconsistent combinations
(a GGUF quantization without a ``.gguf`` file) fail *immediately* with a clear message,
instead of crashing ten minutes into a model download.

Run ``python -m llmbench.config`` to validate ``configs/models.yaml`` and print a summary.
"""

from __future__ import annotations

import sys
from enum import StrEnum
from pathlib import Path
from typing import Any, Self

import yaml
from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CONFIG_DIR = PROJECT_ROOT / "configs"
DATA_DIR = PROJECT_ROOT / "data"
RESULTS_DIR = PROJECT_ROOT / "results"
DEFAULT_MODELS_CONFIG = CONFIG_DIR / "models.yaml"


class ConfigError(Exception):
    """Raised when a configuration file is missing or invalid."""


class Backend(StrEnum):
    """Inference engine used to load a model variant."""

    TRANSFORMERS = "transformers"  # Hugging Face, PyTorch weights (GPU strongly preferred)
    LLAMA_CPP = "llama_cpp"  # llama-cpp-python, GGUF weights (CPU or GPU)


class Quant(StrEnum):
    """Weight format / quantization scheme of a model variant."""

    FP16 = "fp16"
    BNB_8BIT = "bnb_8bit"
    BNB_4BIT = "bnb_4bit"  # NF4, the scheme used by QLoRA
    GPTQ_INT4 = "gptq_int4"
    AWQ_INT4 = "awq_int4"
    GGUF_Q4_K_M = "gguf_q4_k_m"
    GGUF_Q5_K_M = "gguf_q5_k_m"
    GGUF_Q8_0 = "gguf_q8_0"

    @property
    def backend(self) -> Backend:
        """The only backend able to run this format."""
        return Backend.LLAMA_CPP if self.value.startswith("gguf") else Backend.TRANSFORMERS

    @property
    def requires_nvidia_gpu(self) -> bool:
        """True for every Transformers format.

        bitsandbytes, GPTQ and AWQ kernels are CUDA-only. FP16 *can* run on CPU, but
        PyTorch CPU kernels for half precision are slow and partial, so benchmarking it
        there would be misleading: on CPU we use GGUF instead.
        """
        return self.backend is Backend.TRANSFORMERS


class StrictModel(BaseModel):
    """Base model: unknown keys are errors, instances are immutable."""

    model_config = ConfigDict(extra="forbid", frozen=True)


class GenerationParams(StrictModel):
    """Decoding parameters. Kept identical across models for a fair comparison."""

    temperature: float = Field(0.0, ge=0.0, le=2.0)
    top_p: float = Field(1.0, gt=0.0, le=1.0)
    max_new_tokens: int = Field(256, gt=0, le=8192)
    repetition_penalty: float = Field(1.0, ge=1.0, le=2.0)
    seed: int = 42


class Variant(StrictModel):
    """One concrete, loadable artefact of a model (e.g. Qwen2.5-1.5B in GGUF Q4_K_M)."""

    quant: Quant
    repo_id: str | None = Field(
        None, description="Hugging Face repo of this variant; defaults to the model's hf_repo."
    )
    file: str | None = Field(None, description="File inside repo_id (required for GGUF).")
    enabled: bool = True

    @property
    def backend(self) -> Backend:
        return self.quant.backend

    @model_validator(mode="after")
    def _check_file(self) -> Self:
        is_gguf = self.quant.backend is Backend.LLAMA_CPP
        if is_gguf and (self.file is None or not self.file.endswith(".gguf")):
            raise ValueError(f"quant '{self.quant}' needs a 'file' ending in .gguf")
        if not is_gguf and self.file is not None:
            raise ValueError(f"quant '{self.quant}' loads a whole HF repo: remove 'file'")
        return self


class ModelSpec(StrictModel):
    """A model family member and all the variants we benchmark."""

    hf_repo: str = Field(..., pattern=r"^[\w.-]+/[\w.-]+$")
    family: str
    params_b: float = Field(..., gt=0, description="Parameter count, in billions.")
    context_length: int = Field(4096, gt=0)
    gated: bool = Field(False, description="True if the HF repo requires accepting a licence.")
    variants: list[Variant] = Field(..., min_length=1)
    generation: GenerationParams | None = None

    @model_validator(mode="after")
    def _unique_quants(self) -> Self:
        quants = [v.quant for v in self.variants]
        duplicates = {q.value for q in quants if quants.count(q) > 1}
        if duplicates:
            raise ValueError(f"duplicate quant(s) in variants: {sorted(duplicates)}")
        return self

    def variant(self, quant: Quant | str) -> Variant:
        quant = Quant(quant)
        for v in self.variants:
            if v.quant is quant:
                return v
        available = [v.quant.value for v in self.variants]
        raise KeyError(f"quant '{quant}' not configured; available: {available}")


class ModelsConfig(StrictModel):
    """Root of ``configs/models.yaml``."""

    generation: GenerationParams = GenerationParams()
    models: dict[str, ModelSpec] = Field(..., min_length=1)

    def get(self, model_id: str) -> ModelSpec:
        try:
            return self.models[model_id]
        except KeyError:
            raise KeyError(
                f"model '{model_id}' not configured; available: {sorted(self.models)}"
            ) from None

    def generation_for(self, model_id: str) -> GenerationParams:
        """Model-specific generation params, falling back to the global defaults."""
        return self.get(model_id).generation or self.generation

    def iter_variants(
        self, *, gpu_available: bool, include_disabled: bool = False
    ) -> list[tuple[str, Variant]]:
        """All (model_id, variant) pairs runnable on the current hardware."""
        return [
            (model_id, v)
            for model_id, spec in self.models.items()
            for v in spec.variants
            if include_disabled or v.enabled
            if gpu_available or not v.quant.requires_nvidia_gpu
        ]


def _read_yaml(path: Path) -> Any:
    if not path.is_file():
        raise ConfigError(f"config file not found: {path}")
    try:
        with path.open(encoding="utf-8") as fh:
            return yaml.safe_load(fh)
    except yaml.YAMLError as exc:
        raise ConfigError(f"invalid YAML in {path}: {exc}") from exc


def load_models_config(path: Path | str = DEFAULT_MODELS_CONFIG) -> ModelsConfig:
    """Load and validate ``models.yaml``. Raises :class:`ConfigError` on any problem."""
    path = Path(path)
    try:
        return ModelsConfig.model_validate(_read_yaml(path))
    except ValidationError as exc:
        raise ConfigError(f"invalid config {path}:\n{exc}") from exc


def main() -> int:
    """CLI: validate the models config and print one line per variant."""
    try:
        cfg = load_models_config(sys.argv[1] if len(sys.argv) > 1 else DEFAULT_MODELS_CONFIG)
    except ConfigError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    for model_id, v in cfg.iter_variants(gpu_available=True, include_disabled=True):
        flags = ("GPU" if v.quant.requires_nvidia_gpu else "CPU") + ("" if v.enabled else " off")
        print(f"{model_id:<22} {v.quant.value:<12} {v.backend.value:<13} [{flags}]")
    print(f"OK: {len(cfg.models)} models validated")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
